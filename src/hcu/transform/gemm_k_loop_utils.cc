#include "hcu/transform/gemm_k_loop_utils.h"

#include "hcu/op/builtin.h"
#include "hcu/utils/extern_call_checker.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "rocm/op/builtin.h"
#include "transform/common/pipeline_utils.h"

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

bool IsAsyncCopyCall(const CallNode *call) {
  if (!call) {
    return false;
  }
  return call->op.same_as(hcu_cp_async_idxen()) ||
         call->op.same_as(ptx_cp_async());
}

bool IsMmaCall(const CallNode *call) {
  if (!call) {
    return false;
  }
  // The HCU GEMM lowering emits tl.hcu_mmac / tl.hcu_wmma; the ROCm backend
  // emits tl.tvm_mfma / tl.tvm_rdna_wmma through the same tile-op path.
  return call->op.same_as(hcu_mmac()) || call->op.same_as(hcu_mmac_store()) ||
         call->op.same_as(hcu_wmma()) || call->op.same_as(hcu_wmma_store()) ||
         call->op.same_as(tvm_mfma()) || call->op.same_as(tvm_mfma_store()) ||
         call->op.same_as(tvm_rdna_wmma());
}

bool StmtContainsMma(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&found](const ObjectRef &node) {
    if (const auto *call = node.as<CallNode>()) {
      if (IsMmaCall(call)) {
        found = true;
      }
    }
  });
  return found;
}

bool StmtContainsAsyncCopy(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&found](const ObjectRef &node) {
    if (const auto *call = node.as<CallNode>()) {
      if (IsAsyncCopyCall(call)) {
        found = true;
      }
    }
  });
  return found;
}

bool StmtContainsMlsAsyncLoad(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&found](const ObjectRef &node) {
    if (const auto *call = node.as<CallNode>()) {
      if (IsMlsAsyncLoadExternCall(call)) {
        found = true;
      }
    }
  });
  return found;
}

// Compute-only statement: the surrounding K loop and anything that issues an
// async copy belong to the producer side and are not an MMA cluster.
bool IsMmaCluster(const Stmt &stmt) {
  if (!StmtContainsMma(stmt)) {
    return false;
  }
  if (const auto *loop = stmt.as<ForNode>()) {
    if (IsGemmKLoop(loop)) {
      return false;
    }
  }
  return !StmtContainsAsyncCopy(stmt);
}

bool IsSchedBarrierStmt(const Stmt &stmt) {
  const auto *eval = stmt.as<EvaluateNode>();
  if (eval == nullptr) {
    return false;
  }
  const auto *call = eval->value.as<CallNode>();
  if (call == nullptr || !call->op.same_as(builtin::call_extern()) ||
      call->args.empty()) {
    return false;
  }
  if (const auto *s = call->args[0].as<StringImmNode>()) {
    return s->value == "__builtin_amdgcn_sched_barrier";
  }
  return false;
}

std::optional<int64_t> GetConstIntValue(const PrimExpr &expr) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    return imm->value;
  }
  return std::nullopt;
}

namespace {

const BufferLoadNode *PeelBufferLoad(const PrimExpr &value) {
  if (const auto *load = value.as<BufferLoadNode>()) {
    return load;
  }
  if (const auto *cast = value.as<CastNode>()) {
    return cast->value.as<BufferLoadNode>();
  }
  return nullptr;
}

class FeatureChecker : public StmtExprVisitor {
public:
  bool has_mma{false};
  bool has_memory_access{false};
  bool has_global_src{false};

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    has_memory_access = true;
    if (IsGlobalBuffer(op->buffer)) {
      has_global_src = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    has_memory_access = true;
    if (const auto *load = PeelBufferLoad(op->value)) {
      if (IsGlobalBuffer(load->buffer)) {
        has_global_src = true;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (IsMmaCall(op)) {
      has_mma = true;
    }
    if (IsAsyncCopyCall(op)) {
      has_memory_access = true;
      has_global_src = true;
    }
    if (op->op.same_as(ds_read_vector())) {
      has_memory_access = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

class LoopCollector : public StmtVisitor {
public:
  std::vector<const ForNode *> loops;

private:
  void VisitStmt_(const ForNode *op) final {
    if (IsGemmKLoop(op)) {
      loops.push_back(op);
    }
    StmtVisitor::VisitStmt_(op);
  }
};

bool StmtContainsAsyncCommit(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&found](const ObjectRef &node) {
    if (const auto *call = node.as<CallNode>()) {
      if (call->op.same_as(builtin::ptx_commit_group())) {
        found = true;
      }
    }
  });
  return found;
}

} // namespace

GemmKLoopFeatures AnalyzeGemmKLoopBody(const Stmt &body) {
  FeatureChecker checker;
  checker(body);
  return {checker.has_mma, checker.has_memory_access, checker.has_global_src};
}

bool IsGemmKLoop(const ForNode *loop) {
  if (loop == nullptr) {
    return false;
  }
  GemmKLoopFeatures features = AnalyzeGemmKLoopBody(loop->body);
  return features.has_mma && features.has_memory_access &&
         features.has_global_src;
}

std::vector<const ForNode *> CollectGemmKLoops(const Stmt &stmt) {
  LoopCollector collector;
  collector(stmt);
  return collector.loops;
}

bool FuncHasRegisterPipeline(const Stmt &stmt) {
  for (const ForNode *loop : CollectGemmKLoops(stmt)) {
    if (LoopHasAppliedRegisterPipeline(loop)) {
      return true;
    }
  }
  return false;
}

const ForNode *FindOwnedWaitPlanKLoop(const Stmt &stmt) {
  std::vector<const ForNode *> loops = CollectGemmKLoops(stmt);
  if (loops.size() != 1) {
    return nullptr;
  }
  const ForNode *k_loop = loops.front();
  // The plan derives its wait counts from the commit groups it can count in
  // the K loop body, which holds only while every producer it accounts for is
  // one it recognized. A matrix-load producer commits its DMA into the same
  // group counter but is issued by its own lowering, so a K loop carrying one
  // has no wait plan this pass can author: decline ownership and let the
  // in-place lowering of the commit/wait attrs stand.
  if (StmtContainsMlsAsyncLoad(k_loop->body)) {
    return nullptr;
  }
  if (!StmtContainsAsyncCopy(k_loop->body) &&
      !StmtContainsAsyncCommit(k_loop->body)) {
    return nullptr;
  }
  if (LoopHasAppliedRegisterPipeline(k_loop)) {
    return k_loop;
  }
  if (auto ns = GetExplicitPipelinedNumStages(k_loop)) {
    if (static_cast<int>(ns.value()->value) >= 2) {
      return k_loop;
    }
  }
  return nullptr;
}

} // namespace tl
} // namespace tvm
