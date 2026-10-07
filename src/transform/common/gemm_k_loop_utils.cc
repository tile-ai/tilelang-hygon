#include "gemm_k_loop_utils.h"

#include "op/builtin.h"
#include "op/utils.h"

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace tl {

using namespace tirx;

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
  return call->op.same_as(tvm_mfma()) || call->op.same_as(tvm_mfma_store()) ||
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

} // namespace tl
} // namespace tvm
