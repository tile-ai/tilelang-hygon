/*!
 * \file inject_register_pipeline_sched_barrier.cc
 * \brief Wrap MMA clusters with sched_barrier when register pipeline is on.
 */

#include "common/gemm_k_loop_utils.h"
#include "common/pipeline_utils.h"
#include "op/builtin.h"
#include "op/utils.h"

#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {
using namespace tirx;
using ffi::Array;
using ffi::GetRef;

namespace {

constexpr int kAmdgcnSchedBarrierNoRestriction = 0;

bool FuncHasRegisterPipeline(const Stmt &root) {
  for (const ForNode *loop : CollectGemmKLoops(root)) {
    if (LoopHasRegisterPipeline(loop)) {
      return true;
    }
  }
  return false;
}

bool StmtContainsAsyncCopy(const Stmt &stmt) {
  struct Checker : public StmtExprVisitor {
    bool found{false};
    void VisitExpr_(const CallNode *op) override {
      if (IsAsyncCopyCall(op)) {
        found = true;
        return;
      }
      ExprVisitor::VisitExpr_(op);
    }
  };
  Checker checker;
  checker(stmt);
  return checker.found;
}

bool IsMmaCluster(const Stmt &stmt) {
  if (!StmtContainsMma(stmt)) {
    return false;
  }
  if (const auto *for_op = stmt.as<ForNode>()) {
    if (IsGemmKLoop(for_op)) {
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

Stmt MakeSchedBarrierStmt() {
  return Evaluate(Call(DataType::Void(), builtin::call_extern(),
                       {StringImm("__builtin_amdgcn_sched_barrier"),
                        IntImm(DataType::Int(32),
                               kAmdgcnSchedBarrierNoRestriction)}));
}

class SchedBarrierMutator : public StmtExprMutator {
public:
  explicit SchedBarrierMutator(bool enabled) : enabled_(enabled) {}

  Stmt VisitStmt_(const SeqStmtNode *op) override {
    if (!enabled_) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Stmt flattened = SeqStmt::Flatten(GetRef<Stmt>(op));
    const auto *seq_op = flattened.as<SeqStmtNode>();
    if (seq_op == nullptr) {
      return VisitStmt(flattened);
    }

    Array<Stmt> new_seq;
    for (const Stmt &stmt : seq_op->seq) {
      if (IsMmaCluster(stmt)) {
        if (in_mma_stmt_ > 0) {
          new_seq.push_back(VisitStmt(stmt));
          continue;
        }
        ++in_mma_stmt_;
        if (new_seq.empty() || !IsSchedBarrierStmt(new_seq.back())) {
          new_seq.push_back(MakeSchedBarrierStmt());
        }
        new_seq.push_back(VisitStmt(stmt));
        new_seq.push_back(MakeSchedBarrierStmt());
        --in_mma_stmt_;
      } else {
        new_seq.push_back(VisitStmt(stmt));
      }
    }
    if (new_seq.size() == 1) {
      return new_seq[0];
    }
    return SeqStmt(new_seq);
  }

private:
  bool enabled_{false};
  int in_mma_stmt_{0};
};

} // namespace

namespace transform {

tirx::transform::Pass InjectRegisterPipelineSchedBarrier() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, IRModule, PassContext) {
    auto *n = f.CopyOnWrite();
    const bool enabled = FuncHasRegisterPipeline(n->body);
    if (!enabled) {
      return f;
    }
    SchedBarrierMutator mutator(enabled);
    n->body = mutator(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectRegisterPipelineSchedBarrier",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectRegisterPipelineSchedBarrier",
                        InjectRegisterPipelineSchedBarrier);
}

} // namespace transform
} // namespace tl
} // namespace tvm
