/*!
 * \file inject_register_pipeline_sched_barrier.cc
 * \brief Wrap MMA clusters with sched_barrier when register pipeline is on.
 */

#include "common/gemm_k_loop_utils.h"
#include "common/pipeline_utils.h"
#include "hcu/target_utils.h"
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

// llvm.amdgcn.sched.barrier mask: 0 leaves no instruction type allowed to be
// reordered across the barrier, i.e. the strongest barrier (the templates in
// tl_templates/hcu call sched_barrier(0) around s_barrier the same way).
constexpr int kAmdgcnSchedBarrierFull = 0;

bool FuncHasRegisterPipeline(const Stmt &root) {
  for (const ForNode *loop : CollectGemmKLoops(root)) {
    if (LoopHasRegisterPipeline(loop)) {
      return true;
    }
  }
  return false;
}

Stmt MakeSchedBarrierStmt() {
  return Evaluate(Call(DataType::Void(), builtin::call_extern(),
                       {StringImm("__builtin_amdgcn_sched_barrier"),
                        IntImm(DataType::Int(32),
                               kAmdgcnSchedBarrierFull)}));
}

class SchedBarrierMutator : public StmtExprMutator {
public:
  Stmt VisitStmt_(const SeqStmtNode *op) override {
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
  int in_mma_stmt_{0};
};

} // namespace

namespace transform {

tirx::transform::Pass InjectRegisterPipelineSchedBarrier() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, IRModule, PassContext) {
    Target target = f->GetAttr<Target>(tvm::attr::kTarget).value_or(Target());
    if (!target.defined() || !TargetIsHCU(target)) {
      return f;
    }
    auto *n = f.CopyOnWrite();
    if (!FuncHasRegisterPipeline(n->body)) {
      return f;
    }
    SchedBarrierMutator mutator;
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
