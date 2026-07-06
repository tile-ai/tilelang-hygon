/*!
 * \file annotate_mls_gemm_dep.cc
 * \brief Pre-layout pass: annotate matrix_load / ds_read_format / gemm with MLS
 * consumer GEMM facts collected via PropagationTirCollector.
 */

#include "../op/builtin.h"
#include "../op/ds_read_format.h"
#include "../op/gemm.h"
#include "../op/mls.h"
#include "../op/mls_gemm_dep.h"
#include "../op/operator.h"
#include "../op/propagation_tir_collector.h"
#include "../op/propagation_util.h"
#include "../target/utils.h"

#include <tvm/ir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

Map<Var, Buffer> BuildBufferMap(const PrimFunc &f) {
  Map<Var, Buffer> buffer_map;
  for (const auto &[var, buffer] : f->buffer_map) {
    buffer_map.Set(var, buffer);
  }
  return buffer_map;
}

/*! \brief Fast scan for tile-level MLS ops; avoids PropagationTirCollector on
 * non-MLS kernels. */
class MlsTileOpPresenceChecker : public StmtExprVisitor {
public:
  bool found() const { return found_; }

private:
  void VisitStmt(const Stmt &stmt) final {
    if (found_) {
      return;
    }
    StmtExprVisitor::VisitStmt(stmt);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (found_) {
      return;
    }
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.as<OpNode>()) {
        Op tir_op = Downcast<Op>(call->op);
        if (tir_op == MatrixLoad::Get() || tir_op == DsReadFormat::Get()) {
          found_ = true;
          return;
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool found_{false};
};

bool ContainsMlsTileOps(const Stmt &body) {
  MlsTileOpPresenceChecker checker;
  checker(body);
  return checker.found();
}

} // namespace

class AnnotateMlsGemmDepMutator : public StmtExprMutator {
public:
  AnnotateMlsGemmDepMutator(PropagationTirCollector *collector, Target target)
      : collector_(collector), target_(std::move(target)) {}

  static PrimFunc Substitute(PrimFunc f) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsHCU(target.value())) {
      return f;
    }
    if (!ContainsMlsTileOps(f->body)) {
      return f;
    }
    Map<Var, Buffer> buffer_map = BuildBufferMap(f);
    PropagationTirCollector collector(buffer_map);
    collector.Collect(f->body);
    AnnotateMlsGemmDepMutator mutator(&collector, target.value());
    PrimFuncNode *fn = f.CopyOnWrite();
    fn->body = mutator(f->body);
    return f;
  }

private:
  bool LookupSharedMlsTrans(const Buffer &dst, bool *out_trans) {
    auto it = shared_mls_trans_.find(dst);
    if (it != shared_mls_trans_.end()) {
      *out_trans = it->second;
      return true;
    }
    bool trans = true;
    ComputeSharedDstMlsTrans(dst, collector_, &trans);
    shared_mls_trans_[dst] = trans;
    *out_trans = trans;
    return true;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    auto call = op->value.as<CallNode>();
    if (call == nullptr || !call->op.as<OpNode>()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Op tir_op = Downcast<Op>(call->op);

    if (tir_op == MatrixLoad::Get()) {
      auto mls =
          Downcast<MatrixLoad>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
      bool trans = true;
      LookupSharedMlsTrans(mls->dst, &trans);
      auto annotations = call->annotations;
      annotations.Set(attr::kMlsTrans,
                      IntImm(DataType::Int(32), trans ? 1 : 0));
      Call new_call(call->dtype, call->op, call->args, annotations, call->span);
      return Evaluate(new_call);
    }

    if (tir_op == DsReadFormat::Get()) {
      auto ds =
          Downcast<DsReadFormat>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
      auto gemm_with_input = PropagateToFindGemmConsumerOpWithInputAfterCall(
          ds->dst, collector_, call);
      if (gemm_with_input) {
        auto meta = BuildMlsGemmDepMeta(*gemm_with_input, collector_);
        auto annotations = call->annotations;
        annotations.Set(attr::kMlsGemmDep, meta);
        Call new_call(call->dtype, call->op, call->args, annotations,
                      call->span);
        return Evaluate(new_call);
      }
      return StmtExprMutator::VisitStmt_(op);
    }

    if (tir_op.same_as(Gemm::Get())) {
      auto gemm = Downcast<Gemm>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
      auto annotations =
          AnnotateGemmHcuMlsFlags(call->annotations, gemm.get(), collector_);
      Call new_call(call->dtype, call->op, call->args, annotations, call->span);
      return Evaluate(new_call);
    }

    return StmtExprMutator::VisitStmt_(op);
  }

  PropagationTirCollector *collector_;
  Target target_;
  std::unordered_map<Buffer, bool, ObjectPtrHash, ObjectPtrEqual>
      shared_mls_trans_;
};

tvm::transform::Pass AnnotateMlsGemmDep() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    (void)m;
    (void)ctx;
    return AnnotateMlsGemmDepMutator::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateMlsGemmDep", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnnotateMlsGemmDep", AnnotateMlsGemmDep);
}

} // namespace tl
} // namespace tvm
