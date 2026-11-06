// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#include "hcu/op/builtin.h"
#include "hcu/target_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {
using namespace tirx;

// Run after vectorization, before the common final broadcast-hoisting pass.
// Hardware controls and a shared scale remain scalar. This also prevents
// broadcast hoisting from obscuring their compile-time/shared-lane properties.
class ScaledCastLowerer : public StmtExprMutator {
  PrimExpr VisitExpr_(const CallNode *op) final {
    auto call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!call->op.same_as(scaled_cast()))
      return call;
    ICHECK_EQ(call->args.size(), 3U);
    ffi::Array<PrimExpr> args{call->args[0]};
    for (size_t i = 1; i < call->args.size(); ++i) {
      PrimExpr arg = call->args[i];
      while (const auto *broadcast = arg.as<BroadcastNode>())
        arg = broadcast->value;
      args.push_back(arg);
    }
    return Call(call->dtype, hcu_scaled_cast(), args, call->annotations);
  }
};

tvm::transform::Pass LowerHcuScaledCast() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      const tvm::transform::PassContext &) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (target.defined() && TargetIsHCU(target.value()))
      f.CopyOnWrite()->body = ScaledCastLowerer()(f->body);
    return f;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.LowerHcuScaledCast", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("tl.transform.LowerHcuScaledCast",
                                        LowerHcuScaledCast);
}
} // namespace tl
} // namespace tvm
