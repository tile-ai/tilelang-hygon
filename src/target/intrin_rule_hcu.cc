/*!
 * \file intrin_rule_hcu.cc
 * \brief HIP intrinsic rules.
 */
#include <tvm/tir/builtin.h>
#include <tvm/tir/op_attr_types.h>

#include "../support/ffi_aliases.h"
#include "target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
// Add float suffix to the intrinsics, HIP fast math.
using tir::FLowerIntrinsic;

struct HCUMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      case 16: {
        if (name == "fabs") {
          return "hcu_habs";
        } else if (name == "round") {
          return "hrint";
        } else {
          return "h" + name;
        }
      }
      default:
        return "";
      }
    } else if (t.is_bfloat16()) {
      if (name == "fabs") {
        return "hcu_habs";
      } else if (name == "round") {
        return "hrint";
      } else {
        return "h" + name;
      }
    } else if (t.is_int() || t.is_uint()) {
      switch (t.bits()) {
      case 32:
        return "__" + name;
      case 64:
        return "__" + name + "ll";
      default:
        return "";
      }
    }
    return "";
  }
};

struct HCUFastMath : public HCUMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float() && t.bits() == 32) {
      return "__" + name + 'f';
    } else {
      return HCUMath::operator()(t, name);
    }
    return "";
  }
};

struct HCUFastMathTan : public HCUMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      case 16:
        return std::string("h") + name;
      default:
        return "";
      }
    }
    return "";
  }
};

struct HCUPopcount {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_uint()) {
      switch (t.bits()) {
      case 32:
        return "__popc";
      case 64:
        return "__popcll";
      default:
        return "";
      }
    }
    return "";
  }
};

struct HCUWarpIntrinsic {
  const Op operator()(DataType t, const Op &orig_op) const {
    if (orig_op.same_as(builtin::tvm_warp_shuffle())) {
      return Op::Get("tir.hip.__shfl_sync");
    } else if (orig_op.same_as(builtin::tvm_warp_shuffle_up())) {
      return Op::Get("tir.hip.__shfl_up_sync");
    } else {
      ICHECK(orig_op.same_as(builtin::tvm_warp_shuffle_down()));
      return Op::Get("tir.hip.__shfl_down_sync");
    }
  }
};

static PrimExpr DispatchHCUWarpActiveMask(const PrimExpr &e) {
  const CallNode *call = e.as<CallNode>();
  ICHECK(call != nullptr);
  return Call(call->dtype, Op::Get("tir.hip.__activemask"), {});
}

template <typename T> static PrimExpr DispatchHCUShuffle(const PrimExpr &e) {
  // NOLINTBEGIN(clang-analyzer-cplusplus.InnerPointer)
  const CallNode *call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 5); // mask, value, warp_id, width, warp_size
  Array<PrimExpr> hip_args{
      {call->args[0], call->args[1], call->args[2], call->args[3]}};
  return Call(call->dtype, T()(call->dtype, Downcast<Op>(call->op)), hip_args);
  // NOLINTEND(clang-analyzer-cplusplus.InnerPointer)
}

TVM_REGISTER_OP("tir.clz").set_attr<FLowerIntrinsic>(
    "hcu.FLowerIntrinsic",
    DispatchPureExtern<HCUMath, /*dtype_from_arg=*/true>);

TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUMath>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUMath>);

TVM_REGISTER_OP("tir.trunc")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUMath>);

TVM_REGISTER_OP("tir.fabs")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUMath>);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUMath>);

TVM_REGISTER_OP("tir.nearbyint")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUMath>);

TVM_REGISTER_OP("tir.exp").set_attr<FLowerIntrinsic>(
    "hcu.FLowerIntrinsic", DispatchPureExtern<HCUFastMath>);

TVM_REGISTER_OP("tir.exp2")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUMath>);

TVM_REGISTER_OP("tir.exp10")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUFastMath>);

TVM_REGISTER_OP("tir.erf").set_attr<FLowerIntrinsic>(
    "hcu.FLowerIntrinsic", DispatchPureExtern<HCUMath>);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>(
    "hcu.FLowerIntrinsic", DispatchPureExtern<HCUFastMath>);

TVM_REGISTER_OP("tir.log2")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUFastMath>);

TVM_REGISTER_OP("tir.log10")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUFastMath>);

TVM_REGISTER_OP("tir.tan").set_attr<FLowerIntrinsic>(
    "hcu.FLowerIntrinsic", DispatchPureExtern<HCUFastMathTan>);

TVM_REGISTER_OP("tir.cos").set_attr<FLowerIntrinsic>(
    "hcu.FLowerIntrinsic", DispatchPureExtern<HCUFastMath>);

TVM_REGISTER_OP("tir.cosh")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUMath>);

TVM_REGISTER_OP("tir.sin").set_attr<FLowerIntrinsic>(
    "hcu.FLowerIntrinsic", DispatchPureExtern<HCUFastMath>);

TVM_REGISTER_OP("tir.sinh")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUMath>);

TVM_REGISTER_OP("tir.atan")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUMath>);

TVM_REGISTER_OP("tir.tanh")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUMath>);

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUMath>);

TVM_REGISTER_OP("tir.pow").set_attr<FLowerIntrinsic>(
    "hcu.FLowerIntrinsic", DispatchPureExtern<HCUMath>);

TVM_REGISTER_OP("tir.popcount")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUPopcount>);

TVM_REGISTER_OP("tir.tvm_warp_shuffle")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchHCUShuffle<HCUWarpIntrinsic>);

TVM_REGISTER_OP("tir.tvm_warp_shuffle_up")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchHCUShuffle<HCUWarpIntrinsic>);

TVM_REGISTER_OP("tir.tvm_warp_shuffle_down")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchHCUShuffle<HCUWarpIntrinsic>);

TVM_REGISTER_OP("tir.tvm_warp_activemask")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchHCUWarpActiveMask);

TVM_REGISTER_OP("tir.fmod")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUMath>);

} // namespace intrin
} // namespace codegen
} // namespace tvm
