/*!
 * \file intrin_rule_hcu.cc
 * \brief HIP intrinsic rules.
 */
#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>

#include "target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
using tirx::FLowerIntrinsic;
using namespace ffi;

template <typename T, bool dtype_from_arg = false>
inline PrimExpr DispatchPureExternScalarized(const PrimExpr &e) {
  const CallNode *call = e.as<CallNode>();
  ICHECK(call != nullptr);
  DataType dtype = dtype_from_arg ? call->args[0].dtype() : call->dtype;
  if (!dtype.is_vector()) {
    return DispatchPureExtern<T, dtype_from_arg>(e);
  }
  const OpNode *op = call->op.as<OpNode>();
  ICHECK(op != nullptr);
  std::string name = T()(dtype.element_of(), op->name.substr(5));
  if (name.empty()) {
    return e;
  }
  ffi::Array<PrimExpr> new_args = {StringImm(name)};
  for (const auto &arg : call->args) {
    new_args.push_back(arg);
  }
  return Call(call->dtype, builtin::call_pure_extern(), new_args,
              call->annotations);
}

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
      return Op::Get("tirx.hip.__shfl_sync");
    } else if (orig_op.same_as(builtin::tvm_warp_shuffle_up())) {
      return Op::Get("tirx.hip.__shfl_up_sync");
    } else {
      ICHECK(orig_op.same_as(builtin::tvm_warp_shuffle_down()));
      return Op::Get("tirx.hip.__shfl_down_sync");
    }
  }
};

static PrimExpr DispatchHCUWarpActiveMask(const PrimExpr &e) {
  const CallNode *call = e.as<CallNode>();
  ICHECK(call != nullptr);
  return Call(call->dtype, Op::Get("tirx.hip.__activemask"), {});
}

template <typename T> static PrimExpr DispatchHCUShuffle(const PrimExpr &e) {
  // NOLINTBEGIN(clang-analyzer-cplusplus.InnerPointer)
  const CallNode *call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 5); // mask, value, warp_id, width, warp_size
  ffi::Array<PrimExpr> hip_args{
      {call->args[0], call->args[1], call->args[2], call->args[3]}};
  return Call(call->dtype, T()(call->dtype, Downcast<Op>(call->op)), hip_args);
  // NOLINTEND(clang-analyzer-cplusplus.InnerPointer)
}

TVM_REGISTER_OP("tirx.clz")
    .set_attr<FLowerIntrinsic>(
        "hcu.FLowerIntrinsic",
        DispatchPureExtern<HCUMath, /*dtype_from_arg=*/true>);

TVM_REGISTER_OP("tirx.floor")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

TVM_REGISTER_OP("tirx.ceil")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

TVM_REGISTER_OP("tirx.trunc")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

TVM_REGISTER_OP("tirx.fabs")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

TVM_REGISTER_OP("tirx.round")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

TVM_REGISTER_OP("tirx.nearbyint")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

TVM_REGISTER_OP("tirx.exp")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

TVM_REGISTER_OP("tirx.exp2")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

TVM_REGISTER_OP("tirx.exp10")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUFastMath>);

TVM_REGISTER_OP("tirx.erf")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

TVM_REGISTER_OP("tirx.log")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUFastMath>);

TVM_REGISTER_OP("tirx.log2")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUFastMath>);

TVM_REGISTER_OP("tirx.log10")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUFastMath>);

TVM_REGISTER_OP("tirx.tan")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUFastMathTan>);

TVM_REGISTER_OP("tirx.cos")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUFastMath>);

TVM_REGISTER_OP("tirx.cosh")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

TVM_REGISTER_OP("tirx.sin")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUFastMath>);

TVM_REGISTER_OP("tirx.sinh")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

TVM_REGISTER_OP("tirx.atan")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

TVM_REGISTER_OP("tirx.tanh")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

TVM_REGISTER_OP("tirx.sqrt")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

TVM_REGISTER_OP("tirx.pow")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

TVM_REGISTER_OP("tirx.popcount")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExtern<HCUPopcount>);

TVM_REGISTER_OP("tirx.tvm_warp_shuffle")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchHCUShuffle<HCUWarpIntrinsic>);

TVM_REGISTER_OP("tirx.tvm_warp_shuffle_up")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchHCUShuffle<HCUWarpIntrinsic>);

TVM_REGISTER_OP("tirx.tvm_warp_shuffle_down")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchHCUShuffle<HCUWarpIntrinsic>);

TVM_REGISTER_OP("tirx.tvm_warp_activemask")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchHCUWarpActiveMask);

TVM_REGISTER_OP("tirx.fmod")
    .set_attr<FLowerIntrinsic>("hcu.FLowerIntrinsic",
                               DispatchPureExternScalarized<HCUMath>);

} // namespace intrin
} // namespace codegen
} // namespace tvm
