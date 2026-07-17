/*!
 * \file hcu/op/fill.cc
 * \brief HCU implementation for tl.fill lowering.
 */

#include "backend/common/op/fill.h"

#include "hcu/target_utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchHCUFillTarget(Target target) { return TargetIsHCU(target); }

bool RegisterHCUFill() {
  RegisterFillImpl(FillImpl{
      "hcu.Fill",
      MatchHCUFillTarget,
      backend::Fill::Lower,
  });
  return true;
}

const bool hcu_fill_registered = RegisterHCUFill();

} // namespace

} // namespace tl
} // namespace tvm
