/*!
 * \file hcu/op/transpose.cc
 * \brief HCU implementation for tl.transpose lowering.
 */

#include "backend/common/op/transpose.h"

#include "hcu/target_utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchHCUTransposeTarget(Target target) { return TargetIsHCU(target); }

bool RegisterHCUTranspose() {
  RegisterTransposeImpl(TransposeImpl{
      "hcu.Transpose",
      MatchHCUTransposeTarget,
      backend::Transpose::Lower,
  });
  return true;
}

const bool hcu_transpose_registered = RegisterHCUTranspose();

} // namespace

} // namespace tl
} // namespace tvm
