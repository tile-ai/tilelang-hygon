/*!
 * \file hcu/op/atomic_reduce.cc
 * \brief HCU implementation for tl.atomicmax/tl.atomicmin lowering.
 */

#include "backend/common/op/atomic_reduce.h"

#include "hcu/target_utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchHCUAtomicReduceTarget(Target target) { return TargetIsHCU(target); }

bool RegisterHCUAtomicReduce() {
  RegisterAtomicReduceImpl(AtomicReduceImpl{
      "hcu.AtomicReduce",
      MatchHCUAtomicReduceTarget,
      backend::AtomicReduce::InferLayout,
      backend::AtomicReduce::Lower,
  });
  return true;
}

const bool hcu_atomic_reduce_registered = RegisterHCUAtomicReduce();

} // namespace

} // namespace tl
} // namespace tvm
