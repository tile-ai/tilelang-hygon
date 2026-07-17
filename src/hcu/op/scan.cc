/*!
 * \file hcu/op/scan.cc
 * \brief HCU implementation registration for tl scan lowering.
 */

#include "backend/common/op/scan.h"

#include "hcu/target_utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchHCUScanTarget(Target target) { return TargetIsHCU(target); }

bool RegisterHCUScan() {
  RegisterCumSumImpl(CumSumImpl{
      "hcu.CumSum",
      MatchHCUScanTarget,
      backend::scan::LowerCumSum,
  });
  RegisterCumMaxImpl(CumMaxImpl{
      "hcu.CumMax",
      MatchHCUScanTarget,
      backend::scan::LowerCumMax,
  });
  return true;
}

const bool hcu_scan_registered = RegisterHCUScan();

} // namespace

} // namespace tl
} // namespace tvm
