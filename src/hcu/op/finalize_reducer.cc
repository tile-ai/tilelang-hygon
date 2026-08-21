/*!
 * \file hcu/op/finalize_reducer.cc
 * \brief HCU implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "backend/common/op/finalize_reducer.h"

#include "hcu/target_utils.h"
#include "hcu/utils/auto_ebarrier.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace hcu {

struct FinalizeReducer : backend::FinalizeReducerLowerer<FinalizeReducer> {
  static int WarpSize(Target target) { return TargetHcuGetWarpSize(target); }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset,
                                        PrimExpr all_threads, int batch,
                                        int workspace_stride, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (reducing_threads > TargetHcuGetWarpSize(target) &&
        TargetSupportsHcuEBarrier(target)) {
      ss << ", " << kAutoEBarrierPolicyMarker;
    } else {
      ss << ", tl::SyncThreadsBarrier";
    }
    ss << ", " << batch << ", " << workspace_stride << ">::run_batch";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (reducing_threads > TargetHcuGetWarpSize(target) &&
        TargetSupportsHcuEBarrier(target)) {
      ss << ", " << kAutoEBarrierPolicyMarker;
    }
    ss << ">::run";
    return ss.str();
  }
};

} // namespace hcu

namespace {

bool MatchHCUFinalizeReducerTarget(Target target) {
  return TargetIsHCU(target);
}

bool RegisterHCUFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "hcu.FinalizeReducer",
      MatchHCUFinalizeReducerTarget,
      hcu::FinalizeReducer::Lower,
  });
  return true;
}

const bool hcu_finalize_reducer_registered = RegisterHCUFinalizeReducer();

} // namespace

} // namespace tl
} // namespace tvm
