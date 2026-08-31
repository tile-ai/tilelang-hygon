/*!
 * \file hcu/target_utils.h
 * \brief HCU target attribute helpers.
 */
#ifndef TVM_TL_HCU_TARGET_UTILS_H_
#define TVM_TL_HCU_TARGET_UTILS_H_

#include <string>
#include <tvm/target/target.h>

namespace tvm {
namespace tl {

enum class HcuLdsWrapEncoding {
  kNone,
  kFourDword,
  kHybridFourAndOneDword,
  kOneDword,
};

struct HcuLdsWrapConfig {
  int field_bits{0};
  int field_shift{0};
  int lds_offset_bits{0};
  HcuLdsWrapEncoding encoding{HcuLdsWrapEncoding::kNone};
};

bool TargetIsHCU(Target target);
bool TargetSupportsHcuWdra(Target target);
bool TargetSupportsHcuEBarrier(Target target);
bool TargetSupportsHcuABarrier(Target target);
bool TargetHasMmacLitLts(Target target);
std::string GetHcuArchString(Target target);
bool TargetHcuHasAsyncCopy(Target target);
bool IsHCUEnableAutoAsyncCopyTarget(Target target);
bool DefaultEnableAutoAsyncCopy(Target target);
int TargetHcuGetWarpSize(Target target);
int TargetHcuGetLdsBankCount(Target target);
int TargetHcuGetLdsBankWidthBytes(Target target);
HcuLdsWrapConfig TargetHcuGetLdsWrapConfig(Target target);
int TargetHcuGetLdsWrapFieldBits(Target target);
int TargetHcuGetLdsWrapGranularityDwords(Target target);
int TargetHcuGetLdsWrapMaxOffsetDwords(Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_TARGET_UTILS_H_
