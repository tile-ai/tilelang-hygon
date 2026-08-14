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
int TargetHcuGetLdsWrapFieldBits(Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_TARGET_UTILS_H_
