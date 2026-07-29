/*!
 * \file hcu/target_utils.cc
 */
#include "hcu/target_utils.h"
#include "support/check.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target_kind.h>
#include <tvm/tirx/expr.h>

#include <set>
#include <string>

namespace tvm {
namespace tl {
namespace refl = ffi::reflection;
namespace {

const std::set<std::string> &HcuMcpuSet() {
  static const std::set<std::string> kSet = {"gfx928", "gfx936", "gfx938",
                                             "gfx92a", "gfx946"};
  return kSet;
}

bool IsKnownHcuMcpu(const std::string &mcpu) {
  return HcuMcpuSet().find(mcpu) != HcuMcpuSet().end();
}

bool TargetHasMcpu(const Target &target) {
  return target->attrs.count("mcpu") != 0;
}

std::string GetMcpu(Target target) {
  return Downcast<ffi::String>(target->attrs.at("mcpu"));
}

} // namespace

bool TargetIsHCU(Target target) { return target->kind->name == "hcu"; }

bool TargetSupportsHcuWdra(Target target) {
  if (!TargetIsHCU(target) || !TargetHasMcpu(target)) {
    return false;
  }
  return GetMcpu(target) == "gfx946";
}

bool TargetHasMmacLitLts(Target target) {
  if (!TargetIsHCU(target) || !TargetHasMcpu(target)) {
    return false;
  }
  static const std::set<std::string> kSet = {"gfx938", "gfx946"};
  return kSet.count(GetMcpu(target)) > 0;
}

std::string GetHcuArchString(Target target) {
  ICHECK(TargetIsHCU(target)) << "GetHcuArchString requires HCU target";
  ICHECK(TargetHasMcpu(target)) << "HCU target must have mcpu attribute";
  std::string mcpu = GetMcpu(target);
  static const std::set<std::string> kMlsSupported = {"gfx938", "gfx92a",
                                                      "gfx946"};
  ICHECK(kMlsSupported.count(mcpu))
      << "HCU arch " << mcpu
      << " not supported for MLS/GEMM_MLS; supported: gfx938, gfx92a, gfx946";
  return mcpu;
}

bool TargetHcuHasAsyncCopy(Target target) {
  if (!TargetIsHCU(target) || !TargetHasMcpu(target)) {
    return false;
  }
  static const std::set<std::string> kSupported = {"gfx936", "gfx938", "gfx92a",
                                                   "gfx946"};
  return kSupported.count(GetMcpu(target)) > 0;
}

bool IsHCUEnableAutoAsyncCopyTarget(Target target) {
  if (!TargetIsHCU(target) || !TargetHasMcpu(target)) {
    return false;
  }
  static const std::set<std::string> auto_async_whitelist = {};
  return auto_async_whitelist.count(GetMcpu(target)) > 0;
}

bool DefaultEnableAutoAsyncCopy(Target target) {
  if (TargetIsHCU(target)) {
    return IsHCUEnableAutoAsyncCopyTarget(target);
  }
  return true;
}

int TargetHcuGetWarpSize(Target target) {
  ICHECK(TargetIsHCU(target)) << "TargetHcuGetWarpSize requires HCU target";
  if (target->attrs.count("thread_warp_size")) {
    return static_cast<int>(
        Downcast<IntImm>(target->attrs.at("thread_warp_size"))->value);
  }
  return 64;
}

TVM_REGISTER_TARGET_KIND("hcu", kDLROCM)
    .add_attr_option<ffi::String>("mcpu")
    .add_attr_option<ffi::String>("mtriple")
    .add_attr_option<ffi::Array<ffi::String>>("mattr")
    .add_attr_option<int64_t>("max_num_threads",
                              refl::DefaultValue(int64_t{1024}))
    .add_attr_option<int64_t>("max_threads_per_block",
                              refl::DefaultValue(int64_t{1024}))
    .add_attr_option<int64_t>("max_shared_memory_per_block",
                              refl::DefaultValue(int64_t{65536}))
    .add_attr_option<int64_t>("thread_warp_size",
                              refl::DefaultValue(int64_t{64}))
    .set_default_keys({"hcu", "gpu"});

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  refl::GlobalDef()
      .def("tl.TargetIsHCU", [](Target target) { return TargetIsHCU(target); })
      .def("tl.TargetSupportsHcuWdra",
           [](Target target) { return TargetSupportsHcuWdra(target); })
      .def("tl.TargetHasMmacLitLts",
           [](Target target) { return TargetHasMmacLitLts(target); })
      .def("tl.GetHcuArchString",
           [](Target target) { return GetHcuArchString(target); })
      .def("tl.TargetHcuHasAsyncCopy",
           [](Target target) { return TargetHcuHasAsyncCopy(target); })
      .def("tl.IsHCUEnableAutoAsyncCopyTarget",
           [](Target target) { return IsHCUEnableAutoAsyncCopyTarget(target); })
      .def("tl.DefaultEnableAutoAsyncCopy",
           [](Target target) { return DefaultEnableAutoAsyncCopy(target); })
      .def("tl.TargetHcuGetWarpSize",
           [](Target target) { return TargetHcuGetWarpSize(target); });
}

} // namespace tl
} // namespace tvm
