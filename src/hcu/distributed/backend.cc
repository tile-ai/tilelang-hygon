/*!
 * \file hcu/distributed/backend.cc
 * \brief HCU distributed backend registry and target dispatch.
 */
#include "backend.h"

#include "hcu/target_utils.h"

#include <tvm/ffi/error.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {
namespace hcu {

const HcuDistributedBackend& GetIpcDistributedBackend();

bool TargetHasHcuDistributedBackend(const Target& target) {
  return TargetIsHCU(target) && target->attrs.count("dist_backend") != 0;
}

const HcuDistributedBackend& GetHcuDistributedBackend(const Target& target) {
  if (!TargetIsHCU(target)) {
    TVM_FFI_THROW(ValueError) << "HCU distributed backend requires an HCU target, got "
                              << target->kind->name;
  }
  if (!TargetHasHcuDistributedBackend(target)) {
    TVM_FFI_THROW(ValueError)
        << "HCU distributed target requires a non-empty dist_backend attribute";
  }

  ffi::String backend = Downcast<ffi::String>(target->attrs.at("dist_backend"));
  if (backend == "ipc") {
    return GetIpcDistributedBackend();
  }
  TVM_FFI_THROW(ValueError) << "Unsupported HCU distributed backend: " << backend
                            << "; supported backends: ipc";
}

} // namespace hcu
} // namespace tl
} // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace hcu = tvm::tl::hcu;
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.TargetHasHcuDistributedBackend", hcu::TargetHasHcuDistributedBackend)
      .def("tl.GetHcuDistributedBackendName", [](tvm::Target target) {
        return hcu::GetHcuDistributedBackend(target).name();
      })
      .def("tl.HcuDistributedBackendSupportsRankWorldSize", [](tvm::Target target) {
        return hcu::GetHcuDistributedBackend(target).Supports(
            hcu::DistributedCapability::kRankAndWorldSize);
      })
      .def("tl.HcuDistributedBackendSupportsBlockRemoteGet", [](tvm::Target target) {
        return hcu::GetHcuDistributedBackend(target).Supports(
            hcu::DistributedCapability::kBlockRemoteGet);
      });
}
