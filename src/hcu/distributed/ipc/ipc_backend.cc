/*!
 * \file hcu/distributed/ipc/ipc_backend.cc
 * \brief IPC capability declaration for the HCU distributed dispatcher.
 */
#include "hcu/distributed/backend.h"

namespace tvm {
namespace tl {
namespace hcu {
namespace {

class IpcDistributedBackend final : public HcuDistributedBackend {
public:
  ffi::String name() const final { return "ipc"; }

  bool Supports(DistributedCapability capability) const final {
    switch (capability) {
    case DistributedCapability::kRankAndWorldSize:
    case DistributedCapability::kBlockRemoteGet:
      return true;
    }
    return false;
  }
};

} // namespace

const HcuDistributedBackend& GetIpcDistributedBackend() {
  static const IpcDistributedBackend backend;
  return backend;
}

} // namespace hcu
} // namespace tl
} // namespace tvm
