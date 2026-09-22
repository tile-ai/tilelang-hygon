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

  std::string EmitRankExpr() const final { return "tl::ipc_get_rank()"; }

  std::string EmitNumRanksExpr() const final { return "tl::ipc_get_num_ranks()"; }

  std::string ModulePreamble() const final {
    return R"(
extern "C" __device__ uint64_t __tilelang_ipc_metadata[1024];

extern "C" __global__ void __tilelang_init_ipc_metadata(
    const uint64_t* source, int64_t count) {
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
      threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    const int64_t bounded_count = count < 1024 ? count : 1024;
    for (int64_t i = 0; i < bounded_count; ++i) {
      __tilelang_ipc_metadata[i] = source[i];
    }
  }
}
)";
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
