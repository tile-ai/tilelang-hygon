/*!
 * \file hcu/distributed/ipc/ipc_backend.cc
 * \brief IPC capability declaration for the HCU distributed dispatcher.
 */
#include "hcu/distributed/backend.h"
#include "tl_templates/hcu/distributed/ipc_payload.h"
#include "tl_templates/hcu/distributed/metadata.h"
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {
namespace hcu {
namespace {

class IpcDistributedBackend final : public HcuDistributedBackend {
public:
  ffi::String name() const final { return "ipc"; }

  std::string EmitRankExpr() const final {
    return "tl::distributed_get_rank()";
  }

  std::string EmitNumRanksExpr() const final {
    return "tl::distributed_get_num_ranks()";
  }
  std::string EmitBlockGetExpr(const std::string &src, const std::string &dst,
                               const std::string &size,
                               const std::string &src_pe) const final {
    return "tl::ipc_get_block(" + src + ", " + dst + ", " + size + ", " +
           src_pe + ")";
  }

  std::string DeviceIncludes() const final {
    return "#include <tl_templates/hcu/distributed/distributed.h>\n";
  }
  std::string ModulePreamble() const final {
    return EmitDistributedModulePreamble("ipc", "tl::ipc_validate_payload");
  }
};

} // namespace

const HcuDistributedBackend &GetIpcDistributedBackend() {
  static const IpcDistributedBackend backend;
  return backend;
}

} // namespace hcu
} // namespace tl
} // namespace tvm

// Python consumes the compiled ABI, never a separately maintained layout.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace hcu = tvm::tl::hcu;
  hcu::RegisterHcuDistributedBackend(
      "ipc", ::tl::ipc_payload::kBackendId, hcu::GetIpcDistributedBackend,
      {hcu::DistributedCapability::kRankAndWorldSize,
       hcu::DistributedCapability::kBlockRemoteGet});
  tvm::ffi::reflection::GlobalDef().def("tl.hcu.ipc.payload_layout", []() {
    using namespace ::tl::ipc_payload;
    return tvm::ffi::String(
        "{\"invalid_source_count\":" + std::to_string(kInvalidSourceCount) +
        ",\"invalid_source_span\":" + std::to_string(kInvalidSourceSpan) +
        ",\"invalid_source_arena\":" + std::to_string(kInvalidSourceArena) +
        ",\"arena_size\":" + std::to_string(kArenaSize) +
        ",\"peer_bases\":" + std::to_string(kPeerBases) +
        ",\"header_size\":" + std::to_string(kHeaderSize) +
        ",\"max_world_size\":" + std::to_string(kMaxWorldSize) + "}");
  });
}
