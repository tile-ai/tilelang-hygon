/*!
 * \file hcu/distributed/backend.cc
 * \brief HCU distributed backend registry and target dispatch.
 */
#include "backend.h"
#include "tl_templates/hcu/distributed/metadata.h"

#include "hcu/target_utils.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/reflection/registry.h>

#include <iomanip>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <utility>

namespace tvm {
namespace tl {
namespace hcu {

namespace {

struct BackendRegistration {
  uint32_t backend_id;
  HcuDistributedBackendFactory factory;
  std::set<DistributedCapability> capabilities;
};

struct BackendRegistry {
  std::mutex mutex;
  std::map<std::string, BackendRegistration> entries;
};

BackendRegistry &Registry() {
  static BackendRegistry registry;
  return registry;
}

const BackendRegistration &Lookup(const std::string &name) {
  auto &registry = Registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  auto it = registry.entries.find(name);
  if (it != registry.entries.end()) {
    // Entries are append-only; map insertion preserves their addresses.
    return it->second;
  }
  std::ostringstream supported;
  for (const auto &entry : registry.entries) {
    if (supported.tellp() > 0) {
      supported << ", ";
    }
    supported << entry.first;
  }
  TVM_FFI_THROW(ValueError) << "Unsupported HCU distributed backend: " << name
                            << "; supported backends: " << supported.str();
}

} // namespace

void RegisterHcuDistributedBackend(
    const std::string &name, uint32_t backend_id,
    HcuDistributedBackendFactory factory,
    std::vector<DistributedCapability> capabilities) {
  if (name.empty() || backend_id == 0 || !factory) {
    TVM_FFI_THROW(ValueError) << "Distributed backend registration requires a "
                                 "name, nonzero ID and factory";
  }
  std::set<DistributedCapability> declared(capabilities.begin(),
                                           capabilities.end());
  const HcuDistributedBackend &backend = factory();
  if (std::string(backend.name()) != name) {
    TVM_FFI_THROW(ValueError)
        << "Distributed backend factory name mismatch: " << name;
  }
  auto &registry = Registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  if (registry.entries.count(name)) {
    TVM_FFI_THROW(ValueError)
        << "Distributed backend is already registered and cannot be replaced: "
        << name;
  }
  for (const auto &entry : registry.entries) {
    if (entry.second.backend_id == backend_id) {
      TVM_FFI_THROW(ValueError) << "Distributed backend ID " << backend_id
                                << " is already registered to " << entry.first;
    }
  }
  registry.entries.emplace(
      name,
      BackendRegistration{backend_id, std::move(factory), std::move(declared)});
}

uint32_t GetHcuDistributedBackendId(const std::string &name) {
  return Lookup(name).backend_id;
}

std::vector<std::string> ListHcuDistributedBackends() {
  auto &registry = Registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  std::vector<std::string> names;
  for (const auto &entry : registry.entries) {
    names.push_back(entry.first);
  }
  return names;
}

DistributedInitSpec HcuDistributedBackend::InitSpec() const {
  return {{"__tilelang_init_distributed_metadata",
           {DataType::Handle(), DataType::Int(64), DataType::Handle()}}};
}

ffi::String GetHcuDistributedRequirements(const std::string &name) {
  const auto &entry = Lookup(name);
  const auto &backend = entry.factory();
  const auto spec = backend.InitSpec();
  std::ostringstream json;
  json << "{\"schema_version\":"
       << ::tl::distributed_metadata::kRequirementsVersion
       << ",\"backend\":" << std::quoted(name)
       << ",\"backend_id\":" << entry.backend_id
       << ",\"abi_version\":" << ::tl::distributed_metadata::kAbiVersion
       << ",\"capabilities\":[";
  bool first = true;
  for (auto capability : entry.capabilities) {
    if (!first)
      json << ",";
    first = false;
    json << std::quoted(capability == DistributedCapability::kRankAndWorldSize
                            ? "rank_world_size"
                            : "block_remote_get");
  }
  json << "],\"init_spec\":{\"initialize\":{\"name\":"
       << std::quoted(spec.initialize.name) << ",\"argument_types\":[";
  bool first_type = true;
  for (auto type : spec.initialize.argument_types) {
    if (!first_type)
      json << ",";
    first_type = false;
    std::ostringstream dtype;
    dtype << DataType(type);
    json << std::quoted(dtype.str());
  }
  json << "]}}}";
  return ffi::String(json.str());
}

std::string
EmitDistributedModulePreamble(const std::string &name,
                              const std::string &validator,
                              const std::string &private_declarations) {
  const auto &entry = Lookup(name);
  const auto spec = entry.factory().InitSpec();
  std::ostringstream code;
  code << "// tilelang_distributed_abi: "
       << ::tl::distributed_metadata::kAbiVersion << ":" << entry.backend_id
       << "\n";
  code << R"(
extern "C" __device__ volatile uint64_t __tilelang_distributed_metadata[tl::distributed_metadata::kCapacity] = {};
)" << private_declarations;
  code << "\nextern \"C\" __global__ void " << spec.initialize.name << R"((
    const uint64_t* source, int64_t count, uint64_t* result) {
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
      threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    using namespace tl::distributed_metadata;
    result[0] = tl::distributed_validate_envelope(source, count, )"
       << entry.backend_id << R"();
    result[1] = count >= kPayload && source != nullptr ? source[kVersion] : 0;
    if (result[0] != kSuccess) return;
    result[0] = )"
       << validator << R"((source, count);
    if (result[0] != kSuccess) return;
    for (int64_t i = 0; i < count; ++i) {
      __tilelang_distributed_metadata[i] = source[i];
    }
  }
}
)";
  return code.str();
}

const char *DistributedCapabilityName(DistributedCapability capability) {
  static const std::map<DistributedCapability, const char *> wording = {
      {DistributedCapability::kRankAndWorldSize, "rank/world-size intrinsics"},
      {DistributedCapability::kBlockRemoteGet, "block remote get"},
  };
  return wording.at(capability);
}

bool HcuDistributedBackend::Supports(DistributedCapability capability) const {
  return Lookup(std::string(name())).capabilities.count(capability) != 0;
}

void RequireDistributedCapability(const HcuDistributedBackend &backend,
                                  DistributedCapability capability) {
  if (!backend.Supports(capability)) {
    TVM_FFI_THROW(ValueError)
        << "HCU distributed backend " << backend.name() << " does not support "
        << DistributedCapabilityName(capability);
  }
}

bool TargetHasHcuDistributedBackend(const Target &target) {
  return TargetIsHCU(target) && target->attrs.count("dist_backend") != 0;
}

const HcuDistributedBackend &GetHcuDistributedBackend(const Target &target) {
  if (!TargetIsHCU(target)) {
    TVM_FFI_THROW(ValueError)
        << "HCU distributed backend requires an HCU target, got "
        << target->kind->name;
  }
  if (!TargetHasHcuDistributedBackend(target)) {
    TVM_FFI_THROW(ValueError)
        << "HCU distributed target requires a non-empty dist_backend attribute";
  }

  ffi::String backend = Downcast<ffi::String>(target->attrs.at("dist_backend"));
  return Lookup(std::string(backend)).factory();
}

} // namespace hcu
} // namespace tl
} // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace hcu = tvm::tl::hcu;
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.GetHcuDistributedRequirements",
           [](tvm::ffi::String name) {
             return hcu::GetHcuDistributedRequirements(std::string(name));
           })
      .def("tl.hcu.distributed.metadata_layout",
           []() {
             using namespace ::tl::distributed_metadata;
             return tvm::ffi::String(
                 "{\"abi_version\":" + std::to_string(kAbiVersion) +
                 ",\"requirements_version\":" +
                 std::to_string(kRequirementsVersion) +
                 ",\"capacity\":" + std::to_string(kCapacity) +
                 ",\"version\":" + std::to_string(kVersion) +
                 ",\"backend\":" + std::to_string(kBackend) +
                 ",\"rank\":" + std::to_string(kRank) +
                 ",\"world_size\":" + std::to_string(kWorldSize) +
                 ",\"payload_size\":" + std::to_string(kPayloadSize) +
                 ",\"status\":" + std::to_string(kStatus) +
                 ",\"status_detail\":" + std::to_string(kStatusDetail) +
                 ",\"payload\":" + std::to_string(kPayload) +
                 ",\"status_words\":" + std::to_string(kStatusWords) +
                 ",\"success\":" + std::to_string(kSuccess) +
                 ",\"invalid_peer_error\":" + std::to_string(kInvalidPeer) +
                 ",\"incompatible_version\":" +
                 std::to_string(kIncompatibleVersion) +
                 ",\"invalid_envelope\":" + std::to_string(kInvalidEnvelope) +
                 ",\"backend_mismatch\":" + std::to_string(kBackendMismatch) +
                 ",\"invalid_payload\":" + std::to_string(kInvalidPayload) +
                 "}");
           })
      .def("tl.GetHcuDistributedBackendId",
           [](tvm::ffi::String name) {
             return static_cast<int64_t>(
                 hcu::GetHcuDistributedBackendId(std::string(name)));
           })
      .def("tl.ListHcuDistributedBackends",
           []() {
             tvm::ffi::Array<tvm::ffi::String> names;
             for (const auto &name : hcu::ListHcuDistributedBackends()) {
               names.push_back(tvm::ffi::String(name));
             }
             return names;
           })
      .def("tl.TargetHasHcuDistributedBackend",
           hcu::TargetHasHcuDistributedBackend)
      .def("tl.GetHcuDistributedBackendName",
           [](tvm::Target target) {
             return hcu::GetHcuDistributedBackend(target).name();
           })
      .def("tl.HcuDistributedBackendSupportsRankWorldSize",
           [](tvm::Target target) {
             return hcu::GetHcuDistributedBackend(target).Supports(
                 hcu::DistributedCapability::kRankAndWorldSize);
           })
      .def("tl.HcuDistributedBackendSupportsBlockRemoteGet",
           [](tvm::Target target) {
             return hcu::GetHcuDistributedBackend(target).Supports(
                 hcu::DistributedCapability::kBlockRemoteGet);
           });
}
