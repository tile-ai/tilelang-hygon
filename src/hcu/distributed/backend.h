/*!
 * \file hcu/distributed/backend.h
 * \brief Compile-time backend selection for HCU distributed code generation.
 */
#ifndef TILELANG_HCU_DISTRIBUTED_BACKEND_H_
#define TILELANG_HCU_DISTRIBUTED_BACKEND_H_

#include <tvm/ffi/string.h>
#include <tvm/target/target.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tvm {
namespace tl {
namespace hcu {

/*! \brief Public distributed operations a backend can lower. */
enum class DistributedCapability {
  kRankAndWorldSize,
  kBlockRemoteGet,
};

/*! \brief Host runtime helper signatures contributed by a backend. */
struct DistributedHelperSpec {
  std::string name;
  std::vector<DLDataType> argument_types;
};
struct DistributedInitSpec {
  DistributedHelperSpec initialize;
};

/*! \brief Backend-private HCU distributed codegen contract.
 *
 * The dispatcher owns backend selection from Target::attrs["dist_backend"].
 * Backends own capability declarations and, in later implementation units,
 * intrinsic emission, module globals, and helper kernels.  This prevents IPC
 * details from becoming part of the common distributed IR ABI.
 */
class TVM_DLL HcuDistributedBackend {
public:
  virtual ~HcuDistributedBackend() = default;
  virtual ffi::String name() const = 0;
  bool Supports(DistributedCapability capability) const;
  virtual std::string EmitRankExpr() const = 0;
  virtual std::string EmitNumRanksExpr() const = 0;
  virtual std::string EmitBlockGetExpr(const std::string &src,
                                       const std::string &dst,
                                       const std::string &size,
                                       const std::string &src_pe) const = 0;
  virtual std::string ModulePreamble() const = 0;
  virtual std::string DeviceIncludes() const {
    return "#include <tl_templates/hcu/distributed/metadata_runtime.h>\n";
  }
  virtual DistributedInitSpec InitSpec() const;
};

/*! \brief Stable diagnostic wording for a distributed capability. */
const char *DistributedCapabilityName(DistributedCapability capability);
/*! \brief Require a capability without extending the backend virtual interface.
 */
void RequireDistributedCapability(const HcuDistributedBackend &backend,
                                  DistributedCapability capability);

/*! \brief Append-only process registry; names and nonzero IDs are unique.
 *
 * Factories return process-lifetime backend instances. Registration checks the
 * instance's name and cannot replace an existing binding. Capabilities are
 * declared only in this registration. Plugins
 * must remain loaded while their factory or instance can be used.
 */
using HcuDistributedBackendFactory =
    std::function<const HcuDistributedBackend &()>;
TVM_DLL void
RegisterHcuDistributedBackend(const std::string &name, uint32_t backend_id,
                              HcuDistributedBackendFactory factory,
                              std::vector<DistributedCapability> capabilities);
TVM_DLL uint32_t GetHcuDistributedBackendId(const std::string &name);
TVM_DLL std::vector<std::string> ListHcuDistributedBackends();

/*! \brief Common globals/entry points around a backend-private validator. */
TVM_DLL std::string
EmitDistributedModulePreamble(const std::string &backend,
                              const std::string &validator,
                              const std::string &private_declarations = "");
TVM_DLL ffi::String GetHcuDistributedRequirements(const std::string &backend);

/*! \brief Returns true when an HCU target explicitly selects a backend. */
bool TargetHasHcuDistributedBackend(const Target &target);

/*! \brief Resolve the selected backend or throw an actionable ValueError. */
const HcuDistributedBackend &GetHcuDistributedBackend(const Target &target);

} // namespace hcu
} // namespace tl
} // namespace tvm

#endif // TILELANG_HCU_DISTRIBUTED_BACKEND_H_
