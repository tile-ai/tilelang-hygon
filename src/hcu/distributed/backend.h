/*!
 * \file hcu/distributed/backend.h
 * \brief Compile-time backend selection for HCU distributed code generation.
 */
#ifndef TILELANG_HCU_DISTRIBUTED_BACKEND_H_
#define TILELANG_HCU_DISTRIBUTED_BACKEND_H_

#include <tvm/ffi/string.h>
#include <tvm/target/target.h>

#include <string>

namespace tvm {
namespace tl {
namespace hcu {

/*! \brief Public distributed operations a backend can lower. */
enum class DistributedCapability {
  kRankAndWorldSize,
  kBlockRemoteGet,
};

/*! \brief Backend-private HCU distributed codegen contract.
 *
 * The dispatcher owns backend selection from Target::attrs["dist_backend"].
 * Backends own capability declarations and, in later implementation units,
 * intrinsic emission, module globals, and helper kernels.  This prevents IPC
 * details from becoming part of the common distributed IR ABI.
 */
class HcuDistributedBackend {
public:
  virtual ~HcuDistributedBackend() = default;
  virtual ffi::String name() const = 0;
  virtual bool Supports(DistributedCapability capability) const = 0;
  virtual std::string EmitRankExpr() const = 0;
  virtual std::string EmitNumRanksExpr() const = 0;
  virtual std::string ModulePreamble() const = 0;
};

/*! \brief Returns true when an HCU target explicitly selects a backend. */
bool TargetHasHcuDistributedBackend(const Target& target);

/*! \brief Resolve the selected backend or throw an actionable ValueError. */
const HcuDistributedBackend& GetHcuDistributedBackend(const Target& target);

} // namespace hcu
} // namespace tl
} // namespace tvm

#endif // TILELANG_HCU_DISTRIBUTED_BACKEND_H_
