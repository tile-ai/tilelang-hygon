// HIP-only accessors and envelope validation; private payloads stay backend-owned.
#ifndef TILELANG_HCU_DISTRIBUTED_METADATA_RUNTIME_H_
#define TILELANG_HCU_DISTRIBUTED_METADATA_RUNTIME_H_
#include <stdint.h>
#include "metadata.h"
extern "C" __device__ volatile uint64_t
    __tilelang_distributed_metadata[tl::distributed_metadata::kCapacity];
namespace tl {
__device__ __forceinline__ int32_t distributed_get_rank() {
  return static_cast<int32_t>(__tilelang_distributed_metadata[distributed_metadata::kRank]);
}
__device__ __forceinline__ int32_t distributed_get_num_ranks() {
  return static_cast<int32_t>(__tilelang_distributed_metadata[distributed_metadata::kWorldSize]);
}
__device__ __forceinline__ int distributed_validate_envelope(
    const uint64_t *source, int64_t count, uint64_t backend_id) {
  using namespace distributed_metadata;
  if (source == nullptr || count < kPayload || count > kCapacity) {
    return kInvalidEnvelope;
  }
  if (source[kVersion] != kAbiVersion) return kIncompatibleVersion;
  if (source[kBackend] != backend_id) return kBackendMismatch;
  if (source[kWorldSize] == 0 || source[kWorldSize] > INT32_MAX ||
      source[kRank] >= source[kWorldSize] ||
      source[kPayloadSize] != static_cast<uint64_t>(count - kPayload) ||
      source[kStatus] != kSuccess || source[kStatusDetail] != 0) {
    return kInvalidEnvelope;
  }
  return kSuccess;
}
} // namespace tl
#endif // TILELANG_HCU_DISTRIBUTED_METADATA_RUNTIME_H_
