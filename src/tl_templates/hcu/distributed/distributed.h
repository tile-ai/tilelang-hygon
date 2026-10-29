/*!
 * \file tl_templates/hcu/distributed/distributed.h
 * \brief Device-side helpers shared by HCU IPC distributed kernels.
 *
 * Callers complete producer writes on their producing streams, then rendezvous
 * across ranks before peer reads. Open IPC mappings and supported peer access
 * are prerequisites for visibility; compilation/initialization are not
 barriers.
 * get_block has no implicit collective or block barrier. Synchronize block
 * threads before consuming or reusing an LDS/shared destination. Keep sources
 * alive and unchanged until all remote consumers complete, then rendezvous
 * before refilling or releasing them.

 */
#ifndef TILELANG_TEMPLATES_HCU_DISTRIBUTED_DISTRIBUTED_H_
#define TILELANG_TEMPLATES_HCU_DISTRIBUTED_DISTRIBUTED_H_

#include "ipc_payload.h"
#include "metadata_runtime.h"
#include <stdint.h>

namespace tl {

__device__ __forceinline__ int ipc_validate_payload(const uint64_t *source,
                                                    int64_t count) {
  using namespace distributed_metadata;
  using namespace ipc_payload;
  const uint64_t world_size = source[kWorldSize];
  if (world_size > ipc_payload::kMaxWorldSize ||
      count != static_cast<int64_t>(kPayload + kHeaderSize + world_size) ||
      source[kPayload + kArenaSize] == 0) {
    return kInvalidPayload;
  }
  return kSuccess;
}

__device__ __forceinline__ uint64_t ipc_get_peer_base(int32_t peer) {
  return __tilelang_distributed_metadata[distributed_metadata::kPayload +
                                         ipc_payload::kPeerBases + peer];
}

template <typename SrcT, typename DstT>
__device__ __forceinline__ void ipc_get_block(const SrcT *local_src, DstT *dst,
                                              int64_t count, int32_t src_pe) {
  const int64_t thread_id =
      static_cast<int64_t>(threadIdx.x) +
      static_cast<int64_t>(blockDim.x) *
          (static_cast<int64_t>(threadIdx.y) +
           static_cast<int64_t>(blockDim.y) * threadIdx.z);
  const int64_t thread_count =
      static_cast<int64_t>(blockDim.x) * blockDim.y * blockDim.z;
  // The kernel author guarantees a valid peer and symmetric arena layout.
  const uintptr_t local_base =
      static_cast<uintptr_t>(ipc_get_peer_base(distributed_get_rank()));
  const uintptr_t remote_base =
      static_cast<uintptr_t>(ipc_get_peer_base(src_pe));
  const uintptr_t offset = reinterpret_cast<uintptr_t>(local_src) - local_base;
  const SrcT *remote_src = reinterpret_cast<const SrcT *>(remote_base + offset);
  for (int64_t index = thread_id; index < count; index += thread_count) {
    dst[index] = static_cast<DstT>(remote_src[index]);
  }
}

} // namespace tl

#endif // TILELANG_TEMPLATES_HCU_DISTRIBUTED_DISTRIBUTED_H_
