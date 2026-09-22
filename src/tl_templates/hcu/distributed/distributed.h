/*!
 * \file tl_templates/hcu/distributed/distributed.h
 * \brief Device-side helpers shared by HCU IPC distributed kernels.
 */
#ifndef TILELANG_TEMPLATES_HCU_DISTRIBUTED_DISTRIBUTED_H_
#define TILELANG_TEMPLATES_HCU_DISTRIBUTED_DISTRIBUTED_H_

#include <stdint.h>

extern "C" __device__ uint64_t __tilelang_ipc_metadata[1024];

namespace tl {

__device__ __forceinline__ int32_t ipc_get_rank() {
  return static_cast<int32_t>(__tilelang_ipc_metadata[0]);
}

__device__ __forceinline__ int32_t ipc_get_num_ranks() {
  return static_cast<int32_t>(__tilelang_ipc_metadata[1]);
}

__device__ __forceinline__ uint64_t ipc_get_peer_base(int32_t peer) {
  return __tilelang_ipc_metadata[2 + peer];
}

template <typename SrcT, typename DstT>
__device__ __forceinline__ void ipc_get_block(const SrcT* local_src, DstT* dst,
                                              int64_t count, int32_t src_pe) {
  const int64_t index = static_cast<int64_t>(threadIdx.x) +
                        static_cast<int64_t>(blockDim.x) * threadIdx.y;
  if (index >= count) return;
  const int32_t world_size = ipc_get_num_ranks();
  if (src_pe < 0 || src_pe >= world_size) {
    dst[index] = DstT{};
    return;
  }
  const uintptr_t local_base = static_cast<uintptr_t>(ipc_get_peer_base(ipc_get_rank()));
  const uintptr_t remote_base = static_cast<uintptr_t>(ipc_get_peer_base(src_pe));
  const uintptr_t offset = reinterpret_cast<uintptr_t>(local_src) - local_base;
  const SrcT* remote_src = reinterpret_cast<const SrcT*>(remote_base + offset);
  dst[index] = static_cast<DstT>(remote_src[index]);
}

} // namespace tl

#endif // TILELANG_TEMPLATES_HCU_DISTRIBUTED_DISTRIBUTED_H_
