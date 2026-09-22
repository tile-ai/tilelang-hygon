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

} // namespace tl

#endif // TILELANG_TEMPLATES_HCU_DISTRIBUTED_DISTRIBUTED_H_
