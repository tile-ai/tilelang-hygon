/*!
 * \file tl_templates/hcu/distributed/signal.h
 * \brief System-scope signaling on directly accessible HCU global memory.
 */
#ifndef TILELANG_HCU_DISTRIBUTED_SIGNAL_H_
#define TILELANG_HCU_DISTRIBUTED_SIGNAL_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace tl {

// All addresses must be naturally aligned uint64_t global-memory locations.
__device__ __forceinline__ uint64_t signal_load(const uint64_t *signal) {
  return __hip_atomic_load(signal, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <int Operation>
__device__ __forceinline__ void signal_update(uint64_t *signal,
                                              uint64_t value) {
  static_assert(Operation == 0 || Operation == 1,
                "unsupported signal operation");
  if constexpr (Operation == 0) {
    __hip_atomic_store(signal, value, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  } else {
    __hip_atomic_fetch_add(signal, value, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// Comparison IDs follow the NVSHMEM comparison order; update IDs are private.
template <int Compare>
__device__ __forceinline__ bool signal_compare(uint64_t observed,
                                               uint64_t expected) {
  static_assert(Compare >= 0 && Compare <= 5, "unsupported signal comparison");
  if constexpr (Compare == 0)
    return observed == expected;
  if constexpr (Compare == 1)
    return observed != expected;
  if constexpr (Compare == 2)
    return observed > expected;
  if constexpr (Compare == 3)
    return observed <= expected;
  if constexpr (Compare == 4)
    return observed < expected;
  if constexpr (Compare == 5)
    return observed >= expected;
}

template <int Compare>
__device__ __forceinline__ uint64_t signal_wait_until(const uint64_t *signal,
                                                      uint64_t expected) {
  uint64_t observed, polls = 0;
  while (!signal_compare<Compare>(observed = signal_load(signal), expected)) {
    if ((++polls & 15) == 0)
      __builtin_amdgcn_s_sleep(1);
  }
  return observed;
}

} // namespace tl
#endif // TILELANG_HCU_DISTRIBUTED_SIGNAL_H_
