#pragma once

#include "../common.h"

// Specialization for half_t: use __hip_atomic_fetch_add builtin which supports half_t
// Following the pattern from amd_hip_atomic.h for float
template <>
TL_DEVICE void AtomicAdd<half_t, half_t>(half_t *address, half_t val) {
  __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <>
TL_DEVICE void AtomicAdd<half_t, float>(half_t *address, float val) {
  half_t half_val = static_cast<half_t>(val);
  AtomicAdd<half_t, half_t>(address, half_val);
}

// Specialization for AtomicAddRet with half_t
template <>
TL_DEVICE half_t AtomicAddRet<half_t, half_t>(half_t &ref, half_t val) {
  return __hip_atomic_fetch_add(&ref, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <>
TL_DEVICE half_t AtomicAddRet<half_t, float>(half_t &ref, float val) {
  half_t half_val = static_cast<half_t>(val);
  return AtomicAddRet<half_t, half_t>(ref, half_val);
}

template <typename T> TL_DEVICE T AtomicLoad(T &ref, int memory_order) {
  // Use HIP atomic with agent memory scope for cross-block synchronization
  return __hip_atomic_load(&ref, memory_order, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T1, typename T2>
TL_DEVICE void AtomicStore(T1 &ref, T2 value, int memory_order) {
  __hip_atomic_store(&ref, static_cast<T1>(value), memory_order, __HIP_MEMORY_SCOPE_AGENT);
}