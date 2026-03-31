#pragma once

// Device atomics aligned with tl_templates/hip/atomic.h (v0.1.7.post3-style
// standalone header). Trailing `memory_order` matches lowering; HIP
// intrinsics ignore it. half_t uses __hip_atomic_*; AtomicLoad/AtomicStore
// are HCU-oriented helpers.
#include "common.h"

template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 *address, T2 val, int memory_order = 0) {
  (void)memory_order;
  atomicAdd(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
}

template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 &address, T2 val, int memory_order = 0) {
  (void)memory_order;
  atomicAdd(reinterpret_cast<T1 *>(&address), static_cast<T1>(val));
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicAddRet(T1 *ref, T2 val, int memory_order = 0) {
  (void)memory_order;
  return atomicAdd(ref, static_cast<T1>(val));
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMax(T1 *address, T2 val, int memory_order = 0) {
  (void)memory_order;
  atomicMax(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMax(T1 &address, T2 val, int memory_order = 0) {
  (void)memory_order;
  atomicMax(reinterpret_cast<T1 *>(&address), static_cast<T1>(val));
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMin(T1 *address, T2 val, int memory_order = 0) {
  (void)memory_order;
  atomicMin(reinterpret_cast<T1 *>(address), static_cast<T1>(val));
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMin(T1 &address, T2 val, int memory_order = 0) {
  (void)memory_order;
  atomicMin(reinterpret_cast<T1 *>(&address), static_cast<T1>(val));
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMaxRet(T1 *ref, T2 val, int memory_order = 0) {
  (void)memory_order;
  return atomicMax(ref, static_cast<T1>(val));
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMinRet(T1 *ref, T2 val, int memory_order = 0) {
  (void)memory_order;
  return atomicMin(ref, static_cast<T1>(val));
}

TL_DEVICE void AtomicAddx2(float *ref, float *val, int memory_order = 0) {
  (void)memory_order;
  float2 add_val = *reinterpret_cast<float2 *>(val);
  atomicAdd(ref + 0, add_val.x);
  atomicAdd(ref + 1, add_val.y);
}

TL_DEVICE float2 AtomicAddx2Ret(float *ref, float *val, int memory_order = 0) {
  (void)memory_order;
  float2 add_val = *reinterpret_cast<float2 *>(val);
  float2 ret;
  ret.x = atomicAdd(ref + 0, add_val.x);
  ret.y = atomicAdd(ref + 1, add_val.y);
  return ret;
}

TL_DEVICE void AtomicAddx4(float *ref, float *val, int memory_order = 0) {
  (void)memory_order;
  float4 add_val = *reinterpret_cast<float4 *>(val);
  atomicAdd(ref + 0, add_val.x);
  atomicAdd(ref + 1, add_val.y);
  atomicAdd(ref + 2, add_val.z);
  atomicAdd(ref + 3, add_val.w);
}

TL_DEVICE float4 AtomicAddx4Ret(float *ref, float *val, int memory_order = 0) {
  (void)memory_order;
  float4 add_val = *reinterpret_cast<float4 *>(val);
  float4 ret;
  ret.x = atomicAdd(ref + 0, add_val.x);
  ret.y = atomicAdd(ref + 1, add_val.y);
  ret.z = atomicAdd(ref + 2, add_val.z);
  ret.w = atomicAdd(ref + 3, add_val.w);
  return ret;
}

template <>
TL_DEVICE void AtomicAdd<half_t, half_t>(half_t *address, half_t val,
                                         int memory_order) {
  (void)memory_order;
  __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
}

template <>
TL_DEVICE void AtomicAdd<half_t, float>(half_t *address, float val,
                                        int memory_order) {
  (void)memory_order;
  half_t half_val = static_cast<half_t>(val);
  AtomicAdd<half_t, half_t>(address, half_val, memory_order);
}

template <>
TL_DEVICE half_t AtomicAddRet<half_t, half_t>(half_t *ref, half_t val,
                                              int memory_order) {
  (void)memory_order;
  return __hip_atomic_fetch_add(ref, val, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT);
}

template <>
TL_DEVICE half_t AtomicAddRet<half_t, float>(half_t *ref, float val,
                                             int memory_order) {
  (void)memory_order;
  half_t half_val = static_cast<half_t>(val);
  return AtomicAddRet<half_t, half_t>(ref, half_val, memory_order);
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicAddRet(T1 &ref, T2 val, int memory_order = 0) {
  return AtomicAddRet(&ref, val, memory_order);
}

template <typename T> TL_DEVICE T AtomicLoad(T &ref, int memory_order) {
  return __hip_atomic_load(&ref, memory_order, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename T1, typename T2>
TL_DEVICE void AtomicStore(T1 &ref, T2 value, int memory_order) {
  __hip_atomic_store(&ref, static_cast<T1>(value), memory_order,
                     __HIP_MEMORY_SCOPE_AGENT);
}
