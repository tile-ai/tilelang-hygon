#pragma once

#include <cstring>
#include <hip/amd_detail/amd_warp_functions.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <tl_templates/hcu/core.hpp>

#define HIPRT_INF_F __int_as_float(0x7f800000)
#define HIPRT_NEGINF_F __int_as_float(0xff800000)
#define HIPRT_NAN_F __int_as_float(0x7fffffff)
#define HIPRT_MIN_DENORM_F __int_as_float(0x00000001)
#define HIPRT_MAX_NORMAL_F __int_as_float(0x7f7fffff)
#define HIPRT_NEG_ZERO_F __int_as_float(0x80000000)
#define HIPRT_ZERO_F 0.0f
#define HIPRT_ONE_F 1.0f

/* double precision constants */
#define HIPRT_INF __hiloint2double(0x7ff00000, 0x00000000)
#define HIPRT_NAN __hiloint2double(0xfff80000, 0x00000000)

#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short

#define TL_DEVICE_NOINLINE __noinline__ __device__

#define TILELANG_CHECK(stmt)                                                   \
  do {                                                                         \
    hipError_t __err = (stmt);                                                 \
    if (__err != hipSuccess) {                                                 \
      snprintf(error_buf, ERROR_BUF_SIZE, "%s:%d: %s - %s", __FILE__,          \
               __LINE__, hipGetErrorName(__err), hipGetErrorString(__err));    \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILELANG_CHECK_LAST_ERROR(kernel_name)                                 \
  do {                                                                         \
    hipError_t __err = hipGetLastError();                                      \
    if (__err != hipSuccess) {                                                 \
      snprintf(error_buf, ERROR_BUF_SIZE, "kernel_name: %s - %s",              \
               hipGetErrorName(__err), hipGetErrorString(__err));              \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define half _Float16
#define __float2half_rn(x) half(x)

#define hpow __ocml_pown_f16
#define hsqrt __ocml_sqrt_f16

using float16_t = _Float16;
using float16x2 =
    __attribute__((__vector_size__(2 * sizeof(float16_t)))) float16_t;
using float16x4 =
    __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
using float16x8 =
    __attribute__((__vector_size__(8 * sizeof(float16_t)))) float16_t;
using float16x16 =
    __attribute__((__vector_size__(16 * sizeof(float16_t)))) float16_t;

using half_t = float16_t;

using bfloat16_t = __bf16;

struct bfloat16x2 {
  bfloat16_t x, y;
};

struct bfloat16x4 {
  bfloat16_t data[4];
};

struct bfloat16x8 {
  bfloat16_t data[8];
};

struct bfloat16x16 {
  bfloat16_t data[16];
};

typedef
    __attribute__((__vector_size__(4 * sizeof(short)))) short bfloat16x4_vec;
typedef
    __attribute__((__vector_size__(8 * sizeof(short)))) short bfloat16x8_vec;

using int32x2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using float32x2 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using float32x32 = __attribute__((__vector_size__(32 * sizeof(float)))) float;

using int8x4 = __attribute__((__vector_size__(4 * sizeof(int8_t)))) int8_t;

TL_DEVICE half_t hcu_habs(half_t a) {
  float v = static_cast<float>(a);
  return static_cast<half_t>(v >= 0.0f ? v : -v);
}

TL_DEVICE bfloat16_t hcu_habs(bfloat16_t a) {
  float v = static_cast<float>(a);
  return static_cast<bfloat16_t>(v >= 0.0f ? v : -v);
}

// __shfl overload for _Float16/half_t
// There is no half_t version of __shfl in HIP, so we implement it ourselves
// here.
TL_DEVICE half_t __shfl(half_t var, int src_lane,
                        int width = __AMDGCN_WAVEFRONT_SIZE) {
  static_assert(sizeof(half_t) == sizeof(unsigned short), "");
  // Bit-preserving shuffle: use union to preserve exact bit pattern.
  union {
    unsigned short us;
    half_t f;
  } tmp;
  tmp.f = var;
  // Cast to int for __shfl, then cast back to preserve exact bits
  int shuffled = __shfl(static_cast<int>(tmp.us), src_lane, width);
  tmp.us = static_cast<unsigned short>(shuffled);
  return tmp.f;
}

// Pack two half_t values.
TL_DEVICE unsigned __pack_half2(const half_t x, const half_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two bfloat16_t values.
TL_DEVICE unsigned __pack_bfloat162(const bfloat16_t x, const bfloat16_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

namespace tl {

namespace detail {

TL_DEVICE constexpr int default_warp_size() { return 64; }

TL_DEVICE int linear_thread_idx_in_block() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

} // namespace detail

TL_DEVICE int get_lane_idx(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() % warp_size;
}

TL_DEVICE int get_warp_idx_sync(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE int get_warp_idx(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE void sync_warp(unsigned long long mask = ~0ull) {
  (void)mask;
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_amdgcn_s_barrier();
#endif
}

TL_DEVICE unsigned long long activemask() {
  return (unsigned long long)__ballot(1);
}

template <typename T>
TL_DEVICE unsigned long long match_any_sync(unsigned long long mask, T value) {
  constexpr int kWaveSize = 64;
  unsigned long long active = activemask() & mask;
  unsigned long long out = 0ull;
#pragma unroll
  for (int src = 0; src < kWaveSize; ++src) {
    if ((active >> src) & 1ull) {
      T src_value = __shfl(value, src, kWaveSize);
      if (src_value == value) {
        out |= (1ull << src);
      }
    }
  }
  return out & active;
}

// Packed x2 element-wise math helpers (scalar float2).
TL_DEVICE float2 add2(float2 a, float2 b) {
  float2 out;
  out.x = a.x + b.x;
  out.y = a.y + b.y;
  return out;
}

TL_DEVICE float2 sub2(float2 a, float2 b) {
  float2 out;
  out.x = a.x - b.x;
  out.y = a.y - b.y;
  return out;
}

TL_DEVICE float2 mul2(float2 a, float2 b) {
  float2 out;
  out.x = a.x * b.x;
  out.y = a.y * b.y;
  return out;
}

TL_DEVICE float2 fma2(float2 a, float2 b, float2 c) {
  float2 out;
  out.x = a.x * b.x + c.x;
  out.y = a.y * b.y + c.y;
  return out;
}

TL_DEVICE float2 max2(float2 a, float2 b) {
  float2 out;
  out.x = (a.x > b.x) ? a.x : b.x;
  out.y = (a.y > b.y) ? a.y : b.y;
  return out;
}

TL_DEVICE float2 min2(float2 a, float2 b) {
  float2 out;
  out.x = (a.x < b.x) ? a.x : b.x;
  out.y = (a.y < b.y) ? a.y : b.y;
  return out;
}

TL_DEVICE float2 abs2(float2 a) {
  float2 out;
  out.x = (a.x >= 0.0f) ? a.x : -a.x;
  out.y = (a.y >= 0.0f) ? a.y : -a.y;
  return out;
}

template <typename T> TL_DEVICE T from_uint1(uint1 v) {
  T r;
  memcpy(&r, &v, sizeof(T));
  return r;
}

template <typename T> TL_DEVICE uint1 to_uint1(T v) {
  uint1 r;
  memcpy(&r, &v, sizeof(uint1));
  return r;
}

TL_DEVICE bfloat16x2 add2(bfloat16x2 a, bfloat16x2 b) {
  return bfloat16x2{bfloat16_t(float(a.x) + float(b.x)),
                    bfloat16_t(float(a.y) + float(b.y))};
}

TL_DEVICE bfloat16x2 sub2(bfloat16x2 a, bfloat16x2 b) {
  return bfloat16x2{bfloat16_t(float(a.x) - float(b.x)),
                    bfloat16_t(float(a.y) - float(b.y))};
}

TL_DEVICE bfloat16x2 mul2(bfloat16x2 a, bfloat16x2 b) {
  return bfloat16x2{bfloat16_t(float(a.x) * float(b.x)),
                    bfloat16_t(float(a.y) * float(b.y))};
}

TL_DEVICE bfloat16x2 fma2(bfloat16x2 a, bfloat16x2 b, bfloat16x2 c) {
  return bfloat16x2{bfloat16_t(float(a.x) * float(b.x) + float(c.x)),
                    bfloat16_t(float(a.y) * float(b.y) + float(c.y))};
}

TL_DEVICE bfloat16x2 max2(bfloat16x2 a, bfloat16x2 b) {
  return bfloat16x2{float(a.x) > float(b.x) ? a.x : b.x,
                    float(a.y) > float(b.y) ? a.y : b.y};
}

TL_DEVICE bfloat16x2 min2(bfloat16x2 a, bfloat16x2 b) {
  return bfloat16x2{float(a.x) < float(b.x) ? a.x : b.x,
                    float(a.y) < float(b.y) ? a.y : b.y};
}

TL_DEVICE bfloat16x2 abs2(bfloat16x2 a) {
  return bfloat16x2{hcu_habs(a.x), hcu_habs(a.y)};
}

TL_DEVICE float16x2 add2(float16x2 a, float16x2 b) {
  float16x2 out;
  out[0] = half_t(float(a[0]) + float(b[0]));
  out[1] = half_t(float(a[1]) + float(b[1]));
  return out;
}

TL_DEVICE float16x2 sub2(float16x2 a, float16x2 b) {
  float16x2 out;
  out[0] = half_t(float(a[0]) - float(b[0]));
  out[1] = half_t(float(a[1]) - float(b[1]));
  return out;
}

TL_DEVICE float16x2 mul2(float16x2 a, float16x2 b) {
  float16x2 out;
  out[0] = half_t(float(a[0]) * float(b[0]));
  out[1] = half_t(float(a[1]) * float(b[1]));
  return out;
}

TL_DEVICE float16x2 fma2(float16x2 a, float16x2 b, float16x2 c) {
  float16x2 out;
  out[0] = half_t(float(a[0]) * float(b[0]) + float(c[0]));
  out[1] = half_t(float(a[1]) * float(b[1]) + float(c[1]));
  return out;
}

TL_DEVICE float16x2 max2(float16x2 a, float16x2 b) {
  float16x2 out;
  out[0] = float(a[0]) > float(b[0]) ? a[0] : b[0];
  out[1] = float(a[1]) > float(b[1]) ? a[1] : b[1];
  return out;
}

TL_DEVICE float16x2 min2(float16x2 a, float16x2 b) {
  float16x2 out;
  out[0] = float(a[0]) < float(b[0]) ? a[0] : b[0];
  out[1] = float(a[1]) < float(b[1]) ? a[1] : b[1];
  return out;
}

TL_DEVICE float16x2 abs2(float16x2 a) {
  float16x2 out;
  out[0] = hcu_habs(a[0]);
  out[1] = hcu_habs(a[1]);
  return out;
}

template <typename T> TL_DEVICE bool Any(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (a[i]) {
      return true;
    }
  }
  return false;
}

template <typename T> TL_DEVICE bool All(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (!a[i]) {
      return false;
    }
  }
  return true;
}

template <typename T> TL_DEVICE T shfl_xor(T val, int delta) {
  return __shfl_xor(val, delta);
}

template <typename T> TL_DEVICE T shfl_down(T val, int delta) {
  return __shfl_down(val, delta);
}

template <typename T> TL_DEVICE T shfl_up(T val, int delta) {
  return __shfl_up(val, delta);
}

template <typename T> TL_DEVICE T shfl(T val, int srcLane) {
  return __shfl(val, srcLane);
}

template <> TL_DEVICE half_t shfl_xor(half_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_xor(f, delta);
  return half_t(r);
}

template <> TL_DEVICE half_t shfl_down(half_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_down(f, delta);
  return half_t(r);
}

template <> TL_DEVICE half_t shfl_up(half_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_up(f, delta);
  return half_t(r);
}

template <> TL_DEVICE half_t shfl(half_t val, int srcLane) {
  float f = static_cast<float>(val);
  float r = __shfl(f, srcLane);
  return half_t(r);
}

template <> TL_DEVICE bfloat16_t shfl_xor(bfloat16_t val, int laneMask) {
  float f = static_cast<float>(val);
  float r = __shfl_xor(f, laneMask);
  return static_cast<bfloat16_t>(r);
}

template <> TL_DEVICE bfloat16_t shfl_down(bfloat16_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_down(f, delta);
  return static_cast<bfloat16_t>(r);
}

template <> TL_DEVICE bfloat16_t shfl_up(bfloat16_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_up(f, delta);
  return static_cast<bfloat16_t>(r);
}

template <> TL_DEVICE bfloat16_t shfl(bfloat16_t val, int srcLane) {
  float f = static_cast<float>(val);
  float r = __shfl(f, srcLane);
  return static_cast<bfloat16_t>(r);
}

} // namespace tl
