#pragma once

#include <ck_tile/core.hpp>
#include <hip/amd_detail/amd_warp_functions.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

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

#define TL_DEVICE __forceinline__ __device__
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
using bfloat16_t = ck_tile::bf16_t;
using bfloat16x2 = ck_tile::bf16x2_t;
using bfloat16x4 = ck_tile::bf16x4_t;
using bfloat16x8 = ck_tile::bf16x8_t;
using bfloat16x16 = ck_tile::bf16x16_t;

// Tilelang's own bfloat16_t implementation
// This provides true conversion with float and supports conversion to ck_tile::bf16_t (ushort)
struct __align__(2) bf16_cvt_t {
  using raw_type = uint16_t;
  raw_type data;

  // Static method for bit_cast
  TL_DEVICE static constexpr bf16_cvt_t bit_cast(raw_type x) {
    bf16_cvt_t y;
    y.data = x;
    return y;
  }

  // Default constructor
  TL_DEVICE constexpr bf16_cvt_t() : data() {}

  // Construct from float - use ck_tile's conversion function
  #if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx946__))
  TL_DEVICE explicit bf16_cvt_t(const float& x)
    : data(__builtin_hcu_cvt_bf16_f32(x, false/*clamp*/, 0/*dst_sel:WORD_1*/)) {}
  #else
  TL_DEVICE explicit constexpr bf16_cvt_t(const float& x)
    : data(ck_tile::float_to_bf16_raw<ck_tile::bf16_rounding_mode::standard_asm>(x)) {}
  #endif

  // Construct from double - use ck_tile's conversion function
  #if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx946__))
  TL_DEVICE explicit bf16_cvt_t(const double& x)
    : data(__builtin_hcu_cvt_bf16_f32(static_cast<float>(x), false/*clamp*/, 0/*dst_sel:WORD_1*/)) {}
  #else
  TL_DEVICE explicit constexpr bf16_cvt_t(const double& x)
    : data(ck_tile::float_to_bf16_raw<ck_tile::bf16_rounding_mode::standard_asm>(static_cast<float>(x))) {}
  #endif

  // Construct from int
  #if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx946__))
  TL_DEVICE explicit bf16_cvt_t(const int& x)
    : data(__builtin_hcu_cvt_bf16_f32(static_cast<float>(x), false/*clamp*/, 0/*dst_sel:WORD_1*/)) {}
  #else
  TL_DEVICE explicit constexpr bf16_cvt_t(const int& x)
    : data(ck_tile::float_to_bf16_raw<ck_tile::bf16_rounding_mode::standard_asm>(static_cast<float>(x))) {}
  #endif

  // Construct from unsigned int
  #if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx946__))
  TL_DEVICE explicit bf16_cvt_t(const unsigned int& x)
    : data(__builtin_hcu_cvt_bf16_f32(static_cast<float>(x), false/*clamp*/, 0/*dst_sel:WORD_1*/)) {}
  #else
  TL_DEVICE explicit constexpr bf16_cvt_t(const unsigned int& x)
    : data(ck_tile::float_to_bf16_raw<ck_tile::bf16_rounding_mode::standard_asm>(static_cast<float>(x))) {}
  #endif

  // Construct from ck_tile::bf16_t (which is ushort when CK_TILE_USE_CUSTOM_DATA_TYPE is off)
  // Direct bit_cast since ck_tile::bf16_t is just ushort in that case
  TL_DEVICE constexpr bf16_cvt_t(const ck_tile::bf16_t& v)
    : data(ck_tile::bit_cast<uint16_t>(v)) {}

  // Cast to float - use ck_tile's conversion function
  #if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx946__))
  TL_DEVICE explicit operator float() const {
    return __builtin_hcu_cvt_f32_bf16(data, false/*clamp*/, 0/*omod*/, 0/*src0_sel:WORD_1*/);
  }
  #else
  TL_DEVICE explicit constexpr operator float() const {
    return ck_tile::bf16_to_float_raw(data);
  }
  #endif

  // Cast to double - use ck_tile's conversion function
  #if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx946__))
  TL_DEVICE explicit operator double() const {
    return static_cast<double>(__builtin_hcu_cvt_f32_bf16(data, false/*clamp*/, 0/*omod*/, 0/*src0_sel:WORD_1*/));
  }
  #else
  TL_DEVICE explicit constexpr operator double() const {
    return static_cast<double>(ck_tile::bf16_to_float_raw(data));
  }
  #endif

  // Cast to int
  #if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx946__))
  TL_DEVICE explicit operator int() const {
    return static_cast<int>(__builtin_hcu_cvt_f32_bf16(data, false/*clamp*/, 0/*omod*/, 0/*src0_sel:WORD_1*/));
  }
  #else
  TL_DEVICE explicit constexpr operator int() const {
    return static_cast<int>(ck_tile::bf16_to_float_raw(data));
  }
  #endif

  // Conversion to ck_tile::bf16_t (ushort when CK_TILE_USE_CUSTOM_DATA_TYPE is off)
  // This allows seamless interoperability with ck_tile
  TL_DEVICE constexpr operator ck_tile::bf16_t() const {
    return ck_tile::bit_cast<ck_tile::bf16_t>(data);
  }

  // Internal access
  TL_DEVICE constexpr raw_type& get() { return data; }
  TL_DEVICE constexpr raw_type get() const { return data; }
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

// __shfl overload for _Float16/half_t
// There is no half_t version of __shfl in HIP, so we implement it ourselves here.
TL_DEVICE half_t __shfl(half_t var, int src_lane, int width = __AMDGCN_WAVEFRONT_SIZE) {
    static_assert(sizeof(half_t) == sizeof(unsigned short), "");
    // Bit-preserving shuffle: use union to preserve exact bit pattern.
    union { unsigned short us; half_t f; } tmp; tmp.f = var;
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

// Packed x2 element-wise math helpers (scalar float2 fallbacks; matches HIP tl::add2…).

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
  float f = ck_tile::type_convert<float>(val);
  float r = __shfl_xor(f, laneMask);
  return ck_tile::type_convert<bfloat16_t>(r);
}

template <> TL_DEVICE bfloat16_t shfl_down(bfloat16_t val, int delta) {
  float f = ck_tile::type_convert<float>(val);
  float r = __shfl_down(f, delta);
  return ck_tile::type_convert<bfloat16_t>(r);
}

template <> TL_DEVICE bfloat16_t shfl_up(bfloat16_t val, int delta) {
  float f = ck_tile::type_convert<float>(val);
  float r = __shfl_up(f, delta);
  return ck_tile::type_convert<bfloat16_t>(r);
}

template <> TL_DEVICE bfloat16_t shfl(bfloat16_t val, int srcLane) {
  float f = ck_tile::type_convert<float>(val);
  float r = __shfl(f, srcLane);
  return ck_tile::type_convert<bfloat16_t>(r);
}

} // namespace tl
