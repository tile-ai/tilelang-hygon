// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <tl_templates/hcu/common.h>
#include <tl_templates/hcu/core/numeric/pk_fp4.hpp>

namespace tl {
namespace hcu {

// Private codegen IDs: low = fp8/bf8/fp4, high = f16/bf16/f32.
using scale_float2 = float __attribute__((ext_vector_type(2)));
using scale_half2 = _Float16 __attribute__((ext_vector_type(2)));
using scale_short2 = short __attribute__((ext_vector_type(2)));

// Instruction adapters: only this layer names hardware builtins and controls.
// Widen: low -> high (dequantization, multiply by the decoded E8M0 scale).
// Narrow: high -> low (quantization, divide by scale then round).
// Pair means two data elements sharing one scale, not a pair of scales.
template <int Low, int High, bool Clamp>
TL_DEVICE auto scaled_widen_scalar(int source, int scale) {
  static_assert(Low < 2 && (High == 0 || High == 2));
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx946__)
  if constexpr (High == 0) {
    // Current HCU ABI: src, scale, old, src_sel, dst_sel, clamp, scale_sel.
    // dst_sel is absent from the original instruction-table signature.
    if constexpr (Low == 0)
      return __builtin_hcu_cvt_scale_f16_fp8(source, scale, 0, 0, false, Clamp,
                                             0);
    else
      return __builtin_hcu_cvt_scale_f16_bf8(source, scale, 0, 0, false, Clamp,
                                             0);
  } else {
    if constexpr (Low == 0)
      return __builtin_hcu_cvt_scale_f32_fp8(source, scale, 0, Clamp, 0);
    else
      return __builtin_hcu_cvt_scale_f32_bf8(source, scale, 0, Clamp, 0);
  }
#else
  if constexpr (High == 0)
    return _Float16{};
  else
    return float{};
#endif
}

template <int Low, int High, bool Clamp>
TL_DEVICE auto scaled_widen_pair(int source, int scale) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx946__)
  // Match the toolchain's amd_hip_fp4.h: packed FP4 uses selector 4 for
  // byte zero. Selector 0 does not describe the same FP4 encoding.
  constexpr int selector = Low == 2 ? 4 : 0;
#define TL_SCALE_WIDEN(L, NAME)                                                \
  if constexpr (Low == L) {                                                    \
    if constexpr (High == 0)                                                   \
      return __builtin_hcu_cvt_scale_pk_f16_##NAME(source, scale, selector,    \
                                                   Clamp, 0);                  \
    else if constexpr (High == 1)                                              \
      return __builtin_hcu_cvt_scale_pk_bf16_##NAME(source, scale, selector,   \
                                                    Clamp, 0);                 \
    else                                                                       \
      return __builtin_hcu_cvt_scale_pk_f32_##NAME(source, scale, selector,    \
                                                   Clamp, 0);                  \
  }
  TL_SCALE_WIDEN(0, fp8)
  TL_SCALE_WIDEN(1, bf8)
  TL_SCALE_WIDEN(2, fp4)
#undef TL_SCALE_WIDEN
#else
  if constexpr (High == 2)
    return scale_float2{};
  else
    return scale_short2{};
#endif
}

template <int Low, int High, typename Pair>
TL_DEVICE int scaled_narrow_pair(Pair source, int scale) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx946__)
  constexpr int selector = Low == 2 ? 4 : 0;
#define TL_SCALE_NARROW(L, NAME)                                               \
  if constexpr (Low == L) {                                                    \
    if constexpr (High == 0)                                                   \
      return __builtin_hcu_cvt_scale_pk_##NAME##_f16(                          \
          __builtin_bit_cast(scale_half2, source), scale, 0, selector, 0);     \
    else if constexpr (High == 1)                                              \
      return __builtin_hcu_cvt_scale_pk_##NAME##_bf16(                         \
          __builtin_bit_cast(scale_half2, source), scale, 0, selector, 0);     \
    else                                                                       \
      return __builtin_hcu_cvt_scale_pk_##NAME##_f32(source[0], source[1],     \
                                                     scale, 0, selector, 0);   \
  }
  TL_SCALE_NARROW(0, fp8)
  TL_SCALE_NARROW(1, bf8)
  TL_SCALE_NARROW(2, fp4)
#undef TL_SCALE_NARROW
#endif
  return 0;
}

// Register-carrier adapters: pair/extract/assemble values through the
// instruction adapters above; do not select or invoke builtins here.
// FP4 objects follow TIR's contiguous nibble storage. An unpaired element
// uses a native scalar CVT where available; otherwise the unused slot is zero.
template <typename Result, int Lanes, int Low, int High, bool Widen, bool Clamp,
          bool SharedScale, typename Source, typename Scale>
TL_DEVICE Result scaled_cast_convert(Source source, Scale scale) {
  constexpr int low_bits = Low == 2 ? 4 : 8;
  constexpr int high_bytes = High == 2 ? 4 : 2;
  static_assert(sizeof(Scale) == (SharedScale ? 1 : Lanes),
                "scale ABI mismatch");
  static_assert(sizeof(Source) >=
                    (Widen ? (Lanes * low_bits + 7) / 8 : Lanes * high_bytes),
                "source ABI mismatch");
  static_assert(sizeof(Result) >=
                    (Widen ? Lanes * high_bytes : (Lanes * low_bits + 7) / 8),
                "result ABI mismatch");
  Result result{};
  const auto *src = reinterpret_cast<const unsigned char *>(&source);
  const auto *sf = reinterpret_cast<const unsigned char *>(&scale);
  auto *dst = reinterpret_cast<unsigned char *>(&result);
#pragma unroll
  for (int i = 0; i < Lanes; i += (SharedScale ? 2 : 1)) {
    const int j = SharedScale && i + 1 < Lanes ? i + 1 : i;
    const int s = sf[SharedScale ? 0 : i];
    if constexpr (Widen) {
      auto get = [&](int lane) {
        return (src[lane * low_bits / 8] >> (lane * low_bits % 8)) &
               ((1 << low_bits) - 1);
      };
      if constexpr (Low < 2 && High != 1) {
        if (j == i) {
          auto value = scaled_widen_scalar<Low, High, Clamp>(get(i), s);
          __builtin_memcpy(dst + i * high_bytes, &value, high_bytes);
          continue;
        }
      }
      const int packed = get(i) | (j != i ? get(j) << low_bits : 0);
      auto pair = scaled_widen_pair<Low, High, Clamp>(packed, s);
      __builtin_memcpy(dst + i * high_bytes, &pair, high_bytes);
      if (j != i)
        __builtin_memcpy(dst + j * high_bytes,
                         reinterpret_cast<const unsigned char *>(&pair) +
                             high_bytes,
                         high_bytes);
    } else {
      int packed;
      if constexpr (High == 2) {
        scale_float2 pair{};
        __builtin_memcpy(&pair, src + i * high_bytes, high_bytes);
        if (j != i)
          __builtin_memcpy(reinterpret_cast<unsigned char *>(&pair) +
                               high_bytes,
                           src + j * high_bytes, high_bytes);
        packed = scaled_narrow_pair<Low, High>(pair, s);
      } else {
        scale_half2 pair{};
        __builtin_memcpy(&pair, src + i * high_bytes, high_bytes);
        if (j != i)
          __builtin_memcpy(reinterpret_cast<unsigned char *>(&pair) +
                               high_bytes,
                           src + j * high_bytes, high_bytes);
        packed = scaled_narrow_pair<Low, High>(pair, s);
      }
      auto put = [&](int lane, unsigned value) {
        dst[lane * low_bits / 8] |= (value & ((1 << low_bits) - 1))
                                    << (lane * low_bits % 8);
      };
      put(i, packed);
      if (j != i)
        put(j, unsigned(packed) >> low_bits);
    }
  }
  return result;
}

// Input/output objects are packed register carriers. Accessing their object
// representation as unsigned char preserves all FP8/E8M0 encodings, including
// int8 scales with the high bit set. No numeric cast is involved.
template <typename Result, int Lanes, bool BF8, bool BF16, bool Clamp,
          bool SharedScale, typename Source, typename Scale>
TL_DEVICE Result scaled_cast(Source source, Scale scale) {
  static_assert(sizeof(Source) == Lanes, "scaled_cast source ABI mismatch");
  static_assert(sizeof(Scale) == (SharedScale ? 1 : Lanes),
                "scaled_cast scale ABI mismatch");
  static_assert(sizeof(Result) == 2 * Lanes, "scaled_cast result ABI mismatch");
  Result result{};
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx946__)
  const auto *src = reinterpret_cast<const unsigned char *>(&source);
  const auto *sf = reinterpret_cast<const unsigned char *>(&scale);
  auto *dst = reinterpret_cast<unsigned char *>(&result);
#pragma unroll
  for (int i = 0; i < Lanes; i += (SharedScale ? 2 : 1)) {
    const int s = sf[SharedScale ? 0 : i];
    if constexpr (SharedScale && Lanes > 1) {
      if (i + 1 < Lanes) {
        const int packed = int(src[i]) | (int(src[i + 1]) << 8);
        auto converted =
            scaled_widen_pair<BF8 ? 1 : 0, BF16 ? 1 : 0, Clamp>(packed, s);
        __builtin_memcpy(dst + 2 * i, &converted, 4);
        continue;
      }
    }
    if constexpr (BF16) {
      // No scalar BF16 instruction. The unused input slot is zero, not a
      // duplicate of src[i]; only the first converted result is consumed.
      const int packed = int(src[i]);
      auto converted = scaled_widen_pair<BF8 ? 1 : 0, 1, Clamp>(packed, s);
      __builtin_memcpy(dst + 2 * i, &converted, 2);
    } else {
      auto converted =
          scaled_widen_scalar<BF8 ? 1 : 0, 0, Clamp>(int(src[i]), s);
      __builtin_memcpy(dst + 2 * i, &converted, 2);
    }
  }
#elif defined(__HIP_DEVICE_COMPILE__)
  static_assert(sizeof(Result) == 0, "scaled_cast requires HCU gfx946");
#endif
  return result;
}

} // namespace hcu
} // namespace tl
