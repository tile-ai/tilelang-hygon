// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

/*
 * mls_atom_for_tile maps
 * (MlsTileMN, MlsTileK, Trans, SrcBits, DstBits, HcuArch) to the concrete
 * matrix_load atom. Native paths use SrcBits == DstBits; mixed precision paths
 * can add explicit specializations.
 */

#include <tl_templates/hcu/core.hpp>

namespace tl {
namespace mls {

template <::tl::index_t MlsTileMN, ::tl::index_t MlsTileK, bool Trans,
          ::tl::index_t SrcBits, ::tl::index_t DstBits,
          ::tl::hcu_target_enum HcuArch>
struct mls_atom_for_tile;

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
#include <tl_templates/hcu/mls/tl_mls_atom_gfx938.hpp>

template <>
struct mls_atom_for_tile<16, 64, true, 16, 16, ::tl::hcu_target_enum::gfx938> {
  using Type = tl::mls::gfx938_mls_16x64_trans_b16;
};

template <>
struct mls_atom_for_tile<32, 32, true, 16, 16, ::tl::hcu_target_enum::gfx938> {
  using Type = tl::mls::gfx938_mls_32x32_trans_b16;
};

template <>
struct mls_atom_for_tile<16, 32, true, 16, 16, ::tl::hcu_target_enum::gfx938> {
  using Type = tl::mls::gfx938_mls_16x32_trans_b16;
};

template <>
struct mls_atom_for_tile<32, 32, false, 16, 16, ::tl::hcu_target_enum::gfx938> {
  using Type = tl::mls::gfx938_mls_32x32_b16;
};

template <>
struct mls_atom_for_tile<32, 16, false, 16, 16, ::tl::hcu_target_enum::gfx938> {
  using Type = tl::mls::gfx938_mls_32x16_b16;
};

template <>
struct mls_atom_for_tile<64, 16, false, 16, 16, ::tl::hcu_target_enum::gfx938> {
  using Type = tl::mls::gfx938_mls_64x16_b16;
};

template <>
struct mls_atom_for_tile<64, 16, false, 8, 8, ::tl::hcu_target_enum::gfx938> {
  using Type = tl::mls::gfx938_mls_64x16_b8;
};

template <>
struct mls_atom_for_tile<16, 64, true, 8, 8, ::tl::hcu_target_enum::gfx938> {
  using Type = tl::mls::gfx938_mls_16x64_trans_b8;
};

template <>
struct mls_atom_for_tile<64, 32, false, 8, 8, ::tl::hcu_target_enum::gfx938> {
  using Type = tl::mls::gfx938_mls_64x32_b8;
};

template <>
struct mls_atom_for_tile<32, 64, true, 8, 8, ::tl::hcu_target_enum::gfx938> {
  using Type = tl::mls::gfx938_mls_32x64_trans_b8;
};

template <>
struct mls_atom_for_tile<128, 16, false, 8, 8, ::tl::hcu_target_enum::gfx938> {
  using Type = tl::mls::gfx938_mls_128x16_b8;
};

template <>
struct mls_atom_for_tile<16, 128, true, 8, 8, ::tl::hcu_target_enum::gfx938> {
  using Type = tl::mls::gfx938_mls_16x128_trans_b8;
};
#endif

#if !defined(__HIP_DEVICE_COMPILE__) ||                                        \
    (defined(__gfx946__) || defined(__gfx92a__))
#include <tl_templates/hcu/mls/tl_mls_atom_gfx946.hpp>

template <>
struct mls_atom_for_tile<16, 64, true, 16, 16, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_16x64_trans_b16;
};

template <>
struct mls_atom_for_tile<32, 32, true, 16, 16, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_32x32_trans_b16;
};

template <>
struct mls_atom_for_tile<16, 32, true, 16, 16, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_16x32_trans_b16;
};

template <>
struct mls_atom_for_tile<32, 32, false, 16, 16, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_32x32_b16;
};

template <>
struct mls_atom_for_tile<32, 16, false, 16, 16, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_32x16_b16;
};

template <>
struct mls_atom_for_tile<64, 16, false, 16, 16, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_64x16_b16;
};

template <>
struct mls_atom_for_tile<64, 16, false, 8, 8, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_64x16_b8;
};

template <>
struct mls_atom_for_tile<16, 64, true, 8, 8, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_16x64_trans_b8;
};

template <>
struct mls_atom_for_tile<64, 32, false, 8, 8, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_64x32_b8;
};

template <>
struct mls_atom_for_tile<32, 64, true, 8, 8, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_32x64_trans_b8;
};

template <>
struct mls_atom_for_tile<128, 16, false, 8, 8, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_128x16_b8;
};

template <>
struct mls_atom_for_tile<16, 128, true, 8, 8, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_16x128_trans_b8;
};

template <>
struct mls_atom_for_tile<128, 16, false, 4, 4, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_128x16_b4;
};

template <>
struct mls_atom_for_tile<16, 128, true, 4, 4, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_16x128_trans_b4;
};

template <>
struct mls_atom_for_tile<256, 16, false, 4, 4, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_256x16_b4;
};

template <>
struct mls_atom_for_tile<16, 256, true, 4, 4, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_16x256_trans_b4;
};

template <>
struct mls_atom_for_tile<64, 16, false, 4, 8, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_64x16_fp4;
};

template <>
struct mls_atom_for_tile<16, 64, true, 4, 8, ::tl::hcu_target_enum::gfx946> {
  using Type = tl::mls::gfx946_mls_16x64_trans_fp4;
};

template <>
struct mls_atom_for_tile<16, 64, true, 16, 16, ::tl::hcu_target_enum::gfx92a> {
  using Type = tl::mls::gfx946_mls_16x64_trans_b16;
};

template <>
struct mls_atom_for_tile<32, 32, true, 16, 16, ::tl::hcu_target_enum::gfx92a> {
  using Type = tl::mls::gfx946_mls_32x32_trans_b16;
};

template <>
struct mls_atom_for_tile<16, 32, true, 16, 16, ::tl::hcu_target_enum::gfx92a> {
  using Type = tl::mls::gfx946_mls_16x32_trans_b16;
};

template <>
struct mls_atom_for_tile<32, 32, false, 16, 16, ::tl::hcu_target_enum::gfx92a> {
  using Type = tl::mls::gfx946_mls_32x32_b16;
};

template <>
struct mls_atom_for_tile<32, 16, false, 16, 16, ::tl::hcu_target_enum::gfx92a> {
  using Type = tl::mls::gfx946_mls_32x16_b16;
};

template <>
struct mls_atom_for_tile<64, 16, false, 16, 16, ::tl::hcu_target_enum::gfx92a> {
  using Type = tl::mls::gfx946_mls_64x16_b16;
};

template <>
struct mls_atom_for_tile<64, 16, false, 8, 8, ::tl::hcu_target_enum::gfx92a> {
  using Type = tl::mls::gfx946_mls_64x16_b8;
};

template <>
struct mls_atom_for_tile<16, 64, true, 8, 8, ::tl::hcu_target_enum::gfx92a> {
  using Type = tl::mls::gfx946_mls_16x64_trans_b8;
};

template <>
struct mls_atom_for_tile<64, 32, false, 8, 8, ::tl::hcu_target_enum::gfx92a> {
  using Type = tl::mls::gfx946_mls_64x32_b8;
};

template <>
struct mls_atom_for_tile<32, 64, true, 8, 8, ::tl::hcu_target_enum::gfx92a> {
  using Type = tl::mls::gfx946_mls_32x64_trans_b8;
};

template <>
struct mls_atom_for_tile<128, 16, false, 8, 8, ::tl::hcu_target_enum::gfx92a> {
  using Type = tl::mls::gfx946_mls_128x16_b8;
};

template <>
struct mls_atom_for_tile<16, 128, true, 8, 8, ::tl::hcu_target_enum::gfx92a> {
  using Type = tl::mls::gfx946_mls_16x128_trans_b8;
};

template <>
struct mls_atom_for_tile<64, 16, false, 4, 8, ::tl::hcu_target_enum::gfx92a> {
  using Type = tl::mls::gfx946_mls_64x16_fp4;
};

template <>
struct mls_atom_for_tile<16, 64, true, 4, 8, ::tl::hcu_target_enum::gfx92a> {
  using Type = tl::mls::gfx946_mls_16x64_trans_fp4;
};
#endif

// Primary: no specialization
template <::tl::index_t MlsTileMN, ::tl::index_t MlsTileK, bool Trans,
          ::tl::index_t SrcBits, ::tl::index_t DstBits,
          ::tl::hcu_target_enum HcuArch>
struct mls_atom_for_tile {
  static_assert(MlsTileMN != MlsTileMN,
                "Unsupported (MlsTileMN, MlsTileK, Trans, SrcBits, DstBits, "
                "HcuArch). Add specialization in tl_mls_atom_dispatcher.hpp.");
};

} // namespace mls
} // namespace tl
