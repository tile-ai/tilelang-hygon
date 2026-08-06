// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

/*
 * mls_ds_traits: maps (MlsAtom, SrcBits, LdsBits, RegBits, Alt) ->
 * DsFormatInst
 * (ds_read_matrix_format). Uses tilelang DsreadmFormatDispatcher
 * (builtin-based). MlsAtom: tl::mls::gfx938_* or tl::mls::gfx946_* from
 * mls_atom_for_tile.
 */

#include <tl_templates/hcu/core.hpp>

#include <tl_templates/hcu/mls/tl_dsreadm_format_dispatcher.hpp>

namespace tl {
namespace mls {

template <typename MlsAtom, ::tl::index_t SrcBits, ::tl::index_t LdsBits,
          ::tl::index_t RegBits, ::tl::index_t Alt>
struct mls_ds_traits;

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
#include <tl_templates/hcu/mls/tl_mls_atom_gfx938.hpp>
// gfx938 b16
template <::tl::index_t Alt>
struct mls_ds_traits<tl::mls::gfx938_mls_32x16_b16, 16, 16, 16, Alt> {
  using Type = DsreadmFormatDispatcher<2, 32, 16, Alt, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x32_trans_b16, 16, 16, 16, 1> {
  using Type = DsreadmFormatDispatcher<2, 32, 16, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x32_trans_b16, 16, 16, 16, 2> {
  using Type = DsreadmFormatDispatcher<2, 16, 32, 2, true>;
};

template <::tl::index_t Alt>
struct mls_ds_traits<tl::mls::gfx938_mls_32x32_b16, 16, 16, 16, Alt> {
  using Type = DsreadmFormatDispatcher<2, 32, 16, Alt, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_32x32_trans_b16, 16, 16, 16, 1> {
  using Type = DsreadmFormatDispatcher<2, 32, 16, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_32x32_trans_b16, 16, 16, 16, 2> {
  using Type = DsreadmFormatDispatcher<2, 16, 32, 2, true>;
};

template <::tl::index_t Alt>
struct mls_ds_traits<tl::mls::gfx938_mls_64x16_b16, 16, 16, 16, Alt> {
  using Type = DsreadmFormatDispatcher<2, 32, 16, Alt, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x64_trans_b16, 16, 16, 16, 1> {
  using Type = DsreadmFormatDispatcher<2, 32, 16, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x64_trans_b16, 16, 16, 16, 2> {
  using Type = DsreadmFormatDispatcher<2, 16, 32, 2, true>;
};

template <> struct mls_ds_traits<tl::mls::gfx938_mls_64x16_b8, 8, 8, 8, 1> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 1, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx938_mls_64x16_b8, 8, 8, 8, 2> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 2, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx938_mls_64x16_b8, 8, 8, 8, 4> {
  using Type = DsreadmFormatDispatcher<1, 64, 16, 4, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx938_mls_128x16_b8, 8, 8, 8, 1> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 1, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx938_mls_128x16_b8, 8, 8, 8, 2> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 2, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx938_mls_128x16_b8, 8, 8, 8, 4> {
  using Type = DsreadmFormatDispatcher<1, 64, 16, 4, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx938_mls_64x32_b8, 8, 8, 8, 1> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 1, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx938_mls_64x32_b8, 8, 8, 8, 2> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 2, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx938_mls_64x32_b8, 8, 8, 8, 4> {
  using Type = DsreadmFormatDispatcher<1, 64, 16, 4, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x64_trans_b8, 8, 8, 8, 1> {
  using Type = DsreadmFormatDispatcher<1, 16, 64, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x128_trans_b8, 8, 8, 8, 1> {
  using Type = DsreadmFormatDispatcher<1, 16, 64, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x128_trans_b8, 8, 8, 8, 2> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x128_trans_b8, 8, 8, 8, 4> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_32x64_trans_b8, 8, 8, 8, 1> {
  using Type = DsreadmFormatDispatcher<1, 16, 64, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_32x64_trans_b8, 8, 8, 8, 2> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_32x64_trans_b8, 8, 8, 8, 4> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

#endif

#if !defined(__HIP_DEVICE_COMPILE__) ||                                        \
    (defined(__gfx946__) || defined(__gfx92a__))
#include <tl_templates/hcu/mls/tl_mls_atom_gfx946.hpp>

template <::tl::index_t Alt>
struct mls_ds_traits<tl::mls::gfx946_mls_32x16_b16, 16, 16, 16, Alt> {
  using Type = DsreadmFormatDispatcher<2, 32, 16, Alt, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x32_trans_b16, 16, 16, 16, 1> {
  using Type = DsreadmFormatDispatcher<2, 32, 16, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x32_trans_b16, 16, 16, 16, 2> {
  using Type = DsreadmFormatDispatcher<2, 16, 32, 2, true>;
};

template <::tl::index_t Alt>
struct mls_ds_traits<tl::mls::gfx946_mls_32x32_b16, 16, 16, 16, Alt> {
  using Type = DsreadmFormatDispatcher<2, 32, 16, Alt, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_32x32_trans_b16, 16, 16, 16, 1> {
  using Type = DsreadmFormatDispatcher<2, 32, 16, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_32x32_trans_b16, 16, 16, 16, 2> {
  using Type = DsreadmFormatDispatcher<2, 16, 32, 2, true>;
};

template <::tl::index_t Alt>
struct mls_ds_traits<tl::mls::gfx946_mls_64x16_b16, 16, 16, 16, Alt> {
  using Type = DsreadmFormatDispatcher<2, 32, 16, Alt, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x64_trans_b16, 16, 16, 16, 1> {
  using Type = DsreadmFormatDispatcher<2, 32, 16, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x64_trans_b16, 16, 16, 16, 2> {
  using Type = DsreadmFormatDispatcher<2, 16, 32, 2, true>;
};

template <> struct mls_ds_traits<tl::mls::gfx946_mls_64x16_b8, 8, 8, 8, 1> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 1, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx946_mls_64x16_b8, 8, 8, 8, 2> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 2, false>;
};

// gfx946 non-trans b8 packed shapes share the same ds_read_format mapping.
template <> struct mls_ds_traits<tl::mls::gfx946_mls_64x16_b8, 8, 8, 8, 4> {
  using Type = DsreadmFormatDispatcher<1, 64, 16, 4, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx946_mls_128x16_b8, 8, 8, 8, 1> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 1, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx946_mls_128x16_b8, 8, 8, 8, 2> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 2, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx946_mls_128x16_b8, 8, 8, 8, 4> {
  using Type = DsreadmFormatDispatcher<1, 64, 16, 4, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx946_mls_64x32_b8, 8, 8, 8, 1> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 1, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx946_mls_64x32_b8, 8, 8, 8, 2> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 2, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx946_mls_64x32_b8, 8, 8, 8, 4> {
  using Type = DsreadmFormatDispatcher<1, 64, 16, 4, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x64_trans_b8, 8, 8, 8, 1> {
  using Type = DsreadmFormatDispatcher<1, 16, 64, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x128_trans_b8, 8, 8, 8, 1> {
  using Type = DsreadmFormatDispatcher<1, 16, 64, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x128_trans_b8, 8, 8, 8, 2> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x128_trans_b8, 8, 8, 8, 4> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_32x64_trans_b8, 8, 8, 8, 1> {
  using Type = DsreadmFormatDispatcher<1, 16, 64, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_32x64_trans_b8, 8, 8, 8, 2> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_32x64_trans_b8, 8, 8, 8, 4> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

template <> struct mls_ds_traits<tl::mls::gfx946_mls_128x16_b4, 4, 4, 4, 1> {
  using Type = DsreadmFormatDispatcher<4, 32, 64, 1, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x128_trans_b4, 4, 4, 4, 1> {
  using Type = DsreadmFormatDispatcher<4, 32, 64, 1, true>;
};

template <> struct mls_ds_traits<tl::mls::gfx946_mls_256x16_b4, 4, 4, 4, 1> {
  using Type = DsreadmFormatDispatcher<4, 32, 64, 1, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x256_trans_b4, 4, 4, 4, 1> {
  using Type = DsreadmFormatDispatcher<4, 16, 128, 1, true>;
};

template <> struct mls_ds_traits<tl::mls::gfx946_mls_64x16_fp4, 4, 8, 8, 1> {
  using Type = DsreadmFormatDispatcher<1, 32, 32, 1, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x64_trans_fp4, 4, 8, 8, 1> {
  using Type = DsreadmFormatDispatcher<1, 16, 64, 1, true>;
};

template <> struct mls_ds_traits<tl::mls::gfx946_mls_128x16_b4, 4, 4, 8, 1> {
  using Type = DsreadmPadByteDispatcher<32, 32, 0, false>;
};

template <> struct mls_ds_traits<tl::mls::gfx946_mls_256x16_b4, 4, 4, 8, 1> {
  using Type = DsreadmPadByteDispatcher<32, 32, 0, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x128_trans_b4, 4, 4, 8, 1> {
  using Type = DsreadmPadByteDispatcher<32, 32, 0, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x256_trans_b4, 4, 4, 8, 1> {
  using Type = DsreadmPadByteDispatcher<32, 32, 0, true>;
};

#endif

// Primary: no specialization for (MlsAtom, SrcBits, LdsBits, RegBits, Alt)
template <typename MlsAtom, ::tl::index_t SrcBits, ::tl::index_t LdsBits,
          ::tl::index_t RegBits, ::tl::index_t Alt>
struct mls_ds_traits {
  static_assert(sizeof(MlsAtom) == 0,
                "Unsupported (MlsAtom, SrcBits, LdsBits, RegBits, Alt). "
                "Add specialization in mls_ds_traits.hpp. "
                "Known gfx938 b16: gfx938_mls_32x16_b16, 16x32_trans, 32x32, "
                "32x32_trans, "
                "64x16, 16x64_trans.");
};

} // namespace mls
} // namespace tl
