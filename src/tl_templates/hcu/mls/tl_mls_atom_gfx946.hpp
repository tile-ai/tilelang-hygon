// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

/*
 * tilelang MlsAtom for gfx946 - load() uses __builtin_hcu_matrix_load_* with
 * bps parameter. Ported from ck_tile/core/arch/hcu_mls_atom_gfx946.hpp.
 */

#include <tl_templates/hcu/core.hpp>

#include <tl_templates/hcu/mls/tl_mls_atom_gfx938.hpp>

namespace tl {
namespace mls {

struct gfx946_mls_32x16_b16 {
  static constexpr auto TileShape = ::tl::sequence<32, 16>{};

  template <::tl::index_t moffset, bool r, bool bps = false>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>,
                             ::tl::bool_constant<bps> = {}) {
    __builtin_hcu_matrix_load_32X16_b16(rsrc, lds_addr, moffset, false, r,
                                        false, false, bps);
  }
};

struct gfx946_mls_16x32_trans_b16 {
  static constexpr auto TileShape = ::tl::sequence<16, 32>{};

  template <::tl::index_t moffset, bool r, bool bps = false>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>,
                             ::tl::bool_constant<bps> = {}) {
    __builtin_hcu_matrix_load_32X16_b16(rsrc, lds_addr, moffset, true, r, false,
                                        false, bps);
  }
};

struct gfx946_mls_32x32_b16 {
  static constexpr auto TileShape = ::tl::sequence<32, 32>{};

  template <::tl::index_t moffset, bool r, bool bps = false>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>,
                             ::tl::bool_constant<bps> = {}) {
    __builtin_hcu_matrix_load_32X32_b16(rsrc, lds_addr, moffset, false, r,
                                        false, false, bps);
  }
};

struct gfx946_mls_32x32_trans_b16 {
  static constexpr auto TileShape = ::tl::sequence<32, 32>{};

  template <::tl::index_t moffset, bool r, bool bps = false>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>,
                             ::tl::bool_constant<bps> = {}) {
    __builtin_hcu_matrix_load_32X32_b16(rsrc, lds_addr, moffset, true, r, false,
                                        false, bps);
  }
};

struct gfx946_mls_64x16_b16 {
  static constexpr auto TileShape = ::tl::sequence<64, 16>{};

  template <::tl::index_t moffset, bool r, bool bps = false>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>,
                             ::tl::bool_constant<bps> = {}) {
    __builtin_hcu_matrix_load_64X16_b16(rsrc, lds_addr, moffset, false, r,
                                        false, false, bps);
  }
};

struct gfx946_mls_16x64_trans_b16 {
  static constexpr auto TileShape = ::tl::sequence<16, 64>{};

  template <::tl::index_t moffset, bool r, bool bps = false>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>,
                             ::tl::bool_constant<bps> = {}) {
    __builtin_hcu_matrix_load_64X16_b16(rsrc, lds_addr, moffset, true, r, false,
                                        false, bps);
  }
};

struct gfx946_mls_64x16_b8 {
  static constexpr auto TileShape = ::tl::sequence<64, 16>{};

  template <::tl::index_t moffset, bool r, bool bps = false>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>,
                             ::tl::bool_constant<bps> = {}) {
    if constexpr (!bps) {
      gfx938_mls_64x16_b8::load(lds_addr, rsrc, ::tl::number<moffset>{},
                                ::tl::bool_constant<r>{});
    } else {
      __builtin_hcu_matrix_load_64X16_b8(rsrc, lds_addr, moffset, false, r,
                                         false, false, bps);
    }
  }
};

struct gfx946_mls_16x64_trans_b8 {
  static constexpr auto TileShape = ::tl::sequence<16, 64>{};

  template <::tl::index_t moffset, bool r, bool bps = false>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>,
                             ::tl::bool_constant<bps> = {}) {
    if constexpr (!bps) {
      gfx938_mls_16x64_trans_b8::load(lds_addr, rsrc, ::tl::number<moffset>{},
                                      ::tl::bool_constant<r>{});
    } else {
      __builtin_hcu_matrix_load_64X16_b8(rsrc, lds_addr, moffset, true, r,
                                         false, false, bps);
    }
  }
};

struct gfx946_mls_64x32_b8 {
  static constexpr auto TileShape = ::tl::sequence<64, 32>{};

  template <::tl::index_t moffset, bool r, bool bps = false>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>,
                             ::tl::bool_constant<bps> = {}) {
    if constexpr (!bps) {
      gfx938_mls_64x32_b8::load(lds_addr, rsrc, ::tl::number<moffset>{},
                                ::tl::bool_constant<r>{});
    } else {
      __builtin_hcu_matrix_load_64X32_b8(rsrc, lds_addr, moffset, false, r,
                                         false, false, bps);
    }
  }
};

struct gfx946_mls_32x64_trans_b8 {
  static constexpr auto TileShape = ::tl::sequence<32, 64>{};

  template <::tl::index_t moffset, bool r, bool bps = false>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>,
                             ::tl::bool_constant<bps> = {}) {
    if constexpr (!bps) {
      gfx938_mls_32x64_trans_b8::load(lds_addr, rsrc, ::tl::number<moffset>{},
                                      ::tl::bool_constant<r>{});
    } else {
      __builtin_hcu_matrix_load_64X32_b8(rsrc, lds_addr, moffset, true, r,
                                         false, false, bps);
    }
  }
};

struct gfx946_mls_128x16_b8 {
  static constexpr auto TileShape = ::tl::sequence<128, 16>{};

  template <::tl::index_t moffset, bool r, bool bps = false>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>,
                             ::tl::bool_constant<bps> = {}) {
    if constexpr (!bps) {
      gfx938_mls_128x16_b8::load(lds_addr, rsrc, ::tl::number<moffset>{},
                                 ::tl::bool_constant<r>{});
    } else {
      __builtin_hcu_matrix_load_128X16_b8(rsrc, lds_addr, moffset, false, r,
                                          false, false, bps);
    }
  }
};

struct gfx946_mls_16x128_trans_b8 {
  static constexpr auto TileShape = ::tl::sequence<16, 128>{};

  template <::tl::index_t moffset, bool r, bool bps = false>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>,
                             ::tl::bool_constant<bps> = {}) {
    if constexpr (!bps) {
      gfx938_mls_16x128_trans_b8::load(lds_addr, rsrc, ::tl::number<moffset>{},
                                       ::tl::bool_constant<r>{});
    } else {
      __builtin_hcu_matrix_load_128X16_b8(rsrc, lds_addr, moffset, true, r,
                                          false, false, bps);
    }
  }
};

} // namespace mls
} // namespace tl
