#pragma once

/*
 * tilelang MlsAtom for gfx938 - load() uses __builtin_hcu_matrix_load_* instead
 * of inline asm. Ported from ck_tile/core/arch/hcu_mls_atom_gfx938.hpp.
 */

#include <tl_templates/hcu/core.hpp>

namespace tl {
namespace mls {

struct gfx938_mls_32x16_b16 {
  static constexpr auto TileShape = ::tl::sequence<32, 16>{};

  template <::tl::index_t moffset, bool r>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>) {
    __builtin_hcu_matrix_load_32X16_b16(rsrc,
                                        reinterpret_cast<uint32_t *>(lds_addr),
                                        moffset, false, r, false, false);
  }
};

struct gfx938_mls_16x32_trans_b16 {
  static constexpr auto TileShape = ::tl::sequence<16, 32>{};

  template <::tl::index_t moffset, bool r>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>) {
    __builtin_hcu_matrix_load_32X16_b16(
        rsrc,
        reinterpret_cast<uint32_t *>(lds_addr |
                                     TL_MATRIX_LOAD_TRANS_EXTRA_CONFIG),
        moffset, true, r, false, false);
  }
};

struct gfx938_mls_32x32_b16 {
  static constexpr auto TileShape = ::tl::sequence<32, 32>{};

  template <::tl::index_t moffset, bool r>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>) {
    __builtin_hcu_matrix_load_32X32_b16(rsrc,
                                        reinterpret_cast<uint32_t *>(lds_addr),
                                        moffset, false, r, false, false);
  }
};

struct gfx938_mls_32x32_trans_b16 {
  static constexpr auto TileShape = ::tl::sequence<32, 32>{};

  template <::tl::index_t moffset, bool r>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>) {
    __builtin_hcu_matrix_load_32X32_b16(
        rsrc,
        reinterpret_cast<uint32_t *>(lds_addr |
                                     TL_MATRIX_LOAD_TRANS_EXTRA_CONFIG),
        moffset, true, r, false, false);
  }
};

struct gfx938_mls_64x16_b16 {
  static constexpr auto TileShape = ::tl::sequence<64, 16>{};

  template <::tl::index_t moffset, bool r>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>) {
    __builtin_hcu_matrix_load_64X16_b16(rsrc,
                                        reinterpret_cast<uint32_t *>(lds_addr),
                                        moffset, false, r, false, false);
  }
};

struct gfx938_mls_16x64_trans_b16 {
  static constexpr auto TileShape = ::tl::sequence<16, 64>{};

  template <::tl::index_t moffset, bool r>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>) {
#if 0 // TEMP: use inline asm when builtin causes compile error on gfx938
        const auto soffset = static_cast<uint32_t>(
            lds_addr | TL_MATRIX_LOAD_TRANS_EXTRA_CONFIG);
        if constexpr(!r)
        {
            asm volatile("s_mov_b32 m0, %1\n\t"
                         "s_nop 0\n\t"
                         "matrix_load_64x16_b16 %0, m0, moffset:%2, t, lds; \n\t"
                         :
                         : "s"(rsrc), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else
        {
            asm volatile("s_mov_b32 m0, %1\n\t"
                         "s_nop 0\n\t"
                         "matrix_load_64x16_b16 %0, m0, moffset:%2, t r, lds; \n\t"
                         :
                         : "s"(rsrc), "s"(soffset), "n"(moffset)
                         : "memory");
        }
#else
    __builtin_hcu_matrix_load_64X16_b16(
        rsrc,
        reinterpret_cast<uint32_t *>(lds_addr |
                                     TL_MATRIX_LOAD_TRANS_EXTRA_CONFIG),
        moffset, true, r, false, false);
#endif
  }
};

struct gfx938_mls_64x16_b8 {
  static constexpr auto TileShape = ::tl::sequence<64, 16>{};

  template <::tl::index_t moffset, bool r>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>) {
    __builtin_hcu_matrix_load_64X16_b8(rsrc,
                                       reinterpret_cast<uint32_t *>(lds_addr),
                                       moffset, false, r, false, false);
  }
};

struct gfx938_mls_16x64_trans_b8 {
  static constexpr auto TileShape = ::tl::sequence<16, 64>{};

  template <::tl::index_t moffset, bool r>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>) {
    __builtin_hcu_matrix_load_64X16_b8(
        rsrc,
        reinterpret_cast<uint32_t *>(lds_addr |
                                     TL_MATRIX_LOAD_TRANS_EXTRA_CONFIG),
        moffset, true, r, false, false);
  }
};

struct gfx938_mls_64x32_b8 {
  static constexpr auto TileShape = ::tl::sequence<64, 32>{};

  template <::tl::index_t moffset, bool r>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>) {
    __builtin_hcu_matrix_load_64X32_b8(rsrc,
                                       reinterpret_cast<uint32_t *>(lds_addr),
                                       moffset, false, r, false, false);
  }
};

struct gfx938_mls_32x64_trans_b8 {
  static constexpr auto TileShape = ::tl::sequence<32, 64>{};

  template <::tl::index_t moffset, bool r>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>) {
    __builtin_hcu_matrix_load_64X32_b8(
        rsrc,
        reinterpret_cast<uint32_t *>(lds_addr |
                                     TL_MATRIX_LOAD_TRANS_EXTRA_CONFIG),
        moffset, true, r, false, false);
  }
};

struct gfx938_mls_128x16_b8 {
  static constexpr auto TileShape = ::tl::sequence<128, 16>{};

  template <::tl::index_t moffset, bool r>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>) {
    __builtin_hcu_matrix_load_128X16_b8(rsrc,
                                        reinterpret_cast<uint32_t *>(lds_addr),
                                        moffset, false, r, false, false);
  }
};

struct gfx938_mls_16x128_trans_b8 {
  static constexpr auto TileShape = ::tl::sequence<16, 128>{};

  template <::tl::index_t moffset, bool r>
  TL_DEVICE static void load(const uintptr_t lds_addr,
                             const ::tl::int32x4_t &rsrc, ::tl::number<moffset>,
                             ::tl::bool_constant<r>) {
    __builtin_hcu_matrix_load_128X16_b8(
        rsrc,
        reinterpret_cast<uint32_t *>(lds_addr |
                                     TL_MATRIX_LOAD_TRANS_EXTRA_CONFIG),
        moffset, true, r, false, false);
  }
};

} // namespace mls
} // namespace tl
