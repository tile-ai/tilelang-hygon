#pragma once

/*
 * tilelang MlsAtom for gfx946 - load() uses __builtin_hcu_matrix_load_* with bps parameter.
 * Ported from ck_tile/core/arch/hcu_mls_atom_gfx946.hpp.
 */

#include <ck_tile/core.hpp>

#include "tl_mls_atom_gfx938.hpp"

namespace tl {
namespace mls {

struct gfx946_mls_32x16_b16
{
    static constexpr auto TileShape = ck_tile::sequence<32, 16>{};

    template <ck_tile::index_t moffset, bool r, bool bps = false>
    CK_TILE_DEVICE static void load(const uintptr_t lds_addr,
                                    const ck_tile::int32x4_t& rsrc,
                                    ck_tile::number<moffset>,
                                    ck_tile::bool_constant<r>,
                                    ck_tile::bool_constant<bps> = {})
    {
        __builtin_hcu_matrix_load_32X16_b16(rsrc, lds_addr, moffset, false, r, false, false, bps);
    }
};

struct gfx946_mls_16x32_trans_b16
{
    static constexpr auto TileShape = ck_tile::sequence<16, 32>{};

    template <ck_tile::index_t moffset, bool r, bool bps = false>
    CK_TILE_DEVICE static void load(const uintptr_t lds_addr,
                                    const ck_tile::int32x4_t& rsrc,
                                    ck_tile::number<moffset>,
                                    ck_tile::bool_constant<r>,
                                    ck_tile::bool_constant<bps> = {})
    {
        __builtin_hcu_matrix_load_32X16_b16(rsrc, lds_addr, moffset, true, r, false, false, bps);
    }
};

struct gfx946_mls_32x32_b16
{
    static constexpr auto TileShape = ck_tile::sequence<32, 32>{};

    template <ck_tile::index_t moffset, bool r, bool bps = false>
    CK_TILE_DEVICE static void load(const uintptr_t lds_addr,
                                    const ck_tile::int32x4_t& rsrc,
                                    ck_tile::number<moffset>,
                                    ck_tile::bool_constant<r>,
                                    ck_tile::bool_constant<bps> = {})
    {
        __builtin_hcu_matrix_load_32X32_b16(rsrc, lds_addr, moffset, false, r, false, false, bps);
    }
};

struct gfx946_mls_32x32_trans_b16
{
    static constexpr auto TileShape = ck_tile::sequence<32, 32>{};

    template <ck_tile::index_t moffset, bool r, bool bps = false>
    CK_TILE_DEVICE static void load(const uintptr_t lds_addr,
                                    const ck_tile::int32x4_t& rsrc,
                                    ck_tile::number<moffset>,
                                    ck_tile::bool_constant<r>,
                                    ck_tile::bool_constant<bps> = {})
    {
        __builtin_hcu_matrix_load_32X32_b16(rsrc, lds_addr, moffset, true, r, false, false, bps);
    }
};

struct gfx946_mls_64x16_b16
{
    static constexpr auto TileShape = ck_tile::sequence<64, 16>{};

    template <ck_tile::index_t moffset, bool r, bool bps = false>
    CK_TILE_DEVICE static void load(const uintptr_t lds_addr,
                                    const ck_tile::int32x4_t& rsrc,
                                    ck_tile::number<moffset>,
                                    ck_tile::bool_constant<r>,
                                    ck_tile::bool_constant<bps> = {})
    {
        __builtin_hcu_matrix_load_64X16_b16(rsrc, lds_addr, moffset, false, r, false, false, bps);
    }
};

struct gfx946_mls_16x64_trans_b16
{
    static constexpr auto TileShape = ck_tile::sequence<16, 64>{};

    template <ck_tile::index_t moffset, bool r, bool bps = false>
    CK_TILE_DEVICE static void load(const uintptr_t lds_addr,
                                    const ck_tile::int32x4_t& rsrc,
                                    ck_tile::number<moffset>,
                                    ck_tile::bool_constant<r>,
                                    ck_tile::bool_constant<bps> = {})
    {
        __builtin_hcu_matrix_load_64X16_b16(rsrc, lds_addr, moffset, true, r, false, false, bps);
    }
};

struct gfx946_mls_64x16_b8
{
    static constexpr auto TileShape = ck_tile::sequence<64, 16>{};

    template <ck_tile::index_t moffset, bool r, bool bps = false>
    CK_TILE_DEVICE static void load(const uintptr_t lds_addr,
                                    const ck_tile::int32x4_t& rsrc,
                                    ck_tile::number<moffset>,
                                    ck_tile::bool_constant<r>,
                                    ck_tile::bool_constant<bps> = {})
    {
        if constexpr(!bps)
        {
            gfx938_mls_64x16_b8::load(lds_addr, rsrc, ck_tile::number<moffset>{},
                                      ck_tile::bool_constant<r>{});
        }
        else
        {
            __builtin_hcu_matrix_load_64X16_b8(rsrc, lds_addr, moffset, false, r, false, false,
                                               bps);
        }
    }
};

struct gfx946_mls_16x64_trans_b8
{
    static constexpr auto TileShape = ck_tile::sequence<16, 64>{};

    template <ck_tile::index_t moffset, bool r, bool bps = false>
    CK_TILE_DEVICE static void load(const uintptr_t lds_addr,
                                    const ck_tile::int32x4_t& rsrc,
                                    ck_tile::number<moffset>,
                                    ck_tile::bool_constant<r>,
                                    ck_tile::bool_constant<bps> = {})
    {
        if constexpr(!bps)
        {
            gfx938_mls_16x64_trans_b8::load(lds_addr, rsrc, ck_tile::number<moffset>{},
                                            ck_tile::bool_constant<r>{});
        }
        else
        {
            __builtin_hcu_matrix_load_64X16_b8(rsrc, lds_addr, moffset, true, r, false, false, bps);
        }
    }
};

struct gfx946_mls_64x32_b8
{
    static constexpr auto TileShape = ck_tile::sequence<64, 32>{};

    template <ck_tile::index_t moffset, bool r, bool bps = false>
    CK_TILE_DEVICE static void load(const uintptr_t lds_addr,
                                    const ck_tile::int32x4_t& rsrc,
                                    ck_tile::number<moffset>,
                                    ck_tile::bool_constant<r>,
                                    ck_tile::bool_constant<bps> = {})
    {
        if constexpr(!bps)
        {
            gfx938_mls_64x32_b8::load(lds_addr, rsrc, ck_tile::number<moffset>{},
                                      ck_tile::bool_constant<r>{});
        }
        else
        {
            __builtin_hcu_matrix_load_64X32_b8(rsrc, lds_addr, moffset, false, r, false, false,
                                               bps);
        }
    }
};

struct gfx946_mls_32x64_trans_b8
{
    static constexpr auto TileShape = ck_tile::sequence<32, 64>{};

    template <ck_tile::index_t moffset, bool r, bool bps = false>
    CK_TILE_DEVICE static void load(const uintptr_t lds_addr,
                                    const ck_tile::int32x4_t& rsrc,
                                    ck_tile::number<moffset>,
                                    ck_tile::bool_constant<r>,
                                    ck_tile::bool_constant<bps> = {})
    {
        if constexpr(!bps)
        {
            gfx938_mls_32x64_trans_b8::load(lds_addr, rsrc, ck_tile::number<moffset>{},
                                            ck_tile::bool_constant<r>{});
        }
        else
        {
            __builtin_hcu_matrix_load_64X32_b8(rsrc, lds_addr, moffset, true, r, false, false, bps);
        }
    }
};

struct gfx946_mls_128x16_b8
{
    static constexpr auto TileShape = ck_tile::sequence<128, 16>{};

    template <ck_tile::index_t moffset, bool r, bool bps = false>
    CK_TILE_DEVICE static void load(const uintptr_t lds_addr,
                                    const ck_tile::int32x4_t& rsrc,
                                    ck_tile::number<moffset>,
                                    ck_tile::bool_constant<r>,
                                    ck_tile::bool_constant<bps> = {})
    {
        if constexpr(!bps)
        {
            gfx938_mls_128x16_b8::load(lds_addr, rsrc, ck_tile::number<moffset>{},
                                       ck_tile::bool_constant<r>{});
        }
        else
        {
            __builtin_hcu_matrix_load_128X16_b8(rsrc, lds_addr, moffset, false, r, false, false,
                                               bps);
        }
    }
};

struct gfx946_mls_16x128_trans_b8
{
    static constexpr auto TileShape = ck_tile::sequence<16, 128>{};

    template <ck_tile::index_t moffset, bool r, bool bps = false>
    CK_TILE_DEVICE static void load(const uintptr_t lds_addr,
                                    const ck_tile::int32x4_t& rsrc,
                                    ck_tile::number<moffset>,
                                    ck_tile::bool_constant<r>,
                                    ck_tile::bool_constant<bps> = {})
    {
        if constexpr(!bps)
        {
            gfx938_mls_16x128_trans_b8::load(lds_addr, rsrc, ck_tile::number<moffset>{},
                                             ck_tile::bool_constant<r>{});
        }
        else
        {
            __builtin_hcu_matrix_load_128X16_b8(rsrc, lds_addr, moffset, true, r, false, false,
                                                bps);
        }
    }
};

} // namespace mls
} // namespace tl
