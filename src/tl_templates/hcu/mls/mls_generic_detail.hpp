#pragma once

#include <cassert>
#include <tl_templates/hcu/core.hpp>

#include <tl_templates/hcu/mls/tl_mls_atom_dispatcher.hpp>
#include <tl_templates/hcu/mls/tl_mls_traits.hpp>

namespace tl {
namespace mls {
namespace detail {

/*
 * make_lds_desc_generic: builds LDS descriptor from BlockSize and MlsAtom.
 * Template params: MlsAtom, Alt, BlockSizeMN, BlockSizeK, Trans.
 *
 * Layout rules (tile_issue_* loop order = memory layout, outer first):
 * - Trans:   tile_issue_k outer, tile_issue_mn inner.
 * - Non-trans: tile_issue_mn outer, tile_issue_k inner.
 */
template <typename MlsAtom,
          ::tl::index_t Alt,
          ::tl::index_t BlockSizeMN,
          ::tl::index_t BlockSizeK,
          bool Trans>
struct make_lds_desc_generic;

// ========== Trans: 16x64 (PackedShape = (kK0, kMN, kK1)), tile_K = kK0*kK1 = 64 ==========
template <::tl::index_t Alt,
          ::tl::index_t BlockSizeMN,
          ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx938_mls_16x64_trans_b16,
                            Alt,
                            BlockSizeMN,
                            BlockSizeK,
                            true>
{
    using MlsTraits = mls_traits<tl::mls::gfx938_mls_16x64_trans_b16, Alt>;
    static constexpr ::tl::index_t MlsTileMN = 16;
    static constexpr ::tl::index_t MlsTileK  = MlsTraits::kK0 * MlsTraits::kK1; // 64

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        if constexpr(Alt == 1)
        {
            if constexpr(tile_issue_k == 1)
            {
                // 4D: (tile_mn, kK0, kMN, kK1) - single K tile, MN only
                constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                    ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn), MlsTraits::PackedShape));
                return ::tl::transform_tensor_descriptor(
                    lds_desc_raw,
                    ::tl::make_tuple(
                        ::tl::make_merge_transform(
                            ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                        ::tl::make_merge_transform(
                            ::tl::make_tuple(MlsTraits::kK0, MlsTraits::kK1))),
                    ::tl::make_tuple(::tl::sequence<0, 2>{}, ::tl::sequence<1, 3>{}),
                    ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
            }
            else
            {
                // 5D: (tile_k, tile_mn, kK0, kMN, kK1) - K outer, MN inner
                constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                    ::tl::concat_tuple(::tl::make_tuple(tile_issue_k, tile_issue_mn),
                                         MlsTraits::PackedShape));
                return ::tl::transform_tensor_descriptor(
                    lds_desc_raw,
                    ::tl::make_tuple(
                        ::tl::make_merge_transform(
                            ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                        ::tl::make_merge_transform(
                            ::tl::make_tuple(tile_issue_k, MlsTraits::kK0, MlsTraits::kK1))),
                    ::tl::make_tuple(::tl::sequence<1, 3>{}, ::tl::sequence<0, 2, 4>{}),
                    ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
            }
        }
        else
        {
            constexpr auto tile_issue_mn_outer =
                ::tl::number<tile_issue_mn / MlsTraits::kSlots>{};
            if constexpr(tile_issue_k == 1)
            {
                // 5D: (tile_mn_outer, kK0, kSlots, kMN, kK1) - single K tile, PackedShape=(kK0,kSlots,kMN,kK1)
                constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                    ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn_outer),
                                         MlsTraits::PackedShape));
                return ::tl::transform_tensor_descriptor(
                    lds_desc_raw,
                    ::tl::make_tuple(
                        ::tl::make_merge_transform(::tl::make_tuple(
                            tile_issue_mn_outer, MlsTraits::kSlots, MlsTraits::kMN)),
                        ::tl::make_merge_transform(
                            ::tl::make_tuple(MlsTraits::kK0, MlsTraits::kK1))),
                    ::tl::make_tuple(::tl::sequence<0, 2, 3>{}, ::tl::sequence<1, 4>{}),
                    ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
            }
            else
            {
                // 6D: (tile_k, tile_mn_outer, kK0, kSlots, kMN, kK1) - K outer, MN inner, PackedShape=(kK0,kSlots,kMN,kK1)
                constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                    ::tl::concat_tuple(
                        ::tl::make_tuple(tile_issue_k, tile_issue_mn_outer),
                        MlsTraits::PackedShape));
                return ::tl::transform_tensor_descriptor(
                    lds_desc_raw,
                    ::tl::make_tuple(
                        ::tl::make_merge_transform(::tl::make_tuple(
                            tile_issue_mn_outer, MlsTraits::kSlots, MlsTraits::kMN)),
                        ::tl::make_merge_transform(
                            ::tl::make_tuple(tile_issue_k, MlsTraits::kK0, MlsTraits::kK1))),
                    ::tl::make_tuple(::tl::sequence<1, 3, 4>{}, ::tl::sequence<0, 2, 5>{}),
                    ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
            }
        }
    }
};

// ========== Trans: 32x32 (PackedShape = (kMN, kK)), tile 32x32 ==========
template <::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx938_mls_32x32_trans_b16,
                            1,
                            BlockSizeMN,
                            BlockSizeK,
                            true>
{
    using MlsTraits = mls_traits<tl::mls::gfx938_mls_32x32_trans_b16, 1>;
    static constexpr ::tl::index_t MlsTileMN = 32;
    static constexpr ::tl::index_t MlsTileK  = 32;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        if constexpr(tile_issue_k == 1)
        {
            // 3D: (tile_mn, kMN, kK) - single K tile
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn), MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_pass_through_transform(MlsTraits::kK)),
                ::tl::make_tuple(::tl::sequence<0, 1>{}, ::tl::sequence<2>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
        else
        {
            // 4D: (tile_k, tile_mn, kMN, kK) - K outer, MN inner
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_k, tile_issue_mn),
                                     MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
                ::tl::make_tuple(::tl::sequence<1, 2>{}, ::tl::sequence<0, 3>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
    }
};

// ========== Trans: 16x32 (PackedShape = (kMN, kK)), tile 16x32 ==========
template <::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx938_mls_16x32_trans_b16,
                            1,
                            BlockSizeMN,
                            BlockSizeK,
                            true>
{
    using MlsTraits = mls_traits<tl::mls::gfx938_mls_16x32_trans_b16, 1>;
    static constexpr ::tl::index_t MlsTileMN = 16;
    static constexpr ::tl::index_t MlsTileK  = 32;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        if constexpr(tile_issue_k == 1)
        {
            // 3D: (tile_mn, kMN, kK) - single K tile
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn), MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_pass_through_transform(MlsTraits::kK)),
                ::tl::make_tuple(::tl::sequence<0, 1>{}, ::tl::sequence<2>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
        else
        {
            // 4D: (tile_k, tile_mn, kMN, kK) - K outer, MN inner
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_k, tile_issue_mn),
                                     MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
                ::tl::make_tuple(::tl::sequence<1, 2>{}, ::tl::sequence<0, 3>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
    }
};

// ========== Non-trans: 32x32 (PackedShape = (kK, kMN)), tile 32x32 ==========
template <::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx938_mls_32x32_b16,
                            1,
                            BlockSizeMN,
                            BlockSizeK,
                            false>
{
    using MlsTraits = mls_traits<tl::mls::gfx938_mls_32x32_b16, 1>;
    static constexpr ::tl::index_t MlsTileMN = 32;
    static constexpr ::tl::index_t MlsTileK  = 32;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{}; // MN outer
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};   // K inner

        // 4D: (tile_mn, tile_k, kK, kMN) - MN outer, K inner
        constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
            ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn, tile_issue_k),
                                 MlsTraits::PackedShape));
        return ::tl::transform_tensor_descriptor(
            lds_desc_raw,
            ::tl::make_tuple(
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
            ::tl::make_tuple(::tl::sequence<0, 3>{}, ::tl::sequence<1, 2>{}),
            ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
    }
};

// ========== Non-trans: 32x16 (PackedShape = (kK, kMN)), tile 32x16 ==========
template <::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx938_mls_32x16_b16,
                            1,
                            BlockSizeMN,
                            BlockSizeK,
                            false>
{
    using MlsTraits = mls_traits<tl::mls::gfx938_mls_32x16_b16, 1>;
    static constexpr ::tl::index_t MlsTileMN = 32;
    static constexpr ::tl::index_t MlsTileK  = 16;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{}; // MN outer
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};      // K inner

        // 4D: (tile_mn, tile_k, kK, kMN) - MN outer, K inner
        constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
            ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn, tile_issue_k),
                                 MlsTraits::PackedShape));
        return ::tl::transform_tensor_descriptor(
            lds_desc_raw,
            ::tl::make_tuple(
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
            ::tl::make_tuple(::tl::sequence<0, 3>{}, ::tl::sequence<1, 2>{}),
            ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
    }
};

// ========== Non-trans: 64x16 (PackedShape = (kMN0, kK, kMN1)), tile 64x16 ==========
template <::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx938_mls_64x16_b16,
                            1,
                            BlockSizeMN,
                            BlockSizeK,
                            false>
{
    using MlsTraits = mls_traits<tl::mls::gfx938_mls_64x16_b16, 1>;
    static constexpr ::tl::index_t MlsTileMN = MlsTraits::kMN0 * MlsTraits::kMN1; // 64
    static constexpr ::tl::index_t MlsTileK  = 16;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{}; // MN outer
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};    // K inner

        // 5D: (tile_mn, tile_k, kMN0, kK, kMN1) - MN outer, K inner
        constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
            ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn, tile_issue_k),
                                 MlsTraits::PackedShape));
        return ::tl::transform_tensor_descriptor(
            lds_desc_raw,
            ::tl::make_tuple(
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN0, MlsTraits::kMN1)),
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
            ::tl::make_tuple(::tl::sequence<0, 2, 4>{}, ::tl::sequence<1, 3>{}),
            ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
    }
};

// ========== Non-trans b8: 64x16 (PackedShape = (kK, kMN)), tile 64x16 ==========
template <::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx938_mls_64x16_b8,
                            1,
                            BlockSizeMN,
                            BlockSizeK,
                            false>
{
    using MlsTraits = mls_traits<tl::mls::gfx938_mls_64x16_b8, 1>;
    static constexpr ::tl::index_t MlsTileMN = 64;
    static constexpr ::tl::index_t MlsTileK  = 16;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{}; // MN outer
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};   // K inner

        // 4D: (tile_mn, tile_k, kK, kMN) - MN outer, K inner
        constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
            ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn, tile_issue_k),
                                 MlsTraits::PackedShape));
        return ::tl::transform_tensor_descriptor(
            lds_desc_raw,
            ::tl::make_tuple(
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
            ::tl::make_tuple(::tl::sequence<0, 3>{}, ::tl::sequence<1, 2>{}),
            ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
    }
};

// ========== Non-trans b8: 64x32 (PackedShape = (kK, kMN)), tile 64x32 ==========
template <::tl::index_t Alt, ::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx938_mls_64x32_b8, Alt, BlockSizeMN, BlockSizeK, false>
{
    using MlsTraits = mls_traits<tl::mls::gfx938_mls_64x32_b8, Alt>;
    static constexpr ::tl::index_t MlsTileMN = 64;
    static constexpr ::tl::index_t MlsTileK  = 32;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        // (tile_issue_mn, tile_issue_k, kK, kMN)
        constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
            ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn, tile_issue_k),
                                 MlsTraits::PackedShape));
        return ::tl::transform_tensor_descriptor(
            lds_desc_raw,
            ::tl::make_tuple(
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
            ::tl::make_tuple(::tl::sequence<0, 3>{}, ::tl::sequence<1, 2>{}),
            ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
    }
};

// ========== Trans b8: 32x64 (PackedShape = (kMN, kK) or (kMN0,kSlots,kMN1,kK)), tile 32x64 ==========
template <::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx938_mls_32x64_trans_b8,
                            1,
                            BlockSizeMN,
                            BlockSizeK,
                            true>
{
    using MlsTraits = mls_traits<tl::mls::gfx938_mls_32x64_trans_b8, 1>;
    static constexpr ::tl::index_t MlsTileMN = 32;
    static constexpr ::tl::index_t MlsTileK  = 64;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        if constexpr(tile_issue_k == 1)
        {
            // (tile_issue_mn, kMN, kK)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn), MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_pass_through_transform(MlsTraits::kK)),
                ::tl::make_tuple(::tl::sequence<0, 1>{}, ::tl::sequence<2>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
        else
        {
            // (tile_issue_k, tile_issue_mn, kMN, kK)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_k, tile_issue_mn),
                                     MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
                ::tl::make_tuple(::tl::sequence<1, 2>{}, ::tl::sequence<0, 3>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
    }
};

template <::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx938_mls_32x64_trans_b8,
                            2,
                            BlockSizeMN,
                            BlockSizeK,
                            true>
{
    using MlsTraits = mls_traits<tl::mls::gfx938_mls_32x64_trans_b8, 2>;
    static constexpr ::tl::index_t MlsTileMN = 32;
    static constexpr ::tl::index_t MlsTileK  = 64;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        if constexpr(tile_issue_k == 1)
        {
            // (tile_issue_mn, kMN, kK)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn), MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_pass_through_transform(MlsTraits::kK)),
                ::tl::make_tuple(::tl::sequence<0, 1>{}, ::tl::sequence<2>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
        else
        {
            // (tile_issue_k, tile_issue_mn, kMN, kK)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_k, tile_issue_mn),
                                     MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
                ::tl::make_tuple(::tl::sequence<1, 2>{}, ::tl::sequence<0, 3>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
    }
};

// ========== Non-trans b8: 128x16 (PackedShape = (kMN0, kSlots, kK, kMN1)), tile 128x16 ==========
template <::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx938_mls_128x16_b8,
                            1,
                            BlockSizeMN,
                            BlockSizeK,
                            false>
{
    using MlsTraits = mls_traits<tl::mls::gfx938_mls_128x16_b8, 1>;
    static constexpr ::tl::index_t MlsTileMN = MlsTraits::kMN0 * MlsTraits::kMN1;
    static constexpr ::tl::index_t MlsTileK  = 16;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK / MlsTraits::kSlots>{};
        // (tile_issue_mn, tile_issue_k, kMN0, kSlots, kK, kMN1)
        constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
            ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn, tile_issue_k),
                                 MlsTraits::PackedShape));
        return ::tl::transform_tensor_descriptor(
            lds_desc_raw,
            ::tl::make_tuple(
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN0,
                                       MlsTraits::kMN1)),
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_k, MlsTraits::kSlots, MlsTraits::kK))),
            ::tl::make_tuple(::tl::sequence<0, 2, 5>{}, ::tl::sequence<1, 3, 4>{}),
            ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
    }
};

// ========== Trans b8: 16x128 (PackedShape = (kK0, kMN, kK1)), tile 16x128 ==========
template <::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx938_mls_16x128_trans_b8,
                            1,
                            BlockSizeMN,
                            BlockSizeK,
                            true>
{
    using MlsTraits = mls_traits<tl::mls::gfx938_mls_16x128_trans_b8, 1>;
    static constexpr ::tl::index_t MlsTileMN = 16;
    static constexpr ::tl::index_t MlsTileK  = MlsTraits::kK0 * MlsTraits::kK1;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        if constexpr(tile_issue_k == 1)
        {
            // (tile_issue_mn, kK0, kMN, kK1)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn), MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(MlsTraits::kK0, MlsTraits::kK1))),
                ::tl::make_tuple(::tl::sequence<0, 2>{}, ::tl::sequence<1, 3>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
        else
        {
            // (tile_issue_k, tile_issue_mn, kK0, kMN, kK1)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_k, tile_issue_mn),
                                     MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_k, MlsTraits::kK0, MlsTraits::kK1))),
                ::tl::make_tuple(::tl::sequence<1, 3>{}, ::tl::sequence<0, 2, 4>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
    }
};

// ========== Trans b8: 16x64 (PackedShape = (kMN, kK)), tile 16x64 ==========
template <::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx938_mls_16x64_trans_b8,
                            1,
                            BlockSizeMN,
                            BlockSizeK,
                            true>
{
    using MlsTraits = mls_traits<tl::mls::gfx938_mls_16x64_trans_b8, 1>;
    static constexpr ::tl::index_t MlsTileMN = 16;
    static constexpr ::tl::index_t MlsTileK  = 64;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        if constexpr(tile_issue_k == 1)
        {
            // 3D: (tile_mn, kMN, kK) - single K tile
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn), MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_pass_through_transform(MlsTraits::kK)),
                ::tl::make_tuple(::tl::sequence<0, 1>{}, ::tl::sequence<2>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
        else
        {
            // 4D: (tile_k, tile_mn, kMN, kK) - K outer, MN inner
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_k, tile_issue_mn),
                                     MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
                ::tl::make_tuple(::tl::sequence<1, 2>{}, ::tl::sequence<0, 3>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
    }
};

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx946__)
// ========== gfx946: same shapes as gfx938, use tl::mls::mls_traits ==========
#include <tl_templates/hcu/mls/tl_mls_atom_gfx946.hpp>

template <::tl::index_t Alt,
          ::tl::index_t BlockSizeMN,
          ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx946_mls_16x64_trans_b16,
                            Alt,
                            BlockSizeMN,
                            BlockSizeK,
                            true>
{
    using MlsTraits = mls_traits<tl::mls::gfx946_mls_16x64_trans_b16, Alt>;
    static constexpr ::tl::index_t MlsTileMN = 16;
    static constexpr ::tl::index_t MlsTileK  = MlsTraits::kK0 * MlsTraits::kK1;

    static constexpr auto apply()
    {
        // gfx946 16x64_trans: both Alt 1,2 use PackedShape=(kK0, kSlots, kMN, kK1)
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};
        constexpr auto tile_issue_mn_outer =
            ::tl::number<tile_issue_mn / MlsTraits::kSlots>{};

        if constexpr(tile_issue_k == 1)
        {
            // (tile_issue_mn_outer, kK0, kSlots, kMN, kK1)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn_outer),
                                     MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(::tl::make_tuple(
                        tile_issue_mn_outer, MlsTraits::kSlots, MlsTraits::kMN)),
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(MlsTraits::kK0, MlsTraits::kK1))),
                ::tl::make_tuple(::tl::sequence<0, 2, 3>{}, ::tl::sequence<1, 4>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
        else
        {
            // (tile_issue_k, tile_issue_mn_outer, kK0, kSlots, kMN, kK1)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(
                    ::tl::make_tuple(tile_issue_k, tile_issue_mn_outer),
                    MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(::tl::make_tuple(
                        tile_issue_mn_outer, MlsTraits::kSlots, MlsTraits::kMN)),
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_k, MlsTraits::kK0, MlsTraits::kK1))),
                ::tl::make_tuple(::tl::sequence<1, 3, 4>{}, ::tl::sequence<0, 2, 5>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
    }
};

template <::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx946_mls_32x32_trans_b16, 1, BlockSizeMN, BlockSizeK, true>
{
    using MlsTraits = mls_traits<tl::mls::gfx946_mls_32x32_trans_b16, 1>;
    static constexpr ::tl::index_t MlsTileMN = 32;
    static constexpr ::tl::index_t MlsTileK  = 32;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        if constexpr(tile_issue_k == 1)
        {
            // (tile_issue_mn, kMN, kK)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn), MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_pass_through_transform(MlsTraits::kK)),
                ::tl::make_tuple(::tl::sequence<0, 1>{}, ::tl::sequence<2>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
        else
        {
            // (tile_issue_k, tile_issue_mn, kMN, kK)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_k, tile_issue_mn),
                                     MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
                ::tl::make_tuple(::tl::sequence<1, 2>{}, ::tl::sequence<0, 3>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
    }
};

template <::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx946_mls_16x32_trans_b16, 1, BlockSizeMN, BlockSizeK, true>
{
    using MlsTraits = mls_traits<tl::mls::gfx946_mls_16x32_trans_b16, 1>;
    static constexpr ::tl::index_t MlsTileMN = 16;
    static constexpr ::tl::index_t MlsTileK  = 32;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        if constexpr(tile_issue_k == 1)
        {
            // (tile_issue_mn, kMN, kK)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn), MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_pass_through_transform(MlsTraits::kK)),
                ::tl::make_tuple(::tl::sequence<0, 1>{}, ::tl::sequence<2>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
        else
        {
            // (tile_issue_k, tile_issue_mn, kMN, kK)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_k, tile_issue_mn),
                                     MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
                ::tl::make_tuple(::tl::sequence<1, 2>{}, ::tl::sequence<0, 3>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
    }
};

template <::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx946_mls_32x32_b16, 1, BlockSizeMN, BlockSizeK, false>
{
    using MlsTraits = mls_traits<tl::mls::gfx946_mls_32x32_b16, 1>;
    static constexpr ::tl::index_t MlsTileMN = 32;
    static constexpr ::tl::index_t MlsTileK  = 32;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        // (tile_issue_mn, tile_issue_k, kK, kMN)
        constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
            ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn, tile_issue_k),
                                 MlsTraits::PackedShape));
        return ::tl::transform_tensor_descriptor(
            lds_desc_raw,
            ::tl::make_tuple(
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
            ::tl::make_tuple(::tl::sequence<0, 3>{}, ::tl::sequence<1, 2>{}),
            ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
    }
};

template <::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx946_mls_32x16_b16, 1, BlockSizeMN, BlockSizeK, false>
{
    using MlsTraits = mls_traits<tl::mls::gfx946_mls_32x16_b16, 1>;
    static constexpr ::tl::index_t MlsTileMN = 32;
    static constexpr ::tl::index_t MlsTileK  = 16;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        // (tile_issue_mn, tile_issue_k, kK, kMN)
        constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
            ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn, tile_issue_k),
                                 MlsTraits::PackedShape));
        return ::tl::transform_tensor_descriptor(
            lds_desc_raw,
            ::tl::make_tuple(
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
            ::tl::make_tuple(::tl::sequence<0, 3>{}, ::tl::sequence<1, 2>{}),
            ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
    }
};

// gfx946 64x16_b16: PackedShape=(kMN0, kSlots, kK, kMN1), differs from gfx938
template <::tl::index_t Alt, ::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx946_mls_64x16_b16, Alt, BlockSizeMN, BlockSizeK, false>
{
    using MlsTraits = mls_traits<tl::mls::gfx946_mls_64x16_b16, Alt>;
    static constexpr ::tl::index_t MlsTileMN = MlsTraits::kMN0 * MlsTraits::kMN1;
    static constexpr ::tl::index_t MlsTileK  = 16;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};
        constexpr auto tile_issue_mn_outer =
            ::tl::number<tile_issue_mn / MlsTraits::kSlots>{};

        // (tile_issue_mn_outer, tile_issue_k, kMN0, kSlots, kK, kMN1)
        constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
            ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn_outer, tile_issue_k),
                                 MlsTraits::PackedShape));
        return ::tl::transform_tensor_descriptor(
            lds_desc_raw,
            ::tl::make_tuple(
                ::tl::make_merge_transform(::tl::make_tuple(
                    tile_issue_mn_outer, MlsTraits::kSlots, MlsTraits::kMN0, MlsTraits::kMN1)),
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
            ::tl::make_tuple(::tl::sequence<0, 2, 3, 5>{}, ::tl::sequence<1, 4>{}),
            ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
    }
};

// ========== gfx946 b8: 64x16, 16x64_trans, 64x32, 32x64_trans, 128x16, 16x128_trans ==========
template <::tl::index_t Alt, ::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx946_mls_64x16_b8, Alt, BlockSizeMN, BlockSizeK, false>
{
    using MlsTraits = mls_traits<tl::mls::gfx946_mls_64x16_b8, Alt>;
    static constexpr ::tl::index_t MlsTileMN = 64;
    static constexpr ::tl::index_t MlsTileK  = 16;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        // (tile_issue_mn, tile_issue_k, kK, kMN)
        constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
            ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn, tile_issue_k),
                                 MlsTraits::PackedShape));
        return ::tl::transform_tensor_descriptor(
            lds_desc_raw,
            ::tl::make_tuple(
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
            ::tl::make_tuple(::tl::sequence<0, 3>{}, ::tl::sequence<1, 2>{}),
            ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
    }
};

template <::tl::index_t Alt, ::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx946_mls_16x64_trans_b8, Alt, BlockSizeMN, BlockSizeK, true>
{
    using MlsTraits = mls_traits<tl::mls::gfx946_mls_16x64_trans_b8, Alt>;
    static constexpr ::tl::index_t MlsTileMN = 16;
    static constexpr ::tl::index_t MlsTileK  = 64;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        if constexpr(tile_issue_k == 1)
        {
            // (tile_issue_mn, kMN, kK)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn), MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_pass_through_transform(MlsTraits::kK)),
                ::tl::make_tuple(::tl::sequence<0, 1>{}, ::tl::sequence<2>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
        else
        {
            // (tile_issue_k, tile_issue_mn, kMN, kK)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_k, tile_issue_mn),
                                     MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
                ::tl::make_tuple(::tl::sequence<1, 2>{}, ::tl::sequence<0, 3>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
    }
};

template <::tl::index_t Alt, ::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx946_mls_64x32_b8, Alt, BlockSizeMN, BlockSizeK, false>
{
    using MlsTraits = mls_traits<tl::mls::gfx946_mls_64x32_b8, Alt>;
    static constexpr ::tl::index_t MlsTileMN = 64;
    static constexpr ::tl::index_t MlsTileK  = 32;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        // (tile_issue_mn, tile_issue_k, kK, kMN)
        constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
            ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn, tile_issue_k),
                                 MlsTraits::PackedShape));
        return ::tl::transform_tensor_descriptor(
            lds_desc_raw,
            ::tl::make_tuple(
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
            ::tl::make_tuple(::tl::sequence<0, 3>{}, ::tl::sequence<1, 2>{}),
            ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
    }
};

template <::tl::index_t Alt, ::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx946_mls_32x64_trans_b8, Alt, BlockSizeMN, BlockSizeK, true>
{
    using MlsTraits = mls_traits<tl::mls::gfx946_mls_32x64_trans_b8, Alt>;
    static constexpr ::tl::index_t MlsTileMN = 32;
    static constexpr ::tl::index_t MlsTileK  = 64;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};

        if constexpr(tile_issue_k == 1)
        {
            // (tile_issue_mn, kMN, kK)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn), MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_pass_through_transform(MlsTraits::kK)),
                ::tl::make_tuple(::tl::sequence<0, 1>{}, ::tl::sequence<2>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
        else
        {
            // (tile_issue_k, tile_issue_mn, kMN, kK)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_k, tile_issue_mn),
                                     MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
                ::tl::make_tuple(::tl::sequence<1, 2>{}, ::tl::sequence<0, 3>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
    }
};

template <::tl::index_t Alt, ::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx946_mls_128x16_b8, Alt, BlockSizeMN, BlockSizeK, false>
{
    using MlsTraits = mls_traits<tl::mls::gfx946_mls_128x16_b8, Alt>;
    static constexpr ::tl::index_t MlsTileMN = MlsTraits::kMN0 * MlsTraits::kMN1;
    static constexpr ::tl::index_t MlsTileK  = 16;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};
        constexpr auto tile_issue_mn_outer =
            ::tl::number<tile_issue_mn / MlsTraits::kSlots>{};

        // (tile_issue_mn_outer, tile_issue_k, kMN0, kSlots, kK, kMN1)
        constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
            ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn_outer, tile_issue_k),
                                 MlsTraits::PackedShape));
        return ::tl::transform_tensor_descriptor(
            lds_desc_raw,
            ::tl::make_tuple(
                ::tl::make_merge_transform(::tl::make_tuple(
                    tile_issue_mn_outer, MlsTraits::kSlots, MlsTraits::kMN0, MlsTraits::kMN1)),
                ::tl::make_merge_transform(
                    ::tl::make_tuple(tile_issue_k, MlsTraits::kK))),
            ::tl::make_tuple(::tl::sequence<0, 2, 3, 5>{}, ::tl::sequence<1, 4>{}),
            ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
    }
};

template <::tl::index_t Alt, ::tl::index_t BlockSizeMN, ::tl::index_t BlockSizeK>
struct make_lds_desc_generic<tl::mls::gfx946_mls_16x128_trans_b8, Alt, BlockSizeMN, BlockSizeK, true>
{
    using MlsTraits = mls_traits<tl::mls::gfx946_mls_16x128_trans_b8, Alt>;
    static constexpr ::tl::index_t MlsTileMN = 16;
    static constexpr ::tl::index_t MlsTileK  = MlsTraits::kK0 * MlsTraits::kK1;

    static constexpr auto apply()
    {
        constexpr auto tile_issue_mn = ::tl::number<BlockSizeMN / MlsTileMN>{};
        constexpr auto tile_issue_k  = ::tl::number<BlockSizeK / MlsTileK>{};
        constexpr auto tile_issue_k_outer =
            ::tl::number<tile_issue_k / MlsTraits::kSlots>{};

        if constexpr(tile_issue_k_outer == 1)
        {
            // (tile_issue_mn, kK0, kSlots, kMN, kK1)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_mn), MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_merge_transform(::tl::make_tuple(
                        MlsTraits::kK0, MlsTraits::kSlots, MlsTraits::kK1))),
                ::tl::make_tuple(::tl::sequence<0, 3>{}, ::tl::sequence<1, 2, 4>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
        else
        {
            // (tile_issue_k_outer, tile_issue_mn, kK0, kSlots, kMN, kK1)
            constexpr auto lds_desc_raw = ::tl::make_naive_tensor_descriptor_packed(
                ::tl::concat_tuple(::tl::make_tuple(tile_issue_k_outer, tile_issue_mn),
                                     MlsTraits::PackedShape));
            return ::tl::transform_tensor_descriptor(
                lds_desc_raw,
                ::tl::make_tuple(
                    ::tl::make_merge_transform(
                        ::tl::make_tuple(tile_issue_mn, MlsTraits::kMN)),
                    ::tl::make_merge_transform(::tl::make_tuple(
                        tile_issue_k_outer, MlsTraits::kK0, MlsTraits::kSlots, MlsTraits::kK1))),
                ::tl::make_tuple(::tl::sequence<1, 3>{}, ::tl::sequence<0, 2, 4, 5>{}),
                ::tl::make_tuple(::tl::sequence<0>{}, ::tl::sequence<1>{}));
        }
    }
};

#endif

} // namespace detail

/*
 * mls_lds_desc_detail: LDS descriptor only, no warp partitioning.
 * Use when reading LDS with different warp layout than MLS write.
 * Template params: BlockSizeMN, BlockSizeK, MlsAtom, Alt, Trans.
 */
template <::tl::index_t BlockSizeMN,
          ::tl::index_t BlockSizeK,
          typename MlsAtomT,
          ::tl::index_t Alt,
          bool Trans>
struct mls_lds_desc_detail
{
    using MlsAtom   = MlsAtomT;
    using MlsTraits = mls_traits<MlsAtomT, Alt>;

    TL_DEVICE static constexpr auto make_lds_desc()
    {
        return detail::make_lds_desc_generic<MlsAtomT, Alt, BlockSizeMN, BlockSizeK, Trans>::apply();
    }
};

/*
 * mls_generic_detail: generic Detail for tile_window_mls_param_traits.
 */
template <::tl::index_t BlockSizeMN,
          ::tl::index_t BlockSizeK,
          typename MlsAtomT,
          ::tl::index_t WarpMN,
          ::tl::index_t WarpK,
          ::tl::index_t Alt,
          bool Trans>
struct mls_generic_detail
{
    using MlsAtom   = MlsAtomT;  // expose template param for uniform access via Detail::MlsAtom
    using MlsTraits = mls_traits<MlsAtomT, Alt>;

    static constexpr auto WarpCluster = ::tl::sequence<WarpMN, WarpK>{};

    static constexpr auto TileShape =
        ::tl::sequence<BlockSizeMN, BlockSizeK>{};
    static constexpr auto TileLoadWarpPerIssue = MlsAtomT::TileShape;
    static constexpr auto TileLoadWGPerIssue   = WarpCluster * TileLoadWarpPerIssue;

    // When TileLoadWGPerIssue > TileShape in a dim, warps repeat load (same data).
    // WarpMlsIssueSeq_i = 1 when repetition, else TileShape_i / TileLoadWGPerIssue_i.
    static constexpr ::tl::index_t TileLoadWGPerIssueMN = TileLoadWGPerIssue.at(::tl::number<0>{});
    static constexpr ::tl::index_t TileLoadWGPerIssueK  = TileLoadWGPerIssue.at(::tl::number<1>{});
    static constexpr ::tl::index_t WarpMlsIssueSeqMN =
        (BlockSizeMN >= TileLoadWGPerIssueMN) ? (BlockSizeMN / TileLoadWGPerIssueMN) : 1;
    static constexpr ::tl::index_t WarpMlsIssueSeqK =
        (BlockSizeK >= TileLoadWGPerIssueK) ? (BlockSizeK / TileLoadWGPerIssueK) : 1;
    static constexpr auto WarpMlsIssueSeq =
        ::tl::sequence<WarpMlsIssueSeqMN, WarpMlsIssueSeqK>{};

    // Effective warps per dim: unique warp positions (for modulo when repetition).
    static constexpr ::tl::index_t EffectiveWarpMN = BlockSizeMN / TileLoadWarpPerIssue.at(::tl::number<0>{});
    static constexpr ::tl::index_t EffectiveWarpK  = BlockSizeK / TileLoadWarpPerIssue.at(::tl::number<1>{});
    static constexpr auto EffectiveWarpCluster = ::tl::sequence<EffectiveWarpMN, EffectiveWarpK>{};

    using SFC_WarpAccess =
        ::tl::space_filling_curve<decltype(WarpMlsIssueSeq),
                                     ::tl::sequence<1, 0>,
                                     ::tl::sequence<1, 1>,
                                     false>;

    TL_DEVICE static constexpr auto make_lds_desc()
    {
        return detail::make_lds_desc_generic<MlsAtomT, Alt, BlockSizeMN, BlockSizeK, Trans>::apply();
    }
};

} // namespace mls
} // namespace tl
