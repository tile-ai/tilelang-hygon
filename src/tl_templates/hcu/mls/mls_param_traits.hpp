#pragma once

#include <tl_templates/hcu/core.hpp>

#include <tl_templates/hcu/mls/tl_mls_atom_dispatcher.hpp>
#include <tl_templates/hcu/mls/mls_generic_detail.hpp>

namespace tl {
namespace mls {

/*
 * mls_lds_desc_param_traits: LDS descriptor only, no WarpMN/WarpK.
 * Use when reading LDS with different warp layout than MLS write.
 * MlsTileSize: ::tl::sequence<MN, K> - MLS tile dimensions (MN dim, K dim).
 */
template <typename BlockSize,
          typename MlsTileSize,
          ::tl::index_t Bits,
          ::tl::index_t Alt,
          bool Trans,
          ::tl::hcu_target_enum HcuArch>
struct mls_lds_desc_param_traits
{
    static constexpr ::tl::index_t BlockSizeMN = BlockSize::at(::tl::number<0>{});
    static constexpr ::tl::index_t BlockSizeK  = BlockSize::at(::tl::number<1>{});
    static constexpr ::tl::index_t MlsTileMN   = MlsTileSize::at(::tl::number<0>{});
    static constexpr ::tl::index_t MlsTileK   = MlsTileSize::at(::tl::number<1>{});

    using MlsAtom = typename mls_atom_for_tile<MlsTileMN, MlsTileK, Trans, Bits, HcuArch>::Type;

    using LdsDescDetail = mls_lds_desc_detail<BlockSizeMN, BlockSizeK, MlsAtom, Alt, Trans>;

    TL_DEVICE static constexpr auto get_tile_lds_desc() { return LdsDescDetail::make_lds_desc(); }
};

/*
 * tile_window_mls_param_traits: parameterized MLS traits.
 * Template params: BlockSize (::tl::sequence<MN,K>), MlsTileSize (::tl::sequence<MN,K>),
 *                  WarpMN, WarpK, Bits (element size in bits, e.g. 16 for b16), Alt, Trans, HcuArch.
 */
template <typename BlockSize,
          typename MlsTileSize,
          ::tl::index_t WarpMN,
          ::tl::index_t WarpK,
          ::tl::index_t Bits,
          ::tl::index_t Alt,
          bool Trans,
          ::tl::hcu_target_enum HcuArch>
struct tile_window_mls_param_traits
{
    static constexpr ::tl::index_t BlockSizeMN = BlockSize::at(::tl::number<0>{});
    static constexpr ::tl::index_t BlockSizeK  = BlockSize::at(::tl::number<1>{});
    static constexpr ::tl::index_t MlsTileMN   = MlsTileSize::at(::tl::number<0>{});
    static constexpr ::tl::index_t MlsTileK   = MlsTileSize::at(::tl::number<1>{});

    using MlsAtom = typename mls_atom_for_tile<MlsTileMN, MlsTileK, Trans, Bits, HcuArch>::Type;

    using Detail = mls_generic_detail<BlockSizeMN, BlockSizeK, MlsAtom, WarpMN, WarpK, Alt, Trans>;
};

} // namespace mls
} // namespace tl
