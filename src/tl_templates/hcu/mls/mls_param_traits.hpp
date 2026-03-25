#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/core/tensor/tile_window_mls_gfx938.hpp>

#include "tl_mls_atom_dispatcher.hpp"
#include "mls_generic_detail.hpp"

namespace tl {
namespace mls {

/*
 * mls_lds_desc_param_traits: LDS descriptor only, no WarpMN/WarpK.
 * Use when reading LDS with different warp layout than MLS write.
 * MlsTileSize: ck_tile::sequence<MN, K> - MLS tile dimensions (MN dim, K dim).
 */
template <typename BlockSize,
          typename MlsTileSize,
          ck_tile::index_t Bits,
          ck_tile::index_t Alt,
          bool Trans,
          ck_tile::hcu_target_enum HcuArch>
struct mls_lds_desc_param_traits
{
    static constexpr ck_tile::index_t BlockSizeMN = BlockSize::at(ck_tile::number<0>{});
    static constexpr ck_tile::index_t BlockSizeK  = BlockSize::at(ck_tile::number<1>{});
    static constexpr ck_tile::index_t MlsTileMN   = MlsTileSize::at(ck_tile::number<0>{});
    static constexpr ck_tile::index_t MlsTileK   = MlsTileSize::at(ck_tile::number<1>{});

    using MlsAtom = typename mls_atom_for_tile<MlsTileMN, MlsTileK, Trans, Bits, HcuArch>::Type;

    using LdsDescDetail = mls_lds_desc_detail<BlockSizeMN, BlockSizeK, MlsAtom, Alt, Trans>;

    CK_TILE_DEVICE static constexpr auto get_tile_lds_desc() { return LdsDescDetail::make_lds_desc(); }
};

/*
 * tile_window_mls_param_traits: parameterized MLS traits.
 * Template params: BlockSize (ck_tile::sequence<MN,K>), MlsTileSize (ck_tile::sequence<MN,K>),
 *                  WarpMN, WarpK, Bits (element size in bits, e.g. 16 for b16), Alt, Trans, HcuArch.
 */
template <typename BlockSize,
          typename MlsTileSize,
          ck_tile::index_t WarpMN,
          ck_tile::index_t WarpK,
          ck_tile::index_t Bits,
          ck_tile::index_t Alt,
          bool Trans,
          ck_tile::hcu_target_enum HcuArch>
struct tile_window_mls_param_traits
{
    static constexpr ck_tile::index_t BlockSizeMN = BlockSize::at(ck_tile::number<0>{});
    static constexpr ck_tile::index_t BlockSizeK  = BlockSize::at(ck_tile::number<1>{});
    static constexpr ck_tile::index_t MlsTileMN   = MlsTileSize::at(ck_tile::number<0>{});
    static constexpr ck_tile::index_t MlsTileK   = MlsTileSize::at(ck_tile::number<1>{});

    using MlsAtom = typename mls_atom_for_tile<MlsTileMN, MlsTileK, Trans, Bits, HcuArch>::Type;

    using Detail = mls_generic_detail<BlockSizeMN, BlockSizeK, MlsAtom, WarpMN, WarpK, Alt, Trans>;
};

} // namespace mls
} // namespace tl
