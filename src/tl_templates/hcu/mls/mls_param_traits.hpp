// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <tl_templates/hcu/core.hpp>
#include <tl_templates/hcu/core/numeric/pk_fp4.hpp>
#include <tl_templates/hcu/core/numeric/pk_int4.hpp>

#include <tl_templates/hcu/mls/mls_generic_detail.hpp>
#include <tl_templates/hcu/mls/tl_mls_atom_dispatcher.hpp>

namespace tl {
namespace mls {

template <typename T> struct mls_elem_bits {
  static constexpr ::tl::index_t value = sizeof(::tl::remove_cvref_t<T>) * 8;
};

template <> struct mls_elem_bits<::tl::pk_fp4_t> {
  static constexpr ::tl::index_t value = 4;
};

template <> struct mls_elem_bits<::tl::pk_int4_t> {
  static constexpr ::tl::index_t value = 4;
};

template <typename T>
static constexpr ::tl::index_t mls_elem_bits_v =
    mls_elem_bits<::tl::remove_cvref_t<T>>::value;

template <typename T> struct mls_storage_traits {
  using Type = ::tl::remove_cvref_t<T>;
  static constexpr ::tl::index_t LogicalBits = mls_elem_bits_v<Type>;
  static constexpr ::tl::index_t StorageBits = sizeof(Type) * 8;
  static_assert(
      StorageBits % LogicalBits == 0,
      "MLS storage type must hold a whole number of logical elements");
  static constexpr ::tl::index_t LogicalPerStorage = StorageBits / LogicalBits;

  TL_HOST_DEVICE static constexpr ::tl::index_t
  logical_offset_to_storage_offset(::tl::index_t logical_offset) {
    return logical_offset / LogicalPerStorage;
  }

  TL_HOST_DEVICE static constexpr ::tl::index_t
  logical_offset_to_byte_offset(::tl::index_t logical_offset) {
    return logical_offset_to_storage_offset(logical_offset) * sizeof(Type);
  }
};

template <::tl::index_t DstBits> struct mls_lds_physical_storage_traits {
  static_assert(DstBits == 4 || DstBits == 8 || DstBits == 16 || DstBits == 32,
                "Unsupported MLS LDS physical bits");

  TL_HOST_DEVICE static constexpr ::tl::index_t
  logical_offset_to_byte_offset(::tl::index_t logical_offset) {
    return logical_offset * DstBits / 8;
  }
};

/*
 * mls_lds_desc_param_traits: LDS descriptor only, no WarpMN/WarpK.
 * Use when reading LDS with different warp layout than MLS write.
 * MlsTileSize: ::tl::sequence<MN, K> - MLS tile dimensions (MN dim, K dim).
 */
template <typename BlockSize, typename MlsTileSize, ::tl::index_t SrcBits,
          ::tl::index_t DstBits, ::tl::index_t Alt, bool Trans,
          ::tl::hcu_target_enum HcuArch>
struct mls_lds_desc_param_traits {
  static constexpr ::tl::index_t BlockSizeMN = BlockSize::at(::tl::number<0>{});
  static constexpr ::tl::index_t BlockSizeK = BlockSize::at(::tl::number<1>{});
  static constexpr ::tl::index_t MlsTileMN = MlsTileSize::at(::tl::number<0>{});
  static constexpr ::tl::index_t MlsTileK = MlsTileSize::at(::tl::number<1>{});

  using MlsAtom = typename mls_atom_for_tile<MlsTileMN, MlsTileK, Trans,
                                             SrcBits, DstBits, HcuArch>::Type;

  using LdsDescDetail =
      mls_lds_desc_detail<BlockSizeMN, BlockSizeK, MlsAtom, Alt, Trans>;

  TL_DEVICE static constexpr auto get_tile_lds_desc() {
    return LdsDescDetail::make_lds_desc();
  }
};

/*
 * tile_window_mls_param_traits: parameterized MLS traits.
 * Template params: BlockSize (::tl::sequence<MN,K>), MlsTileSize
 * (::tl::sequence<MN,K>), WarpMN, WarpK, SrcBits/DstBits (element size in
 * bits, e.g. 16 for b16), Alt, Trans, HcuArch.
 */
template <typename BlockSize, typename MlsTileSize, ::tl::index_t WarpMN,
          ::tl::index_t WarpK, ::tl::index_t SrcBits, ::tl::index_t DstBits,
          ::tl::index_t Alt, bool Trans, ::tl::hcu_target_enum HcuArch>
struct tile_window_mls_param_traits {
  static constexpr ::tl::index_t BlockSizeMN = BlockSize::at(::tl::number<0>{});
  static constexpr ::tl::index_t BlockSizeK = BlockSize::at(::tl::number<1>{});
  static constexpr ::tl::index_t MlsTileMN = MlsTileSize::at(::tl::number<0>{});
  static constexpr ::tl::index_t MlsTileK = MlsTileSize::at(::tl::number<1>{});

  using MlsAtom = typename mls_atom_for_tile<MlsTileMN, MlsTileK, Trans,
                                             SrcBits, DstBits, HcuArch>::Type;

  using Detail = mls_generic_detail<BlockSizeMN, BlockSizeK, MlsAtom, WarpMN,
                                    WarpK, Alt, Trans>;
};

} // namespace mls
} // namespace tl
