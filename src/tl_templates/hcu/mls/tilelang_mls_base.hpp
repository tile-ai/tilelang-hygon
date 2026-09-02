// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

/*
 * tilelang_mls_base: simplified MLS base without BottomTensorView.
 *
 * Uses DataType* + runtime (ptr, mls_stride, mn_length_raw, k_length_raw).
 * pad_k = align_up(k_length_raw, BlockSizeK) - k_length_raw, computed in
 * constructor. mls_stride = stride in major-order direction. Trans=true (K
 * major):   offset = mn * mls_stride + k Trans=false (MN major): offset = k *
 * mls_stride + mn
 *
 * Template params: BlockSize (sequence<MN,K>), MlsTileSize (sequence<MN,K>),
 * WarpMN, WarpK, DataType, Alt, Trans, HcuArch. Uses scoped warp id
 * (get_warp_id() - warp_id_offset) for partition inside thread branches.
 */

#include <tl_templates/hcu/core.hpp>
#include <tl_templates/hcu/mls/mls_resource.hpp>

#include <tl_templates/hcu/mls/tile_window_mls.h>

namespace tl {
namespace mls {

enum class mls_resource_axis {
  auto_select,
  mn,
  k,
};

enum class mls_address_mode {
  absolute_rebase,
  forward_delta,
};

template <mls_address_mode Mode, ::tl::index_t NumAccess>
struct mls_base_addr_storage {};

template <::tl::index_t NumAccess>
struct mls_base_addr_storage<mls_address_mode::absolute_rebase, NumAccess> {
  ::tl::array<uintptr_t, NumAccess> mls_base_addr_;
};

template <typename Traits, mls_resource_axis RequestedAxis>
struct mls_resource_axis_policy {
  static constexpr auto NumWarpAccessMN =
      Traits::Detail::SFC_WarpAccess::access_lengths.at(::tl::number<0>{});
  static constexpr auto NumWarpAccessK =
      Traits::Detail::SFC_WarpAccess::access_lengths.at(::tl::number<1>{});
  // The one-shot lowering has no loop-nesting information.  Prefer the axis
  // requiring fewer descriptors, and preserve the historical MN choice on a
  // tie.  Hoist may override the requested axis when loop information makes
  // the tie-break (or the forward direction) cheaper.
  static constexpr mls_resource_axis Axis =
      RequestedAxis != mls_resource_axis::auto_select
          ? RequestedAxis
          : (NumWarpAccessK < NumWarpAccessMN ? mls_resource_axis::k
                                              : mls_resource_axis::mn);
  static constexpr auto NumResourceAccess =
      Axis == mls_resource_axis::mn ? NumWarpAccessMN : NumWarpAccessK;
};

template <typename BlockSize, typename MlsTileSize, ::tl::index_t WarpMN,
          ::tl::index_t WarpK, typename DataType, ::tl::index_t Alt, bool Trans,
          ::tl::hcu_target_enum HcuArch,
          ::tl::index_t DstBits = mls_elem_bits_v<DataType>,
          mls_resource_axis ResourceAxis = mls_resource_axis::auto_select,
          mls_address_mode AddressMode = mls_address_mode::absolute_rebase>
struct tilelang_mls_base
    : private mls_base_addr_storage<
          AddressMode,
          mls_resource_axis_policy<
              tile_window_mls_param_traits<BlockSize, MlsTileSize, WarpMN,
                                           WarpK, mls_elem_bits_v<DataType>,
                                           DstBits, Alt, Trans, HcuArch>,
              ResourceAxis>::NumResourceAccess> {
  using Traits = tile_window_mls_param_traits<BlockSize, MlsTileSize, WarpMN,
                                              WarpK, mls_elem_bits_v<DataType>,
                                              DstBits, Alt, Trans, HcuArch>;
  using Detail = typename Traits::Detail;
  using MlsAtom = typename Detail::MlsAtom; // both generic and ck_tile Detail
                                            // expose MlsAtom

  static constexpr auto BlockSizeMN = BlockSize::at(::tl::number<0>{});
  static constexpr auto BlockSizeK = BlockSize::at(::tl::number<1>{});

  static constexpr auto WarpCluster = Detail::WarpCluster;
  static constexpr auto EffectiveWarpCluster = Detail::EffectiveWarpCluster;
  static constexpr auto TileLoadWarpPerIssue = Detail::TileLoadWarpPerIssue;
  static constexpr auto TileLoadWGPerIssue = Detail::TileLoadWGPerIssue;

  static constexpr auto TileLoadWarpPerIssueMN =
      TileLoadWarpPerIssue.at(::tl::number<0>{});
  static constexpr auto TileLoadWarpPerIssueK =
      TileLoadWarpPerIssue.at(::tl::number<1>{});

  static constexpr auto TileLoadWGPerIssueMN =
      TileLoadWGPerIssue.at(::tl::number<0>{});
  static constexpr auto TileLoadWGPerIssueK =
      TileLoadWGPerIssue.at(::tl::number<1>{});

  using SFC_WarpAccess = typename Detail::SFC_WarpAccess;

  static constexpr auto NumWarpAccess = SFC_WarpAccess::get_num_of_access();
  static constexpr auto NumWarpAccessMN =
      SFC_WarpAccess::access_lengths.at(::tl::number<0>{});
  static constexpr auto NumWarpAccessK =
      SFC_WarpAccess::access_lengths.at(::tl::number<1>{});

  using ResourcePolicy = mls_resource_axis_policy<Traits, ResourceAxis>;
  static constexpr auto ResolvedResourceAxis = ResourcePolicy::Axis;
  static constexpr auto NumResourceAccess = ResourcePolicy::NumResourceAccess;
  static constexpr bool ResourceAlongMN =
      ResolvedResourceAxis == mls_resource_axis::mn;

  using BaseAddrStorage = mls_base_addr_storage<AddressMode, NumResourceAccess>;

  TL_DEVICE static constexpr auto get_num_of_access() { return NumWarpAccess; }

  TL_DEVICE static constexpr auto get_tile_lds_desc() {
    return Detail::make_lds_desc();
  }

  // HIP C++ clamp, no inline asm. Prefer `?:` over if/`v_med3`/bit-hacks:
  // on this hipcc it stays SALU (`s_max_i32` then `s_min_u32`) with no
  // EXEC branch. Input readfirstlane is dead (coord/tile/length already
  // SGPR). `if` and signed bit min/max both InstCombine to v_med3.
  TL_DEVICE static ::tl::index_t clamp_filter_uniform(::tl::index_t coord,
                                                      ::tl::index_t tile,
                                                      ::tl::index_t length) {
    ::tl::index_t value = coord + tile - length;
    value = value < 0 ? 0 : value;
    value = value > tile ? tile : value;
    return value;
  }

  TL_DEVICE tilelang_mls_base(DataType *p_data, ::tl::index_t mls_stride,
                              ::tl::index_t mn_length_raw,
                              ::tl::index_t k_length_raw,
                              ::tl::index_t warp_id_offset = 0)
      : p_data_(p_data), mls_stride_(mls_stride), mn_length_raw_(mn_length_raw),
        k_length_raw_(k_length_raw), warp_id_offset_(warp_id_offset) {
    init();
  }

  TL_DEVICE auto get_warp_cluster_idx() {
    constexpr auto warp_cluster_to_id_adaptor =
        ::tl::make_single_stage_tensor_adaptor(
            ::tl::make_tuple(::tl::make_merge_transform(WarpCluster)),
            ::tl::make_tuple(
                typename ::tl::arithmetic_sequence_gen<0, WarpCluster.size(),
                                                       1>::type{}),
            ::tl::make_tuple(::tl::sequence<0>{}));

    // The scoped warp id is uniform within a wave.  Scalarize it before
    // running the coordinate adaptor so MLS origins, filters, and resource
    // descriptors stay on the SALU path instead of creating VALU temporaries.
    const auto scoped_warp_id =
        __builtin_amdgcn_readfirstlane(::tl::get_warp_id() - warp_id_offset_);
    return warp_cluster_to_id_adaptor.calculate_bottom_index(
        ::tl::make_multi_index(scoped_warp_id));
  }

  template <typename WarpClusterIdx, typename AccessIdx>
  TL_DEVICE static auto
  calculate_tile_warp_coord(const WarpClusterIdx &warp_cluster_idx,
                            const AccessIdx &access_idx) {
    return ::tl::generate_tuple(
        [&](auto ii) {
          const auto eff_idx =
              (EffectiveWarpCluster.at(ii) != WarpCluster.at(ii))
                  ? (warp_cluster_idx[ii] % EffectiveWarpCluster.at(ii))
                  : warp_cluster_idx[ii];
          return eff_idx * TileLoadWarpPerIssue.at(ii) +
                 access_idx[ii] * TileLoadWGPerIssue.at(ii);
        },
        ::tl::number<2>{});
  }

  TL_DEVICE void init() {
    constexpr auto tile_lds_desc = get_tile_lds_desc();
    const auto warp_cluster_idx = get_warp_cluster_idx();

    ::tl::static_for<0, NumWarpAccess, 1>{}([&](auto i) {
      constexpr auto access_idx = SFC_WarpAccess::get_index(i);

      const auto tile_warp_coord =
          calculate_tile_warp_coord(warp_cluster_idx, access_idx);

      mls_lds_offset_(i) =
          __builtin_amdgcn_readfirstlane(tile_lds_desc.calculate_offset(
              ::tl::to_multi_index(tile_warp_coord)));
    });
  }

  TL_DEVICE void
  init(const ::tl::array<::tl::index_t, 2> &block_window_origin) {
    const auto warp_cluster_idx = get_warp_cluster_idx();
    const auto origin_mn = block_window_origin.at(::tl::number<0>{});
    const auto origin_k = block_window_origin.at(::tl::number<1>{});
    ::tl::static_for<0, NumResourceAccess, 1>{}([&](auto i) {
      constexpr auto access_idx = [&]() {
        if constexpr (ResourceAlongMN)
          return ::tl::make_tuple(i, ::tl::number<0>{});
        else
          return ::tl::make_tuple(::tl::number<0>{}, i);
      }();

      const auto tile_warp_coord =
          calculate_tile_warp_coord(warp_cluster_idx, access_idx);

      const auto tile_mn = tile_warp_coord[::tl::number<0>{}];
      const auto tile_k = tile_warp_coord[::tl::number<1>{}];
      resource_mn_offset_(i) = tile_mn;
      resource_k_offset_(i) = tile_k;

      const auto warp_coord_mn = origin_mn + tile_mn;
      const auto warp_coord_k = origin_k + tile_k;

      ::tl::index_t mls_mn_filter = 0;
      ::tl::index_t mls_k_filter = 0;
      if constexpr (ResourceAlongMN) {
        mls_mn_filter = clamp_filter_uniform(
            warp_coord_mn, TileLoadWarpPerIssueMN, mn_length_raw_);
      } else {
        // Resource-K: only the K field is stored in the descriptor.  MN
        // filters are applied per moffset issue at load time.
        // Keep the K clamp explicit: eliding a known-zero filter miscompiles
        // on the current HCU backend/model.
        mls_k_filter = clamp_filter_uniform(warp_coord_k, TileLoadWarpPerIssueK,
                                            k_length_raw_);
      }

      constexpr auto mfmt = ::tl::mls::detail::mfmt_traits<Alt>::value;

      uint8_t *ptr_offset = reinterpret_cast<uint8_t *>(p_data_);
      if constexpr (Trans) {
        const ::tl::long_index_t offset_elems =
            static_cast<::tl::long_index_t>(warp_coord_mn) * mls_stride_ +
            warp_coord_k;
        ptr_offset += ::tl::mls::mls_storage_traits<
            DataType>::logical_offset_to_byte_offset(offset_elems);
        mls_res_(i) = ::tl::mls::make_mls_resource(
            static_cast<const void *>(ptr_offset), mls_stride_,
            ResourceAlongMN ? 0 : mls_k_filter,
            ResourceAlongMN ? mls_mn_filter : 0, mfmt);
      } else {
        const ::tl::long_index_t offset_elems =
            static_cast<::tl::long_index_t>(warp_coord_k) * mls_stride_ +
            warp_coord_mn;
        ptr_offset += ::tl::mls::mls_storage_traits<
            DataType>::logical_offset_to_byte_offset(offset_elems);
        mls_res_(i) = ::tl::mls::make_mls_resource(
            static_cast<const void *>(ptr_offset), mls_stride_,
            ResourceAlongMN ? mls_mn_filter : 0,
            ResourceAlongMN ? 0 : mls_k_filter, mfmt);
      }
      if constexpr (AddressMode == mls_address_mode::absolute_rebase) {
        this->mls_base_addr_(i) = reinterpret_cast<uintptr_t>(ptr_offset);
      }
    });
  }

  TL_DEVICE void
  set_window_origin(const ::tl::array<::tl::index_t, 2> &block_window_origin) {
    init(block_window_origin);
  }

  TL_DEVICE void set_k_filter(::tl::int32x4_t &mls_res, uint32_t filter) {
    if constexpr (Trans) {
      mls_res.w = (mls_res.w & ~static_cast<uint32_t>(0xff)) | filter;
    } else {
      mls_res.w = (mls_res.w & ~static_cast<uint32_t>(0xff00)) | (filter << 8);
    }
  }

  TL_DEVICE void set_mn_filter(::tl::int32x4_t &mls_res, uint32_t filter) {
    if constexpr (Trans) {
      mls_res.w = (mls_res.w & ~static_cast<uint32_t>(0xff00)) | (filter << 8);
    } else {
      mls_res.w = (mls_res.w & ~static_cast<uint32_t>(0xff)) | filter;
    }
  }

  TL_DEVICE uint32_t get_mn_filter(const ::tl::index_t block_mn_base,
                                   const ::tl::index_t access_idx_mn) {
    const auto warp_coord_mn =
        block_mn_base + resource_mn_offset_(access_idx_mn);
    return static_cast<uint32_t>(clamp_filter_uniform(
        warp_coord_mn, TileLoadWarpPerIssueMN, mn_length_raw_));
  }

  TL_DEVICE uint32_t get_k_filter(const ::tl::index_t block_k_base,
                                  const ::tl::index_t access_idx_k) {
    const auto warp_coord_k = block_k_base + resource_k_offset_(access_idx_k);
    return static_cast<uint32_t>(clamp_filter_uniform(
        warp_coord_k, TileLoadWarpPerIssueK, k_length_raw_));
  }

  TL_DEVICE void refresh_k_filter(::tl::index_t block_k_base) {
    if constexpr (!ResourceAlongMN) {
      ::tl::static_for<0, NumResourceAccess, 1>{}([&](auto i) {
        set_k_filter(mls_res_(i), get_k_filter(block_k_base, i));
      });
    }
  }

  TL_DEVICE void refresh_mn_filter(::tl::index_t block_mn_base) {
    ::tl::static_for<0, NumResourceAccess, 1>{}([&](auto i) {
      set_mn_filter(mls_res_(i), get_mn_filter(block_mn_base, i));
    });
  }

  template <typename T, bool bps = false>
  TL_DEVICE void async_mls_load_asm_impl_ct(TL_LDS_ADDR T *smem,
                                            ::tl::bool_constant<bps> = {}) {
    ::tl::static_for<0, NumWarpAccess, 1>{}([&](auto i) {
      constexpr auto access_idx = SFC_WarpAccess::get_index(i);
      constexpr auto access_idx_mn = access_idx[::tl::number<0>{}];
      constexpr auto access_idx_k = access_idx[::tl::number<1>{}];
      constexpr auto resource_idx =
          ResourceAlongMN ? access_idx_mn : access_idx_k;
      constexpr auto moffset_value = ResourceAlongMN
                                         ? access_idx_k * TileLoadWGPerIssueK
                                         : access_idx_mn * TileLoadWGPerIssueMN;
      constexpr auto moffset = ::tl::number<moffset_value>{};
      static_assert(moffset_value >= -512 && moffset_value <= 511,
                    "MLS moffset exceeds its signed 10-bit element range");

      if constexpr (ResourceAlongMN) {
        if constexpr (AddressMode == mls_address_mode::absolute_rebase) {
          set_k_filter(mls_res_(resource_idx), 0);
        }
      } else if constexpr (AddressMode == mls_address_mode::absolute_rebase) {
        set_mn_filter(mls_res_(resource_idx), 0);
      }

      if constexpr (HcuArch == ::tl::hcu_target_enum::gfx938) {
        MlsAtom::template load<moffset, ResourceAlongMN>(
            reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
            mls_res_[resource_idx], moffset,
            ::tl::bool_constant<ResourceAlongMN>{});
      } else if constexpr (HcuArch == ::tl::hcu_target_enum::gfx946 ||
                           HcuArch == ::tl::hcu_target_enum::gfx92a) {
        TL_LDS_ADDR uint8_t *smem_bytes =
            reinterpret_cast<TL_LDS_ADDR uint8_t *>(smem);
        const auto lds_byte_offset = ::tl::mls::mls_lds_physical_storage_traits<
            DstBits>::logical_offset_to_byte_offset(mls_lds_offset_[i]);
        MlsAtom::template load<moffset, ResourceAlongMN, bps>(
            reinterpret_cast<uintptr_t>(smem_bytes + lds_byte_offset),
            mls_res_[resource_idx], moffset,
            ::tl::bool_constant<ResourceAlongMN>{}, ::tl::bool_constant<bps>{});
      }
    });
  }

  template <typename T, bool bps>
  TL_DEVICE void async_mls_load_asm_impl_rt(TL_LDS_ADDR T *smem,
                                            ::tl::bool_constant<bps>,
                                            ::tl::index_t block_k_base,
                                            ::tl::index_t block_mn_base) {
    const auto warp_cluster_idx = get_warp_cluster_idx();
    ::tl::static_for<0, NumWarpAccess, 1>{}([&](auto i) {
      constexpr auto access_idx = SFC_WarpAccess::get_index(i);
      constexpr auto access_idx_mn = access_idx[::tl::number<0>{}];
      constexpr auto access_idx_k = access_idx[::tl::number<1>{}];

      constexpr auto resource_idx =
          ResourceAlongMN ? access_idx_mn : access_idx_k;
      constexpr auto moffset_value = ResourceAlongMN
                                         ? access_idx_k * TileLoadWGPerIssueK
                                         : access_idx_mn * TileLoadWGPerIssueMN;
      constexpr auto moffset = ::tl::number<moffset_value>{};
      static_assert(moffset_value >= -512 && moffset_value <= 511,
                    "MLS moffset exceeds its signed 10-bit element range");

      const auto tile_warp_coord =
          calculate_tile_warp_coord(warp_cluster_idx, access_idx);
      if constexpr (ResourceAlongMN) {
        const auto coord = block_k_base + tile_warp_coord[::tl::number<1>{}];
        set_k_filter(mls_res_(resource_idx),
                     static_cast<uint32_t>(clamp_filter_uniform(
                         coord, TileLoadWarpPerIssueK, k_length_raw_)));
      } else {
        const auto coord = block_mn_base + tile_warp_coord[::tl::number<0>{}];
        set_mn_filter(mls_res_(resource_idx),
                      static_cast<uint32_t>(clamp_filter_uniform(
                          coord, TileLoadWarpPerIssueMN, mn_length_raw_)));
      }

      if constexpr (HcuArch == ::tl::hcu_target_enum::gfx938) {
        MlsAtom::template load<moffset, ResourceAlongMN>(
            reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
            mls_res_[resource_idx], moffset,
            ::tl::bool_constant<ResourceAlongMN>{});
      } else if constexpr (HcuArch == ::tl::hcu_target_enum::gfx946 ||
                           HcuArch == ::tl::hcu_target_enum::gfx92a) {
        TL_LDS_ADDR uint8_t *smem_bytes =
            reinterpret_cast<TL_LDS_ADDR uint8_t *>(smem);
        const auto lds_byte_offset = ::tl::mls::mls_lds_physical_storage_traits<
            DstBits>::logical_offset_to_byte_offset(mls_lds_offset_[i]);
        MlsAtom::template load<moffset, ResourceAlongMN, bps>(
            reinterpret_cast<uintptr_t>(smem_bytes + lds_byte_offset),
            mls_res_[resource_idx], moffset,
            ::tl::bool_constant<ResourceAlongMN>{}, ::tl::bool_constant<bps>{});
      }
    });
  }

  template <typename T, bool refresh_k = true, bool refresh_mn = true>
  TL_DEVICE void async_mls_load_asm(TL_LDS_ADDR void *smem,
                                    ::tl::index_t block_k_base,
                                    ::tl::index_t block_mn_base) {
    auto *typed_smem = reinterpret_cast<TL_LDS_ADDR T *>(smem);
    constexpr bool refresh_filter = ResourceAlongMN ? refresh_k : refresh_mn;
    if constexpr (refresh_filter) {
      async_mls_load_asm_impl_rt<T, false>(typed_smem,
                                           ::tl::bool_constant<false>{},
                                           block_k_base, block_mn_base);
    } else {
      async_mls_load_asm_impl_ct<T, false>(typed_smem);
    }
  }

  template <bool refresh_k = true>
  TL_DEVICE void move_k_base(const ::tl::index_t delta_k,
                             const ::tl::index_t next_k_base) {
    ::tl::long_index_t addr_byte_offset;
    if constexpr (Trans) {
      addr_byte_offset = ::tl::mls::mls_storage_traits<
          DataType>::logical_offset_to_byte_offset(delta_k);
    } else {
      addr_byte_offset = ::tl::mls::mls_storage_traits<DataType>::
          logical_offset_to_byte_offset(
              static_cast<::tl::long_index_t>(delta_k) * mls_stride_);
    }
    ::tl::static_for<0, NumResourceAccess, 1>{}([&](auto i) {
      ::tl::mls::move_mls_addr_base(mls_res_(i), addr_byte_offset);
    });
    if constexpr (!ResourceAlongMN) {
      if constexpr (refresh_k)
        refresh_k_filter(next_k_base);
    }
  }

  template <bool refresh_mn = true>
  TL_DEVICE void move_mn_base(const ::tl::index_t delta_mn,
                              const ::tl::index_t next_mn_base) {
    ::tl::long_index_t addr_byte_offset;
    if constexpr (Trans) {
      addr_byte_offset = ::tl::mls::mls_storage_traits<DataType>::
          logical_offset_to_byte_offset(
              static_cast<::tl::long_index_t>(delta_mn) * mls_stride_);
    } else {
      addr_byte_offset = ::tl::mls::mls_storage_traits<
          DataType>::logical_offset_to_byte_offset(delta_mn);
    }
    ::tl::static_for<0, NumResourceAccess, 1>{}([&](auto i) {
      ::tl::mls::move_mls_addr_base(mls_res_(i), addr_byte_offset);
    });
    if constexpr (ResourceAlongMN) {
      if constexpr (refresh_mn)
        refresh_mn_filter(next_mn_base);
    }
  }

  template <bool refresh_k = true>
  TL_DEVICE void update_k_base(const ::tl::index_t block_k_base) {
    static_assert(AddressMode == mls_address_mode::absolute_rebase,
                  "update_k_base requires absolute_rebase MLS address mode");
    apply_k_addr_from_window(block_k_base);
    if constexpr (!ResourceAlongMN) {
      if constexpr (refresh_k)
        refresh_k_filter(block_k_base);
      else
        ::tl::static_for<0, NumResourceAccess, 1>{}(
            [&](auto i) { set_k_filter(mls_res_(i), 0); });
    }
  }

  template <bool refresh_mn = true>
  TL_DEVICE void update_mn_base(const ::tl::index_t block_mn_base) {
    static_assert(AddressMode == mls_address_mode::absolute_rebase,
                  "update_mn_base requires absolute_rebase MLS address mode");
    apply_mn_addr_from_window(block_mn_base);
    if constexpr (ResourceAlongMN) {
      if constexpr (refresh_mn)
        refresh_mn_filter(block_mn_base);
      else
        ::tl::static_for<0, NumResourceAccess, 1>{}(
            [&](auto i) { set_mn_filter(mls_res_(i), 0); });
    }
  }

  TL_DEVICE void apply_k_addr_from_window(::tl::index_t block_k_base) {
    ::tl::long_index_t addr_byte_offset;
    if constexpr (Trans) {
      addr_byte_offset = ::tl::mls::mls_storage_traits<
          DataType>::logical_offset_to_byte_offset(block_k_base);
    } else {
      addr_byte_offset = ::tl::mls::mls_storage_traits<DataType>::
          logical_offset_to_byte_offset(
              static_cast<::tl::long_index_t>(block_k_base) * mls_stride_);
    }
    ::tl::static_for<0, NumResourceAccess, 1>{}([&](auto i) {
      update_mls_addr_base(mls_res_(i), this->mls_base_addr_(i),
                           addr_byte_offset);
    });
  }

  TL_DEVICE void apply_mn_addr_from_window(::tl::index_t block_mn_base) {
    ::tl::long_index_t addr_byte_offset;
    if constexpr (Trans) {
      addr_byte_offset = ::tl::mls::mls_storage_traits<DataType>::
          logical_offset_to_byte_offset(
              static_cast<::tl::long_index_t>(block_mn_base) * mls_stride_);
    } else {
      addr_byte_offset = ::tl::mls::mls_storage_traits<
          DataType>::logical_offset_to_byte_offset(block_mn_base);
    }
    ::tl::static_for<0, NumResourceAccess, 1>{}([&](auto i) {
      update_mls_addr_base(mls_res_(i), this->mls_base_addr_(i),
                           addr_byte_offset);
    });
  }

public:
  DataType *p_data_;
  ::tl::array<::tl::int32x4_t, NumResourceAccess> mls_res_;
  ::tl::array<::tl::index_t, NumResourceAccess> resource_mn_offset_;
  ::tl::array<::tl::index_t, NumResourceAccess> resource_k_offset_;

  ::tl::index_t mls_stride_;
  ::tl::index_t mn_length_raw_;
  ::tl::index_t k_length_raw_;
  ::tl::index_t warp_id_offset_{0};

  ::tl::array<::tl::index_t, NumWarpAccess> mls_lds_offset_;
};

/*
 * mls_load_tile: one-shot MLS load (no hoist).
 * Flow: construct -> set_window_origin(mn, k) -> async_mls_load_asm.
 * refresh_k / refresh_mn are established results (user or proof).
 */
template <typename BlockSize, typename MlsTileSize, ::tl::index_t WarpMN,
          ::tl::index_t WarpK, typename DataType, ::tl::index_t Alt, bool Trans,
          ::tl::hcu_target_enum HcuArch,
          ::tl::index_t DstBits = mls_elem_bits_v<DataType>,
          bool refresh_k = true, bool refresh_mn = true>
TL_DEVICE void
mls_load_tile(DataType *p_data, ::tl::index_t mls_stride,
              ::tl::index_t mn_length_raw, ::tl::index_t k_length_raw,
              ::tl::index_t block_mn_base, ::tl::index_t block_k_base,
              TL_LDS_ADDR void *smem, ::tl::index_t warp_id_offset = 0) {
  using MlsBase = tilelang_mls_base<BlockSize, MlsTileSize, WarpMN, WarpK,
                                    DataType, Alt, Trans, HcuArch, DstBits>;
  MlsBase mls(p_data, mls_stride, mn_length_raw, k_length_raw, warp_id_offset);
  mls.set_window_origin(
      ::tl::make_array<::tl::index_t>(block_mn_base, block_k_base));
  auto *typed_smem = reinterpret_cast<TL_LDS_ADDR DataType *>(smem);
  mls.template async_mls_load_asm<DataType, refresh_k, refresh_mn>(
      typed_smem, block_k_base, block_mn_base);
}

} // namespace mls
} // namespace tl
