#pragma once

/*
 * tilelang_mls_base: simplified MLS base without BottomTensorView.
 *
 * Uses DataType* + runtime (ptr, mls_stride, mn_length_raw, k_length_raw).
 * pad_k = align_up(k_length_raw, BlockSizeK) - k_length_raw, computed in constructor.
 * mls_stride = stride in major-order direction.
 * Trans=true (K major):   offset = mn * mls_stride + k
 * Trans=false (MN major): offset = k * mls_stride + mn
 *
 * Template params: BlockSize (sequence<MN,K>), MlsTileSize (sequence<MN,K>), WarpMN, WarpK,
 *                  DataType, Alt, Trans, HcuArch.
 * Uses ::tl::get_warp_id() internally (standard block layout).
 */

#include <tl_templates/hcu/core.hpp>
#include <tl_templates/hcu/mls/mls_resource.hpp>

#include <tl_templates/hcu/mls/tile_window_mls.h>

namespace tl {
namespace mls {

template <typename BlockSize,
          typename MlsTileSize,
          ::tl::index_t WarpMN,
          ::tl::index_t WarpK,
          typename DataType,
          ::tl::index_t Alt,
          bool Trans,
          ::tl::hcu_target_enum HcuArch>
struct tilelang_mls_base
{
    using Traits  = tile_window_mls_param_traits<BlockSize,
                                                MlsTileSize,
                                                WarpMN,
                                                WarpK,
                                                sizeof(DataType) * 8,
                                                Alt,
                                                Trans,
                                                HcuArch>;
    using Detail  = typename Traits::Detail;
    using MlsAtom = typename Detail::MlsAtom;  // both generic and ck_tile Detail expose MlsAtom

    static constexpr auto BlockSizeMN = BlockSize::at(::tl::number<0>{});
    static constexpr auto BlockSizeK  = BlockSize::at(::tl::number<1>{});

    static constexpr auto WarpCluster          = Detail::WarpCluster;
    static constexpr auto EffectiveWarpCluster = Detail::EffectiveWarpCluster;
    static constexpr auto TileLoadWarpPerIssue = Detail::TileLoadWarpPerIssue;
    static constexpr auto TileLoadWGPerIssue   = Detail::TileLoadWGPerIssue;

    static constexpr auto TileLoadWarpPerIssueMN = TileLoadWarpPerIssue.at(::tl::number<0>{});
    static constexpr auto TileLoadWarpPerIssueK  = TileLoadWarpPerIssue.at(::tl::number<1>{});

    static constexpr auto TileLoadWGPerIssueMN = TileLoadWGPerIssue.at(::tl::number<0>{});
    static constexpr auto TileLoadWGPerIssueK  = TileLoadWGPerIssue.at(::tl::number<1>{});

    using SFC_WarpAccess = typename Detail::SFC_WarpAccess;

    static constexpr auto NumWarpAccess   = SFC_WarpAccess::get_num_of_access();
    static constexpr auto NumWarpAccessMN = SFC_WarpAccess::access_lengths.at(::tl::number<0>{});
    static constexpr auto NumWarpAccessK  = SFC_WarpAccess::access_lengths.at(::tl::number<1>{});

    TL_DEVICE static constexpr auto get_num_of_access() { return NumWarpAccess; }

    TL_DEVICE static constexpr auto get_tile_lds_desc() { return Detail::make_lds_desc(); }

    TL_DEVICE tilelang_mls_base(DataType* p_data,
                                     ::tl::index_t mls_stride,
                                     ::tl::index_t mn_length_raw,
                                     ::tl::index_t k_length_raw)
        : p_data_(p_data),
          mls_stride_(mls_stride),
          mn_length_raw_(mn_length_raw),
          k_length_raw_(k_length_raw),
          last_block_remain_k(k_length_raw % BlockSizeK)
    {
        init();
    }

    TL_DEVICE auto get_warp_cluster_idx()
    {
        constexpr auto warp_cluster_to_id_adaptor = ::tl::make_single_stage_tensor_adaptor(
            ::tl::make_tuple(::tl::make_merge_transform(WarpCluster)),
            ::tl::make_tuple(
                typename ::tl::arithmetic_sequence_gen<0, WarpCluster.size(), 1>::type{}),
            ::tl::make_tuple(::tl::sequence<0>{}));

        return warp_cluster_to_id_adaptor.calculate_bottom_index(
            ::tl::make_multi_index(::tl::get_warp_id()));
    }

    TL_DEVICE void init()
    {
        constexpr auto tile_lds_desc = get_tile_lds_desc();
        const auto warp_cluster_idx  = get_warp_cluster_idx();

        ::tl::static_for<0, NumWarpAccess, 1>{}([&](auto i) {
            constexpr auto access_idx = SFC_WarpAccess::get_index(i);

            const auto tile_warp_coord = ::tl::generate_tuple(
                [&](auto ii) {
                    const auto eff_idx = (EffectiveWarpCluster.at(ii) != WarpCluster.at(ii))
                                            ? (warp_cluster_idx[ii] % EffectiveWarpCluster.at(ii))
                                            : warp_cluster_idx[ii];
                    return eff_idx * TileLoadWarpPerIssue.at(ii) +
                           access_idx[ii] * TileLoadWGPerIssue.at(ii);
                },
                ::tl::number<2>{});

            mls_lds_offset_(i) = __builtin_amdgcn_readfirstlane(
                tile_lds_desc.calculate_offset(::tl::to_multi_index(tile_warp_coord)));
        });
    }

    TL_DEVICE void init(const ::tl::array<::tl::index_t, 2>& block_window_origin)
    {
        const auto warp_cluster_idx = get_warp_cluster_idx();
        const auto origin_mn        = block_window_origin.at(::tl::number<0>{});
        const auto origin_k         = block_window_origin.at(::tl::number<1>{});
        mls_k_origin_               = origin_k;

        ::tl::static_for<0, NumWarpAccessMN, 1>{}([&](auto i) {
            constexpr auto access_idx = ::tl::make_tuple(i, ::tl::number<0>{});

            const auto tile_warp_coord = ::tl::generate_tuple(
                [&](auto ii) {
                    const auto eff_idx = (EffectiveWarpCluster.at(ii) != WarpCluster.at(ii))
                                            ? (warp_cluster_idx[ii] % EffectiveWarpCluster.at(ii))
                                            : warp_cluster_idx[ii];
                    return eff_idx * TileLoadWarpPerIssue.at(ii) +
                           access_idx[ii] * TileLoadWGPerIssue.at(ii);
                },
                ::tl::number<2>{});

            const auto tile_mn = tile_warp_coord[::tl::number<0>{}];
            const auto tile_k  = tile_warp_coord[::tl::number<1>{}];
            mls_mn_offset_(i) = tile_mn;
            mls_k_offset_(i) = tile_k;

            const auto warp_coord_mn = origin_mn + tile_mn;
            const auto warp_coord_k  = origin_k + tile_k;

            // const ::tl::index_t mls_mn_filter =
            //     warp_coord_mn + TileLoadWarpPerIssueMN > mn_length_raw_
            //         ? __builtin_amdgcn_readfirstlane(::tl::min(
            //               TileLoadWarpPerIssueMN,
            //               warp_coord_mn + TileLoadWarpPerIssueMN - mn_length_raw_))
            //         : 0;

            const ::tl::index_t mls_mn_filter =
                ::tl::min(TileLoadWarpPerIssueMN,
                    ::tl::max(0,
                                 warp_coord_mn + TileLoadWarpPerIssueMN - mn_length_raw_));

            constexpr auto mfmt = ::tl::mls::detail::mfmt_traits<Alt>::value;

            DataType* ptr_offset = p_data_;
            if constexpr(Trans)
            {
                const ::tl::index_t offset_elems =
                    warp_coord_mn * mls_stride_ + warp_coord_k;
                ptr_offset += offset_elems;
                mls_res_(i) = ::tl::mls::make_mls_resource(
                    static_cast<const void*>(ptr_offset), mls_stride_, 0, mls_mn_filter, mfmt);
            }
            else
            {
                const ::tl::index_t offset_elems =
                    warp_coord_k * mls_stride_ + warp_coord_mn;
                ptr_offset += offset_elems;
                mls_res_(i) = ::tl::mls::make_mls_resource(
                    static_cast<const void*>(ptr_offset), mls_stride_, mls_mn_filter, 0, mfmt);
            }
            mls_base_addr_(i) = reinterpret_cast<uintptr_t>(ptr_offset);
        });

        ::tl::static_for<0, NumWarpAccessK, 1>{}([&](auto i) {
            constexpr auto access_idx = ::tl::make_tuple(::tl::number<0>{}, i);

            const auto tile_warp_coord = ::tl::generate_tuple(
                [&](auto ii) {
                    const auto eff_idx = (EffectiveWarpCluster.at(ii) != WarpCluster.at(ii))
                                            ? (warp_cluster_idx[ii] % EffectiveWarpCluster.at(ii))
                                            : warp_cluster_idx[ii];
                    return eff_idx * TileLoadWarpPerIssue.at(ii) +
                           access_idx[ii] * TileLoadWGPerIssue.at(ii);
                },
                ::tl::number<2>{});
            const auto tile_warp_coord_k = tile_warp_coord[::tl::number<1>{}];

            // mls_k_filter_(i) =
            //     tile_warp_coord_k + TileLoadWarpPerIssueK > BlockSizeK - pad_k_
            //         ? __builtin_amdgcn_readfirstlane(
            //               ::tl::min(TileLoadWarpPerIssueK,
            //                            tile_warp_coord_k + TileLoadWarpPerIssueK -
            //                                (BlockSizeK - pad_k_)))
            //         : 0;
            mls_k_filter_(i) =
                ::tl::min(TileLoadWarpPerIssueK,
                    ::tl::max(0,
                                tile_warp_coord_k + TileLoadWarpPerIssueK - last_block_remain_k));
        });
    }

    TL_DEVICE void set_window_origin(const ::tl::array<::tl::index_t, 2>& block_window_origin)
    {
        init(block_window_origin);
    }

    TL_DEVICE void set_k_filter(::tl::int32x4_t& mls_res, uint32_t filter)
    {
        if constexpr(Trans)
        {
            mls_res.w = (mls_res.w & ~static_cast<uint32_t>(0xff)) | filter;
        }
        else
        {
            mls_res.w = (mls_res.w & ~static_cast<uint32_t>(0xff00)) | (filter << 8);
        }
    }

    TL_DEVICE void set_mn_filter(::tl::int32x4_t& mls_res, uint32_t filter)
    {
        if constexpr(Trans)
        {
            mls_res.w = (mls_res.w & ~static_cast<uint32_t>(0xff00)) | (filter << 8);
        }
        else
        {
            mls_res.w = (mls_res.w & ~static_cast<uint32_t>(0xff)) | filter;
        }
    }

    TL_DEVICE uint32_t get_mn_filter(const ::tl::index_t block_mn_base,
                                          const ::tl::index_t access_idx_mn)
    {
        const auto warp_coord_mn = block_mn_base + mls_mn_offset_(access_idx_mn);
        return static_cast<uint32_t>(
            ::tl::min(TileLoadWarpPerIssueMN,
                ::tl::max(0,
                             warp_coord_mn + TileLoadWarpPerIssueMN - mn_length_raw_)));
    }

    template <typename T, bool bps = false, bool last_load = false>
    TL_DEVICE void async_mls_load_asm_impl_ct(TL_LDS_ADDR T* smem,
                                                  ::tl::bool_constant<bps> = {},
                                                  ::tl::bool_constant<last_load> = {})
    {
        ::tl::static_for<0, NumWarpAccess, 1>{}([&](auto i) {
            constexpr auto access_idx    = SFC_WarpAccess::get_index(i);
            constexpr auto access_idx_mn = access_idx[::tl::number<0>{}];
            constexpr auto access_idx_k  = access_idx[::tl::number<1>{}];

            // moffset is always on K dimension
            constexpr auto moffset =
                ::tl::number<access_idx_k * TileLoadWGPerIssue.at(::tl::number<1>{})>{};

            if constexpr(last_load)
            {
                set_k_filter(mls_res_(access_idx_mn),
                             static_cast<uint32_t>(mls_k_filter_[access_idx_k]));
            }

            if constexpr(HcuArch == ::tl::hcu_target_enum::gfx938)
            {
                MlsAtom::template load<moffset, true>(
                    reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
                    mls_res_[access_idx_mn],
                    moffset,
                    ::tl::bool_constant<true>{});
            }
            else if constexpr(HcuArch == ::tl::hcu_target_enum::gfx946)
            {
                MlsAtom::template load<moffset, true, bps>(
                    reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
                    mls_res_[access_idx_mn],
                    moffset,
                    ::tl::bool_constant<true>{},
                    ::tl::bool_constant<bps>{});
            }
        });
    }

    template <typename T, bool bps>
    TL_DEVICE void async_mls_load_asm_impl_rt(TL_LDS_ADDR T* smem,
                                                   ::tl::bool_constant<bps>,
                                                   bool last_load)
    {
        ::tl::static_for<0, NumWarpAccess, 1>{}([&](auto i) {
            constexpr auto access_idx    = SFC_WarpAccess::get_index(i);
            constexpr auto access_idx_mn = access_idx[::tl::number<0>{}];
            constexpr auto access_idx_k  = access_idx[::tl::number<1>{}];

            // moffset is always on K dimension
            constexpr auto moffset =
                ::tl::number<access_idx_k * TileLoadWGPerIssue.at(::tl::number<1>{})>{};

            if (last_load)
            {
                set_k_filter(mls_res_(access_idx_mn),
                             static_cast<uint32_t>(mls_k_filter_[access_idx_k]));
            }

            if constexpr(HcuArch == ::tl::hcu_target_enum::gfx938)
            {
                MlsAtom::template load<moffset, true>(
                    reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
                    mls_res_[access_idx_mn],
                    moffset,
                    ::tl::bool_constant<true>{});
            }
            else if constexpr(HcuArch == ::tl::hcu_target_enum::gfx946)
            {
                MlsAtom::template load<moffset, true, bps>(
                    reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
                    mls_res_[access_idx_mn],
                    moffset,
                    ::tl::bool_constant<true>{},
                    ::tl::bool_constant<bps>{});
            }
        });
    }

    template <typename T, bool bps, bool check_k_filter, bool check_mn_filter>
    TL_DEVICE void async_mls_load_asm_impl_rt_mn(TL_LDS_ADDR T* smem,
                                                      ::tl::bool_constant<bps>,
                                                      ::tl::index_t block_mn_base,
                                                      bool last_mn_load)
    {
        ::tl::static_for<0, NumWarpAccess, 1>{}([&](auto i) {
            constexpr auto access_idx    = SFC_WarpAccess::get_index(i);
            constexpr auto access_idx_mn = access_idx[::tl::number<0>{}];
            constexpr auto access_idx_k  = access_idx[::tl::number<1>{}];

            // moffset is always on K dimension
            constexpr auto moffset =
                ::tl::number<access_idx_k * TileLoadWGPerIssue.at(::tl::number<1>{})>{};

            if constexpr(check_k_filter)
            {
                if (mls_k_origin_ + BlockSizeK > k_length_raw_)
                {
                    set_k_filter(mls_res_(access_idx_mn),
                                 static_cast<uint32_t>(mls_k_filter_[access_idx_k]));
                }
            }
            if constexpr(check_mn_filter)
            {
                if (last_mn_load)
                {
                    set_mn_filter(mls_res_(access_idx_mn),
                                  get_mn_filter(block_mn_base, access_idx_mn));
                }
            }

            if constexpr(HcuArch == ::tl::hcu_target_enum::gfx938)
            {
                MlsAtom::template load<moffset, true>(
                    reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
                    mls_res_[access_idx_mn],
                    moffset,
                    ::tl::bool_constant<true>{});
            }
            else if constexpr(HcuArch == ::tl::hcu_target_enum::gfx946)
            {
                MlsAtom::template load<moffset, true, bps>(
                    reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
                    mls_res_[access_idx_mn],
                    moffset,
                    ::tl::bool_constant<true>{},
                    ::tl::bool_constant<bps>{});
            }
        });
    }

    template <typename T, bool check_last_load = true, bool last_load = false>
    TL_DEVICE void async_mls_load_asm(TL_LDS_ADDR T* smem,
                                          ::tl::index_t block_k_base)
    {
        if constexpr(check_last_load)
        {
            const bool last_load_rt = (block_k_base + BlockSizeK > k_length_raw_);
            async_mls_load_asm_impl_rt(smem, ::tl::bool_constant<false>{}, last_load_rt);
        }
        else
        {
            async_mls_load_asm_impl_ct<T, false, last_load>(smem);
        }
    }

    template <typename T,
              bool bps = false,
              bool check_k_filter = true,
              bool check_mn_filter = true>
    TL_DEVICE void async_mls_load_asm_mn(TL_LDS_ADDR T* smem,
                                             ::tl::index_t block_mn_base)
    {
        bool last_mn_load_rt = false;
        if constexpr(check_mn_filter)
        {
            last_mn_load_rt = (block_mn_base + BlockSizeMN > mn_length_raw_);
        }
        async_mls_load_asm_impl_rt_mn<T, bps, check_k_filter, check_mn_filter>(
            smem,
            ::tl::bool_constant<bps>{},
            block_mn_base,
            last_mn_load_rt);
    }

    TL_DEVICE void move_base(const ::tl::index_t block_k_base)
    {
        ::tl::index_t addr_byte_offset;
        if constexpr(Trans)
        {
            addr_byte_offset = block_k_base * ::tl::number<sizeof(DataType)>{};
        }
        else
        {
            addr_byte_offset =
                block_k_base * mls_stride_ * ::tl::number<sizeof(DataType)>{};
        }
        ::tl::static_for<0, NumWarpAccessMN, 1>{}(
            [&](auto i) { ::tl::mls::move_mls_addr_base(mls_res_(i), addr_byte_offset); });
    }

    TL_DEVICE void update_base(const ::tl::index_t block_k_base)
    {
        ::tl::index_t addr_byte_offset;
        if constexpr(Trans)
        {
            addr_byte_offset = block_k_base * ::tl::number<sizeof(DataType)>{};
        }
        else
        {
            addr_byte_offset =
                block_k_base * mls_stride_ * ::tl::number<sizeof(DataType)>{};
        }
        ::tl::static_for<0, NumWarpAccessMN, 1>{}(
            [&](auto i) { update_mls_addr_base(mls_res_(i), mls_base_addr_(i), addr_byte_offset); });
    }

    TL_DEVICE void update_mn_base(const ::tl::index_t block_mn_base)
    {
        ::tl::static_for<0, NumWarpAccessMN, 1>{}([&](auto i) {
            const auto warp_coord_mn = block_mn_base + mls_mn_offset_(i);

            DataType* ptr_offset = p_data_;
            if constexpr(Trans)
            {
                ptr_offset += warp_coord_mn * mls_stride_ + mls_k_origin_ + mls_k_offset_(i);
            }
            else
            {
                ptr_offset += (mls_k_origin_ + mls_k_offset_(i)) * mls_stride_ + warp_coord_mn;
            }
            mls_base_addr_(i) = reinterpret_cast<uintptr_t>(ptr_offset);
            update_mls_addr_base(mls_res_(i), mls_base_addr_(i), 0);
        });
    }

    DataType* p_data_;
    ::tl::array<::tl::int32x4_t, NumWarpAccessMN> mls_res_;
    ::tl::array<uintptr_t, NumWarpAccessMN> mls_base_addr_;
    ::tl::array<::tl::index_t, NumWarpAccessMN> mls_mn_offset_;
    ::tl::array<::tl::index_t, NumWarpAccessMN> mls_k_offset_;
    ::tl::array<uint8_t, NumWarpAccessK> mls_k_filter_;

    ::tl::index_t mls_stride_;
    ::tl::index_t mn_length_raw_;
    ::tl::index_t k_length_raw_;
    ::tl::index_t mls_k_origin_{0};
    ::tl::index_t last_block_remain_k;

    ::tl::array<::tl::index_t, NumWarpAccess> mls_lds_offset_;
};

/*
 * mls_load_tile: one-shot MLS load for block K loop.
 * Flow: construct(k_length_raw) -> set_window_origin(block_mn_base, 0) -> move_base_to(block_k_base)
 *       -> async_mls_load_asm(smem, block_k_base).
 * check_last_load=true (default): compute last_load at runtime, may cause branch.
 * check_last_load=false: use last_load template param (compile-time), no branch.
 */
template <typename BlockSize,
          typename MlsTileSize,
          ::tl::index_t WarpMN,
          ::tl::index_t WarpK,
          typename DataType,
          ::tl::index_t Alt,
          bool Trans,
          ::tl::hcu_target_enum HcuArch,
          bool check_last_load = true,
          bool last_load       = false>
TL_DEVICE void mls_load_tile(DataType* p_data,
                                  ::tl::index_t mls_stride,
                                  ::tl::index_t mn_length_raw,
                                  ::tl::index_t k_length_raw,
                                  ::tl::index_t block_mn_base,
                                  ::tl::index_t block_k_base,
                                  TL_LDS_ADDR DataType* smem)
{
    using MlsBase = tilelang_mls_base<BlockSize,
                                      MlsTileSize,
                                      WarpMN,
                                      WarpK,
                                      DataType,
                                      Alt,
                                      Trans,
                                      HcuArch>;
    MlsBase mls(p_data, mls_stride, mn_length_raw, k_length_raw);
    // mls.set_window_origin(::tl::make_array<::tl::index_t>(block_mn_base, ::tl::number<0>{}));
    // mls.update_base(block_k_base);
    mls.set_window_origin(::tl::make_array<::tl::index_t>(block_mn_base, block_k_base));
    mls.template async_mls_load_asm<DataType, check_last_load, last_load>(smem, block_k_base);
}

} // namespace mls
} // namespace tl
