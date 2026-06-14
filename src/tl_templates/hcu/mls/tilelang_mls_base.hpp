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
 * Uses ck_tile::get_warp_id() internally (standard block layout).
 */

#include <ck_tile/core.hpp>
#include <ck_tile/core/arch/hcu_matrix_addressing.hpp>

#include "tile_window_mls.h"

namespace tl {
namespace mls {

/*
 * update_mls_addr_base: set mls_res addr to base_addr + addr_byte_offset (absolute, not add).
 * Used by move_base_to for LICM-friendly load pattern.
 */
CK_TILE_DEVICE void update_mls_addr_base(ck_tile::int32x4_t& mls_res,
                                         uintptr_t base_addr,
                                         ck_tile::index_t addr_byte_offset)
{
    ck_tile::mls_addr_union addr_union;
    addr_union.addr = base_addr + addr_byte_offset;
    mls_res.x       = addr_union.addr_lo;
    mls_res.y       = addr_union.addr_hi;
}

template <typename BlockSize,
          typename MlsTileSize,
          ck_tile::index_t WarpMN,
          ck_tile::index_t WarpK,
          typename DataType,
          ck_tile::index_t Alt,
          bool Trans,
          ck_tile::hcu_target_enum HcuArch>
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

    static constexpr auto BlockSizeMN = BlockSize::at(ck_tile::number<0>{});
    static constexpr auto BlockSizeK  = BlockSize::at(ck_tile::number<1>{});

    static constexpr auto WarpCluster          = Detail::WarpCluster;
    static constexpr auto EffectiveWarpCluster = Detail::EffectiveWarpCluster;
    static constexpr auto TileLoadWarpPerIssue = Detail::TileLoadWarpPerIssue;
    static constexpr auto TileLoadWGPerIssue   = Detail::TileLoadWGPerIssue;

    static constexpr auto TileLoadWarpPerIssueMN = TileLoadWarpPerIssue.at(ck_tile::number<0>{});
    static constexpr auto TileLoadWarpPerIssueK  = TileLoadWarpPerIssue.at(ck_tile::number<1>{});

    static constexpr auto TileLoadWGPerIssueMN = TileLoadWGPerIssue.at(ck_tile::number<0>{});
    static constexpr auto TileLoadWGPerIssueK  = TileLoadWGPerIssue.at(ck_tile::number<1>{});

    using SFC_WarpAccess = typename Detail::SFC_WarpAccess;

    static constexpr auto NumWarpAccess   = SFC_WarpAccess::get_num_of_access();
    static constexpr auto NumWarpAccessMN = SFC_WarpAccess::access_lengths.at(ck_tile::number<0>{});
    static constexpr auto NumWarpAccessK  = SFC_WarpAccess::access_lengths.at(ck_tile::number<1>{});

    CK_TILE_DEVICE static constexpr auto get_num_of_access() { return NumWarpAccess; }

    CK_TILE_DEVICE static constexpr auto get_tile_lds_desc() { return Detail::make_lds_desc(); }

    CK_TILE_DEVICE tilelang_mls_base(DataType* p_data,
                                     ck_tile::index_t mls_stride,
                                     ck_tile::index_t mn_length_raw,
                                     ck_tile::index_t k_length_raw)
        : p_data_(p_data),
          mls_stride_(mls_stride),
          mn_length_raw_(mn_length_raw),
          k_length_raw_(k_length_raw),
          last_block_remain_k(k_length_raw % BlockSizeK)
    {
        init();
    }

    CK_TILE_DEVICE auto get_warp_cluster_idx()
    {
        constexpr auto warp_cluster_to_id_adaptor = ck_tile::make_single_stage_tensor_adaptor(
            ck_tile::make_tuple(ck_tile::make_merge_transform(WarpCluster)),
            ck_tile::make_tuple(
                typename ck_tile::arithmetic_sequence_gen<0, WarpCluster.size(), 1>::type{}),
            ck_tile::make_tuple(ck_tile::sequence<0>{}));

        return warp_cluster_to_id_adaptor.calculate_bottom_index(
            ck_tile::make_multi_index(ck_tile::get_warp_id()));
    }

    CK_TILE_DEVICE void init()
    {
        constexpr auto tile_lds_desc = get_tile_lds_desc();
        const auto warp_cluster_idx  = get_warp_cluster_idx();

        ck_tile::static_for<0, NumWarpAccess, 1>{}([&](auto i) {
            constexpr auto access_idx = SFC_WarpAccess::get_index(i);

            const auto tile_warp_coord = ck_tile::generate_tuple(
                [&](auto ii) {
                    const auto eff_idx = (EffectiveWarpCluster.at(ii) != WarpCluster.at(ii))
                                            ? (warp_cluster_idx[ii] % EffectiveWarpCluster.at(ii))
                                            : warp_cluster_idx[ii];
                    return eff_idx * TileLoadWarpPerIssue.at(ii) +
                           access_idx[ii] * TileLoadWGPerIssue.at(ii);
                },
                ck_tile::number<2>{});

            mls_lds_offset_(i) = __builtin_amdgcn_readfirstlane(
                tile_lds_desc.calculate_offset(ck_tile::to_multi_index(tile_warp_coord)));
        });
    }

    CK_TILE_DEVICE void init(const ck_tile::array<ck_tile::index_t, 2>& block_window_origin)
    {
        const auto warp_cluster_idx = get_warp_cluster_idx();
        const auto origin_mn        = block_window_origin.at(ck_tile::number<0>{});
        const auto origin_k         = block_window_origin.at(ck_tile::number<1>{});
        mls_k_origin_               = origin_k;

        ck_tile::static_for<0, NumWarpAccessMN, 1>{}([&](auto i) {
            constexpr auto access_idx = ck_tile::make_tuple(i, ck_tile::number<0>{});

            const auto tile_warp_coord = ck_tile::generate_tuple(
                [&](auto ii) {
                    const auto eff_idx = (EffectiveWarpCluster.at(ii) != WarpCluster.at(ii))
                                            ? (warp_cluster_idx[ii] % EffectiveWarpCluster.at(ii))
                                            : warp_cluster_idx[ii];
                    return eff_idx * TileLoadWarpPerIssue.at(ii) +
                           access_idx[ii] * TileLoadWGPerIssue.at(ii);
                },
                ck_tile::number<2>{});

            const auto tile_mn = tile_warp_coord[ck_tile::number<0>{}];
            const auto tile_k  = tile_warp_coord[ck_tile::number<1>{}];
            mls_mn_offset_(i) = tile_mn;
            mls_k_offset_(i) = tile_k;

            const auto warp_coord_mn = origin_mn + tile_mn;
            const auto warp_coord_k  = origin_k + tile_k;

            // const ck_tile::index_t mls_mn_filter =
            //     warp_coord_mn + TileLoadWarpPerIssueMN > mn_length_raw_
            //         ? __builtin_amdgcn_readfirstlane(ck_tile::min(
            //               TileLoadWarpPerIssueMN,
            //               warp_coord_mn + TileLoadWarpPerIssueMN - mn_length_raw_))
            //         : 0;

            const ck_tile::index_t mls_mn_filter =
                ck_tile::min(TileLoadWarpPerIssueMN,
                    ck_tile::max(0,
                                 warp_coord_mn + TileLoadWarpPerIssueMN - mn_length_raw_));

            constexpr auto mfmt = ck_tile::detail::mfmt_traits<Alt>::value;

            DataType* ptr_offset = p_data_;
            if constexpr(Trans)
            {
                const ck_tile::index_t offset_elems =
                    warp_coord_mn * mls_stride_ + warp_coord_k;
                ptr_offset += offset_elems;
                mls_res_(i) = ck_tile::make_mls_resource(
                    static_cast<const void*>(ptr_offset), mls_stride_, 0, mls_mn_filter, mfmt);
            }
            else
            {
                const ck_tile::index_t offset_elems =
                    warp_coord_k * mls_stride_ + warp_coord_mn;
                ptr_offset += offset_elems;
                mls_res_(i) = ck_tile::make_mls_resource(
                    static_cast<const void*>(ptr_offset), mls_stride_, mls_mn_filter, 0, mfmt);
            }
            mls_base_addr_(i) = reinterpret_cast<uintptr_t>(ptr_offset);
        });

        ck_tile::static_for<0, NumWarpAccessK, 1>{}([&](auto i) {
            constexpr auto access_idx = ck_tile::make_tuple(ck_tile::number<0>{}, i);

            const auto tile_warp_coord = ck_tile::generate_tuple(
                [&](auto ii) {
                    const auto eff_idx = (EffectiveWarpCluster.at(ii) != WarpCluster.at(ii))
                                            ? (warp_cluster_idx[ii] % EffectiveWarpCluster.at(ii))
                                            : warp_cluster_idx[ii];
                    return eff_idx * TileLoadWarpPerIssue.at(ii) +
                           access_idx[ii] * TileLoadWGPerIssue.at(ii);
                },
                ck_tile::number<2>{});
            const auto tile_warp_coord_k = tile_warp_coord[ck_tile::number<1>{}];

            // mls_k_filter_(i) =
            //     tile_warp_coord_k + TileLoadWarpPerIssueK > BlockSizeK - pad_k_
            //         ? __builtin_amdgcn_readfirstlane(
            //               ck_tile::min(TileLoadWarpPerIssueK,
            //                            tile_warp_coord_k + TileLoadWarpPerIssueK -
            //                                (BlockSizeK - pad_k_)))
            //         : 0;
            mls_k_filter_(i) =
                ck_tile::min(TileLoadWarpPerIssueK,
                    ck_tile::max(0,
                                tile_warp_coord_k + TileLoadWarpPerIssueK - last_block_remain_k));
        });
    }

    CK_TILE_DEVICE void set_window_origin(const ck_tile::array<ck_tile::index_t, 2>& block_window_origin)
    {
        init(block_window_origin);
    }

    CK_TILE_DEVICE void set_k_filter(ck_tile::int32x4_t& mls_res, uint32_t filter)
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

    CK_TILE_DEVICE void set_mn_filter(ck_tile::int32x4_t& mls_res, uint32_t filter)
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

    CK_TILE_DEVICE uint32_t get_mn_filter(const ck_tile::index_t block_mn_base,
                                          const ck_tile::index_t access_idx_mn)
    {
        const auto warp_coord_mn = block_mn_base + mls_mn_offset_(access_idx_mn);
        return static_cast<uint32_t>(
            ck_tile::min(TileLoadWarpPerIssueMN,
                ck_tile::max(0,
                             warp_coord_mn + TileLoadWarpPerIssueMN - mn_length_raw_)));
    }

    template <typename T, bool bps = false, bool last_load = false>
    CK_TILE_DEVICE void async_mls_load_asm_impl_ct(CK_TILE_LDS_ADDR T* smem,
                                                  ck_tile::bool_constant<bps> = {},
                                                  ck_tile::bool_constant<last_load> = {})
    {
        ck_tile::static_for<0, NumWarpAccess, 1>{}([&](auto i) {
            constexpr auto access_idx    = SFC_WarpAccess::get_index(i);
            constexpr auto access_idx_mn = access_idx[ck_tile::number<0>{}];
            constexpr auto access_idx_k  = access_idx[ck_tile::number<1>{}];

            // moffset is always on K dimension
            constexpr auto moffset =
                ck_tile::number<access_idx_k * TileLoadWGPerIssue.at(ck_tile::number<1>{})>{};

            if constexpr(last_load)
            {
                set_k_filter(mls_res_(access_idx_mn),
                             static_cast<uint32_t>(mls_k_filter_[access_idx_k]));
            }

            if constexpr(HcuArch == ck_tile::hcu_target_enum::gfx938)
            {
                MlsAtom::template load<moffset, true>(
                    reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
                    mls_res_[access_idx_mn],
                    moffset,
                    ck_tile::bool_constant<true>{});
            }
            else if constexpr(HcuArch == ck_tile::hcu_target_enum::gfx946)
            {
                MlsAtom::template load<moffset, true, bps>(
                    reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
                    mls_res_[access_idx_mn],
                    moffset,
                    ck_tile::bool_constant<true>{},
                    ck_tile::bool_constant<bps>{});
            }
        });
    }

    template <typename T, bool bps>
    CK_TILE_DEVICE void async_mls_load_asm_impl_rt(CK_TILE_LDS_ADDR T* smem,
                                                   ck_tile::bool_constant<bps>,
                                                   bool last_load)
    {
        ck_tile::static_for<0, NumWarpAccess, 1>{}([&](auto i) {
            constexpr auto access_idx    = SFC_WarpAccess::get_index(i);
            constexpr auto access_idx_mn = access_idx[ck_tile::number<0>{}];
            constexpr auto access_idx_k  = access_idx[ck_tile::number<1>{}];

            // moffset is always on K dimension
            constexpr auto moffset =
                ck_tile::number<access_idx_k * TileLoadWGPerIssue.at(ck_tile::number<1>{})>{};

            if (last_load)
            {
                set_k_filter(mls_res_(access_idx_mn),
                             static_cast<uint32_t>(mls_k_filter_[access_idx_k]));
            }

            if constexpr(HcuArch == ck_tile::hcu_target_enum::gfx938)
            {
                MlsAtom::template load<moffset, true>(
                    reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
                    mls_res_[access_idx_mn],
                    moffset,
                    ck_tile::bool_constant<true>{});
            }
            else if constexpr(HcuArch == ck_tile::hcu_target_enum::gfx946)
            {
                MlsAtom::template load<moffset, true, bps>(
                    reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
                    mls_res_[access_idx_mn],
                    moffset,
                    ck_tile::bool_constant<true>{},
                    ck_tile::bool_constant<bps>{});
            }
        });
    }

    template <typename T, bool bps, bool check_k_filter, bool check_mn_filter>
    CK_TILE_DEVICE void async_mls_load_asm_impl_rt_mn(CK_TILE_LDS_ADDR T* smem,
                                                      ck_tile::bool_constant<bps>,
                                                      ck_tile::index_t block_mn_base,
                                                      bool last_mn_load)
    {
        ck_tile::static_for<0, NumWarpAccess, 1>{}([&](auto i) {
            constexpr auto access_idx    = SFC_WarpAccess::get_index(i);
            constexpr auto access_idx_mn = access_idx[ck_tile::number<0>{}];
            constexpr auto access_idx_k  = access_idx[ck_tile::number<1>{}];

            // moffset is always on K dimension
            constexpr auto moffset =
                ck_tile::number<access_idx_k * TileLoadWGPerIssue.at(ck_tile::number<1>{})>{};

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

            if constexpr(HcuArch == ck_tile::hcu_target_enum::gfx938)
            {
                MlsAtom::template load<moffset, true>(
                    reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
                    mls_res_[access_idx_mn],
                    moffset,
                    ck_tile::bool_constant<true>{});
            }
            else if constexpr(HcuArch == ck_tile::hcu_target_enum::gfx946)
            {
                MlsAtom::template load<moffset, true, bps>(
                    reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
                    mls_res_[access_idx_mn],
                    moffset,
                    ck_tile::bool_constant<true>{},
                    ck_tile::bool_constant<bps>{});
            }
        });
    }

    template <typename T, bool check_last_load = true, bool last_load = false>
    CK_TILE_DEVICE void async_mls_load_asm(CK_TILE_LDS_ADDR T* smem,
                                          ck_tile::index_t block_k_base)
    {
        if constexpr(check_last_load)
        {
            const bool last_load_rt = (block_k_base + BlockSizeK > k_length_raw_);
            async_mls_load_asm_impl_rt(smem, ck_tile::bool_constant<false>{}, last_load_rt);
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
    CK_TILE_DEVICE void async_mls_load_asm_mn(CK_TILE_LDS_ADDR T* smem,
                                             ck_tile::index_t block_mn_base)
    {
        bool last_mn_load_rt = false;
        if constexpr(check_mn_filter)
        {
            last_mn_load_rt = (block_mn_base + BlockSizeMN > mn_length_raw_);
        }
        async_mls_load_asm_impl_rt_mn<T, bps, check_k_filter, check_mn_filter>(
            smem,
            ck_tile::bool_constant<bps>{},
            block_mn_base,
            last_mn_load_rt);
    }

    CK_TILE_DEVICE void move_base(const ck_tile::index_t block_k_base)
    {
        ck_tile::index_t addr_byte_offset;
        if constexpr(Trans)
        {
            addr_byte_offset = block_k_base * ck_tile::number<sizeof(DataType)>{};
        }
        else
        {
            addr_byte_offset =
                block_k_base * mls_stride_ * ck_tile::number<sizeof(DataType)>{};
        }
        ck_tile::static_for<0, NumWarpAccessMN, 1>{}(
            [&](auto i) { ck_tile::move_mls_addr_base(mls_res_(i), addr_byte_offset); });
    }

    CK_TILE_DEVICE void update_base(const ck_tile::index_t block_k_base)
    {
        ck_tile::index_t addr_byte_offset;
        if constexpr(Trans)
        {
            addr_byte_offset = block_k_base * ck_tile::number<sizeof(DataType)>{};
        }
        else
        {
            addr_byte_offset =
                block_k_base * mls_stride_ * ck_tile::number<sizeof(DataType)>{};
        }
        ck_tile::static_for<0, NumWarpAccessMN, 1>{}(
            [&](auto i) { update_mls_addr_base(mls_res_(i), mls_base_addr_(i), addr_byte_offset); });
    }

    CK_TILE_DEVICE void update_mn_base(const ck_tile::index_t block_mn_base)
    {
        ck_tile::static_for<0, NumWarpAccessMN, 1>{}([&](auto i) {
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
    ck_tile::array<ck_tile::int32x4_t, NumWarpAccessMN> mls_res_;
    ck_tile::array<uintptr_t, NumWarpAccessMN> mls_base_addr_;
    ck_tile::array<ck_tile::index_t, NumWarpAccessMN> mls_mn_offset_;
    ck_tile::array<ck_tile::index_t, NumWarpAccessMN> mls_k_offset_;
    ck_tile::array<uint8_t, NumWarpAccessK> mls_k_filter_;

    ck_tile::index_t mls_stride_;
    ck_tile::index_t mn_length_raw_;
    ck_tile::index_t k_length_raw_;
    ck_tile::index_t mls_k_origin_{0};
    ck_tile::index_t last_block_remain_k;

    ck_tile::array<ck_tile::index_t, NumWarpAccess> mls_lds_offset_;
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
          ck_tile::index_t WarpMN,
          ck_tile::index_t WarpK,
          typename DataType,
          ck_tile::index_t Alt,
          bool Trans,
          ck_tile::hcu_target_enum HcuArch,
          bool check_last_load = true,
          bool last_load       = false>
CK_TILE_DEVICE void mls_load_tile(DataType* p_data,
                                  ck_tile::index_t mls_stride,
                                  ck_tile::index_t mn_length_raw,
                                  ck_tile::index_t k_length_raw,
                                  ck_tile::index_t block_mn_base,
                                  ck_tile::index_t block_k_base,
                                  CK_TILE_LDS_ADDR DataType* smem)
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
    // mls.set_window_origin(ck_tile::make_array<ck_tile::index_t>(block_mn_base, ck_tile::number<0>{}));
    // mls.update_base(block_k_base);
    mls.set_window_origin(ck_tile::make_array<ck_tile::index_t>(block_mn_base, block_k_base));
    mls.template async_mls_load_asm<DataType, check_last_load, last_load>(smem, block_k_base);
}

} // namespace mls
} // namespace tl
