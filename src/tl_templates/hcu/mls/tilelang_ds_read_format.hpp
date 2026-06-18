#pragma once

#include <algorithm>
#include <tl_templates/hcu/core.hpp>

#include <tl_templates/hcu/mls/mls_ds_traits.hpp>
#include <tl_templates/hcu/mls/mls_param_traits.hpp>

namespace tl {
namespace mls {

/*
 * ds_read_format_traits: traits for reading MLS LDS with warp-chunk layout.
 * Block is divided into WarpMN x WarpK chunks; each warp loads its chunk.
 * Within chunk: traverse MN first, then K (MNIterPerWarp, KIterPerWarp).
 *
 * Template params: BlockSize, MlsTileSize (sequence<MN,K>), WarpMN, WarpK,
 *                  DataType, Alt, Trans, HcuArch.
 */
template <typename BlockSize, typename MlsTileSize, ::tl::index_t WarpMN,
          ::tl::index_t WarpK, typename DataType, ::tl::index_t Alt, bool Trans,
          ::tl::hcu_target_enum HcuArch>
struct ds_read_format_traits {
  static constexpr ::tl::index_t BlockSizeMN = BlockSize::at(::tl::number<0>{});
  static constexpr ::tl::index_t BlockSizeK = BlockSize::at(::tl::number<1>{});

  static constexpr ::tl::index_t Bits = sizeof(DataType) * 8;

  using LdsTraits = mls_lds_desc_param_traits<BlockSize, MlsTileSize, Bits, Alt,
                                              Trans, HcuArch>;
  using MlsAtom = typename LdsTraits::MlsAtom;
  using DsFormatInst =
      typename mls_ds_traits<MlsAtom, sizeof(DataType), Alt>::Type;

  static constexpr auto LdsDesc = LdsTraits::get_tile_lds_desc();

  // Per-warp chunk size
  static constexpr ::tl::index_t PerWarpMN = BlockSizeMN / WarpMN;
  static constexpr ::tl::index_t PerWarpK = BlockSizeK / WarpK;

  // Iterations per warp (tiles of DsFormatInst::kMN x DsFormatInst::kK)
  static constexpr ::tl::index_t MNIterPerWarp = PerWarpMN / DsFormatInst::kMN;
  static constexpr ::tl::index_t KIterPerWarp = PerWarpK / DsFormatInst::kK;

  static_assert(PerWarpMN % DsFormatInst::kMN == 0,
                "PerWarpMN must be divisible by DsFormatInst::kMN");
  static_assert(PerWarpK % DsFormatInst::kK == 0,
                "PerWarpK must be divisible by DsFormatInst::kK");

  using SFC =
      ::tl::space_filling_curve<::tl::sequence<MNIterPerWarp, KIterPerWarp>,
                                ::tl::sequence<1, 0>, ::tl::sequence<1, 1>,
                                false>;

  static constexpr ::tl::index_t NumAccess = SFC::get_num_of_access();

  // Gemm layout: (ki, Mi) -> offset (ki * warp_rows + Mi) * vec_size
  // Matches gemm.h body_rr: a_ptr = A_local + (ki * warp_rows + Mi) * vec_size
  static constexpr ::tl::index_t MmacMNSize = 16;
  static constexpr ::tl::index_t MmacKSize =
      32 / sizeof(DataType); // for b16/fp16
  static constexpr ::tl::index_t GemmWarpRows = PerWarpMN / MmacMNSize;
  static constexpr ::tl::index_t GemmInnerK = PerWarpK / MmacKSize;
  static constexpr ::tl::index_t VecSize = 8 / sizeof(DataType);
  static constexpr ::tl::index_t GemmTensorSize =
      GemmInnerK * GemmWarpRows * VecSize;
};

/*
 * ds_read_format_tensor: read MLS LDS into target buffer with gemm body_rr
 * layout. Indexing: target[(ki * warp_mmac_rows + Mi) * vec_size].
 *
 * warp_mn_idx, warp_k_idx: indices of current warp in the block
 * (caller-provided). target: output buffer, must have at least GemmTensorSize
 * elements (caller-allocated).
 *
 * Each DsFormatInst read yields kMN x kK; we treat it as (kMN/mmacMN) x
 * (kK/mmacK) mmac blocks and scatter them to target positions.
 */
template <typename BlockSize, typename MlsTileSize, ::tl::index_t WarpMN,
          ::tl::index_t WarpK, typename DataType, ::tl::index_t Alt, bool Trans,
          ::tl::hcu_target_enum HcuArch>
TL_DEVICE void
ds_read_format_tensor(TL_LDS_ADDR DataType *smem_ptr, DataType *target,
                      ::tl::index_t warp_mn_idx, ::tl::index_t warp_k_idx) {
  using Traits = ds_read_format_traits<BlockSize, MlsTileSize, WarpMN, WarpK,
                                       DataType, Alt, Trans, HcuArch>;

  using DsFormatInst = typename Traits::DsFormatInst;
  using vector_t = ::tl::ext_vector_t<DataType, DsFormatInst::kVectorLength>;

  constexpr ::tl::index_t NumMMAC_MN = DsFormatInst::kMN / Traits::MmacMNSize;
  constexpr ::tl::index_t NumMMAC_K = DsFormatInst::kK / Traits::MmacKSize;
  constexpr ::tl::index_t NumMMACPerRead = NumMMAC_MN * NumMMAC_K;

  static_assert(DsFormatInst::kVectorLength == NumMMACPerRead * Traits::VecSize,
                "DsFormatInst vector length must match mmac blocks * vec_size");

  const ::tl::index_t warp_lds_elem_offset =
      Traits::LdsDesc.calculate_offset(::tl::make_multi_index(
          warp_mn_idx * Traits::PerWarpMN, warp_k_idx * Traits::PerWarpK));

  ::tl::static_for<0, Traits::NumAccess, 1>{}([&](auto i) {
    constexpr auto idx = Traits::SFC::get_index(i);

    constexpr auto mn_iter = idx.at(::tl::number<0>{});
    constexpr auto k_iter = idx.at(::tl::number<1>{});

    constexpr auto immed_offset =
        Traits::LdsDesc.calculate_offset(::tl::make_multi_index(
            mn_iter * DsFormatInst::kMN, k_iter * DsFormatInst::kK)) *
        sizeof(DataType);

    auto ret = DsFormatInst{}(smem_ptr + warp_lds_elem_offset,
                              ::tl::number<immed_offset>{});
    vector_t vec_value = ret.template get_as<vector_t>()[::tl::number<0>{}];

    // Scatter each mmac block to target.
    // DsFormatInst vector order: row dim first. Trans=false -> row=MN;
    // Trans=true -> row=K.
    ::tl::static_for<0, NumMMACPerRead, 1>{}([&](auto block_idx) {
      constexpr auto mmac_k = [](auto idx) {
        if constexpr (Trans)
          return idx % NumMMAC_K;
        else
          return idx / NumMMAC_MN;
      }(block_idx);
      constexpr auto mmac_mn = [](auto idx) {
        if constexpr (Trans)
          return idx / NumMMAC_K;
        else
          return idx % NumMMAC_MN;
      }(block_idx);
      constexpr auto target_ki = k_iter * NumMMAC_K + mmac_k;
      constexpr auto target_mni = mn_iter * NumMMAC_MN + mmac_mn;
      constexpr auto target_offset =
          (target_ki * Traits::GemmWarpRows + target_mni) * Traits::VecSize;
      constexpr auto vec_offset = block_idx * Traits::VecSize;

#if defined(__HIP__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
      for (::tl::index_t j = 0; j < Traits::VecSize; j++) {
        target[target_offset + j] = vec_value[vec_offset + j];
      }
    });
  });
}

/*
 * ds_read_format_tensor_a: load A (BlockSize = M x K) in gemm layout.
 * Warp id (gemm.h): warp_m = warp_id % WarpM.
 * WarpK must be 1 (warp_k_idx = 0).
 * MlsTileA: ::tl::sequence<MlsTileM, MlsTileKA> - MLS tile for A.
 * target: output buffer, must have at least GemmTensorSize elements
 * (caller-allocated).
 */
template <typename BlockSize, typename MlsTileA, ::tl::index_t WarpM,
          ::tl::index_t WarpK, typename DataType, ::tl::index_t Alt, bool Trans,
          ::tl::hcu_target_enum HcuArch>
TL_DEVICE void ds_read_format_tensor_a(TL_LDS_ADDR DataType *smem_ptr,
                                       DataType *target) {
  static_assert(WarpK == 1, "WarpK must be 1");
  const ::tl::index_t warp_id = ::tl::get_warp_id();
  const ::tl::index_t warp_m_idx = warp_id % WarpM;
  const ::tl::index_t warp_k_idx = 0;
  ds_read_format_tensor<BlockSize, MlsTileA, WarpM, WarpK, DataType, Alt, Trans,
                        HcuArch>(smem_ptr, target, warp_m_idx, warp_k_idx);
}

/*
 * ds_read_format_tensor_common: for output not directly fed to gemm.
 * warp_mn_idx = warp_id % WarpMN_no_recompute;
 * warp_k_idx  = warp_id / WarpMN_no_recompute;
 * When WarpMN_no_recompute != WarpMN: warp_k_idx %= WarpK.
 * WarpMN_no_recompute = min(WarpMN, BlockSizeMN / DsFormatInst::kMN).
 */
template <typename BlockSize, typename MlsTileSize, ::tl::index_t WarpMN,
          ::tl::index_t WarpK, typename DataType, ::tl::index_t Alt, bool Trans,
          ::tl::hcu_target_enum HcuArch>
TL_DEVICE void ds_read_format_tensor_common(TL_LDS_ADDR DataType *smem_ptr,
                                            DataType *target) {
  static constexpr ::tl::index_t BlockSizeMN = BlockSize::at(::tl::number<0>{});
  static constexpr ::tl::index_t Bits = sizeof(DataType) * 8;

  using LdsTraits = mls_lds_desc_param_traits<BlockSize, MlsTileSize, Bits, Alt,
                                              Trans, HcuArch>;
  using MlsAtom = typename LdsTraits::MlsAtom;
  using DsFormatInst =
      typename mls_ds_traits<MlsAtom, sizeof(DataType), Alt>::Type;

  static constexpr ::tl::index_t WarpMN_no_recompute =
      std::min(WarpMN, BlockSizeMN / DsFormatInst::kMN);

  const ::tl::index_t warp_id = ::tl::get_warp_id();
  ::tl::index_t warp_mn_idx = warp_id % WarpMN_no_recompute;
  ::tl::index_t warp_k_idx = warp_id / WarpMN_no_recompute;
  if constexpr (WarpMN_no_recompute != WarpMN) {
    warp_k_idx = warp_k_idx % WarpK;
  }

  ds_read_format_tensor<BlockSize, MlsTileSize, WarpMN_no_recompute, WarpK,
                        DataType, Alt, Trans, HcuArch>(smem_ptr, target,
                                                       warp_mn_idx, warp_k_idx);
}

/*
 * ds_read_format_tensor_b: load B (BlockSize = N x K) in gemm layout.
 * TotalWarp = WarpM * WarpN * WarpK; WarpM = TotalWarp / (WarpN * WarpK).
 * Warp id (gemm.h): warp_n = warp_id / WarpM.
 * WarpN_no_recompute = min(WarpN, BlockN / MinNPerWarp).
 * MinNPerWarp: Trans=true -> 16; Trans=false (B_from_mls && !B_mls_trans)
 * -> 32. When WarpN_no_recompute != WarpN: pass WarpN_no_recompute as WarpMN.
 * WarpK must be 1 (warp_k_idx = 0).
 * MlsTileB: ::tl::sequence<MlsTileN, MlsTileKB> - MLS tile for B.
 * target: output buffer, must have at least GemmTensorSize elements
 * (caller-allocated).
 */
template <typename BlockSize, typename MlsTileB, ::tl::index_t TotalWarp,
          ::tl::index_t WarpN, ::tl::index_t WarpK, typename DataType,
          ::tl::index_t Alt, bool Trans, ::tl::hcu_target_enum HcuArch>
TL_DEVICE void ds_read_format_tensor_b(TL_LDS_ADDR DataType *smem_ptr,
                                       DataType *target) {
  static_assert(WarpK == 1, "WarpK must be 1");
  static constexpr ::tl::index_t WarpM = TotalWarp / (WarpN * WarpK);
  const ::tl::index_t warp_id = ::tl::get_warp_id();
  const ::tl::index_t warp_k_idx = 0;

  ::tl::index_t warp_n_idx = warp_id / WarpM;
  constexpr ::tl::index_t BlockN = BlockSize::at(::tl::number<0>{});
  constexpr ::tl::index_t MinNPerWarp = Trans ? 16 : 32;
  constexpr ::tl::index_t WarpN_no_recompute =
      std::min(WarpN, BlockN / MinNPerWarp);
  if constexpr (WarpN_no_recompute != WarpN) {
    warp_n_idx = warp_n_idx % WarpN_no_recompute;
  }
  ds_read_format_tensor<BlockSize, MlsTileB, WarpN_no_recompute, WarpK,
                        DataType, Alt, Trans, HcuArch>(smem_ptr, target,
                                                       warp_n_idx, warp_k_idx);
}

} // namespace mls
} // namespace tl
