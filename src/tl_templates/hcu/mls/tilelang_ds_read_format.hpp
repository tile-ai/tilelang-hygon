// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <tl_templates/hcu/core.hpp>
#include <type_traits>

#include <tl_templates/hcu/mls/mls_ds_traits.hpp>
#include <tl_templates/hcu/mls/mls_param_traits.hpp>

namespace tl {
namespace mls {

TL_DEVICE float32x4 ds_read_m32x16_b16_builtin(TL_LDS_ADDR half_t *ptr,
                                               int offset) {
  return __builtin_hcu_ds_read_m32x16_f16(ptr, offset);
}

TL_DEVICE float32x4 ds_read_m32x16_b16_builtin(
    TL_LDS_ADDR bfloat16_t *ptr, int offset) {
  return __builtin_hcu_ds_read_m32x16_bf16(ptr, offset);
}

/*
 * ds_read_format_traits: traits for reading MLS LDS with warp-chunk layout.
 * Block is divided into WarpMN x WarpK chunks; each warp loads its chunk.
 * Within chunk: traverse MN first, then K (MNIterPerWarp, KIterPerWarp).
 *
 * LdsBlockSize: full (non-sliced) LDS MN×K used to build LdsDesc (must match
 * matrix_load write shape).
 * DsReadBlockSize: ds_read_format MN×K extent (slice shape / fragment tile);
 * drives PerWarp / SFC.
 *
 * Template params: LdsBlockSize, DsReadBlockSize, MlsTileSize
 *                  (sequence<MN,K>), WarpMN, WarpK, DataType, Alt, Trans,
 *                  HcuArch.
 */
template <typename LdsBlockSize, typename DsReadBlockSize, typename MlsTileSize,
          ::tl::index_t WarpMN, ::tl::index_t WarpK, typename DataType,
          ::tl::index_t Alt, bool Trans, ::tl::hcu_target_enum HcuArch,
          typename TargetType = DataType,
          ::tl::index_t LdsBits = mls_elem_bits_v<DataType>,
          ::tl::index_t RegBits = mls_elem_bits_v<TargetType>>
struct ds_read_format_traits {
  static constexpr ::tl::index_t ReadBlockSizeMN =
      DsReadBlockSize::at(::tl::number<0>{});
  static constexpr ::tl::index_t ReadBlockSizeK =
      DsReadBlockSize::at(::tl::number<1>{});

  static constexpr ::tl::index_t Bits = mls_elem_bits_v<DataType>;
  static constexpr ::tl::index_t LdsPhysicalBits = LdsBits;

  using LdsTraits = mls_lds_desc_param_traits<LdsBlockSize, MlsTileSize, Bits,
                                              LdsBits, Alt, Trans, HcuArch>;
  using MlsAtom = typename LdsTraits::MlsAtom;
  using DsFormatInst = typename mls_ds_traits_with_target<
      MlsAtom, Bits, LdsBits, RegBits, Alt,
      ::tl::remove_cvref_t<TargetType>>::Type;

  static constexpr auto LdsDesc = LdsTraits::get_tile_lds_desc();

  // Per-warp chunk size (over the *read* extent)
  static constexpr ::tl::index_t PerWarpMN = ReadBlockSizeMN / WarpMN;
  static constexpr ::tl::index_t PerWarpK = ReadBlockSizeK / WarpK;

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
  static constexpr ::tl::index_t MmacKSize = RegBits == 4    ? 64
                                             : RegBits == 8  ? 32
                                             : RegBits == 16 ? 16
                                             : RegBits == 32 ? 8
                                                             : 0;
  static_assert(MmacKSize != 0, "Unsupported ds_read_format register bitwidth");
  static constexpr ::tl::index_t GemmWarpRows = PerWarpMN / MmacMNSize;
  static constexpr ::tl::index_t GemmInnerK = PerWarpK / MmacKSize;
  // VecSize is in storage elements; for b4 it counts packed bytes.
  static constexpr ::tl::index_t VecSize = RegBits == 4 ? 8 : MmacKSize / 4;
  static constexpr ::tl::index_t GemmTensorSize =
      GemmInnerK * GemmWarpRows * VecSize;
};

/*
 * ds_read_format_tensor: read MLS LDS into target buffer with gemm body_rr
 * layout. Indexing: target[(ki * warp_mmac_rows + Mi) * vec_size].
 *
 * smem_ptr: base of the *full* LDS buffer (leading-dim offset already applied).
 * origin_mn / origin_k: logical origin of this slice inside the full LDS tile.
 * Absolute logical coord = origin + local; physical via full LdsDesc.
 *
 * warp_mn_idx, warp_k_idx: indices of current warp in the block
 * (caller-provided). target: output buffer, must have at least GemmTensorSize
 * elements (caller-allocated).
 *
 * Each DsFormatInst read yields kMN x kK; we treat it as (kMN/mmacMN) x
 * (kK/mmacK) mmac blocks and scatter them to target positions.
 */
template <typename LdsBlockSize, typename DsReadBlockSize, typename MlsTileSize,
          ::tl::index_t WarpMN, ::tl::index_t WarpK, typename DataType,
          ::tl::index_t Alt, bool Trans, ::tl::hcu_target_enum HcuArch,
          typename TargetType = DataType,
          ::tl::index_t LdsBits = mls_elem_bits_v<DataType>,
          ::tl::index_t RegBits = mls_elem_bits_v<TargetType>>
TL_DEVICE void
ds_read_format_tensor(TL_LDS_ADDR DataType *smem_ptr, void *target,
                      ::tl::index_t warp_mn_idx, ::tl::index_t warp_k_idx,
                      ::tl::index_t origin_mn = 0, ::tl::index_t origin_k = 0) {
  using Traits =
      ds_read_format_traits<LdsBlockSize, DsReadBlockSize, MlsTileSize, WarpMN,
                            WarpK, DataType, Alt, Trans, HcuArch, TargetType,
                            LdsBits, RegBits>;

  using DsFormatInst = typename Traits::DsFormatInst;
  using StorageTargetType = typename ds_read_format_storage_type<
      ::tl::remove_cvref_t<TargetType>>::type;
  using vector_t =
      ::tl::ext_vector_t<StorageTargetType, DsFormatInst::kVectorLength>;
  TargetType *target_typed = reinterpret_cast<TargetType *>(target);

  constexpr ::tl::index_t NumMMAC_MN = DsFormatInst::kMN / Traits::MmacMNSize;
  constexpr ::tl::index_t NumMMAC_K = DsFormatInst::kK / Traits::MmacKSize;
  constexpr ::tl::index_t NumMMACPerRead = NumMMAC_MN * NumMMAC_K;

  static_assert(DsFormatInst::kVectorLength == NumMMACPerRead * Traits::VecSize,
                "DsFormatInst vector length must match mmac blocks * vec_size");

  const ::tl::index_t warp_lds_byte_offset =
      ::tl::mls::mls_lds_physical_storage_traits<Traits::LdsPhysicalBits>::
          logical_offset_to_byte_offset(
              Traits::LdsDesc.calculate_offset(::tl::make_multi_index(
                  origin_mn + warp_mn_idx * Traits::PerWarpMN,
                  origin_k + warp_k_idx * Traits::PerWarpK)));
  TL_LDS_ADDR uint8_t *smem_bytes =
      reinterpret_cast<TL_LDS_ADDR uint8_t *>(smem_ptr);

  ::tl::static_for<0, Traits::NumAccess, 1>{}([&](auto i) {
    constexpr auto idx = Traits::SFC::get_index(i);

    constexpr auto mn_iter = idx.at(::tl::number<0>{});
    constexpr auto k_iter = idx.at(::tl::number<1>{});

    // Immediate is relative to the warp base pointer. Relies on LdsDesc
    // delta(w, local) == offset(local) for this access pattern; origin is
    // folded only into the warp base pointer.
    constexpr auto immed_offset =
        ::tl::mls::mls_lds_physical_storage_traits<Traits::LdsPhysicalBits>::
            logical_offset_to_byte_offset(
                Traits::LdsDesc.calculate_offset(::tl::make_multi_index(
                    mn_iter * DsFormatInst::kMN, k_iter * DsFormatInst::kK)));

    auto ret = DsFormatInst{}(reinterpret_cast<TL_LDS_ADDR DataType *>(
                                  smem_bytes + warp_lds_byte_offset),
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
        target_typed[target_offset + j] =
            static_cast<TargetType>(vec_value[vec_offset + j]);
      }
    });
  });
}

/*
 * ds_read_format_tensor_a: load A (DsReadBlockSize = M x K) in gemm layout.
 * LdsBlockSize is the full MLS write shape used for LdsDesc.
 * Warp id (gemm.h): warp_m = warp_id % WarpM.
 * WarpK must be 1 (warp_k_idx = 0).
 * MlsTileA: ::tl::sequence<MlsTileM, MlsTileKA> - MLS tile for A.
 * origin_mn / origin_k: logical origin of the read slice in the full LDS tile.
 * target: output buffer, must have at least GemmTensorSize elements
 * (caller-allocated).
 */
template <typename LdsBlockSize, typename DsReadBlockSize, typename MlsTileA,
          ::tl::index_t WarpM, ::tl::index_t WarpK, typename DataType,
          ::tl::index_t Alt, bool Trans, ::tl::hcu_target_enum HcuArch,
          typename TargetType = DataType,
          ::tl::index_t LdsBits = mls_elem_bits_v<DataType>,
          ::tl::index_t RegBits = mls_elem_bits_v<TargetType>>
TL_DEVICE void ds_read_format_tensor_a(TL_LDS_ADDR void *smem_ptr, void *target,
                                       ::tl::index_t warp_id_offset = 0,
                                       ::tl::index_t origin_mn = 0,
                                       ::tl::index_t origin_k = 0) {
  static_assert(WarpK == 1, "WarpK must be 1");
  const ::tl::index_t warp_id = ::tl::get_warp_id() - warp_id_offset;
  const ::tl::index_t warp_m_idx = warp_id % WarpM;
  const ::tl::index_t warp_k_idx = 0;
  ds_read_format_tensor<LdsBlockSize, DsReadBlockSize, MlsTileA, WarpM, WarpK,
                        DataType, Alt, Trans, HcuArch, TargetType, LdsBits,
                        RegBits>(
      reinterpret_cast<TL_LDS_ADDR DataType *>(smem_ptr), target, warp_m_idx,
      warp_k_idx, origin_mn, origin_k);
}

/*
 * ds_read_format_tensor_common: for output not directly fed to gemm.
 * warp_mn_idx = warp_id % WarpMN_no_recompute;
 * warp_k_idx  = warp_id / WarpMN_no_recompute;
 * When WarpMN_no_recompute != WarpMN: warp_k_idx %= WarpK.
 * WarpMN_no_recompute = min(WarpMN, ReadBlockSizeMN / DsFormatInst::kMN).
 * origin_mn / origin_k: logical origin of the read slice in the full LDS tile.
 */
template <typename LdsBlockSize, typename DsReadBlockSize, typename MlsTileSize,
          ::tl::index_t WarpMN, ::tl::index_t WarpK, typename DataType,
          ::tl::index_t Alt, bool Trans, ::tl::hcu_target_enum HcuArch,
          typename TargetType = DataType,
          ::tl::index_t LdsBits = mls_elem_bits_v<DataType>,
          ::tl::index_t RegBits = mls_elem_bits_v<TargetType>>
TL_DEVICE void ds_read_format_tensor_common(TL_LDS_ADDR void *smem_ptr,
                                            TargetType *target,
                                            ::tl::index_t warp_id_offset = 0,
                                            ::tl::index_t origin_mn = 0,
                                            ::tl::index_t origin_k = 0) {
  static constexpr ::tl::index_t ReadBlockSizeMN =
      DsReadBlockSize::at(::tl::number<0>{});
  using Traits =
      ds_read_format_traits<LdsBlockSize, DsReadBlockSize, MlsTileSize, WarpMN,
                            WarpK, DataType, Alt, Trans, HcuArch, TargetType,
                            LdsBits, RegBits>;
  using DsFormatInst = typename Traits::DsFormatInst;

  static constexpr ::tl::index_t WarpMN_no_recompute =
      std::min(WarpMN, ReadBlockSizeMN / DsFormatInst::kMN);

  const ::tl::index_t warp_id = ::tl::get_warp_id() - warp_id_offset;
  ::tl::index_t warp_mn_idx = warp_id % WarpMN_no_recompute;
  ::tl::index_t warp_k_idx = warp_id / WarpMN_no_recompute;
  if constexpr (WarpMN_no_recompute != WarpMN) {
    warp_k_idx = warp_k_idx % WarpK;
  }

  ds_read_format_tensor<LdsBlockSize, DsReadBlockSize, MlsTileSize,
                        WarpMN_no_recompute, WarpK, DataType, Alt, Trans,
                        HcuArch, TargetType, LdsBits, RegBits>(
      reinterpret_cast<TL_LDS_ADDR DataType *>(smem_ptr), target, warp_mn_idx,
      warp_k_idx, origin_mn, origin_k);
}

/* Linear LDS B[K,N] to GEMM-B VGPR layout using independent 32N x 16K panels. */
template <typename BlockSize, ::tl::index_t TotalWarp,
          ::tl::index_t WarpN, typename DataType>
TL_DEVICE void ds_read_format_tensor_b_linear(
    TL_LDS_ADDR DataType *smem_ptr, DataType *target,
    ::tl::index_t warp_id_offset = 0) {
  static_assert(sizeof(DataType) == 2);
  static constexpr ::tl::index_t BlockN =
      BlockSize::at(::tl::number<0>{});
  static constexpr ::tl::index_t BlockK =
      BlockSize::at(::tl::number<1>{});
  static_assert(BlockK % 16 == 0 && BlockN % 32 == 0);
  static constexpr ::tl::index_t WarpM = TotalWarp / WarpN;
  static constexpr ::tl::index_t WarpNNoRecompute =
      std::min(WarpN, BlockN / 32);
  static constexpr ::tl::index_t PerWarpN = BlockN / WarpNNoRecompute;
  static_assert(PerWarpN % 32 == 0);
  static constexpr ::tl::index_t NumNAccess = PerWarpN / 32;
  static constexpr ::tl::index_t NumKAccess = BlockK / 16;
  static constexpr ::tl::index_t FragmentLength = 4;
  using FragmentType = ::tl::ext_vector_t<DataType, FragmentLength>;

  const ::tl::index_t warp_id = ::tl::get_warp_id() - warp_id_offset;
  ::tl::index_t warp_n_idx = warp_id / WarpM;
  if constexpr (WarpNNoRecompute != WarpN)
    warp_n_idx %= WarpNNoRecompute;
  const ::tl::index_t lane = ::tl::get_lane_id();

  ::tl::static_for<0, NumNAccess, 1>{}([&](auto n_panel) {
    ::tl::static_for<0, NumKAccess, 1>{}([&](auto k_tile) {
      const ::tl::index_t panel_id =
          warp_n_idx * (PerWarpN / 32) + n_panel;
      const ::tl::index_t k_addr = k_tile * 16 + (lane >> 2);
      const ::tl::index_t n_addr = panel_id * 32 + (lane & 3) * 8;
      auto value = ::tl::bit_cast<float32x4>(
          ds_read_m32x16_b16_builtin(smem_ptr + k_addr * BlockN + n_addr, 0));
      *reinterpret_cast<FragmentType *>(
          target + (k_tile * PerWarpN / 16 + n_panel * 2) * FragmentLength) =
          reinterpret_cast<FragmentType *>(&value)[0];
      *reinterpret_cast<FragmentType *>(
          target +
          (k_tile * PerWarpN / 16 + n_panel * 2 + 1) * FragmentLength) =
          reinterpret_cast<FragmentType *>(&value)[1];
    });
  });
}

/*
 * ds_read_format_tensor_b: load B (DsReadBlockSize = N x K) in gemm layout.
 * LdsBlockSize is the full MLS write shape used for LdsDesc.
 * TotalWarp = WarpM * WarpN * WarpK; WarpM = TotalWarp / (WarpN * WarpK).
 * Warp id (gemm.h): warp_n = warp_id / WarpM.
 * WarpN_no_recompute = min(WarpN, ReadN / MinNPerWarp).
 * MinNPerWarp is the final floor resolved by ComputeWarpPartitionHCU. Its
 * default preserves legacy standalone calls (Trans=true -> 16, otherwise 32).
 * When
 * WarpN_no_recompute != WarpN: pass WarpN_no_recompute as WarpMN.
 * WarpK must be 1 (warp_k_idx = 0).
 * MlsTileB: ::tl::sequence<MlsTileN, MlsTileKB> - MLS tile for B.
 * origin_mn / origin_k: logical origin of the read slice in the full LDS tile.
 * target: output buffer, must have at least GemmTensorSize elements
 * (caller-allocated).
 */
template <typename LdsBlockSize, typename DsReadBlockSize, typename MlsTileB,
          ::tl::index_t TotalWarp, ::tl::index_t WarpN, ::tl::index_t WarpK,
          typename DataType, ::tl::index_t Alt, bool Trans,
          ::tl::hcu_target_enum HcuArch, typename TargetType = DataType,
          ::tl::index_t LdsBits = mls_elem_bits_v<DataType>,
          ::tl::index_t RegBits = mls_elem_bits_v<TargetType>,
          ::tl::index_t MinNPerWarp = (Trans ? 16 : 32)>
TL_DEVICE void ds_read_format_tensor_b(TL_LDS_ADDR void *smem_ptr, void *target,
                                       ::tl::index_t warp_id_offset = 0,
                                       ::tl::index_t origin_mn = 0,
                                       ::tl::index_t origin_k = 0) {
  static_assert(WarpK == 1, "WarpK must be 1");
  static constexpr ::tl::index_t WarpM = TotalWarp / (WarpN * WarpK);
  const ::tl::index_t warp_id = ::tl::get_warp_id() - warp_id_offset;
  const ::tl::index_t warp_k_idx = 0;

  ::tl::index_t warp_n_idx = warp_id / WarpM;
  constexpr ::tl::index_t ReadN = DsReadBlockSize::at(::tl::number<0>{});
  constexpr ::tl::index_t WarpN_no_recompute =
      std::min(WarpN, ReadN / MinNPerWarp);
  if constexpr (WarpN_no_recompute != WarpN) {
    warp_n_idx = warp_n_idx % WarpN_no_recompute;
  }
  ds_read_format_tensor<LdsBlockSize, DsReadBlockSize, MlsTileB,
                        WarpN_no_recompute, WarpK, DataType, Alt, Trans,
                        HcuArch, TargetType, LdsBits, RegBits>(
      reinterpret_cast<TL_LDS_ADDR DataType *>(smem_ptr), target, warp_n_idx,
      warp_k_idx, origin_mn, origin_k);
}

} // namespace mls
} // namespace tl
