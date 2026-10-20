// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <tl_templates/hcu/core.hpp>
#include <type_traits>

#include <tl_templates/hcu/mls/mls_ds_traits.hpp>
#include <tl_templates/hcu/mls/mls_param_traits.hpp>
#include <tl_templates/hcu/mmac_micro_k.hpp>

namespace tl {
namespace mls {

TL_DEVICE float32x4 ds_read_m32x16_b16_builtin(TL_LDS_ADDR half_t *ptr,
                                               int offset) {
  return __builtin_hcu_ds_read_m32x16_f16(ptr, offset);
}

TL_DEVICE float32x4 ds_read_m32x16_b16_builtin(TL_LDS_ADDR bfloat16_t *ptr,
                                               int offset) {
  return __builtin_hcu_ds_read_m32x16_bf16(ptr, offset);
}

} // namespace mls

// Generic CodeGen entry points for layout-aware LDS-to-VGPR reads.
TL_DEVICE void ds_read_vector(float32x4 &dst, TL_LDS_ADDR half_t *ptr) {
  dst = mls::ds_read_m32x16_b16_builtin(ptr, 0);
}

TL_DEVICE void ds_read_vector(float32x4 &dst, TL_LDS_ADDR bfloat16_t *ptr) {
  dst = mls::ds_read_m32x16_b16_builtin(ptr, 0);
}

namespace mls {

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
  // FP32 MMAC K is arch-specific (gfx92a/gfx946: 16x16x4; gfx938: 16x16x8).
  // TF32 is the `int` specialization and stays K=8 on those arches.
  static constexpr ::tl::index_t MmacKSize =
      std::is_same_v<::tl::remove_cvref_t<TargetType>, float>
          ? static_cast<::tl::index_t>(::tl::MmacMicroKDim<float>::value)
          : (RegBits == 4    ? 64
             : RegBits == 8  ? 32
             : RegBits == 16 ? 16
             : RegBits == 32 ? 8
                             : 0);
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
    // Copy each MMAC fragment at storage granularity so clang does not
    // legalize i8 `vec[j]` extracts. gfx92a/gfx946 FP32 fragments are 4B
    // (K=4); TF32/fp8/bf16/fp16/b4 and gfx938 FP32 are 8B.
    // Do not use const thread_buffer::get_as() on 8-bit types (union copy).
    constexpr auto kFragBytes = sizeof(StorageTargetType) * Traits::VecSize;
    static_assert(kFragBytes == 4 || kFragBytes == 8,
                  "MMAC fragment must be 4B (fp32 x4) or 8B");
    using packed_t =
        std::conditional_t<kFragBytes == 8, ::tl::ext_vector_t<uint32_t, 2>,
                           uint32_t>;
    auto *src_frags = reinterpret_cast<packed_t *>(&ret);

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

      *reinterpret_cast<packed_t *>(target_typed + target_offset) =
          src_frags[block_idx];
    });
  });
}

/*
 * ds_read_format_tensor_a: load A (DsReadBlockSize = M x K) in gemm layout.
 * LdsBlockSize is the full MLS write shape used for LdsDesc.
 * Warp id (gemm.h): traversal order warp_m -> warp_n -> warp_k (innermost to
 * outermost), so warp_m = warp_id % WarpM and warp_k = warp_id / (WarpM*WarpN).
 * WarpN is required (not folded into WarpM) precisely so the K index survives
 * an N partition: A has no N axis, but its K slice is still warp_k's.
 * WarpK == 1 collapses to warp_k_idx = 0 (warp_id < WarpM*WarpN).
 * MlsTileA: ::tl::sequence<MlsTileM, MlsTileKA> - MLS tile for A.
 * origin_mn / origin_k: logical origin of the read slice in the full LDS tile.
 * target: output buffer, must have at least GemmTensorSize elements
 * (caller-allocated).
 */
template <typename LdsBlockSize, typename DsReadBlockSize, typename MlsTileA,
          ::tl::index_t WarpM, ::tl::index_t WarpN, ::tl::index_t WarpK,
          typename DataType, ::tl::index_t Alt, bool Trans,
          ::tl::hcu_target_enum HcuArch, typename TargetType = DataType,
          ::tl::index_t LdsBits = mls_elem_bits_v<DataType>,
          ::tl::index_t RegBits = mls_elem_bits_v<TargetType>>
TL_DEVICE void ds_read_format_tensor_a(TL_LDS_ADDR void *smem_ptr, void *target,
                                       ::tl::index_t warp_id_offset = 0,
                                       ::tl::index_t origin_mn = 0,
                                       ::tl::index_t origin_k = 0) {
  const ::tl::index_t warp_id = ::tl::get_warp_id() - warp_id_offset;
  const ::tl::index_t warp_m_idx = warp_id % WarpM;
  const ::tl::index_t warp_k_idx = warp_id / (WarpM * WarpN);
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

/* Linear LDS B[K,N] to GEMM-B VGPR layout using independent 32N x 16K panels.
 */
template <typename BlockSize, ::tl::index_t TotalWarp, ::tl::index_t WarpN,
          typename DataType>
TL_DEVICE void
ds_read_format_tensor_b_linear(TL_LDS_ADDR DataType *smem_ptr, DataType *target,
                               ::tl::index_t warp_id_offset = 0) {
  static_assert(sizeof(DataType) == 2);
  static constexpr ::tl::index_t BlockN = BlockSize::at(::tl::number<0>{});
  static constexpr ::tl::index_t BlockK = BlockSize::at(::tl::number<1>{});
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
      const ::tl::index_t panel_id = warp_n_idx * (PerWarpN / 32) + n_panel;
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
 * Warp id (gemm.h): traversal order warp_m -> warp_n -> warp_k (innermost to
 * outermost), so warp_n = (warp_id/WarpM) % WarpN and
 * warp_k = (warp_id/WarpM) / WarpN.
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
  static constexpr ::tl::index_t WarpM = TotalWarp / (WarpN * WarpK);
  const ::tl::index_t warp_id = ::tl::get_warp_id() - warp_id_offset;
  const ::tl::index_t warp_mn_idx = warp_id / WarpM;
  const ::tl::index_t warp_k_idx = warp_mn_idx / WarpN;

  ::tl::index_t warp_n_idx = warp_mn_idx % WarpN;
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

/*
 * b_vgpr_direct_load: hipblas-style B-operand load straight from global memory
 * into the same ``B_local`` VGPR layout that ds_read_format_tensor_b produces
 * (so the downstream mmac is byte-identical and unchanged). Skips the whole
 * B global->LDS->ds_read round trip (no LDS write, no B __syncthreads).
 *
 * Layout (verified against make_gemm_fragment_b_hcu + the mmac's
 * ``B_local[(ki*WarpCols+j)*8]`` consumption): for run r in 0..KGrp*WarpCols-1,
 *   n  = (lane & 15) + (r % WarpCols)*16 + warp_n_idx*NBlkTile
 *   k0 = (r / WarpCols)*32 + (lane >> 4)*8
 *   B_local[r*8 + b] = B[n][k0 + b],  b in 0..7
 * Each run is 8 CONSECUTIVE-K fp8 at a fixed n (8 bytes = buffer_load_dwordx2).
 * NBlkTile = ReadN / WarpN_no_recompute (n rows per n-warp); WarpCols = that /16
 * (n-panels per n-warp, must equal the mmac warp_cols); KGrp = ReadK / 32.
 * Run r = k_group*WarpCols + n_panel, matching the mmac (ki=k_group, j=n_panel).
 *
 * global_base: pointer to the block's B[n=0][k=0]. k_stride is the FULL global
 * row length of B in elements (K_total, e.g. 5120) — NOT the block k-extent —
 * because B is (N, K_total) N-major; this is exactly hipblas's s44=5120.
 * dtype must be fp8 (8-bit); NBlkTile must be a multiple of 16.
 */
template <::tl::index_t TotalWarp, ::tl::index_t WarpN, ::tl::index_t WarpK,
          ::tl::index_t ReadN, ::tl::index_t ReadK, ::tl::index_t MinNPerWarp,
          ::tl::index_t WarpCols>
TL_DEVICE void b_vgpr_direct_load(const void *global_base, void *target,
                                  ::tl::index_t k_stride,
                                  ::tl::index_t k_offset_elem = 0,
                                  ::tl::index_t warp_id_offset = 0) {
  static_assert(WarpK == 1, "b_vgpr_direct_load requires WarpK == 1");
  static constexpr ::tl::index_t WarpM = TotalWarp / (WarpN * WarpK);
  constexpr ::tl::index_t WarpN_no_recompute =
      std::min(WarpN, ReadN / MinNPerWarp);
  constexpr ::tl::index_t NBlkTile = ReadN / WarpN_no_recompute; // n per n-warp
  static_assert(NBlkTile % 16 == 0, "n-warp tile must be a multiple of 16");
  static_assert(NBlkTile / 16 == WarpCols, "WarpCols must equal n-panels per n-warp");
  constexpr ::tl::index_t KGrp = ReadK / 32;     // 32-wide k-groups
  constexpr ::tl::index_t NumRuns = KGrp * WarpCols;

  const ::tl::index_t warp_id = ::tl::get_warp_id() - warp_id_offset;
  ::tl::index_t warp_n_idx = warp_id / WarpM;
  if constexpr (WarpN_no_recompute != WarpN)
    warp_n_idx = warp_n_idx % WarpN_no_recompute;
  // n-warp's n-block; within it, (r % WarpCols) picks the 16-row panel,
  // (r / WarpCols) the 32-wide k-group. Run order matches the mmac's
  // B_local[(ki * WarpCols + j) * 8] consumption.
  const ::tl::index_t nblock_base = warp_n_idx * NBlkTile;

  const ::tl::index_t lane = ::tl::get_lane_id();
  const ::tl::index_t nrow = (lane & 15);
  const ::tl::index_t kbase = (lane >> 4) * 8;

  const ::tl::fp8_t *base = reinterpret_cast<const ::tl::fp8_t *>(global_base);
  // ``target`` is the caller's ``B_local`` (T.alloc_local => VGPR registers, not
  // memory), so each write is a register move, not a v_store. ``operator[]``
  // returns the thread_buffer's logical byte order (b4-7 garbage would show up
  // here as a wrong byte, not a memory issue). Copy all 8 bytes of the 8-byte
  // (8-consec-k) run so the mmac's ``*(int32x2*)(B_local + R*8)`` sees the same
  // bytes ds_read_format_tensor_b would have produced.
  ::tl::fp8_t *out = reinterpret_cast<::tl::fp8_t *>(target);
  const int32x4_t res = ::tl::make_wave_buffer_resource(base);
  ::tl::static_for<0, NumRuns, 1>{}([&](auto R) {
    const ::tl::index_t n = nrow + (R % WarpCols) * 16 + nblock_base;
    const ::tl::index_t k0 = (R / WarpCols) * 32 + kbase + k_offset_elem;
    auto v = ::tl::hcu_buffer_load_impl<::tl::fp8_t, 8>(
        res, static_cast<::tl::index_t>(n * k_stride + k0), 0);
#pragma unroll
    for (int b = 0; b < 8; ++b) out[R * 8 + b] = v[b];
  });
}


  // Pipelined stage of b_vgpr_direct_load: one 32-wide K-slice of the same
  // hipblas-shaped B global->VGPR direct load, issued into the caller's full-k
  // B_local while the MMAC chain of the previous slice creates the s_waitcnt
  // (vmcnt) window (hipblas LoopBeginL_1 structure). Byte-identical issue
  // pattern to the full-k loader; the slice shift is folded into the address
  // as ``+ k_offset_elem`` so the warp partition can never diverge, and the
  // destination is offset by ``dst_offset_elems`` (slice s = s*warp_cols*
  // local_size_b) so the producer slot is disjoint from the slice the current
  // mmac chain is consuming. Caller owns the fence (s_waitcnt vmcnt(0)) at the
  // iteration bottom.
template <::tl::index_t TotalWarp, ::tl::index_t WarpN, ::tl::index_t WarpK,
          ::tl::index_t ReadN, ::tl::index_t ReadK, ::tl::index_t MinNPerWarp,
          ::tl::index_t WarpCols>
TL_DEVICE void b_vgpr_direct_load_stage(const void *global_base, void *target,
                                        ::tl::index_t k_stride,
                                        ::tl::index_t k_offset_elem = 0,
                                        ::tl::index_t dst_offset_elems = 0,
                                        ::tl::index_t warp_id_offset = 0) {
  ::tl::fp8_t *dst =
      reinterpret_cast<::tl::fp8_t *>(target) + dst_offset_elems;
  b_vgpr_direct_load<TotalWarp, WarpN, WarpK, ReadN, ReadK, MinNPerWarp,
                     WarpCols>(global_base, dst, k_stride, k_offset_elem,
                               warp_id_offset);
}

/*
 * b_vgpr_direct_load16: hipblas-shaped B-operand load straight from global
 * memory using 16-byte `buffer_load_dwordx4` per lane (4 loads for a 128-wide
 * K stage). Unlike `b_vgpr_direct_load` (which emits 8B dwordx2 runs matching
 * the current 8-consec-k MMAC fragment), each 16B load fetches 16 CONSECUTIVE-k
 * fp8 at a fixed n, so the per-lane (n,k) ownership is DIFFERENT from the
 * `ds_read_format_tensor_b` / mmac fragment: one lane owns 8 disjoint 16-wide k
 * windows at a single n, and no lane owns the mmac's "8-consec-k at fixed n with
 * k-groups 32 apart" pattern.
 *
 * Per lane (for a 128-wide K stage, ReadK=128):
 *   n_row = lane & 31                      // 32 n-rows of the n-warp
 *   n     = n_row + nblock_base
 *   lane_hi = lane >> 5                      // 0 or 1
 *   for w in 0..3:                          // 4 dwordx4 loads
 *     kwin = lane_hi*4 + w                   // 8 windows of 16-wide k total
 *     k0   = kwin*16
 *     dst[w*16 + b] = B[n][k0 + b], b in 0..15
 * Destination is a 64-elem per-lane buffer (4 x 16B). The four loads and the
 * single `s_waitcnt vmcnt(0)` are emitted in ONE opaque asm block so the backend
 * cannot insert per-load VMEM drains (the whole point of the hipblas shape).
 */
template <::tl::index_t TotalWarp, ::tl::index_t WarpN, ::tl::index_t WarpK,
          ::tl::index_t ReadN, ::tl::index_t ReadK, ::tl::index_t MinNPerWarp,
          ::tl::index_t WarpCols>
TL_DEVICE void b_vgpr_direct_load16(const void *global_base, void *target,
                                    ::tl::index_t k_stride,
                                    ::tl::index_t k_offset_elem = 0,
                                    ::tl::index_t dst_offset_elems = 0,
                                    ::tl::index_t warp_id_offset = 0) {
  static_assert(WarpK == 1, "b_vgpr_direct_load16 requires WarpK == 1");
  static_assert(ReadK == 128, "b_vgpr_direct_load16 requires ReadK == 128 (4 x 16B per lane)");
  static constexpr ::tl::index_t WarpM = TotalWarp / (WarpN * WarpK);
  constexpr ::tl::index_t WarpN_no_recompute =
      std::min(WarpN, ReadN / MinNPerWarp);
  constexpr ::tl::index_t NBlkTile = ReadN / WarpN_no_recompute; // n per n-warp
  static_assert(NBlkTile % 16 == 0, "n-warp tile must be a multiple of 16");

  const ::tl::index_t warp_id = ::tl::get_warp_id() - warp_id_offset;
  ::tl::index_t warp_n_idx = warp_id / WarpM;
  if constexpr (WarpN_no_recompute != WarpN)
    warp_n_idx = warp_n_idx % WarpN_no_recompute;
  const ::tl::index_t nblock_base = warp_n_idx * NBlkTile;

  const ::tl::index_t lane = ::tl::get_lane_id();
  const ::tl::index_t n_row = lane & 31;         // 32 n-rows of the n-warp
  const ::tl::index_t n = n_row + nblock_base;
  const ::tl::index_t lane_hi = lane >> 5;          // 0 or 1

  const ::tl::fp8_t *base = reinterpret_cast<const ::tl::fp8_t *>(global_base);
  ::tl::fp8_t *dst =
      reinterpret_cast<::tl::fp8_t *>(target) + dst_offset_elems;
  const int32x4_t res = ::tl::make_wave_buffer_resource(base);

  // Four 16B addresses (byte offsets into the wave buffer resource).
  ::tl::index_t a0 = static_cast<::tl::index_t>(n * k_stride + (lane_hi * 4 + 0) * 16 + k_offset_elem);
  ::tl::index_t a1 = static_cast<::tl::index_t>(n * k_stride + (lane_hi * 4 + 1) * 16 + k_offset_elem);
  ::tl::index_t a2 = static_cast<::tl::index_t>(n * k_stride + (lane_hi * 4 + 2) * 16 + k_offset_elem);
  ::tl::index_t a3 = static_cast<::tl::index_t>(n * k_stride + (lane_hi * 4 + 3) * 16 + k_offset_elem);

  fp32x4_t d0, d1, d2, d3;
  asm volatile(
      "buffer_load_dwordx4 %[d0], %[a0], %[res], 0 offen;\n\t"
      "buffer_load_dwordx4 %[d1], %[a1], %[res], 0 offen;\n\t"
      "buffer_load_dwordx4 %[d2], %[a2], %[res], 0 offen;\n\t"
      "buffer_load_dwordx4 %[d3], %[a3], %[res], 0 offen;\n\t"
      "s_waitcnt vmcnt(0);\n\t"
      : [d0] "=&v"(d0), [d1] "=&v"(d1), [d2] "=&v"(d2), [d3] "=&v"(d3)
      : [a0] "v"(a0), [a1] "v"(a1), [a2] "v"(a2), [a3] "v"(a3),
        [res] "s"(res)
      : "memory");

  ::tl::fp8_t *p0 = reinterpret_cast<::tl::fp8_t *>(&d0);
  ::tl::fp8_t *p1 = reinterpret_cast<::tl::fp8_t *>(&d1);
  ::tl::fp8_t *p2 = reinterpret_cast<::tl::fp8_t *>(&d2);
  ::tl::fp8_t *p3 = reinterpret_cast<::tl::fp8_t *>(&d3);
#pragma unroll
  for (int b = 0; b < 16; ++b) {
    dst[0 * 16 + b] = p0[b];
    dst[1 * 16 + b] = p1[b];
    dst[2 * 16 + b] = p2[b];
    dst[3 * 16 + b] = p3[b];
  }
}

} // namespace mls
} // namespace tl
