// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file scale_buffer.hpp
 * \brief HCU scale_buffer helpers: ds_scale_copy + mmac_scale_fp4_body (v10).
 *
 * Each scale buffer is split into MnWarps MN segments (the GEMM's m_seg or
 * n_seg count):
 *   rows_mn = ScaleRowsMN<ScaleShapeMN, GranularityMN>::value
 *   phys_k = ScaleShapeK * ((MmacK == 64 && GranularityK >= 64) ? 2 : 1)
 *   rows_mn_per_seg = rows_mn / MnWarps
 *   logical_rows_per_seg = rows_mn_per_seg * phys_k
 *   aligned_rows_per_seg = Align8(logical_rows_per_seg)
 * The identity-format row within a segment is mn_local * phys_k + k_phys;
 * interleaved formats use ScaleFormatLogicalToRow / their copy policy.
 *
 * TotalWarps fill all segments: warp_id % MnWarps selects the segment and
 * warp_id / MnWarps selects a copy warp within it. MMAC offsets are relative
 * to m0, which points at the consuming wave's aligned segment base.
 *
 * ScalePhysK, ScaleLogicalToRow, ds_scale_copy, and mmac_scale_fp4_body share
 * the same MmacK parameter. MmacK=64 uses the FP4 intrinsic; MmacK=32 uses the
 * f8f6f4 intrinsic selected by RealABType.
 *
 * ds_scale_copy dispatches ScaleFormat to a format-specific policy after
 * validating OpCtrl, MmacK, ScaleShapeK, and layout constraints.
 * DsScaleCopyFill schedules fill batches; Policy::Map converts each batch to
 * its destination scale-buffer row and source LDS byte offset.
 */
#pragma once

#include <tl_templates/hcu/common.h>
#include <tl_templates/hcu/core.hpp>
#include <tl_templates/hcu/core/numeric/pk_fp4.hpp>

namespace tl {
namespace hcu {

template <char OpCtrl, char Offset0 = 0, char Offset1 = 0>
TL_DEVICE void ds_scale_copy_ds2buf(TL_LDS_ADDR int *lds_data, int soffset) {
  __builtin_hcu_ds_scale_copy_ds2buf(lds_data, soffset, OpCtrl, Offset0,
                                     Offset1);
}

constexpr TL_DEVICE int ScaleAlign8(int rows) { return (rows + 7) / 8 * 8; }

template <int ScaleShapeMN, int GranularityMN> struct ScaleRowsMN {
  static_assert(GranularityMN > 0, "GranularityMN must be positive");
  static constexpr int MMAC_MN = 16;
  static constexpr int value = (GranularityMN >= MMAC_MN)
                                   ? ScaleShapeMN
                                   : (ScaleShapeMN * GranularityMN / MMAC_MN);
};

template <int ScaleMNIdx, int GranularityMN> struct ScaleMNToRow {
  static constexpr int MMAC_MN = 16;
  static constexpr int value = (GranularityMN >= MMAC_MN)
                                   ? ScaleMNIdx
                                   : (ScaleMNIdx * GranularityMN / MMAC_MN);
};

template <int ScaleShapeK, int GranularityK, int MmacK = 64> struct ScalePhysK {
  // When mmac_k==64 and gran_k>=64, one scale covers the whole K atom → dup×2
  // rows.
  static constexpr int k_dup = (MmacK == 64 && GranularityK >= 64) ? 2 : 1;
  static constexpr int value = ScaleShapeK * k_dup;
};

enum class DsScaleFormat : int {
  kIdentity = 0,
  kK2Interleave = 1,
  kK4Interleave = 2,
  kK2MN2Interleave = 3,
  kMN2Interleave = 4,
  kMN4Interleave = 5,
};

/*! Local row inside one MN segment (relative to segment base / m0). */
template <int ScaleMNIdx, int ScaleKIdx, int GranularityMN, int ScaleShapeK,
          int GranularityK, int MmacK = 64>
struct ScaleLogicalToRow {
  static constexpr int phys_k =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::value;
  static constexpr int k_dup =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::k_dup;
  static constexpr int mn_local =
      ScaleMNToRow<ScaleMNIdx, GranularityMN>::value;
  static constexpr int value = mn_local * phys_k + ScaleKIdx * k_dup;
};

template <int ScaleFormatId, int ScaleMNIdx, int ScaleKIdx, int GranularityMN,
          int ScaleShapeK, int GranularityK, int MmacK = 64>
struct ScaleFormatLogicalToRow {
  static constexpr int mn_local =
      ScaleMNToRow<ScaleMNIdx, GranularityMN>::value;
  static constexpr DsScaleFormat format =
      static_cast<DsScaleFormat>(ScaleFormatId);
  static constexpr int value = []() constexpr {
    if constexpr (format == DsScaleFormat::kMN2Interleave) {
      return ((mn_local / 2) * ScaleShapeK + ScaleKIdx) * 2 + (mn_local % 2);
    } else if constexpr (format == DsScaleFormat::kMN4Interleave) {
      return ((mn_local / 4) * ScaleShapeK + ScaleKIdx) * 4 + (mn_local % 4);
    } else if constexpr (format == DsScaleFormat::kK2MN2Interleave) {
      static_assert(ScaleShapeK % 2 == 0, "K2MN2 scale shape K must be even");
      return ((mn_local / 2) * (ScaleShapeK / 2) + ScaleKIdx / 2) * 4 +
             (mn_local % 2) * 2 + ScaleKIdx % 2;
    } else {
      return ScaleLogicalToRow<ScaleMNIdx, ScaleKIdx, GranularityMN,
                               ScaleShapeK, GranularityK, MmacK>::value;
    }
  }();
};

/*!
 * Per-scenario fill policies: specialize fill_batch → (dst_row, lds_byte).
 * DsScaleCopyFill only does seg/warp scheduling; index math lives here.
 *
 *   mmac_k | op | K condition | policy
 *   64/32  | 0  | any         | Opctrl0Linear
 *   64/32  | 1  | %2==0       | Opctrl1K2Interleave   (pure K)
 *   32     | 1  | any         | Opctrl1MN2Interleave  (explicit MN2 format)
 *   64/32  | 2  | %4==0       | Opctrl2K4Interleave   (pure K)
 *   64/32  | 2  | ==2         | Opctrl2K2MN2Interleave
 *   32     | 2  | ==1         | Opctrl2MN4Interleave
 */

template <int GranularityMN> TL_DEVICE int ScaleMnIdx(int mn_row, int tx16) {
  if constexpr (GranularityMN >= 16) {
    return mn_row;
  } else {
    constexpr int pack = 16 / GranularityMN;
    return mn_row * pack + tx16 / GranularityMN;
  }
}

// Default descriptor used by the plain-Buffer compatibility path.  Explicit
// ScaleView inputs substitute a generated functor with the same interface.
template <int ParentScaleMN, int ParentScaleK, int ScaleKMajor>
struct ScaleLdsLinearLayout {
  TL_DEVICE static int CalculateOffset(int k, int mn, int, int, int) {
    if constexpr (ScaleKMajor) {
      return mn * ParentScaleK + k;
    } else {
      return k * ParentScaleMN + mn;
    }
  }
};

// LDS offsets are computed in the parent last-2 plane.  The pointer is its
// base; origin_mn/origin_k identify the copied tile inside that plane.
template <int ParentScaleMN, int ParentScaleK, int ScaleShapeMN,
          int ScaleShapeK, int GranularityMN, int GranularityK, int ScaleKMajor,
          int MmacK, typename StorageLayout>
struct DsScaleLdsOpctrl0Linear {
  static constexpr int kOpCtrl = 0;
  static constexpr int kRowsPerFill = 1;
  static constexpr int kPhysK =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::value;
  static constexpr int kDup =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::k_dup;

  TL_DEVICE static void Map(int fill_batch, int seg_row_base, int mn_row0,
                            int tx16, int origin_mn, int origin_k, int *dst_row,
                            int *lds_byte) {
    *dst_row = seg_row_base + fill_batch;
    const int mn_idx =
        ScaleMnIdx<GranularityMN>(mn_row0 + fill_batch / kPhysK, tx16);
    const int k_rel = (fill_batch % kPhysK) / kDup;
    *lds_byte = StorageLayout::CalculateOffset(origin_k + k_rel,
                                               origin_mn + mn_idx, 0, 0, 0);
  }
};

// op_ctrl=1 pure K: fill_batch = mn_local * k_atoms + k_atom.
// LDS: m0k0 m0k1 | m1k0 m1k1 | ... then next k_atom: m0k2 m0k3 | ...
template <int ParentScaleMN, int ParentScaleK, int ScaleShapeMN,
          int ScaleShapeK, int GranularityMN, int GranularityK, int MmacK,
          typename StorageLayout>
struct DsScaleLdsOpctrl1K2Interleave {
  static_assert(ScaleShapeK % 2 == 0,
                "op_ctrl=1 K2 interleave requires scaleShapeK % 2 == 0");
  static constexpr int kOpCtrl = 1;
  static constexpr int kRowsPerFill = 2;
  static constexpr int kPhysK =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::value;
  static constexpr int kAtoms = kPhysK / kRowsPerFill;

  TL_DEVICE static void Map(int fill_batch, int seg_row_base, int mn_row0,
                            int tx16, int origin_mn, int origin_k, int *dst_row,
                            int *lds_byte) {
    if constexpr (kAtoms == 1) {
      // scaleShapeK==2: one K-atom per MN; fill_batch == mn_local.
      *dst_row = seg_row_base + fill_batch * kPhysK;
      const int mn_idx = ScaleMnIdx<GranularityMN>(mn_row0 + fill_batch, tx16);
      *lds_byte = StorageLayout::CalculateOffset(origin_k / 2,
                                                 origin_mn + mn_idx, 0, 0, 0);
    } else {
      const int mn_local = fill_batch / kAtoms;
      const int k_atom = fill_batch % kAtoms;
      *dst_row = seg_row_base + mn_local * kPhysK + k_atom * kRowsPerFill;
      const int mn_idx = ScaleMnIdx<GranularityMN>(mn_row0 + mn_local, tx16);
      *lds_byte = StorageLayout::CalculateOffset(origin_k / 2 + k_atom,
                                                 origin_mn + mn_idx, 0, 0, 0);
    }
  }
};

// op_ctrl=1 MN2 atom: fill_batch = pair slot along MN (m | m+16).
// LDS: m0k0 m16k0 | m1k0 m17k0 | ...  — each thread +2B from pair slot.
template <int ParentScaleMN, int ParentScaleK, int ScaleShapeMN,
          int ScaleShapeK, int GranularityMN, int GranularityK, int MmacK,
          typename StorageLayout>
struct DsScaleLdsOpctrl1MN2Interleave {
  static constexpr int kOpCtrl = 1;
  static constexpr int kRowsPerFill = 2;
  static constexpr int kPhysK =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::value;

  TL_DEVICE static void Map(int fill_batch, int seg_row_base, int mn_row0,
                            int tx16, int origin_mn, int origin_k, int *dst_row,
                            int *lds_byte) {
    const int mn_pair = fill_batch / kPhysK;
    const int k_rel = fill_batch % kPhysK;
    *dst_row = seg_row_base + fill_batch * kRowsPerFill;
    *lds_byte = StorageLayout::CalculateOffset(
        origin_k + k_rel,
        origin_mn / 2 + ScaleMnIdx<GranularityMN>(mn_row0 / 2 + mn_pair, tx16),
        0, 0, 0);
  }
};

// op_ctrl=2 pure K: fill_batch = mn_local * k_groups + k_group (4 rows /
// group). LDS: m0k0..k3 | m1k0..k3 | ... then next group.
template <int ParentScaleMN, int ParentScaleK, int ScaleShapeMN,
          int ScaleShapeK, int GranularityMN, int GranularityK, int MmacK,
          typename StorageLayout>
struct DsScaleLdsOpctrl2K4Interleave {
  static_assert(ScaleShapeK % 4 == 0,
                "op_ctrl=2 K4 interleave requires scaleShapeK % 4 == 0");
  static constexpr int kOpCtrl = 2;
  static constexpr int kRowsPerFill = 4;
  static constexpr int kPhysK =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::value;
  static constexpr int kGroups = kPhysK / kRowsPerFill;

  TL_DEVICE static void Map(int fill_batch, int seg_row_base, int mn_row0,
                            int tx16, int origin_mn, int origin_k, int *dst_row,
                            int *lds_byte) {
    if constexpr (kGroups == 1) {
      // scaleShapeK==4: one 4-row group per MN; fill_batch == mn_local.
      *dst_row = seg_row_base + fill_batch * kPhysK;
      const int mn_idx = ScaleMnIdx<GranularityMN>(mn_row0 + fill_batch, tx16);
      *lds_byte = StorageLayout::CalculateOffset(origin_k / 4,
                                                 origin_mn + mn_idx, 0, 0, 0);
    } else {
      const int mn_local = fill_batch / kGroups;
      const int k_group = fill_batch % kGroups;
      *dst_row = seg_row_base + mn_local * kPhysK + k_group * kRowsPerFill;
      const int mn_idx = ScaleMnIdx<GranularityMN>(mn_row0 + mn_local, tx16);
      *lds_byte = StorageLayout::CalculateOffset(origin_k / 4 + k_group,
                                                 origin_mn + mn_idx, 0, 0, 0);
    }
  }
};

// op_ctrl=2 K2MN2 atom: fill_batch = pair slot (m|m+16) × one K-atom.
// LDS: m0k0 m0k1 m16k0 m16k1 | m1... — each thread +4B.
template <int ParentScaleMN, int ParentScaleK, int ScaleShapeMN,
          int ScaleShapeK, int GranularityMN, int GranularityK, int MmacK,
          typename StorageLayout>
struct DsScaleLdsOpctrl2K2MN2Interleave {
  static_assert(ScaleShapeK % 2 == 0,
                "op_ctrl=2 K2MN2 interleave requires even scaleShapeK");
  static constexpr int kOpCtrl = 2;
  static constexpr int kRowsPerFill = 4;
  static constexpr int kPhysK =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::value;
  static constexpr int kAtoms = kPhysK / 2;

  TL_DEVICE static void Map(int fill_batch, int seg_row_base, int mn_row0,
                            int tx16, int origin_mn, int origin_k, int *dst_row,
                            int *lds_byte) {
    const int mn_pair = fill_batch / kAtoms;
    const int k_atom = fill_batch % kAtoms;
    *dst_row = seg_row_base + fill_batch * kRowsPerFill;
    const int pair =
        origin_mn / 2 + ScaleMnIdx<GranularityMN>(mn_row0 / 2 + mn_pair, tx16);
    *lds_byte = StorageLayout::CalculateOffset(origin_k / 2 + k_atom, pair / 16,
                                               pair % 16, 0, 0);
  }
};

// op_ctrl=2 MN4 atom: fill_batch = quad slot (m|m+16|m+32|m+48).
// LDS: m0k0 m16k0 m32k0 m48k0 | m1... — each thread +4B.
template <int ParentScaleMN, int ParentScaleK, int ScaleShapeMN,
          int ScaleShapeK, int GranularityMN, int GranularityK, int MmacK,
          typename StorageLayout>
struct DsScaleLdsOpctrl2MN4Interleave {
  static constexpr int kOpCtrl = 2;
  static constexpr int kRowsPerFill = 4;
  static constexpr int kPhysK =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::value;

  TL_DEVICE static void Map(int fill_batch, int seg_row_base, int mn_row0,
                            int tx16, int origin_mn, int origin_k, int *dst_row,
                            int *lds_byte) {
    const int mn_quad = fill_batch / kPhysK;
    const int k_rel = fill_batch % kPhysK;
    *dst_row = seg_row_base + fill_batch * kRowsPerFill;
    *lds_byte = StorageLayout::CalculateOffset(
        origin_k + k_rel,
        origin_mn / 4 + ScaleMnIdx<GranularityMN>(mn_row0 / 4 + mn_quad, tx16),
        0, 0, 0);
  }
};

/*!
 * Shared fill: MN seg + TotalWarps schedule only. Policy::Map does index math.
 */
template <int ScaleShapeMN, int ScaleShapeK, int GranularityMN,
          int GranularityK, int MnWarps, int TotalWarps, int MmacK,
          typename Policy>
TL_DEVICE void DsScaleCopyFill(const void *lds_scale, int dst_row_base,
                               int warp_id, int origin_mn, int origin_k) {
  static_assert(MnWarps >= 1 && TotalWarps >= 1, "warps >= 1");
  static_assert(GranularityK % 32 == 0, "GranularityK must be divisible by 32");
  static_assert(TotalWarps % MnWarps == 0,
                "TotalWarps must be divisible by MnWarps");

  constexpr int kFillGroupsPerWarp = 4;
  constexpr int kRowsPerFill = Policy::kRowsPerFill;
  constexpr int kPhysK = Policy::kPhysK;
  constexpr int rows_mn = ScaleRowsMN<ScaleShapeMN, GranularityMN>::value;
  static_assert(rows_mn % MnWarps == 0, "rows_mn must be divisible by MnWarps");
  constexpr int rows_mn_per_seg = rows_mn / MnWarps;
  constexpr int logical_rows_per_seg = rows_mn_per_seg * kPhysK;
  constexpr int aligned_rows_per_seg = ScaleAlign8(logical_rows_per_seg);
  constexpr int copy_warps_per_seg = TotalWarps / MnWarps;
  static_assert(logical_rows_per_seg % kRowsPerFill == 0,
                "logical_rows_per_seg must be divisible by RowsPerFill");

  const int lane = static_cast<int>(threadIdx.x) & 63;
  const int tx16 = lane & 15;
  const int fill_group = lane >> 4;
  auto *src_u8 = reinterpret_cast<const uint8_t *>(lds_scale);

  const int seg = warp_id % MnWarps;
  const int seg_local_warp = warp_id / MnWarps;

  constexpr int num_fill_batches_seg = logical_rows_per_seg / kRowsPerFill;
  constexpr int fill_batches_per_copy_warp =
      (num_fill_batches_seg + copy_warps_per_seg - 1) / copy_warps_per_seg;
  constexpr int full_iters = fill_batches_per_copy_warp / kFillGroupsPerWarp;
  constexpr int rem_fill_groups =
      fill_batches_per_copy_warp % kFillGroupsPerWarp;
  constexpr bool has_ceil_tail =
      (copy_warps_per_seg * fill_batches_per_copy_warp) != num_fill_batches_seg;

  const int seg_row_base = dst_row_base + seg * aligned_rows_per_seg;
  const int mn_row0 = seg * rows_mn_per_seg;

  auto issue_fill_batch = [&](int fill_batch) {
    if constexpr (has_ceil_tail) {
      if (fill_batch >= num_fill_batches_seg) {
        return;
      }
    }
    int dst_row = 0;
    int lds_byte = 0;
    Policy::Map(fill_batch, seg_row_base, mn_row0, tx16, origin_mn, origin_k,
                &dst_row, &lds_byte);
    auto *lds_ptr = reinterpret_cast<TL_LDS_ADDR int *>(
        const_cast<uint8_t *>(src_u8 + lds_byte));
    ds_scale_copy_ds2buf<static_cast<char>(Policy::kOpCtrl)>(lds_ptr, dst_row);
  };

  int fill_batch = seg_local_warp * fill_batches_per_copy_warp + fill_group;
  tl::static_for<0, full_iters, 1>{}([&](auto) {
    issue_fill_batch(fill_batch);
    fill_batch += kFillGroupsPerWarp;
  });

  if constexpr (rem_fill_groups > 0) {
    if (fill_group < rem_fill_groups) {
      issue_fill_batch(fill_batch);
    }
  }
}

/*!
 * Public entry: dispatch by the ScaleView format selected by the kernel.
 * MnWarps = this buffer's side (m_seg / n_seg); TotalWarps = CTA fillers.
 */
template <int OpCtrl, int ScaleFormatId, int ParentScaleMN, int ParentScaleK,
          int ScaleShapeMN, int ScaleShapeK, int GranularityMN,
          int GranularityK, int ScaleKMajor, int MnWarps, int TotalWarps,
          typename StorageLayout =
              ScaleLdsLinearLayout<ParentScaleMN, ParentScaleK, ScaleKMajor>,
          int MmacK = 64>
TL_DEVICE void ds_scale_copy(const void *lds_scale, int dst_row_base,
                             int warp_id, int origin_mn, int origin_k) {
  static_assert(OpCtrl >= 0 && OpCtrl <= 2, "OpCtrl must be 0/1/2");
  static_assert(ScaleFormatId >= 0 && ScaleFormatId <= 5,
                "unknown ScaleFormat");
  constexpr DsScaleFormat ScaleFormat =
      static_cast<DsScaleFormat>(ScaleFormatId);
  static_assert(MmacK == 64 || MmacK == 32, "MmacK must be 64 or 32");
  static_assert(OpCtrl == 0 || ScaleKMajor == 0,
                "op_ctrl!=0 requires MN-major scale (scale_k_major=false)");
  // mmac_k=64 + op_ctrl>0: only gran_k==32 (gran_k>=64 rejects interleave).
  // mmac_k=32 + op_ctrl>0: K-first interleave allowed (gran still %32==0 in
  // Fill).
  static_assert(OpCtrl == 0 || MmacK != 64 || GranularityK == 32,
                "mmac_k=64 op_ctrl>0 requires GranularityK==32");
  static_assert(ParentScaleMN >= ScaleShapeMN && ParentScaleK >= ScaleShapeK,
                "parent scale plane must cover the copied tile");

  if constexpr (ScaleFormat == DsScaleFormat::kIdentity) {
    static_assert(OpCtrl == 0, "identity format requires OpCtrl==0");
    using Policy =
        DsScaleLdsOpctrl0Linear<ParentScaleMN, ParentScaleK, ScaleShapeMN,
                                ScaleShapeK, GranularityMN, GranularityK,
                                ScaleKMajor, MmacK, StorageLayout>;
    DsScaleCopyFill<ScaleShapeMN, ScaleShapeK, GranularityMN, GranularityK,
                    MnWarps, TotalWarps, MmacK, Policy>(
        lds_scale, dst_row_base, warp_id, origin_mn, origin_k);
  } else if constexpr (ScaleFormat == DsScaleFormat::kK2Interleave) {
    static_assert(OpCtrl == 1, "K2 format requires OpCtrl==1");
    static_assert(ScaleShapeK % 2 == 0,
                  "K2 format requires ScaleShapeK divisible by 2");
    using Policy =
        DsScaleLdsOpctrl1K2Interleave<ParentScaleMN, ParentScaleK, ScaleShapeMN,
                                      ScaleShapeK, GranularityMN, GranularityK,
                                      MmacK, StorageLayout>;
    DsScaleCopyFill<ScaleShapeMN, ScaleShapeK, GranularityMN, GranularityK,
                    MnWarps, TotalWarps, MmacK, Policy>(
        lds_scale, dst_row_base, warp_id, origin_mn, origin_k);
  } else if constexpr (ScaleFormat == DsScaleFormat::kMN2Interleave) {
    static_assert(OpCtrl == 1 && MmacK == 32,
                  "MN2 format requires OpCtrl==1 and MmacK==32");
    using Policy =
        DsScaleLdsOpctrl1MN2Interleave<ParentScaleMN, ParentScaleK,
                                       ScaleShapeMN, ScaleShapeK, GranularityMN,
                                       GranularityK, MmacK, StorageLayout>;
    DsScaleCopyFill<ScaleShapeMN, ScaleShapeK, GranularityMN, GranularityK,
                    MnWarps, TotalWarps, MmacK, Policy>(
        lds_scale, dst_row_base, warp_id, origin_mn, origin_k);
  } else if constexpr (ScaleFormat == DsScaleFormat::kK4Interleave) {
    static_assert(OpCtrl == 2, "K4 format requires OpCtrl==2");
    static_assert(ScaleShapeK % 4 == 0,
                  "K4 format requires ScaleShapeK divisible by 4");
    using Policy =
        DsScaleLdsOpctrl2K4Interleave<ParentScaleMN, ParentScaleK, ScaleShapeMN,
                                      ScaleShapeK, GranularityMN, GranularityK,
                                      MmacK, StorageLayout>;
    DsScaleCopyFill<ScaleShapeMN, ScaleShapeK, GranularityMN, GranularityK,
                    MnWarps, TotalWarps, MmacK, Policy>(
        lds_scale, dst_row_base, warp_id, origin_mn, origin_k);
  } else if constexpr (ScaleFormat == DsScaleFormat::kK2MN2Interleave) {
    static_assert(OpCtrl == 2, "K2MN2 format requires OpCtrl==2");
    static_assert(ScaleShapeK % 2 == 0,
                  "K2MN2 format requires even ScaleShapeK");
    using Policy = DsScaleLdsOpctrl2K2MN2Interleave<
        ParentScaleMN, ParentScaleK, ScaleShapeMN, ScaleShapeK, GranularityMN,
        GranularityK, MmacK, StorageLayout>;
    DsScaleCopyFill<ScaleShapeMN, ScaleShapeK, GranularityMN, GranularityK,
                    MnWarps, TotalWarps, MmacK, Policy>(
        lds_scale, dst_row_base, warp_id, origin_mn, origin_k);
  } else if constexpr (ScaleFormat == DsScaleFormat::kMN4Interleave) {
    static_assert(OpCtrl == 2 && MmacK == 32,
                  "MN4 format requires OpCtrl==2 and MmacK==32");
    using Policy =
        DsScaleLdsOpctrl2MN4Interleave<ParentScaleMN, ParentScaleK,
                                       ScaleShapeMN, ScaleShapeK, GranularityMN,
                                       GranularityK, MmacK, StorageLayout>;
    DsScaleCopyFill<ScaleShapeMN, ScaleShapeK, GranularityMN, GranularityK,
                    MnWarps, TotalWarps, MmacK, Policy>(
        lds_scale, dst_row_base, warp_id, origin_mn, origin_k);
  } else {
    static_assert(OpCtrl == -1, "unsupported ScaleFormat for ds_scale_copy");
  }
}

template <int RealABType, typename AVec, typename BVec, typename CVec>
TL_DEVICE CVec mmac_scale_f8f6f4(AVec a, BVec b, CVec c, int off_a, int off_b) {
  static_assert(RealABType >= 0 && RealABType < 25,
                "f8f6f4 RealABType must be in [0, 24]");
  return __builtin_hcu_mmac_scale_f32_16x16x32_f8f6f4_lit_lts(
      a, b, c, RealABType, /*lit=*/true, /*lts=*/false, off_a, off_b);
}

template <int WarpRows, int WarpCols, int NumKAtoms, int GranularityM,
          int GranularityK_A, int GranularityN, int GranularityK_B,
          int ScaleShapeM, int ScaleShapeKA, int ScaleShapeN, int ScaleShapeKB,
          int ScaleFormatA = 0, int ScaleFormatB = 0, int KPack = 1,
          int MmacK = 64, int RealABType = 24>
TL_DEVICE void mmac_scale_fp4_body(const void *a, const void *b, float *c,
                                   int m0_wave_base) {
  constexpr int MMAC_MN = 16;
  constexpr int MMAC_K = MmacK;
  static_assert(KPack > 0, "scale MMAC KPack must be positive");
  static_assert(MMAC_K == 32 || MMAC_K == 64, "scale MMAC K must be 32/64");
  static_assert(GranularityK_A % 32 == 0 && GranularityK_B % 32 == 0,
                "granularity_k % 32 == 0");

  asm volatile("s_mov_b32 m0, %0\n s_nop 0\n" ::"s"(m0_wave_base));
  __builtin_amdgcn_sched_barrier(0);

  using a_vec_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
  using b_vec_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
  using c_vec_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;

  auto *a_v = reinterpret_cast<const a_vec_t *>(a);
  auto *b_v = reinterpret_cast<const b_vec_t *>(b);
  auto *c_v = reinterpret_cast<c_vec_t *>(c);

  tl::static_for<0, NumKAtoms, 1>{}([&](auto ki) {
    tl::static_for<0, KPack, 1>{}([&](auto kp) {
      tl::static_for<0, WarpRows, 1>{}([&](auto i) {
        tl::static_for<0, WarpCols, 1>{}([&](auto j) {
          constexpr int kii = decltype(ki)::value;
          constexpr int kpp = decltype(kp)::value;
          constexpr int ii = decltype(i)::value;
          constexpr int jj = decltype(j)::value;
          constexpr int k_atom = kii * KPack + kpp;
          constexpr int scale_m = (ii * MMAC_MN) / GranularityM;
          constexpr int scale_ka = (k_atom * MMAC_K) / GranularityK_A;
          constexpr int scale_n = (jj * MMAC_MN) / GranularityN;
          constexpr int scale_kb = (k_atom * MMAC_K) / GranularityK_B;
          constexpr int off_a =
              ScaleFormatLogicalToRow<ScaleFormatA, scale_m, scale_ka,
                                      GranularityM, ScaleShapeKA,
                                      GranularityK_A, MMAC_K>::value;
          constexpr int off_b =
              ScaleFormatLogicalToRow<ScaleFormatB, scale_n, scale_kb,
                                      GranularityN, ScaleShapeKB,
                                      GranularityK_B, MMAC_K>::value;
          a_vec_t a_vec = a_v[(kii * WarpRows + ii) * KPack + kpp];
          b_vec_t b_vec = b_v[(kii * WarpCols + jj) * KPack + kpp];
          if constexpr (MMAC_K == 64) {
            static_assert(off_a % 2 == 0 && off_b % 2 == 0,
                          "mmac_k=64 scale offsets must be 2-row aligned");
            c_v[ii * WarpCols + jj] =
                __builtin_hcu_mmac_scale_f32_16x16x64_fp4_lit_lts(
                    a_vec, b_vec, c_v[ii * WarpCols + jj], /*lit=*/true,
                    /*lts=*/false, off_a, off_b);
          } else {
            c_v[ii * WarpCols + jj] = mmac_scale_f8f6f4<RealABType>(
                a_vec, b_vec, c_v[ii * WarpCols + jj], off_a, off_b);
          }
        });
      });
    });
  });
}

} // namespace hcu
} // namespace tl
