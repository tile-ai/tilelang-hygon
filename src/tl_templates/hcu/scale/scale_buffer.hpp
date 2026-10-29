// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file scale_buffer.hpp
 * \brief HCU descriptor-driven scale copy and page-grouped scale MMAC.
 *
 * Each scale buffer is split into MnWarps MN segments (the GEMM's m_seg or
 * n_seg count):
 *   rows_mn = ScaleRowsMN<ScaleShapeMN, GranularityMN>::value
 *   phys_k = ScaleShapeK * ((MmacK == 64 && GranularityK >= 64) ? 2 : 1)
 *   rows_mn_per_seg = rows_mn / MnWarps
 *   logical_rows_per_seg = rows_mn_per_seg * phys_k
 *   physical_rows_per_seg = RowLayout::SegmentRows
 * ScaleRowLayout describes DenseXor, KPhasePacked or FixedBank placement.
 * Copy inverts physical rows to the original format/RowOrder stream before
 * applying its source policy; MMAC maps logical coordinates directly through
 * the same descriptor. Base addresses never encode polarity or layout mode.
 *
 * TotalWarps belong to the copy thread domain, not the consumer GEMM domain.
 * Copy waves round-robin (issue, segment) jobs without filling another
 * segment's alignment padding. When TotalWarps is divisible by MnWarps,
 * each copy wave retains a fixed segment across all issue rounds.
 * MMAC offsets are five-bit immediates relative to an eight-aligned m0 base;
 * large segments are consumed in compile-time page groups at each K atom.
 *
 * ScalePhysK, ScaleLogicalToRow, ds_scale_copy, and mmac_scale_fp4_body share
 * the same MmacK parameter. MmacK=64 uses the FP4 intrinsic; MmacK=32 uses the
 * f8f6f4 intrinsic selected by RealABType.
 *
 * ds_scale_copy dispatches ScaleFormat to a format-specific policy after
 * validating OpCtrl, MmacK, ScaleShapeK, and layout constraints.
 * DsScaleCopyFill schedules fill batches; Policy::Map converts each batch to
 * the original format stream and source LDS byte offset.
 */
#pragma once

#include "scale_row_layout.hpp"
#include <type_traits>

#include <tl_templates/hcu/common.h>
#include <tl_templates/hcu/core.hpp>
#include <tl_templates/hcu/core/numeric/pk_fp4.hpp>

namespace tl {
namespace hcu {

// Model m0 as a named physical register so that scale MMAC's implicit m0 use
// forms a scheduler-visible dependency on this write.
__device__ register unsigned int scale_m0_reg asm("m0");

TL_DEVICE void set_scale_m0(unsigned int value) { scale_m0_reg = value; }

template <char OpCtrl, char Offset0 = 0, char Offset1 = 0>
TL_DEVICE void ds_scale_copy_ds2buf(TL_LDS_ADDR int *lds_data, int soffset) {
  __builtin_hcu_ds_scale_copy_ds2buf(lds_data, soffset, OpCtrl, Offset0,
                                     Offset1);
}

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

enum class ScaleBufferRowOrder : int {
  kKInner = 0,
  kMNInner = 1,
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

/*! Canonical scale-buffer row mapping shared by copy and MMAC.
 *
 * ScaleFormat fixes the atom shape and inner byte/row order.  RowOrder only
 * transposes the two outer atom coordinates; it never splits an atom.  The
 * identity format uses a K2 consumption atom for MmacK=64 even though
 * op_ctrl=0 writes its two rows with separate instructions.
 */
template <int ScaleFormatId, int RowOrderId, int RowsMNPerSeg, int ScaleMNIdx,
          int ScaleKIdx, int GranularityMN, int ScaleShapeK, int GranularityK,
          int MmacK = 64>
struct ScaleBufferLogicalToRow {
  static constexpr DsScaleFormat format =
      static_cast<DsScaleFormat>(ScaleFormatId);
  static constexpr ScaleBufferRowOrder order =
      static_cast<ScaleBufferRowOrder>(RowOrderId);
  static constexpr int mn_row = ScaleMNToRow<ScaleMNIdx, GranularityMN>::value;
  static constexpr int k_dup =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::k_dup;
  static constexpr int phys_k =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::value;
  static constexpr int k_phys = ScaleKIdx * k_dup;
  static constexpr int atom_mn =
      format == DsScaleFormat::kMN4Interleave
          ? 4
          : ((format == DsScaleFormat::kMN2Interleave ||
              format == DsScaleFormat::kK2MN2Interleave)
                 ? 2
                 : 1);
  static constexpr int atom_k =
      format == DsScaleFormat::kK4Interleave
          ? 4
          : ((format == DsScaleFormat::kK2Interleave ||
              format == DsScaleFormat::kK2MN2Interleave ||
              (format == DsScaleFormat::kIdentity && MmacK == 64))
                 ? 2
                 : 1);
  static_assert(RowsMNPerSeg % atom_mn == 0,
                "scale MN segment must contain complete atoms");
  static_assert(phys_k % atom_k == 0,
                "scale K extent must contain complete atoms");
  static constexpr int mn_atoms = RowsMNPerSeg / atom_mn;
  static constexpr int k_atoms = phys_k / atom_k;
  static constexpr int mn_atom = mn_row / atom_mn;
  static constexpr int k_atom = k_phys / atom_k;
  static constexpr int inner_mn = mn_row % atom_mn;
  static constexpr int inner_k = k_phys % atom_k;
  static constexpr int inner = inner_mn * atom_k + inner_k;
  static constexpr int atom_ordinal = order == ScaleBufferRowOrder::kMNInner
                                          ? k_atom * mn_atoms + mn_atom
                                          : mn_atom * k_atoms + k_atom;
  static constexpr int value = atom_ordinal * atom_mn * atom_k + inner;
};

template <int RowOrderId>
TL_DEVICE void DecodeScaleAtomOrdinal(int ordinal, int mn_atoms, int k_atoms,
                                      int *mn_atom, int *k_atom) {
  if constexpr (static_cast<ScaleBufferRowOrder>(RowOrderId) ==
                ScaleBufferRowOrder::kMNInner) {
    *k_atom = ordinal / mn_atoms;
    *mn_atom = ordinal % mn_atoms;
  } else {
    *mn_atom = ordinal / k_atoms;
    *k_atom = ordinal % k_atoms;
  }
}

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
  TL_DEVICE static int CalculateOffset(int i0, int i1, int, int, int) {
    if constexpr (ScaleKMajor) {
      return i0 * ParentScaleK + i1;
    } else {
      return i0 * ParentScaleMN + i1;
    }
  }
};

// StorageLayout follows the physical LDS Buffer dimension order.  Scale copy
// policies use logical (k, mn) coordinates, so reverse the first two inputs for
// a plain Identity K-major buffer whose physical shape is [MN, K].
template <int ScaleKMajor, typename StorageLayout>
TL_DEVICE int ScaleLdsOffset(int k, int mn, int i2 = 0, int i3 = 0,
                             int i4 = 0) {
  if constexpr (ScaleKMajor) {
    return StorageLayout::CalculateOffset(mn, k, i2, i3, i4);
  } else {
    return StorageLayout::CalculateOffset(k, mn, i2, i3, i4);
  }
}

// LDS offsets are computed in the parent last-2 plane.  The pointer is its
// base; origin_mn/origin_k identify the copied tile inside that plane.
template <int ParentScaleMN, int ParentScaleK, int ScaleShapeMN,
          int ScaleShapeK, int GranularityMN, int GranularityK, int ScaleKMajor,
          int MmacK, int RowOrder, typename StorageLayout>
struct DsScaleLdsOpctrl0Linear {
  static constexpr int kOpCtrl = 0;
  static constexpr int kRowsPerFill = 1;
  static constexpr int kPhysK =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::value;
  static constexpr int kDup =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::k_dup;

  TL_DEVICE static void Map(int fill_batch, int seg_row_base, int mn_row0,
                            int rows_mn_per_seg, int tx16, int origin_mn,
                            int origin_k, int *dst_row, int *lds_byte) {
    *dst_row = seg_row_base + fill_batch;
    constexpr int atom_k = MmacK == 64 ? 2 : 1;
    constexpr int k_atoms = kPhysK / atom_k;
    const int atom_ordinal = fill_batch / atom_k;
    const int inner = fill_batch % atom_k;
    int mn_local = 0, k_atom = 0;
    DecodeScaleAtomOrdinal<RowOrder>(atom_ordinal, rows_mn_per_seg, k_atoms,
                                     &mn_local, &k_atom);
    const int mn_idx = ScaleMnIdx<GranularityMN>(mn_row0 + mn_local, tx16);
    const int k_rel = (k_atom * atom_k + inner) / kDup;
    *lds_byte = ScaleLdsOffset<ScaleKMajor, StorageLayout>(origin_k + k_rel,
                                                           origin_mn + mn_idx);
  }
};

// A linear Identity [MN, K] LDS buffer already places adjacent K scales in
// consecutive bytes.  op_ctrl=1/2 can therefore transfer 2/4 distinct scales
// directly, without a K2/K4 physical ScaleView reformat.
template <int OpCtrl, int ParentScaleMN, int ParentScaleK, int ScaleShapeMN,
          int ScaleShapeK, int GranularityMN, int GranularityK, int MmacK,
          int RowOrder, typename StorageLayout>
struct DsScaleLdsKMajorIdentity {
  static_assert(OpCtrl == 1 || OpCtrl == 2,
                "K-major Identity policy requires op_ctrl=1/2");
  static constexpr int kOpCtrl = OpCtrl;
  static constexpr int kRowsPerFill = 1 << OpCtrl;
  static constexpr int kDup =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::k_dup;
  static_assert(kDup == 1,
                "K-major op_ctrl=1/2 cannot duplicate a reused K scale");
  static constexpr int kPhysK = ScaleShapeK;
  static_assert(ScaleShapeK % kRowsPerFill == 0,
                "K-major ScaleShapeK must be divisible by op_ctrl width");
  static constexpr int kGroups = ScaleShapeK / kRowsPerFill;

  TL_DEVICE static void Map(int fill_batch, int seg_row_base, int mn_row0,
                            int rows_mn_per_seg, int tx16, int origin_mn,
                            int origin_k, int *dst_row, int *lds_byte) {
    int mn_local = 0, k_group = 0;
    DecodeScaleAtomOrdinal<RowOrder>(fill_batch, rows_mn_per_seg, kGroups,
                                     &mn_local, &k_group);
    *dst_row = seg_row_base + fill_batch * kRowsPerFill;
    const int mn_idx = ScaleMnIdx<GranularityMN>(mn_row0 + mn_local, tx16);
    *lds_byte = ScaleLdsOffset<1, StorageLayout>(
        origin_k + k_group * kRowsPerFill, origin_mn + mn_idx);
  }
};

// op_ctrl=1 pure K: fill_batch = mn_local * k_atoms + k_atom.
// LDS: m0k0 m0k1 | m1k0 m1k1 | ... then next k_atom: m0k2 m0k3 | ...
template <int ParentScaleMN, int ParentScaleK, int ScaleShapeMN,
          int ScaleShapeK, int GranularityMN, int GranularityK, int MmacK,
          int RowOrder, typename StorageLayout>
struct DsScaleLdsOpctrl1K2Interleave {
  static_assert(ScaleShapeK % 2 == 0,
                "op_ctrl=1 K2 interleave requires scaleShapeK % 2 == 0");
  static constexpr int kOpCtrl = 1;
  static constexpr int kRowsPerFill = 2;
  static constexpr int kPhysK =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::value;
  static constexpr int kAtoms = kPhysK / kRowsPerFill;

  TL_DEVICE static void Map(int fill_batch, int seg_row_base, int mn_row0,
                            int rows_mn_per_seg, int tx16, int origin_mn,
                            int origin_k, int *dst_row, int *lds_byte) {
    int mn_local = 0, k_atom = 0;
    DecodeScaleAtomOrdinal<RowOrder>(fill_batch, rows_mn_per_seg, kAtoms,
                                     &mn_local, &k_atom);
    *dst_row = seg_row_base + fill_batch * kRowsPerFill;
    const int mn_idx = ScaleMnIdx<GranularityMN>(mn_row0 + mn_local, tx16);
    *lds_byte = StorageLayout::CalculateOffset(origin_k / 2 + k_atom,
                                               origin_mn + mn_idx, 0, 0, 0);
  }
};

// op_ctrl=1 MN2 atom: fill_batch = pair slot along MN (m | m+16).
// LDS: m0k0 m16k0 | m1k0 m17k0 | ...  — each thread +2B from pair slot.
template <int ParentScaleMN, int ParentScaleK, int ScaleShapeMN,
          int ScaleShapeK, int GranularityMN, int GranularityK, int MmacK,
          int RowOrder, typename StorageLayout>
struct DsScaleLdsOpctrl1MN2Interleave {
  static constexpr int kOpCtrl = 1;
  static constexpr int kRowsPerFill = 2;
  static constexpr int kPhysK =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::value;

  TL_DEVICE static void Map(int fill_batch, int seg_row_base, int mn_row0,
                            int rows_mn_per_seg, int tx16, int origin_mn,
                            int origin_k, int *dst_row, int *lds_byte) {
    int mn_pair = 0, k_rel = 0;
    DecodeScaleAtomOrdinal<RowOrder>(fill_batch, rows_mn_per_seg / 2, kPhysK,
                                     &mn_pair, &k_rel);
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
          int RowOrder, typename StorageLayout>
struct DsScaleLdsOpctrl2K4Interleave {
  static_assert(ScaleShapeK % 4 == 0,
                "op_ctrl=2 K4 interleave requires scaleShapeK % 4 == 0");
  static constexpr int kOpCtrl = 2;
  static constexpr int kRowsPerFill = 4;
  static constexpr int kPhysK =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::value;
  static constexpr int kGroups = kPhysK / kRowsPerFill;

  TL_DEVICE static void Map(int fill_batch, int seg_row_base, int mn_row0,
                            int rows_mn_per_seg, int tx16, int origin_mn,
                            int origin_k, int *dst_row, int *lds_byte) {
    int mn_local = 0, k_group = 0;
    DecodeScaleAtomOrdinal<RowOrder>(fill_batch, rows_mn_per_seg, kGroups,
                                     &mn_local, &k_group);
    *dst_row = seg_row_base + fill_batch * kRowsPerFill;
    const int mn_idx = ScaleMnIdx<GranularityMN>(mn_row0 + mn_local, tx16);
    *lds_byte = StorageLayout::CalculateOffset(origin_k / 4 + k_group,
                                               origin_mn + mn_idx, 0, 0, 0);
  }
};

// op_ctrl=2 K2MN2 atom: fill_batch = pair slot (m|m+16) × one K-atom.
// LDS: m0k0 m0k1 m16k0 m16k1 | m1... — each thread +4B.
template <int ParentScaleMN, int ParentScaleK, int ScaleShapeMN,
          int ScaleShapeK, int GranularityMN, int GranularityK, int MmacK,
          int RowOrder, typename StorageLayout>
struct DsScaleLdsOpctrl2K2MN2Interleave {
  static_assert(ScaleShapeK % 2 == 0,
                "op_ctrl=2 K2MN2 interleave requires even scaleShapeK");
  static constexpr int kOpCtrl = 2;
  static constexpr int kRowsPerFill = 4;
  static constexpr int kPhysK =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::value;
  static constexpr int kAtoms = kPhysK / 2;

  TL_DEVICE static void Map(int fill_batch, int seg_row_base, int mn_row0,
                            int rows_mn_per_seg, int tx16, int origin_mn,
                            int origin_k, int *dst_row, int *lds_byte) {
    int mn_pair = 0, k_atom = 0;
    DecodeScaleAtomOrdinal<RowOrder>(fill_batch, rows_mn_per_seg / 2, kAtoms,
                                     &mn_pair, &k_atom);
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
          int RowOrder, typename StorageLayout>
struct DsScaleLdsOpctrl2MN4Interleave {
  static constexpr int kOpCtrl = 2;
  static constexpr int kRowsPerFill = 4;
  static constexpr int kPhysK =
      ScalePhysK<ScaleShapeK, GranularityK, MmacK>::value;

  TL_DEVICE static void Map(int fill_batch, int seg_row_base, int mn_row0,
                            int rows_mn_per_seg, int tx16, int origin_mn,
                            int origin_k, int *dst_row, int *lds_byte) {
    int mn_quad = 0, k_rel = 0;
    DecodeScaleAtomOrdinal<RowOrder>(fill_batch, rows_mn_per_seg / 4, kPhysK,
                                     &mn_quad, &k_rel);
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
          typename Policy, typename RowLayout>
TL_DEVICE void DsScaleCopyFill(const void *lds_scale, int dst_row_base,
                               int warp_id, int origin_mn, int origin_k) {
  static_assert(MnWarps >= 1 && TotalWarps >= 1, "warps >= 1");
  static_assert(GranularityK % 32 == 0, "GranularityK must be divisible by 32");

  constexpr int kRowsPerFill = Policy::kRowsPerFill;
  constexpr int kPhysK = Policy::kPhysK;
  constexpr int rows_mn = ScaleRowsMN<ScaleShapeMN, GranularityMN>::value;
  static_assert(rows_mn % MnWarps == 0, "rows_mn must be divisible by MnWarps");
  constexpr int rows_mn_per_seg = rows_mn / MnWarps;
  constexpr int logical_rows_per_seg = rows_mn_per_seg * kPhysK;
  constexpr auto layout = RowLayout::descriptor;
  static_assert(logical_rows_per_seg % kRowsPerFill == 0,
                "logical_rows_per_seg must be divisible by RowsPerFill");

  const int lane = static_cast<int>(threadIdx.x) & 63;
  const int tx16 = lane & 15;
  const int fill_group = lane >> 4;
  // Convert the shared base once, before adding per-issue byte offsets.
  // Keeping arithmetic in the LDS address space avoids repeated generic-to-LDS
  // conversions obscuring common lane addresses after issue unrolling. Do not
  // assume the storage layout is affine: Policy::Map still computes each
  // offset.
  auto *src_u8 =
      reinterpret_cast<TL_LDS_ADDR uint8_t *>(const_cast<void *>(lds_scale));

  constexpr int groups_per_issue = layout.CopyIssueGroups(kRowsPerFill);
  constexpr int chunks_per_issue = layout.CopyIssueChunks(kRowsPerFill);
  constexpr int chunks_per_seg = layout.CopyChunkCount();
  constexpr int issues_per_seg =
      (chunks_per_seg + chunks_per_issue - 1) / chunks_per_issue;
  constexpr int total_issues = issues_per_seg * MnWarps;
  constexpr int iterations = (total_issues + TotalWarps - 1) / TotalWarps;
  constexpr bool partial_round = total_issues % TotalWarps != 0;
  const int work_rank =
      partial_round ? ScaleCopyWaveRank(warp_id, TotalWarps) : warp_id;

  // soffset is wave-uniform. Combine physically compatible groups in one
  // issue; inactive groups must never write padding or the other bank.
  tl::static_for<0, iterations, 1>{}([&](auto iteration) {
    // Flatten (issue, consumer segment) work across the copy scope. Even a
    // single copy wave can fill multiple consumer segments; the two domains
    // need not have the same wave count or divide one another.
    const int job = work_rank + iteration * TotalWarps;
    const int seg = job % MnWarps;
    const int ordinal = job / MnWarps;
    const int seg_row_base = dst_row_base + RowLayout::SegmentOffset(seg);
    const int mn_row0 = seg * rows_mn_per_seg;
    const int chunk = layout.CopyChunk(ordinal * chunks_per_issue);
    const int physical_row = chunk * 4 + fill_group * kRowsPerFill;
    constexpr bool full_chunks = layout.CopyHalfRows(0) % 4 == 0 &&
                                 layout.CopyHalfRows(1) % 4 == 0 &&
                                 chunks_per_seg % chunks_per_issue == 0;
    constexpr bool fixed_k4 =
        layout.mode == ScaleRowMode::FixedBank && kRowsPerFill == 4;
    // Each useful half-chunk is assigned once. The only lane predicate is
    // the instruction group mask (and a genuine tail, if one exists).
    // In particular never form a -1 raw row then divide it as a signed int.
    if (job < total_issues && fill_group < groups_per_issue &&
        (!fixed_k4 || fill_group % 2 == 0) &&
        (full_chunks ||
         layout.CopyGroupValid(physical_row / 4, physical_row % 4))) {
      const unsigned raw_row = layout.CopyRawRow(physical_row);
      const int fill_batch = raw_row / kRowsPerFill;
      int logical_row = 0;
      int lds_byte = 0;
      Policy::Map(fill_batch, 0, mn_row0, rows_mn_per_seg, tx16, origin_mn,
                  origin_k, &logical_row, &lds_byte);
      auto *lds_ptr = reinterpret_cast<TL_LDS_ADDR int *>(src_u8 + lds_byte);
      if constexpr (fixed_k4 && iterations > 1 && TotalWarps % MnWarps == 0) {
        // Anchor repeated K4 issues at the first valid lane address. Keep the
        // exact per-issue difference: an explicit source layout may be XORed,
        // so a constant linear stride must not be assumed. Evaluate the anchor
        // inside the active-group guard, never for padding or idle waves.
        const int first_chunk =
            layout.CopyChunk((work_rank / MnWarps) * chunks_per_issue);
        const unsigned first_raw =
            layout.CopyRawRow(first_chunk * 4 + fill_group * kRowsPerFill);
        int first_logical_row = 0;
        int first_lds_byte = 0;
        Policy::Map(first_raw / kRowsPerFill, 0, mn_row0, rows_mn_per_seg, tx16,
                    origin_mn, origin_k, &first_logical_row, &first_lds_byte);
        auto *lane_src = src_u8 + first_lds_byte;
        lds_ptr = reinterpret_cast<TL_LDS_ADDR int *>(
            lane_src + (lds_byte - first_lds_byte));
      }
      ds_scale_copy_ds2buf<static_cast<char>(Policy::kOpCtrl)>(
          lds_ptr, seg_row_base + chunk * 4);
    }
  });
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
          int MmacK = 64, int RowOrder = 0, typename RowLayout = void>
TL_DEVICE void ds_scale_copy(const void *lds_scale, int dst_row_base,
                             int warp_id, int origin_mn, int origin_k) {
  static_assert(OpCtrl >= 0 && OpCtrl <= 2, "OpCtrl must be 0/1/2");
  static_assert(ScaleFormatId >= 0 && ScaleFormatId <= 5,
                "unknown ScaleFormat");
  constexpr DsScaleFormat ScaleFormat =
      static_cast<DsScaleFormat>(ScaleFormatId);
  static_assert(MmacK == 64 || MmacK == 32, "MmacK must be 64 or 32");
  static_assert(OpCtrl == 0 || ScaleKMajor == 0 ||
                    ScaleFormat == DsScaleFormat::kIdentity,
                "K-major op_ctrl=1/2 requires Identity LDS format");
  static_assert(ScaleKMajor == 0 || OpCtrl == 0 || RowOrder == 0,
                "K-major op_ctrl=1/2 requires KInner row order");
  // mmac_k=64 + op_ctrl>0: only gran_k==32 (gran_k>=64 rejects interleave).
  // mmac_k=32 + op_ctrl>0: K-first interleave allowed (gran still %32==0 in
  // Fill).
  static_assert(OpCtrl == 0 || MmacK != 64 || GranularityK == 32,
                "mmac_k=64 op_ctrl>0 requires GranularityK==32");
  static_assert(ParentScaleMN >= ScaleShapeMN && ParentScaleK >= ScaleShapeK,
                "parent scale plane must cover the copied tile");

  if constexpr (ScaleFormat == DsScaleFormat::kIdentity) {
    if constexpr (OpCtrl == 0) {
      using Policy =
          DsScaleLdsOpctrl0Linear<ParentScaleMN, ParentScaleK, ScaleShapeMN,
                                  ScaleShapeK, GranularityMN, GranularityK,
                                  ScaleKMajor, MmacK, RowOrder, StorageLayout>;
      DsScaleCopyFill<ScaleShapeMN, ScaleShapeK, GranularityMN, GranularityK,
                      MnWarps, TotalWarps, MmacK, Policy, RowLayout>(
          lds_scale, dst_row_base, warp_id, origin_mn, origin_k);
    } else {
      static_assert(ScaleKMajor == 1,
                    "Identity op_ctrl=1/2 requires K-major scale");
      using Policy = DsScaleLdsKMajorIdentity<
          OpCtrl, ParentScaleMN, ParentScaleK, ScaleShapeMN, ScaleShapeK,
          GranularityMN, GranularityK, MmacK, RowOrder, StorageLayout>;
      DsScaleCopyFill<ScaleShapeMN, ScaleShapeK, GranularityMN, GranularityK,
                      MnWarps, TotalWarps, MmacK, Policy, RowLayout>(
          lds_scale, dst_row_base, warp_id, origin_mn, origin_k);
    }
  } else if constexpr (ScaleFormat == DsScaleFormat::kK2Interleave) {
    static_assert(OpCtrl == 1, "K2 format requires OpCtrl==1");
    static_assert(ScaleShapeK % 2 == 0,
                  "K2 format requires ScaleShapeK divisible by 2");
    using Policy =
        DsScaleLdsOpctrl1K2Interleave<ParentScaleMN, ParentScaleK, ScaleShapeMN,
                                      ScaleShapeK, GranularityMN, GranularityK,
                                      MmacK, RowOrder, StorageLayout>;
    DsScaleCopyFill<ScaleShapeMN, ScaleShapeK, GranularityMN, GranularityK,
                    MnWarps, TotalWarps, MmacK, Policy, RowLayout>(
        lds_scale, dst_row_base, warp_id, origin_mn, origin_k);
  } else if constexpr (ScaleFormat == DsScaleFormat::kMN2Interleave) {
    static_assert(OpCtrl == 1 && MmacK == 32,
                  "MN2 format requires OpCtrl==1 and MmacK==32");
    using Policy = DsScaleLdsOpctrl1MN2Interleave<
        ParentScaleMN, ParentScaleK, ScaleShapeMN, ScaleShapeK, GranularityMN,
        GranularityK, MmacK, RowOrder, StorageLayout>;
    DsScaleCopyFill<ScaleShapeMN, ScaleShapeK, GranularityMN, GranularityK,
                    MnWarps, TotalWarps, MmacK, Policy, RowLayout>(
        lds_scale, dst_row_base, warp_id, origin_mn, origin_k);
  } else if constexpr (ScaleFormat == DsScaleFormat::kK4Interleave) {
    static_assert(OpCtrl == 2, "K4 format requires OpCtrl==2");
    static_assert(ScaleShapeK % 4 == 0,
                  "K4 format requires ScaleShapeK divisible by 4");
    using Policy =
        DsScaleLdsOpctrl2K4Interleave<ParentScaleMN, ParentScaleK, ScaleShapeMN,
                                      ScaleShapeK, GranularityMN, GranularityK,
                                      MmacK, RowOrder, StorageLayout>;
    DsScaleCopyFill<ScaleShapeMN, ScaleShapeK, GranularityMN, GranularityK,
                    MnWarps, TotalWarps, MmacK, Policy, RowLayout>(
        lds_scale, dst_row_base, warp_id, origin_mn, origin_k);
  } else if constexpr (ScaleFormat == DsScaleFormat::kK2MN2Interleave) {
    static_assert(OpCtrl == 2, "K2MN2 format requires OpCtrl==2");
    static_assert(ScaleShapeK % 2 == 0,
                  "K2MN2 format requires even ScaleShapeK");
    using Policy = DsScaleLdsOpctrl2K2MN2Interleave<
        ParentScaleMN, ParentScaleK, ScaleShapeMN, ScaleShapeK, GranularityMN,
        GranularityK, MmacK, RowOrder, StorageLayout>;
    DsScaleCopyFill<ScaleShapeMN, ScaleShapeK, GranularityMN, GranularityK,
                    MnWarps, TotalWarps, MmacK, Policy, RowLayout>(
        lds_scale, dst_row_base, warp_id, origin_mn, origin_k);
  } else if constexpr (ScaleFormat == DsScaleFormat::kMN4Interleave) {
    static_assert(OpCtrl == 2 && MmacK == 32,
                  "MN4 format requires OpCtrl==2 and MmacK==32");
    using Policy = DsScaleLdsOpctrl2MN4Interleave<
        ParentScaleMN, ParentScaleK, ScaleShapeMN, ScaleShapeK, GranularityMN,
        GranularityK, MmacK, RowOrder, StorageLayout>;
    DsScaleCopyFill<ScaleShapeMN, ScaleShapeK, GranularityMN, GranularityK,
                    MnWarps, TotalWarps, MmacK, Policy, RowLayout>(
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
          int ScaleShapeKA, int ScaleShapeKB, int KPack, int MmacK,
          int RealABType, typename RowLayoutA, typename RowLayoutB,
          bool ClearAccum = false>
TL_DEVICE void mmac_scale_fp4_body(const void *a, const void *b, float *c,
                                   int m0_wave_base) {
  constexpr int MMAC_MN = 16;
  constexpr int MMAC_K = MmacK;
  static_assert(KPack > 0, "scale MMAC KPack must be positive");
  static_assert(MMAC_K == 32 || MMAC_K == 64, "scale MMAC K must be 32/64");
  static_assert(GranularityK_A % 32 == 0 && GranularityK_B % 32 == 0,
                "granularity_k % 32 == 0");
  constexpr int rows_mn_a =
      GranularityM >= MMAC_MN ? (WarpRows * MMAC_MN / GranularityM) : WarpRows;
  constexpr int rows_mn_b =
      GranularityN >= MMAC_MN ? (WarpCols * MMAC_MN / GranularityN) : WarpCols;
  static_assert(rows_mn_a > 0 && rows_mn_b > 0,
                "each MMAC wave must consume at least one scale MN row");
  // Hardware requires BOTH m0 row fields to be eight-aligned. The lowering
  // checks the allocated base; this check preserves alignment for every wave.
  // Bank polarity stays in RowOffset, not in the base (in particular no +4).
  static_assert(RowLayoutA::SegmentRows % 8 == 0 &&
                    RowLayoutB::SegmentRows % 8 == 0,
                "scale m0 segment strides must be 8-row aligned");

  using a_vec_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
  using b_vec_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
  using c_vec_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;

  auto *a_v = reinterpret_cast<const a_vec_t *>(a);
  auto *b_v = reinterpret_cast<const b_vec_t *>(b);
  auto *c_v = reinterpret_cast<c_vec_t *>(c);

  // Canonical addressing uses the actual consumer shape and row order, not
  // an assumed 2x2 wave partition. The ordinal is a compile-time ki/kp/i/j
  // coordinate; it does not dictate the paged path's M/N emission order.
  constexpr auto physical_offsets = [](auto ordinal) {
    constexpr int atom = decltype(ordinal)::value;
    constexpr int k_atom = atom / (WarpRows * WarpCols);
    constexpr int i = atom / WarpCols % WarpRows;
    constexpr int j = atom % WarpCols;
    constexpr int mn_a =
        ScaleMNToRow<i * MMAC_MN / GranularityM, GranularityM>::value;
    constexpr int mn_b =
        ScaleMNToRow<j * MMAC_MN / GranularityN, GranularityN>::value;
    constexpr int dup_a =
        ScalePhysK<ScaleShapeKA, GranularityK_A, MMAC_K>::k_dup;
    constexpr int dup_b =
        ScalePhysK<ScaleShapeKB, GranularityK_B, MMAC_K>::k_dup;
    constexpr int physical_a =
        RowLayoutA::RowOffset(mn_a, k_atom * MMAC_K / GranularityK_A * dup_a);
    constexpr int physical_b =
        RowLayoutB::RowOffset(mn_b, k_atom * MMAC_K / GranularityK_B * dup_b);
    return physical_a | (physical_b << 16);
  };

  // All layout modes use the same five-bit immediate protection.
  constexpr int pages_a = (RowLayoutA::SegmentRows + 31) / 32;
  constexpr int pages_b = (RowLayoutB::SegmentRows + 31) / 32;
  static_assert(pages_a <= 16 && pages_b <= 16,
                "scale segment exceeds the 512-row physical address space");

  // Hoist page selection across K only after proving that each output atom
  // stays in the same A/B page pair for the entire reduction. Independent
  // outputs can then be grouped by page while preserving every accumulator's
  // exact ki/kp order. This is not generally valid for K-dependent layouts.
  constexpr bool pages_stable_across_k = [=]() {
    bool stable = true;
    tl::static_for<0, NumKAtoms * KPack, 1>{}([&](auto k) {
      tl::static_for<0, WarpRows * WarpCols, 1>{}([&](auto mn) {
        constexpr int first = decltype(mn)::value;
        constexpr int ordinal =
            decltype(k)::value * WarpRows * WarpCols + first;
        constexpr int initial =
            physical_offsets(std::integral_constant<int, first>{});
        constexpr int current =
            physical_offsets(std::integral_constant<int, ordinal>{});
        stable &= (initial & ~0x001f001f) == (current & ~0x001f001f);
      });
    });
    return stable;
  }();

  constexpr int outer_pages = pages_stable_across_k ? pages_a * pages_b : 1;
  tl::static_for<0, outer_pages, 1>{}([&](auto page) {
    constexpr int outer_delta = (decltype(page)::value / pages_b * 32) |
                                (decltype(page)::value % pages_b * 32 << 16);
    constexpr bool occupied = [=]() {
      if constexpr (!pages_stable_across_k)
        return true;
      bool found = false;
      tl::static_for<0, WarpRows * WarpCols, 1>{}([&](auto mn) {
        constexpr int offsets = physical_offsets(mn);
        found |= (offsets & ~0x001f001f) == outer_delta;
      });
      return found;
    }();
    if constexpr (occupied) {
      if constexpr (pages_stable_across_k) {
        static_assert((outer_delta & 0x00070007) == 0,
                      "scale paging must preserve both m0 alignments");
        set_scale_m0(m0_wave_base + outer_delta);
      }

      tl::static_for<0, NumKAtoms, 1>{}([&](auto ki) {
        tl::static_for<0, KPack, 1>{}([&](auto kp) {
          constexpr int k_atom =
              decltype(ki)::value * KPack + decltype(kp)::value;
          auto emit_atom = [&](auto i, auto j) {
            constexpr int kii = decltype(ki)::value;
            constexpr int kpp = decltype(kp)::value;
            constexpr int ii = decltype(i)::value;
            constexpr int jj = decltype(j)::value;
            constexpr int ordinal = (k_atom * WarpRows + ii) * WarpCols + jj;
            constexpr int offsets =
                physical_offsets(std::integral_constant<int, ordinal>{});
            // Scale offset modifiers encode only five bits. Move each 32-row
            // page into the eight-aligned m0 base, retaining bank selection in
            // the immediate. All page arithmetic here is compile-time constant.
            constexpr int off_a = offsets & 31;
            constexpr int off_b = (offsets >> 16) & 31;
            a_vec_t a_vec = a_v[(kii * WarpRows + ii) * KPack + kpp];
            b_vec_t b_vec = b_v[(kii * WarpCols + jj) * KPack + kpp];
            // Each output atom starts from zero only on its first K issue,
            // including when scale pages reorder independent M/N atoms.
            c_vec_t accumulator;
            if constexpr (ClearAccum && k_atom == 0) {
              accumulator = c_vec_t{};
            } else {
              accumulator = c_v[ii * WarpCols + jj];
            }
            if constexpr (MMAC_K == 64) {
              static_assert(off_a % 2 == 0 && off_b % 2 == 0,
                            "mmac_k=64 scale offsets must be 2-row aligned");
              c_v[ii * WarpCols + jj] =
                  __builtin_hcu_mmac_scale_f32_16x16x64_fp4_lit_lts(
                      a_vec, b_vec, accumulator, /*lit=*/true,
                      /*lts=*/false, off_a, off_b);
            } else {
              c_v[ii * WarpCols + jj] = mmac_scale_f8f6f4<RealABType>(
                  a_vec, b_vec, accumulator, off_a, off_b);
            }
          };
          if constexpr (pages_stable_across_k) {
            tl::static_for<0, WarpRows, 1>{}([&](auto i) {
              tl::static_for<0, WarpCols, 1>{}([&](auto j) {
                constexpr int ordinal =
                    (k_atom * WarpRows + decltype(i)::value) * WarpCols +
                    decltype(j)::value;
                constexpr int offsets =
                    physical_offsets(std::integral_constant<int, ordinal>{});
                if constexpr ((offsets & ~0x001f001f) == outer_delta)
                  emit_atom(i, j);
              });
            });
          } else {
            // Determine occupied pages independently for this K atom. A depends
            // only on i, B only on j; their occupied-page Cartesian product is
            // exactly the set of nonempty MMAC groups. No runtime masks or
            // tests.
            constexpr unsigned a_mask = [=]() {
              unsigned mask = 0;
              tl::static_for<0, WarpRows, 1>{}([&](auto i) {
                constexpr int ordinal =
                    (k_atom * WarpRows + decltype(i)::value) * WarpCols;
                constexpr int offset =
                    physical_offsets(std::integral_constant<int, ordinal>{}) &
                    0xffff;
                mask |= 1u << (offset / 32);
              });
              return mask;
            }();
            constexpr unsigned b_mask = [=]() {
              unsigned mask = 0;
              tl::static_for<0, WarpCols, 1>{}([&](auto j) {
                constexpr int ordinal =
                    k_atom * WarpRows * WarpCols + decltype(j)::value;
                constexpr int offset =
                    physical_offsets(std::integral_constant<int, ordinal>{}) >>
                    16;
                mask |= 1u << (offset / 32);
              });
              return mask;
            }();
            // Page grouping is outside the M/N compute loops. Reorder only
            // independent output atoms at fixed K: every accumulator retains
            // exactly the original ki/kp reduction order. Moving a page loop
            // outside K would break that invariant for some ScaleFormats.
            tl::static_for<0, pages_a, 1>{}([&](auto pa) {
              constexpr int page_a = decltype(pa)::value;
              if constexpr ((a_mask & (1u << page_a)) != 0) {
                tl::static_for<0, pages_b, 1>{}([&](auto pb) {
                  constexpr int page_b = decltype(pb)::value;
                  if constexpr ((b_mask & (1u << page_b)) != 0) {
                    constexpr int page_delta =
                        (page_a * 32) | (page_b * 32 << 16);
                    static_assert(
                        (page_delta & 0x00070007) == 0,
                        "scale paging must preserve both m0 alignments");
                    // The first group of this K atom may reuse the final
                    // page of the preceding atom. Compare compile-time page
                    // values, not a runtime m0 shadow or an assumed K stride.
                    constexpr bool reuse_previous = [=]() {
                      if constexpr (k_atom == 0)
                        return false;
                      constexpr bool first =
                          (a_mask & ((1u << page_a) - 1)) == 0 &&
                          (b_mask & ((1u << page_b) - 1)) == 0;
                      int previous_last = 0;
                      if constexpr (k_atom > 0 && first) {
                        tl::static_for<0, WarpRows * WarpCols, 1>{}(
                            [&](auto mn) {
                              constexpr int ordinal =
                                  (k_atom - 1) * WarpRows * WarpCols +
                                  decltype(mn)::value;
                              constexpr int delta =
                                  physical_offsets(
                                      std::integral_constant<int, ordinal>{}) &
                                  ~0x001f001f;
                              previous_last =
                                  delta > previous_last ? delta : previous_last;
                            });
                      }
                      return first && previous_last == page_delta;
                    }();
                    if constexpr (!reuse_previous)
                      set_scale_m0(m0_wave_base + page_delta);
                    tl::static_for<0, WarpRows, 1>{}([&](auto i) {
                      tl::static_for<0, WarpCols, 1>{}([&](auto j) {
                        constexpr int ordinal =
                            (k_atom * WarpRows + decltype(i)::value) *
                                WarpCols +
                            decltype(j)::value;
                        constexpr int offsets = physical_offsets(
                            std::integral_constant<int, ordinal>{});
                        if constexpr ((offsets & ~0x001f001f) == page_delta) {
                          emit_atom(i, j);
                        }
                      });
                    });
                  }
                });
              }
            });
          }
        });
      });
    }
  });
}

} // namespace hcu
} // namespace tl
