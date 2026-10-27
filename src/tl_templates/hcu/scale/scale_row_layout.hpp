// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

// This header deliberately has no TVM or GPU-runtime dependencies. The host
// planner and generated device templates use exactly the same row algebra.
#if defined(__HIPCC__) || defined(__CUDACC__)
#define TL_SCALE_ROW_HD __host__ __device__
#else
#define TL_SCALE_ROW_HD
#endif

namespace tl {
namespace hcu {

enum class ScaleRowMode : int { FixedBank = 0, DenseXor = 1, KPhasePacked = 2 };

// Neighboring physical waves share an LDS port (0/1, 2/3, ...). When an
// issue round is underfilled, assign useful work in physical order
// 0,2,4,...,1,3,5,... instead of packing it into adjacent waves. This is a
// copy-work rank only; it must never change the GEMM's wave/segment mapping.
TL_SCALE_ROW_HD constexpr int ScaleCopyWaveRank(int wave, int total_warps) {
  return (wave % 2) * ((total_warps + 1) / 2) + wave / 2;
}

struct ScaleRowLayoutDesc {
  ScaleRowMode mode;
  int polarity;
  int rows_mn;
  int physical_k;
  int atom_mn;
  int atom_k;
  int row_order; // 0: KInner, 1: MNInner, outside the indivisible atom.
  int phase_k_rows;

  TL_SCALE_ROW_HD constexpr bool Valid() const {
    const int length = atom_mn * atom_k;
    return (mode == ScaleRowMode::FixedBank || mode == ScaleRowMode::DenseXor ||
            mode == ScaleRowMode::KPhasePacked) &&
           (polarity == 0 || polarity == 1) && rows_mn > 0 && physical_k > 0 &&
           atom_mn > 0 && atom_k > 0 &&
           (length == 1 || length == 2 || length == 4) &&
           rows_mn % atom_mn == 0 && physical_k % atom_k == 0 &&
           (row_order == 0 || row_order == 1) && phase_k_rows >= atom_k &&
           phase_k_rows % atom_k == 0;
  }

  TL_SCALE_ROW_HD constexpr int RawRow(int mn, int k) const {
    const int ordinal =
        row_order == 0 ? (mn / atom_mn) * (physical_k / atom_k) + k / atom_k
                       : (k / atom_k) * (rows_mn / atom_mn) + mn / atom_mn;
    return ordinal * atom_mn * atom_k + (mn % atom_mn) * atom_k + k % atom_k;
  }

  // Number of K atoms of phase h in a possibly incomplete final phase pair.
  TL_SCALE_ROW_HD constexpr int PhaseAtoms(int h) const {
    const int count = physical_k / atom_k;
    const int width = phase_k_rows / atom_k;
    const int tail = count % (2 * width) - h * width;
    return count / (2 * width) * width +
           (tail <= 0 ? 0 : (tail < width ? tail : width));
  }

  TL_SCALE_ROW_HD constexpr int RowOffset(int mn, int k) const {
    const int raw = RawRow(mn, k);
    if (mode == ScaleRowMode::FixedBank) {
      return (raw / 4) * 8 + raw % 4 + 4 * polarity;
    }
    if (mode == ScaleRowMode::DenseXor) {
      return raw ^ (4 * polarity);
    }
    const int width = phase_k_rows / atom_k;
    const int ka = k / atom_k;
    const int h = ka / width % 2;
    const int rank_k = ka / (2 * width) * width + ka % width;
    const int j = row_order == 0 ? (mn / atom_mn) * PhaseAtoms(h) + rank_k
                                 : rank_k * (rows_mn / atom_mn) + mn / atom_mn;
    const int length = atom_mn * atom_k;
    const int per_half = 4 / length;
    return 8 * (j / per_half) + 4 * (h ^ polarity) + length * (j % per_half) +
           (mn % atom_mn) * atom_k + k % atom_k;
  }

  TL_SCALE_ROW_HD constexpr int SegmentRows() const {
    if (mode == ScaleRowMode::FixedBank) {
      return 8 * ((rows_mn * physical_k + 3) / 4);
    }
    if (mode == ScaleRowMode::DenseXor) {
      return 8 * ((rows_mn * physical_k + 7) / 8);
    }
    const int n0 = PhaseAtoms(0) * (rows_mn / atom_mn);
    const int n1 = PhaseAtoms(1) * (rows_mn / atom_mn);
    const int per_half = 4 / (atom_mn * atom_k);
    return 8 * (((n0 > n1 ? n0 : n1) + per_half - 1) / per_half);
  }

  // Occupied rows in each bank, compact within that bank's four-row chunks.
  // Enumerating these chunks (not the reserved span) avoids padding issues
  // and distributes useful work evenly across the segment's copy waves.
  TL_SCALE_ROW_HD constexpr int CopyHalfRows(int bank) const {
    const int count = rows_mn * physical_k;
    if (mode == ScaleRowMode::FixedBank)
      return bank == polarity ? count : 0;
    if (mode == ScaleRowMode::DenseXor) {
      const int tail = count % 8 - (bank ^ polarity) * 4;
      return count / 8 * 4 + (tail <= 0 ? 0 : (tail < 4 ? tail : 4));
    }
    return PhaseAtoms(bank ^ polarity) * rows_mn * atom_k;
  }
  TL_SCALE_ROW_HD constexpr int CopyChunkCount() const {
    return (CopyHalfRows(0) + 3) / 4 + (CopyHalfRows(1) + 3) / 4;
  }
  // A full wave writes 4 * rows_per_fill rows. Dense balanced chunks can
  // use all four groups. FixedBank K4 uses groups 0 and 2 to fill two
  // same-bank half-chunks at once, leaving the intervening bank untouched.
  TL_SCALE_ROW_HD constexpr int CopyIssueChunks(int rows_per_fill) const {
    if (CopyHalfRows(0) == CopyHalfRows(1) && CopyHalfRows(0) % 4 == 0)
      return rows_per_fill;
    return mode == ScaleRowMode::FixedBank && rows_per_fill == 4 ? 2 : 1;
  }
  TL_SCALE_ROW_HD constexpr int CopyIssueGroups(int rows_per_fill) const {
    return CopyIssueChunks(rows_per_fill) > 1 ? 4 : 4 / rows_per_fill;
  }
  TL_SCALE_ROW_HD constexpr int CopyChunk(int ordinal) const {
    const int n0 = (CopyHalfRows(0) + 3) / 4;
    const int n1 = (CopyHalfRows(1) + 3) / 4;
    const int paired = n0 < n1 ? n0 : n1;
    return ordinal < paired * 2 ? ordinal
                                : (ordinal - paired) * 2 + (n1 > n0 ? 1 : 0);
  }
  TL_SCALE_ROW_HD constexpr bool CopyGroupValid(int chunk,
                                                int row_in_half) const {
    return chunk / 2 * 4 + row_in_half < CopyHalfRows(chunk % 2);
  }

  // Invert a physical row to the original format/RowOrder stream. Copy uses
  // this before its format-specific source policy. -1 denotes padding: do
  // not issue a source read or a destination write for that group.
  TL_SCALE_ROW_HD constexpr int RawRowFromPhysical(int row) const {
    if (row < 0 || row >= SegmentRows() || !CopyGroupValid(row / 4, row % 4)) {
      return -1;
    }
    return CopyRawRow(row);
  }

  // Caller proves that this is an occupied group. No -1 sentinel, bank or
  // range tests enter device address arithmetic; use unsigned division in
  // the copy template after checking the group's validity.
  TL_SCALE_ROW_HD constexpr int CopyRawRow(int row) const {
    if (mode == ScaleRowMode::FixedBank) {
      return row / 8 * 4 + row % 4;
    }
    if (mode == ScaleRowMode::DenseXor) {
      return row ^ (4 * polarity);
    }
    const int length = atom_mn * atom_k;
    const int h = (row / 4 % 2) ^ polarity;
    const int j = row / 8 * (4 / length) + (row % 4) / length;
    const int mn_atoms = rows_mn / atom_mn;
    const int k_count = PhaseAtoms(h);
    const int ma = row_order == 0 ? j / k_count : j % mn_atoms;
    const int rank_k = row_order == 0 ? j % k_count : j / mn_atoms;
    const int width = phase_k_rows / atom_k;
    const int ka = (rank_k / width * 2 + h) * width + rank_k % width;
    const int ordinal =
        row_order == 0 ? ma * (physical_k / atom_k) + ka : ka * mn_atoms + ma;
    return ordinal * length + row % length;
  }

  // DenseXor is a shortcut only when its bank bit agrees with the chosen K
  // phase for every MN group, not just for the first row of the segment.
  TL_SCALE_ROW_HD constexpr bool DensePhaseCompatible() const {
    for (int mn = 0; mn < rows_mn; ++mn) {
      for (int k = 0; k < physical_k; ++k) {
        if ((RawRow(mn, k) / 4 % 2) != (k / phase_k_rows % 2)) {
          return false;
        }
      }
    }
    return true;
  }
};

template <int Mode, int Polarity, int RowsMN, int PhysicalK, int AtomMN,
          int AtomK, int RowOrder, int PhaseKRows>
struct ScaleRowLayout {
  static constexpr ScaleRowLayoutDesc descriptor{
      static_cast<ScaleRowMode>(Mode),
      Polarity,
      RowsMN,
      PhysicalK,
      AtomMN,
      AtomK,
      RowOrder,
      PhaseKRows};
  static_assert(descriptor.Valid(), "invalid scale row layout descriptor");
  static_assert(Mode != static_cast<int>(ScaleRowMode::DenseXor) ||
                    descriptor.DensePhaseCompatible(),
                "DenseXor bank bit must depend only on the common K phase");
  static constexpr int SegmentRows = descriptor.SegmentRows();
  TL_SCALE_ROW_HD static constexpr int RowOffset(int mn, int k) {
    // Do not pass the static member's address to a device method: HIP can
    // retain descriptor loads and runtime mode branches/integer divisions.
    // An automatic constexpr value keeps the fields visible to folding.
    constexpr ScaleRowLayoutDesc local = descriptor;
    return local.RowOffset(mn, k);
  }
  TL_SCALE_ROW_HD static constexpr int SegmentOffset(int segment) {
    return segment * SegmentRows;
  }
  TL_SCALE_ROW_HD static constexpr int RawRowFromPhysical(int row) {
    constexpr ScaleRowLayoutDesc local = descriptor;
    return local.RawRowFromPhysical(row);
  }
};

} // namespace hcu
} // namespace tl

#undef TL_SCALE_ROW_HD
