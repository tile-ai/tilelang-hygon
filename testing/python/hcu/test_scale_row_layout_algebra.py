# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Host-only exhaustive checks of the actual shared host/device row algebra."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_scale_row_layout_algebra(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required")
    root = Path(__file__).resolve().parents[3]
    source = r"""
#include "tl_templates/hcu/scale/scale_row_layout.hpp"
#include <cassert>
#include <set>
using namespace tl::hcu;
template <typename Layout> void CheckDeviceWrapper() {
  constexpr auto d = Layout::descriptor;
  for (int mn = 0; mn < d.rows_mn; ++mn)
    for (int k = 0; k < d.physical_k; ++k)
      assert(Layout::RowOffset(mn, k) == d.RowOffset(mn, k));
  for (int row = -1; row <= Layout::SegmentRows; ++row)
    assert(Layout::RawRowFromPhysical(row) == d.RawRowFromPhysical(row));
}
int main() {
  for (int waves = 1; waves <= 16; ++waves) {
    std::set<int> ranks;
    for (int wave = 0; wave < waves; ++wave)
      assert(ranks.insert(ScaleCopyWaveRank(wave, waves)).second);
    assert(*ranks.begin() == 0 && *ranks.rbegin() == waves - 1);
    for (int segments = 1; segments <= waves; ++segments) {
      if (waves % segments) continue;
      int per_segment = waves / segments;
      for (int issues = 1; issues <= 8; ++issues) {
        std::set<int> work, active;
        bool partial = issues % per_segment != 0;
        for (int wave = 0; wave < waves; ++wave) {
          int rank = partial ? ScaleCopyWaveRank(wave, waves) : wave;
          int segment = rank % segments;
          for (int issue = rank / segments; issue < issues; issue += per_segment) {
            assert(work.insert(segment * issues + issue).second);
            active.insert(wave);
          }
        }
        assert(work.size() == static_cast<unsigned>(segments * issues));
        if (waves == 4 && segments * issues == 2)
          assert(active == std::set<int>({0, 2}));
      }
    }
  }
  // Exercise runtime arguments too: device wrappers must retain the same
  // algebra when materializing an automatic constexpr descriptor for HIP.
  CheckDeviceWrapper<ScaleRowLayout<0,1,2,4,1,2,0,4>>();
  CheckDeviceWrapper<ScaleRowLayout<1,1,2,8,1,2,0,4>>();
  CheckDeviceWrapper<ScaleRowLayout<2,1,3,12,1,2,1,4>>();
  // Identity K32/K64, K2/K4, MN2/MN4 and K2MN2 atom geometries.
  const int atoms[][2] = {{1,1}, {1,2}, {1,4}, {2,1}, {4,1}, {2,2}};
  for (auto &atom : atoms)
    for (int mn = atom[0]; mn <= 16; mn += atom[0])
      for (int k = atom[1]; k <= 32; k += atom[1])
        for (int order = 0; order < 2; ++order)
          for (int width = atom[1]; width <= 32; width += atom[1])
            for (int mode = 0; mode < 3; ++mode)
              for (int polarity = 0; polarity < 2; ++polarity) {
                ScaleRowLayoutDesc d{static_cast<ScaleRowMode>(mode), polarity,
                                    mn, k, atom[0], atom[1], order, width};
                assert(d.Valid());
                if (mode == 1 && !d.DensePhaseCompatible()) continue;
                assert(d.SegmentRows() % 8 == 0);
                std::set<int> occupied;
                for (int m = 0; m < mn; ++m)
                  for (int s = 0; s < k; ++s) {
                    int row = d.RowOffset(m, s);
                    assert(row >= 0 && row < d.SegmentRows());
                    assert(occupied.insert(row).second);
                    assert(d.RawRowFromPhysical(row) == d.RawRow(m, s));
                    int phase = mode == 0 ? 0 : (s / width) % 2;
                    assert((row / 4 % 2) == (phase ^ polarity));
                    // Every atom remains contiguous, aligned and in one bank.
                    int inner = m % atom[0] * atom[1] + s % atom[1];
                    int first = d.RowOffset(m - m % atom[0], s - s % atom[1]);
                    assert(row == first + inner);
                    assert(first % (atom[0] * atom[1]) == 0);
                  }
                for (int row = 0; row < d.SegmentRows(); ++row)
                  assert((d.RawRowFromPhysical(row) >= 0) == occupied.count(row));
                std::set<int> copied;
                int previous_chunk = -1;
                for (int issue = 0; issue < d.CopyChunkCount(); ++issue) {
                  int chunk = d.CopyChunk(issue);
                  assert(chunk > previous_chunk);
                  previous_chunk = chunk;
                  bool useful = false;
                  for (int group_row = 0; group_row < 4; ++group_row) {
                    if (!d.CopyGroupValid(chunk, group_row)) continue;
                    int row = chunk * 4 + group_row;
                    assert(copied.insert(row).second);
                    assert(d.CopyRawRow(row) == d.RawRowFromPhysical(row));
                    useful = true;
                  }
                  assert(useful);  // No issue is assigned to a padding chunk.
                }
                assert(copied == occupied);
                for (int width : {1, 2, 4}) {
                  if ((atom[0] * atom[1]) % width) continue;
                  std::set<int> issued;
                  int stride = d.CopyIssueChunks(width);
                  int count = (d.CopyChunkCount() + stride - 1) / stride;
                  for (int issue = 0; issue < count; ++issue) {
                    int base = d.CopyChunk(issue * stride) * 4;
                    for (int group = 0; group < d.CopyIssueGroups(width); ++group) {
                      int row = base + group * width;
                      if (d.RawRowFromPhysical(row) < 0) continue;
                      assert(d.CopyRawRow(row) % width == 0);
                      for (int inner = 0; inner < width; ++inner) {
                        assert(d.RawRowFromPhysical(row + inner) == d.CopyRawRow(row) + inner);
                        assert(issued.insert(row + inner).second);
                      }
                    }
                  }
                  assert(issued == occupied);
                }
                assert(d.RawRowFromPhysical(-1) == -1);
                assert(d.RawRowFromPhysical(d.SegmentRows()) == -1);
              }
  // Different MN lengths and opposite RowOrders must still pair by K.
  ScaleRowLayoutDesc a{ScaleRowMode::KPhasePacked,0,8,12,1,2,0,4};
  ScaleRowLayoutDesc b{ScaleRowMode::KPhasePacked,1,2,12,1,2,1,4};
  for (int m = 0; m < 8; ++m)
    for (int n = 0; n < 2; ++n)
      for (int k = 0; k < 12; ++k)
        assert((a.RowOffset(m,k)/4%2) != (b.RowOffset(n,k)/4%2));
  // ScaleK=4 plus multiple MN groups is not blindly DenseXor-compatible.
  ScaleRowLayoutDesc bad{ScaleRowMode::DenseXor,0,2,4,1,4,0,4};
  assert(!bad.DensePhaseCompatible());
  using Large = ScaleRowLayout<0,1,16,8,1,2,0,4>;
  static_assert(Large::RowOffset(15,6) / 32 > 0);
  constexpr int off = Large::RowOffset(15,6);
  static_assert((off / 32) * 32 + off % 32 == off);
}
"""
    executable = tmp_path / "scale_row_layout_algebra"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(root / "src"), "-x", "c++", "-", "-o", str(executable)],
        input=source,
        text=True,
        check=True,
        capture_output=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True)
