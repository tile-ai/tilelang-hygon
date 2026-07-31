// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file scale_lds_layout.cc
 * \brief Automatic bank-aware Scale LDS layout selection utility.
 */

#include "hcu/utils/scale_lds_layout.h"
#include "hcu/op/copy_scale.h"
#include "hcu/op/gemm_partition.h"
#include "hcu/target_utils.h"
#include "layout/layout.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <set>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

constexpr int kBanks = 64;
constexpr int kBankBytes = 4;

bool IsPowerOfTwo(int value) { return value > 0 && (value & (value - 1)) == 0; }

struct ConflictScore {
  int max_multiplicity{1};
  int serialized_dwords{0};
  int conflicting_banks{0};

  bool operator<(const ConflictScore &other) const {
    if (max_multiplicity != other.max_multiplicity)
      return max_multiplicity < other.max_multiplicity;
    if (serialized_dwords != other.serialized_dwords)
      return serialized_dwords < other.serialized_dwords;
    return conflicting_banks < other.conflicting_banks;
  }
};

struct AutoLayoutConfig {
  int op_ctrl{0};
  ScaleLdsFormat format{ScaleLdsFormat::kIdentity};
  int parent_k{0};
  int parent_mn{0};
  int tile_k{0};
  int tile_mn{0};
  int origin_k{0};
  int origin_mn{0};
  int gran_mn{0};
  int gran_k{0};
  int mn_warps{0};
  int total_warps{0};
  std::vector<int> physical_shape;
};

int ScaleMnIndex(int mn_row, int tx16, int gran_mn) {
  if (gran_mn >= 16)
    return mn_row;
  return mn_row * (16 / gran_mn) + tx16 / gran_mn;
}

int FlattenPhysical(const AutoLayoutConfig &cfg, std::vector<int> coord,
                    int xor_shift) {
  if (xor_shift != 0 && cfg.format != ScaleLdsFormat::kK2MN2Interleave) {
    const int outer_dim = 0;
    const int mn_dim = 1;
    const int mn_extent = cfg.physical_shape[mn_dim];
    const int phase = (coord[outer_dim] * xor_shift) & (mn_extent - 1);
    coord[mn_dim] ^= phase;
  }
  int offset = 0;
  for (size_t i = 0; i < coord.size(); ++i)
    offset = offset * cfg.physical_shape[i] + coord[i];
  return offset;
}

void AppendGroupWords(const AutoLayoutConfig &cfg, int fill_batch, int mn_row0,
                      int xor_shift, std::set<int> *word_ids) {
  if (fill_batch < 0)
    return;
  const int rows_per_fill = 1 << cfg.op_ctrl;
  const int k_atoms = cfg.tile_k / rows_per_fill;
  for (int tx16 = 0; tx16 < 16; ++tx16) {
    std::vector<int> coord;
    if (cfg.format == ScaleLdsFormat::kIdentity) {
      const int mn_local = fill_batch / cfg.tile_k;
      const int k_rel = fill_batch % cfg.tile_k;
      const int mn =
          cfg.origin_mn + ScaleMnIndex(mn_row0 + mn_local, tx16, cfg.gran_mn);
      coord = {cfg.origin_k + k_rel, mn};
    } else if (cfg.format == ScaleLdsFormat::kK2Interleave) {
      const int mn_local = fill_batch / k_atoms;
      const int k_atom = fill_batch % k_atoms;
      const int mn =
          cfg.origin_mn + ScaleMnIndex(mn_row0 + mn_local, tx16, cfg.gran_mn);
      coord = {cfg.origin_k / 2 + k_atom, mn, 0};
    } else if (cfg.format == ScaleLdsFormat::kK4Interleave) {
      const int mn_local = fill_batch / k_atoms;
      const int k_group = fill_batch % k_atoms;
      const int mn =
          cfg.origin_mn + ScaleMnIndex(mn_row0 + mn_local, tx16, cfg.gran_mn);
      coord = {cfg.origin_k / 4 + k_group, mn, 0};
    } else if (cfg.format == ScaleLdsFormat::kK2MN2Interleave) {
      const int mixed_k_atoms = cfg.tile_k / 2;
      const int mn_pair = fill_batch / mixed_k_atoms;
      const int k_atom = fill_batch % mixed_k_atoms;
      const int pair = cfg.origin_mn / 2 +
                       ScaleMnIndex(mn_row0 / 2 + mn_pair, tx16, cfg.gran_mn);
      coord = {cfg.origin_k / 2 + k_atom, pair / 16, pair % 16, 0, 0};
    } else if (cfg.format == ScaleLdsFormat::kMN2Interleave) {
      const int mn_pair = fill_batch / cfg.tile_k;
      const int k_rel = fill_batch % cfg.tile_k;
      const int pair = cfg.origin_mn / 2 +
                       ScaleMnIndex(mn_row0 / 2 + mn_pair, tx16, cfg.gran_mn);
      coord = {cfg.origin_k + k_rel, pair, 0};
    } else {
      const int mn_quad = fill_batch / cfg.tile_k;
      const int k_rel = fill_batch % cfg.tile_k;
      const int quad = cfg.origin_mn / 4 +
                       ScaleMnIndex(mn_row0 / 4 + mn_quad, tx16, cfg.gran_mn);
      coord = {cfg.origin_k + k_rel, quad, 0};
    }

    const int byte_offset = FlattenPhysical(cfg, coord, xor_shift);
    const int bytes_per_lane = rows_per_fill;
    const int first_word = byte_offset / kBankBytes;
    const int last_word = (byte_offset + bytes_per_lane - 1) / kBankBytes;
    for (int word = first_word; word <= last_word; ++word)
      word_ids->insert(word);
  }
}

void AccumulateBatch(const std::set<int> &word_ids, ConflictScore *score) {
  std::array<int, kBanks> bank_counts{};
  for (int word : word_ids)
    ++bank_counts[word % kBanks];
  for (int count : bank_counts) {
    score->max_multiplicity = std::max(score->max_multiplicity, count);
    if (count > 1) {
      score->serialized_dwords += count - 1;
      ++score->conflicting_banks;
    }
  }
}

ConflictScore ScoreLayout(const AutoLayoutConfig &cfg, int xor_shift) {
  ConflictScore score;
  const int rows_mn =
      cfg.gran_mn >= 16 ? cfg.tile_mn : cfg.tile_mn * cfg.gran_mn / 16;
  const int rows_per_fill = 1 << cfg.op_ctrl;
  const int rows_mn_per_seg = rows_mn / cfg.mn_warps;
  const int batches_per_seg = rows_mn_per_seg * cfg.tile_k / rows_per_fill;
  const int copy_warps_per_seg = cfg.total_warps / cfg.mn_warps;
  const int batches_per_warp =
      (batches_per_seg + copy_warps_per_seg - 1) / copy_warps_per_seg;
  const int issue_iters = (batches_per_warp + 3) / 4;

  for (int warp = 0; warp < cfg.total_warps; ++warp) {
    const int seg = warp % cfg.mn_warps;
    const int local_warp = warp / cfg.mn_warps;
    const int mn_row0 = seg * rows_mn_per_seg;
    for (int iter = 0; iter < issue_iters; ++iter) {
      const int group_base = local_warp * batches_per_warp + iter * 4;
      const int hardware_batches = cfg.op_ctrl == 2 ? 2 : 1;
      for (int hw = 0; hw < hardware_batches; ++hw) {
        std::set<int> words;
        const int group_begin = cfg.op_ctrl == 2 ? hw * 2 : 0;
        const int group_end = cfg.op_ctrl == 2 ? group_begin + 2 : 4;
        for (int group = group_begin; group < group_end; ++group) {
          const int fill_batch = group_base + group;
          if (fill_batch < batches_per_seg)
            AppendGroupWords(cfg, fill_batch, mn_row0, xor_shift, &words);
        }
        if (!words.empty())
          AccumulateBatch(words, &score);
      }
    }
  }
  return score;
}

Layout MakeXorLayout(const Buffer &buffer, ScaleLdsFormat format,
                     int xor_shift) {
  Array<IterVar> vars;
  Array<PrimExpr> output;
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    vars.push_back(
        IterVar(Range(0, buffer->shape[i]),
                Var("scale_i" + std::to_string(i), buffer->DefaultIndexType()),
                IterVarType::kDataPar));
    output.push_back(vars.back()->var);
  }
  if (xor_shift != 0 && format != ScaleLdsFormat::kK2MN2Interleave) {
    const int outer_dim = 0;
    const int mn_dim = 1;
    const PrimExpr phase =
        FloorMod(vars[outer_dim]->var * xor_shift, buffer->shape[mn_dim]);
    output.Set(mn_dim, vars[mn_dim]->var ^ phase);
  }
  return Layout(vars, output);
}

Optional<Layout> SelectAutoLayout(const CopyScaleNode *op, int block_size,
                                  Target target) {
  if (!op->has_scale_view_ || op->scale_k_major_ != 0 ||
      op->src->dtype.bits() * op->src->dtype.lanes() != 8)
    return std::nullopt;

  auto get_int = [](const PrimExpr &expr) -> int {
    const int64_t *value = as_const_int(expr);
    return value ? static_cast<int>(*value) : -1;
  };
  AutoLayoutConfig cfg;
  cfg.op_ctrl = op->op_ctrl_;
  cfg.format = static_cast<ScaleLdsFormat>(op->scale_format_);
  cfg.parent_k = get_int(op->parent_k_);
  cfg.parent_mn = get_int(op->parent_mn_);
  cfg.tile_k = get_int(op->tile_k_);
  cfg.tile_mn = get_int(op->tile_mn_);
  cfg.origin_k = get_int(op->origin_k_);
  cfg.origin_mn = get_int(op->origin_mn_);
  cfg.gran_mn = op->granularity_mn_;
  cfg.gran_k = op->granularity_k_;
  if (cfg.parent_k <= 0 || cfg.parent_mn <= 0 || cfg.tile_k <= 0 ||
      cfg.tile_mn <= 0 || cfg.origin_k < 0 || cfg.origin_mn < 0 ||
      cfg.gran_mn <= 0 || cfg.gran_k != 32 || !IsPowerOfTwo(cfg.parent_mn))
    return std::nullopt;
  if (cfg.gran_mn < 16 && 16 % cfg.gran_mn != 0)
    return std::nullopt;
  for (const PrimExpr &shape : op->src->shape) {
    const int value = get_int(shape);
    if (value <= 0)
      return std::nullopt;
    cfg.physical_shape.push_back(value);
  }
  if ((cfg.format == ScaleLdsFormat::kIdentity &&
       cfg.physical_shape.size() != 2) ||
      ((cfg.format == ScaleLdsFormat::kK2Interleave ||
        cfg.format == ScaleLdsFormat::kK4Interleave ||
        cfg.format == ScaleLdsFormat::kMN2Interleave ||
        cfg.format == ScaleLdsFormat::kMN4Interleave) &&
       cfg.physical_shape.size() != 3) ||
      (cfg.format == ScaleLdsFormat::kK2MN2Interleave &&
       cfg.physical_shape.size() != 5))
    return std::nullopt;
  if ((cfg.format == ScaleLdsFormat::kIdentity ||
       cfg.format == ScaleLdsFormat::kK2Interleave ||
       cfg.format == ScaleLdsFormat::kK4Interleave) &&
      cfg.physical_shape[1] != cfg.parent_mn)
    return std::nullopt;
  if (cfg.format == ScaleLdsFormat::kMN2Interleave &&
      cfg.physical_shape[1] * 2 != cfg.parent_mn)
    return std::nullopt;
  if (cfg.format == ScaleLdsFormat::kMN4Interleave &&
      cfg.physical_shape[1] * 4 != cfg.parent_mn)
    return std::nullopt;
  if (cfg.format == ScaleLdsFormat::kK2MN2Interleave &&
      (cfg.physical_shape[0] * 2 != cfg.parent_k ||
       cfg.physical_shape[1] * 32 != cfg.parent_mn))
    return std::nullopt;

  auto policy = GemmWarpPolicy(op->gemm_policy_);
  hcu::ScaleWarpSeg seg = hcu::ComputeScaleWarpSeg(
      *policy.get(), op->gemm_m_, op->gemm_n_, op->gemm_k_, op->gemm_k_pack_,
      op->gemm_elem_bits_, block_size, target, op->a_from_mls_ != 0,
      op->b_from_mls_ != 0, op->a_mls_trans_ != 0, op->b_mls_trans_ != 0,
      op->min_m_per_warp_, op->min_n_per_warp_);
  cfg.mn_warps = op->role_ == 0 ? seg.m_seg : seg.n_seg;
  cfg.total_warps = seg.total_warps;
  // These are not StorageLayout legality checks. They are prerequisites for
  // reproducing the exact DsScaleCopyFill issue grouping used by ScoreLayout.
  if (cfg.mn_warps <= 0 || cfg.total_warps % cfg.mn_warps != 0)
    return std::nullopt;
  const int rows_mn =
      cfg.gran_mn >= 16 ? cfg.tile_mn : cfg.tile_mn * cfg.gran_mn / 16;
  const int rows_per_fill = 1 << cfg.op_ctrl;
  if (rows_mn <= 0 || rows_mn % cfg.mn_warps != 0 ||
      (rows_mn / cfg.mn_warps) * cfg.tile_k % rows_per_fill != 0)
    return std::nullopt;

  const ConflictScore linear_score = ScoreLayout(cfg, 0);
  ConflictScore best_score = linear_score;
  int best_shift = 0;
  if (cfg.format != ScaleLdsFormat::kK2MN2Interleave) {
    for (int shift = 1; shift < cfg.parent_mn; shift <<= 1) {
      ConflictScore candidate = ScoreLayout(cfg, shift);
      if (candidate < best_score) {
        best_score = candidate;
        best_shift = shift;
      }
    }
  }
  if (best_shift == 0)
    return std::nullopt;
  DLOG(INFO) << "AutoScaleLdsLayout buffer=" << op->src->name
             << " selected xor_shift=" << best_shift << " score "
             << linear_score.max_multiplicity << "/"
             << linear_score.serialized_dwords << "/"
             << linear_score.conflicting_banks << " -> "
             << best_score.max_multiplicity << "/"
             << best_score.serialized_dwords << "/"
             << best_score.conflicting_banks;
  return MakeXorLayout(op->src, cfg.format, best_shift);
}

} // namespace

namespace hcu {

Optional<Layout> SelectAutoScaleLdsLayout(const CopyScaleNode *op,
                                          const LayoutInferArgs &args) {
  const int64_t *block_size = as_const_int(args.thread_bounds->extent);
  if (block_size == nullptr)
    return std::nullopt;
  return SelectAutoLayout(op, static_cast<int>(*block_size), args.target);
}

} // namespace hcu
} // namespace tl
} // namespace tvm
