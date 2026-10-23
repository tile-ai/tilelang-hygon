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
  int mmac_k{64};
  ScaleBufferRowOrder row_order{ScaleBufferRowOrder::kKInner};
  size_t leading_stage_dims{0};
  std::vector<int> physical_shape;
  Optional<Layout> resolved_layout;
  mutable bool addresses_known{true};
};

size_t ScaleFormatPhysicalRank(ScaleLdsFormat format) {
  if (format == ScaleLdsFormat::kIdentity)
    return 2;
  if (format == ScaleLdsFormat::kK2MN2Interleave)
    return 5;
  return 3;
}

int ScaleMnIndex(int mn_row, int tx16, int gran_mn) {
  if (gran_mn >= 16)
    return mn_row;
  return mn_row * (16 / gran_mn) + tx16 / gran_mn;
}

int FlattenPhysical(const AutoLayoutConfig &cfg, std::vector<int> coord,
                    int xor_shift) {
  if (cfg.resolved_layout.defined()) {
    Array<PrimExpr> inputs;
    for (size_t i = 0; i < cfg.leading_stage_dims; ++i)
      inputs.push_back(Integer(0));
    for (int value : coord)
      inputs.push_back(Integer(value));
    const auto output = cfg.resolved_layout.value()->Forward(inputs);
    const auto shape = cfg.resolved_layout.value()->OutputShape();
    PrimExpr offset = Integer(0);
    for (size_t i = 0; i < output.size(); ++i)
      offset = offset * shape[i] + output[i];
    arith::Analyzer analyzer;
    offset = analyzer.Simplify(offset);
    const int64_t *value = as_const_int(offset);
    if (!value) {
      cfg.addresses_known = false;
      return -1;
    }
    return static_cast<int>(*value);
  }
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

void DecodeAtomOrdinal(int ordinal, int mn_atoms, int k_atoms,
                       ScaleBufferRowOrder order, int *mn_atom, int *k_atom) {
  if (order == ScaleBufferRowOrder::kMNInner) {
    *k_atom = ordinal / mn_atoms;
    *mn_atom = ordinal % mn_atoms;
  } else {
    *mn_atom = ordinal / k_atoms;
    *k_atom = ordinal % k_atoms;
  }
}

void AppendGroupWords(const AutoLayoutConfig &cfg, int fill_batch, int mn_row0,
                      int rows_mn_per_seg, int xor_shift,
                      std::set<int> *word_ids) {
  if (fill_batch < 0)
    return;
  const int rows_per_fill = 1 << cfg.op_ctrl;
  for (int tx16 = 0; tx16 < 16; ++tx16) {
    std::vector<int> coord;
    if (cfg.format == ScaleLdsFormat::kIdentity) {
      const int k_dup = cfg.mmac_k == 64 && cfg.gran_k >= 64 ? 2 : 1;
      const int phys_k = cfg.tile_k * k_dup;
      const int atom_k = cfg.mmac_k == 64 ? 2 : 1;
      const int k_atoms = phys_k / atom_k;
      const int atom_ordinal = fill_batch / atom_k;
      const int inner = fill_batch % atom_k;
      int mn_local = 0, k_atom = 0;
      DecodeAtomOrdinal(atom_ordinal, rows_mn_per_seg, k_atoms, cfg.row_order,
                        &mn_local, &k_atom);
      const int k_rel = (k_atom * atom_k + inner) / k_dup;
      const int mn =
          cfg.origin_mn + ScaleMnIndex(mn_row0 + mn_local, tx16, cfg.gran_mn);
      coord = {cfg.origin_k + k_rel, mn};
    } else if (cfg.format == ScaleLdsFormat::kK2Interleave) {
      const int k_atoms = cfg.tile_k / 2;
      int mn_local = 0, k_atom = 0;
      DecodeAtomOrdinal(fill_batch, rows_mn_per_seg, k_atoms, cfg.row_order,
                        &mn_local, &k_atom);
      const int mn =
          cfg.origin_mn + ScaleMnIndex(mn_row0 + mn_local, tx16, cfg.gran_mn);
      coord = {cfg.origin_k / 2 + k_atom, mn, 0};
    } else if (cfg.format == ScaleLdsFormat::kK4Interleave) {
      const int k_atoms = cfg.tile_k / 4;
      int mn_local = 0, k_group = 0;
      DecodeAtomOrdinal(fill_batch, rows_mn_per_seg, k_atoms, cfg.row_order,
                        &mn_local, &k_group);
      const int mn =
          cfg.origin_mn + ScaleMnIndex(mn_row0 + mn_local, tx16, cfg.gran_mn);
      coord = {cfg.origin_k / 4 + k_group, mn, 0};
    } else if (cfg.format == ScaleLdsFormat::kK2MN2Interleave) {
      const int mixed_k_atoms = cfg.tile_k / 2;
      const int mn_atoms = rows_mn_per_seg / 2;
      int mn_pair = 0, k_atom = 0;
      DecodeAtomOrdinal(fill_batch, mn_atoms, mixed_k_atoms, cfg.row_order,
                        &mn_pair, &k_atom);
      const int pair = cfg.origin_mn / 2 +
                       ScaleMnIndex(mn_row0 / 2 + mn_pair, tx16, cfg.gran_mn);
      coord = {cfg.origin_k / 2 + k_atom, pair / 16, pair % 16, 0, 0};
    } else if (cfg.format == ScaleLdsFormat::kMN2Interleave) {
      const int mn_atoms = rows_mn_per_seg / 2;
      int mn_pair = 0, k_rel = 0;
      DecodeAtomOrdinal(fill_batch, mn_atoms, cfg.tile_k, cfg.row_order,
                        &mn_pair, &k_rel);
      const int pair = cfg.origin_mn / 2 +
                       ScaleMnIndex(mn_row0 / 2 + mn_pair, tx16, cfg.gran_mn);
      coord = {cfg.origin_k + k_rel, pair, 0};
    } else {
      const int mn_atoms = rows_mn_per_seg / 4;
      int mn_quad = 0, k_rel = 0;
      DecodeAtomOrdinal(fill_batch, mn_atoms, cfg.tile_k, cfg.row_order,
                        &mn_quad, &k_rel);
      const int quad = cfg.origin_mn / 4 +
                       ScaleMnIndex(mn_row0 / 4 + mn_quad, tx16, cfg.gran_mn);
      coord = {cfg.origin_k + k_rel, quad, 0};
    }

    const int byte_offset = FlattenPhysical(cfg, coord, xor_shift);
    if (byte_offset < 0)
      continue;
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

ConflictScore
ScoreLayout(const AutoLayoutConfig &cfg, int xor_shift,
            const ::tl::hcu::ScaleRowLayoutDesc *layout = nullptr) {
  ConflictScore score;
  const int rows_mn =
      cfg.gran_mn >= 16 ? cfg.tile_mn : cfg.tile_mn * cfg.gran_mn / 16;
  const int rows_per_fill = 1 << cfg.op_ctrl;
  const int rows_mn_per_seg = rows_mn / cfg.mn_warps;
  const int k_dup = cfg.mmac_k == 64 && cfg.gran_k >= 64 ? 2 : 1;
  const int logical_rows = rows_mn_per_seg * cfg.tile_k * k_dup;
  const int batches_per_seg = logical_rows / rows_per_fill;
  const int copy_warps_per_seg = cfg.total_warps / cfg.mn_warps;
  const int groups_per_issue =
      layout ? layout->CopyIssueGroups(rows_per_fill) : 4 / rows_per_fill;
  const int chunks_per_issue =
      layout ? layout->CopyIssueChunks(rows_per_fill) : 1;
  const int chunks_per_seg =
      layout ? layout->CopyChunkCount() : (logical_rows + 3) / 4;
  const int issues_per_seg =
      (chunks_per_seg + chunks_per_issue - 1) / chunks_per_issue;
  const int issue_iters =
      (issues_per_seg + copy_warps_per_seg - 1) / copy_warps_per_seg;

  for (int warp = 0; warp < cfg.total_warps; ++warp) {
    const int work_rank =
        issues_per_seg % copy_warps_per_seg != 0
            ? ::tl::hcu::ScaleCopyWaveRank(warp, cfg.total_warps)
            : warp;
    const int seg = work_rank % cfg.mn_warps;
    const int local_warp = work_rank / cfg.mn_warps;
    const int mn_row0 = seg * rows_mn_per_seg;
    for (int iter = 0; iter < issue_iters; ++iter) {
      const int ordinal = local_warp + iter * copy_warps_per_seg;
      if (ordinal >= issues_per_seg)
        continue;
      const int chunk =
          layout ? layout->CopyChunk(ordinal * chunks_per_issue) : ordinal;
      std::set<int> words;
      for (int group = 0; group < groups_per_issue; ++group) {
        const int raw =
            layout
                ? layout->RawRowFromPhysical(chunk * 4 + group * rows_per_fill)
                : chunk * 4 + group * rows_per_fill;
        const int fill_batch = raw / rows_per_fill;
        if (raw >= 0 && fill_batch < batches_per_seg)
          AppendGroupWords(cfg, fill_batch, mn_row0, rows_mn_per_seg, xor_shift,
                           &words);
      }
      if (!words.empty())
        AccumulateBatch(words, &score);
    }
  }
  return score;
}

Layout MakeXorLayout(const Buffer &buffer, ScaleLdsFormat format,
                     size_t leading_stage_dims, int xor_shift) {
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
    const size_t outer_dim = leading_stage_dims;
    const size_t mn_dim = leading_stage_dims + 1;
    const PrimExpr phase =
        FloorMod(vars[outer_dim]->var * xor_shift, buffer->shape[mn_dim]);
    output.Set(mn_dim, vars[mn_dim]->var ^ phase);
  }
  return Layout(vars, output);
}

Optional<AutoLayoutConfig> BuildConfig(const CopyScaleNode *op, int block_size,
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
  cfg.mmac_k = op->gemm_elem_bits_ == 8 ? 32 : 64;
  if (cfg.parent_k <= 0 || cfg.parent_mn <= 0 || cfg.tile_k <= 0 ||
      cfg.tile_mn <= 0 || cfg.origin_k < 0 || cfg.origin_mn < 0 ||
      cfg.gran_mn <= 0 || cfg.gran_k != 32 || !IsPowerOfTwo(cfg.parent_mn))
    return std::nullopt;
  if (cfg.gran_mn < 16 && 16 % cfg.gran_mn != 0)
    return std::nullopt;
  const size_t format_rank = ScaleFormatPhysicalRank(cfg.format);
  if (op->src->shape.size() < format_rank)
    return std::nullopt;
  cfg.leading_stage_dims = op->src->shape.size() - format_rank;
  for (const PrimExpr &shape : op->src->shape) {
    const int value = get_int(shape);
    if (value <= 0)
      return std::nullopt;
  }
  for (size_t i = cfg.leading_stage_dims; i < op->src->shape.size(); ++i)
    cfg.physical_shape.push_back(get_int(op->src->shape[i]));
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
  const int atom_mn =
      cfg.format == ScaleLdsFormat::kMN4Interleave
          ? 4
          : (cfg.format == ScaleLdsFormat::kMN2Interleave ||
                     cfg.format == ScaleLdsFormat::kK2MN2Interleave
                 ? 2
                 : 1);
  const int rows_mn_per_seg = rows_mn / cfg.mn_warps;
  if (rows_mn_per_seg % atom_mn != 0)
    return std::nullopt;
  return cfg;
}

Optional<Layout> SelectAutoLayout(const CopyScaleNode *op, int block_size,
                                  Target target) {
  Optional<AutoLayoutConfig> maybe_cfg = BuildConfig(op, block_size, target);
  if (!maybe_cfg.has_value())
    return std::nullopt;
  AutoLayoutConfig cfg = maybe_cfg.value();

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
  return MakeXorLayout(op->src, cfg.format, cfg.leading_stage_dims, best_shift);
}

ScaleBufferRowOrder SelectRowOrder(const CopyScaleNode *op, int block_size,
                                   Target target) {
  Optional<AutoLayoutConfig> maybe_cfg = BuildConfig(op, block_size, target);
  if (!maybe_cfg.has_value())
    return ScaleBufferRowOrder::kKInner;
  AutoLayoutConfig cfg = maybe_cfg.value();
  cfg.row_order = ScaleBufferRowOrder::kKInner;
  const ConflictScore k_score = ScoreLayout(cfg, 0);
  cfg.row_order = ScaleBufferRowOrder::kMNInner;
  const ConflictScore mn_score = ScoreLayout(cfg, 0);
  if (mn_score < k_score) {
    DLOG(INFO) << "ScaleBufferRowOrder buffer=" << op->src->name
               << " selected MNInner score " << k_score.max_multiplicity << "/"
               << k_score.serialized_dwords << "/" << k_score.conflicting_banks
               << " -> " << mn_score.max_multiplicity << "/"
               << mn_score.serialized_dwords << "/"
               << mn_score.conflicting_banks;
    return ScaleBufferRowOrder::kMNInner;
  }
  return ScaleBufferRowOrder::kKInner;
}

} // namespace

namespace hcu {

std::optional<std::array<int, 3>>
ScoreScaleRowLayout(const CopyScaleNode *op,
                    const ::tl::hcu::ScaleRowLayoutDesc &layout, int block_size,
                    Target target, Optional<Layout> resolved_layout) {
  // Async provenance guarantees a linear LDS anchor. Do not guess an
  // already-inferred explicit/swizzled layout for ordinary-copy producers.
  if (!op->lds_from_async_copy_ && !resolved_layout.defined())
    return std::nullopt;
  auto config = BuildConfig(op, block_size, target);
  if (!config.has_value())
    return std::nullopt;
  AutoLayoutConfig cfg = config.value();
  cfg.resolved_layout = resolved_layout;
  cfg.row_order = static_cast<ScaleBufferRowOrder>(layout.row_order);
  const auto score = ScoreLayout(cfg, 0, &layout);
  if (!cfg.addresses_known)
    return std::nullopt;
  return std::array<int, 3>{score.max_multiplicity, score.serialized_dwords,
                            score.conflicting_banks};
}

Optional<Layout> SelectAutoScaleLdsLayout(const CopyScaleNode *op,
                                          const LayoutInferArgs &args) {
  const int64_t *block_size = as_const_int(args.thread_bounds->extent);
  if (block_size == nullptr)
    return std::nullopt;
  return SelectAutoLayout(op, static_cast<int>(*block_size), args.target);
}

ScaleBufferRowOrder SelectScaleBufferRowOrder(const CopyScaleNode *op,
                                              int block_size, Target target) {
  return SelectRowOrder(op, block_size, target);
}

} // namespace hcu
} // namespace tl
} // namespace tvm
