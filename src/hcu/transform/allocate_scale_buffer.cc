// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file allocate_scale_buffer.cc
 * \brief Allocate physical scale_buffer row bases and rewrite row_base vars.
 *
 * Must run while copy_scale/gemm tileops still exist (before LowerTileOp).
 * Uses AnnotateScaleGemmDep clues + threadIdx.x extent to ComputeScaleWarpSeg
 * and size each shared.scale — no separate PlanScaleBufferRows pass.
 *
 * Row layout is explicit metadata; allocation owns its complete physical
 * interval, including padding. Missing warp clues use a single segment.
 */

#include "hcu/op/builtin.h"
#include "hcu/op/copy_scale.h"
#include "hcu/op/gemm_partition.h"
#include "hcu/target_utils.h"
#include "hcu/utils/scale_gemm_dep.h"
#include "hcu/utils/scale_lds_layout.h"
#include "op/gemm.h"
#include "op/gemm_blockscaled.h"
#include "op/operator.h"
#include "tl_templates/hcu/scale/scale_row_layout.hpp"

#include <tvm/ir/transform.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <limits>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

constexpr int kMmacMn = 16;
constexpr int kRowsPerSlot = 16;
constexpr int kMaxScaleSlots = 32;

Optional<int> GetIntAnn(const Map<String, ObjectRef> &annotations,
                        const char *key) {
  if (auto val = annotations.Get(key)) {
    if (const auto *imm = val->as<IntImmNode>()) {
      return static_cast<int>(imm->value);
    }
  }
  return Optional<int>();
}

struct ScaleBufferPlan {
  Buffer buffer;
  int scale_shape_mn{0};
  int scale_shape_k{0};
  int granularity_mn{1};
  int granularity_k{1};
  int scale_k_major{0};
  int role{0};
  int gemm_m{0};
  int gemm_n{0};
  int gemm_k{0};
  int gemm_policy{0};
  int gemm_k_pack{1};
  int gemm_elem_bits{4};
  int a_from_mls{0};
  int b_from_mls{0};
  int a_mls_trans{1};
  int b_mls_trans{1};
  int min_m_per_warp{0};
  int min_n_per_warp{0};
  int reserved_rows{0};
  int start_row{0};
  bool used_fallback{false};
  int mn_warps{1};
  int scale_format{0};
  int row_order{0};
  int op_ctrl{0};
  int gemm_threads{0};
  bool resident{false};
  ::tl::hcu::ScaleRowLayoutDesc layout{};
  CopyScale copy_op;
  Optional<Layout> source_layout;
};

int ComputeRowsMn(int scale_shape_mn, int granularity_mn) {
  if (granularity_mn >= kMmacMn) {
    return scale_shape_mn;
  }
  ICHECK(kMmacMn % granularity_mn == 0) << "granularity_mn=" << granularity_mn
                                        << " must divide mmac_mn=" << kMmacMn;
  return scale_shape_mn * granularity_mn / kMmacMn;
}

std::pair<int, int> ParseScaleShape(const Buffer &buf, int scale_k_major) {
  ICHECK_EQ(buf->shape.size(), 2u)
      << "allocate_scale_buffer requires 2D scale_buffer, got rank="
      << buf->shape.size();
  if (scale_k_major) {
    return {static_cast<int>(*as_const_int(buf->shape[0])),
            static_cast<int>(*as_const_int(buf->shape[1]))};
  }
  return {static_cast<int>(*as_const_int(buf->shape[1])),
          static_cast<int>(*as_const_int(buf->shape[0]))};
}

class ScaleBufferMetaCollector : public StmtExprVisitor {
public:
  std::unordered_map<Buffer, ScaleBufferPlan, ObjectPtrHash, ObjectPtrEqual>
      plans;

private:
  LayoutMap source_layouts_;

  void VisitStmt_(const SBlockNode *op) final {
    if (auto layouts = op->annotations.Get(attr::kLayoutMap)) {
      for (const auto &[buffer, layout] : Downcast<LayoutMap>(layouts.value()))
        source_layouts_.Set(buffer, layout);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void RecordFromCopyScale(const CopyScaleNode *cs,
                           const Map<String, ObjectRef> &ann) {
    ScaleBufferPlan plan;
    plan.buffer = cs->dst;
    plan.copy_op = tvm::ffi::GetRef<CopyScale>(cs);
    plan.source_layout = source_layouts_.Get(cs->src);
    plan.scale_format = cs->scale_format_;
    plan.row_order = cs->row_order_;
    plan.op_ctrl = cs->op_ctrl_;
    plan.gemm_threads = cs->gemm_threads_;
    plan.resident = GetIntAnn(ann, attr::kScaleResident).value_or(0) != 0;
    plan.granularity_mn =
        GetIntAnn(ann, attr::kScaleGranularityMN).value_or(cs->granularity_mn_);
    plan.granularity_k =
        GetIntAnn(ann, attr::kScaleGranularityK).value_or(cs->granularity_k_);
    plan.scale_k_major =
        GetIntAnn(ann, attr::kScaleKMajor).value_or(cs->scale_k_major_);
    plan.role = GetIntAnn(ann, attr::kScaleRole).value_or(cs->role_);
    plan.gemm_m = GetIntAnn(ann, attr::kScaleGemmM).value_or(cs->gemm_m_);
    plan.gemm_n = GetIntAnn(ann, attr::kScaleGemmN).value_or(cs->gemm_n_);
    plan.gemm_k = GetIntAnn(ann, attr::kScaleGemmK).value_or(cs->gemm_k_);
    plan.gemm_policy =
        GetIntAnn(ann, attr::kScaleGemmPolicy).value_or(cs->gemm_policy_);
    plan.gemm_k_pack =
        GetIntAnn(ann, attr::kScaleGemmKPack).value_or(cs->gemm_k_pack_);
    plan.gemm_elem_bits =
        GetIntAnn(ann, attr::kScaleGemmElemBits).value_or(cs->gemm_elem_bits_);
    plan.a_from_mls =
        GetIntAnn(ann, attr::kScaleAFromMls).value_or(cs->a_from_mls_);
    plan.b_from_mls =
        GetIntAnn(ann, attr::kScaleBFromMls).value_or(cs->b_from_mls_);
    plan.min_m_per_warp =
        GetIntAnn(ann, attr::kScaleMinMPerWarp).value_or(cs->min_m_per_warp_);
    plan.min_n_per_warp =
        GetIntAnn(ann, attr::kScaleMinNPerWarp).value_or(cs->min_n_per_warp_);
    plan.a_mls_trans =
        GetIntAnn(ann, attr::kScaleAMlsTrans).value_or(cs->a_mls_trans_);
    plan.b_mls_trans =
        GetIntAnn(ann, attr::kScaleBMlsTrans).value_or(cs->b_mls_trans_);
    auto shape = ParseScaleShape(cs->dst, plan.scale_k_major);
    plan.scale_shape_mn = shape.first;
    plan.scale_shape_k = shape.second;
    MergePlan(plan);
  }

  void RecordFromGemm(const GemmBlockScaledNode *gemm,
                      const Map<String, ObjectRef> &ann, bool is_a) {
    BufferRegion region = is_a ? gemm->sfaRegion_ : gemm->sfbRegion_;
    if (!region.defined()) {
      return;
    }
    ScaleBufferPlan plan;
    plan.buffer = region->buffer;
    if (is_a) {
      plan.granularity_mn =
          GetIntAnn(ann, "tl.hcu.sf_a_granularity_m").value_or(1);
      plan.granularity_k = GetIntAnn(ann, "sf_a_granularity_k").value_or(1);
      plan.scale_k_major = GetIntAnn(ann, "tl.hcu.a_scale_k_major").value_or(0);
    } else {
      plan.granularity_mn =
          GetIntAnn(ann, "tl.hcu.sf_b_granularity_n").value_or(1);
      plan.granularity_k = GetIntAnn(ann, "sf_b_granularity_k").value_or(1);
      plan.scale_k_major = GetIntAnn(ann, "tl.hcu.b_scale_k_major").value_or(0);
    }
    auto shape = ParseScaleShape(plan.buffer, plan.scale_k_major);
    plan.scale_shape_mn = shape.first;
    plan.scale_shape_k = shape.second;
    MergePlan(plan);
  }

  void MergePlan(const ScaleBufferPlan &incoming) {
    auto it = plans.find(incoming.buffer);
    if (it == plans.end()) {
      plans.emplace(incoming.buffer, incoming);
      return;
    }
    ScaleBufferPlan &existing = it->second;
    if (existing.scale_shape_mn == 0) {
      existing.scale_shape_mn = incoming.scale_shape_mn;
    }
    if (existing.scale_shape_k == 0) {
      existing.scale_shape_k = incoming.scale_shape_k;
    }
    if (existing.granularity_mn == 1 && incoming.granularity_mn != 1) {
      existing.granularity_mn = incoming.granularity_mn;
    }
    if (existing.granularity_k == 1 && incoming.granularity_k != 1) {
      existing.granularity_k = incoming.granularity_k;
    }
    existing.scale_k_major = incoming.scale_k_major;
    existing.resident |= incoming.resident;
    if (incoming.gemm_m > 0) {
      existing.copy_op = incoming.copy_op;
      existing.source_layout = incoming.source_layout;
      existing.scale_format = incoming.scale_format;
      existing.row_order = incoming.row_order;
      existing.op_ctrl = incoming.op_ctrl;
      existing.gemm_threads = incoming.gemm_threads;
      existing.role = incoming.role;
      existing.gemm_m = incoming.gemm_m;
      existing.gemm_n = incoming.gemm_n;
      existing.gemm_k = incoming.gemm_k;
      existing.gemm_policy = incoming.gemm_policy;
      existing.gemm_k_pack = incoming.gemm_k_pack;
      existing.gemm_elem_bits = incoming.gemm_elem_bits;
      existing.a_from_mls = incoming.a_from_mls;
      existing.b_from_mls = incoming.b_from_mls;
      existing.a_mls_trans = incoming.a_mls_trans;
      existing.b_mls_trans = incoming.b_mls_trans;
      existing.min_m_per_warp =
          std::max(existing.min_m_per_warp, incoming.min_m_per_warp);
      existing.min_n_per_warp =
          std::max(existing.min_n_per_warp, incoming.min_n_per_warp);
    }
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.as<OpNode>()) {
        Op tir_op = Downcast<Op>(call->op);
        if (tir_op == CopyScale::Get()) {
          auto cs =
              Downcast<CopyScale>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
          RecordFromCopyScale(cs.get(), call->annotations);
        } else if (tir_op == GemmBlockScaled::Get()) {
          auto gemm = Downcast<GemmBlockScaled>(
              ParseOperator(tvm::ffi::GetRef<Call>(call)));
          RecordFromGemm(gemm.get(), gemm->annotations_, true);
          RecordFromGemm(gemm.get(), gemm->annotations_, false);
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

class ScaleAllocCollector : public StmtExprVisitor {
public:
  std::vector<Buffer> scale_buffers;

private:
  void MaybeAdd(const Buffer &buf) {
    if (!buf.defined() || buf.scope() != "shared.scale") {
      return;
    }
    for (const Buffer &existing : scale_buffers) {
      if (existing.same_as(buf)) {
        return;
      }
    }
    scale_buffers.push_back(buf);
  }

  // TileLang allocates via SBlock.alloc_buffers (printed as
  // sblock_alloc_buffer).
  void VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buf : op->alloc_buffers) {
      MaybeAdd(buf);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  // Fallback for flat AllocBuffer form.
  void VisitStmt_(const AllocBufferNode *op) final {
    MaybeAdd(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  // Also pick up scale buffers referenced by copy_scale / gemm_blockscaled.
  void VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.as<OpNode>()) {
        Op tir_op = Downcast<Op>(call->op);
        if (tir_op == CopyScale::Get()) {
          auto cs =
              Downcast<CopyScale>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
          MaybeAdd(cs->dst);
        } else if (tir_op == GemmBlockScaled::Get()) {
          auto gemm = Downcast<GemmBlockScaled>(
              ParseOperator(tvm::ffi::GetRef<Call>(call)));
          if (gemm->sfaRegion_.defined()) {
            MaybeAdd(gemm->sfaRegion_->buffer);
          }
          if (gemm->sfbRegion_.defined()) {
            MaybeAdd(gemm->sfbRegion_->buffer);
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

/*!
 * Linearize copy_scale / gemm_blockscaled touches for scale-buffer liveness.
 *
 * Lifetime (design + user confirm): live from first ``copy_scale`` write until
 * last consuming ``gemm_blockscaled`` (then free for reuse). Ordinary LDS merge
 * is the same shape: collect sizes → liveness over linearized schedule → pack
 * with freelist reuse → rewrite offsets.
 */
class ScaleLivenessFinder : public StmtExprVisitor {
public:
  struct LiveRange {
    int gen{-1};  // first copy_scale
    int kill{-1}; // last gemm consume (exclusive end = kill+1)
  };

  std::unordered_map<Buffer, LiveRange, ObjectPtrHash, ObjectPtrEqual> ranges;
  std::vector<std::pair<Buffer, Buffer>> consumers;

private:
  int stmt_index_{0};

  void TouchWrite(const Buffer &buf) {
    auto &r = ranges[buf];
    if (r.gen < 0) {
      r.gen = stmt_index_;
    }
  }

  void TouchRead(const Buffer &buf) {
    auto &r = ranges[buf];
    r.kill = stmt_index_;
    if (r.gen < 0) {
      // No copy_scale seen yet; keep gen at this point so range is non-empty.
      r.gen = stmt_index_;
    }
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.as<OpNode>()) {
        Op tir_op = Downcast<Op>(call->op);
        if (tir_op == CopyScale::Get()) {
          auto cs =
              Downcast<CopyScale>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
          TouchWrite(cs->dst);
          ++stmt_index_;
          return;
        }
        if (tir_op == GemmBlockScaled::Get()) {
          auto gemm = Downcast<GemmBlockScaled>(
              ParseOperator(tvm::ffi::GetRef<Call>(call)));
          if (gemm->sfaRegion_.defined() && gemm->sfbRegion_.defined()) {
            consumers.emplace_back(gemm->sfaRegion_->buffer,
                                   gemm->sfbRegion_->buffer);
            TouchRead(gemm->sfaRegion_->buffer);
            TouchRead(gemm->sfbRegion_->buffer);
            ++stmt_index_;
            return;
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

struct FreeInterval {
  int start{0};
  int rows{0};
};

// First-fit among free intervals that can hold ``need`` rows (already
// 8-aligned). If nothing fits but a free hole abuts high_water (start+rows ==
// high_water), reuse that hole and grow high_water only by the shortfall.
int AllocateFromFreelist(std::vector<FreeInterval> *free_list, int need,
                         int *high_water) {
  for (size_t i = 0; i < free_list->size(); ++i) {
    FreeInterval &slot = (*free_list)[i];
    if (slot.rows < need) {
      continue;
    }
    const int start = slot.start;
    if (slot.rows == need) {
      free_list->erase(free_list->begin() + static_cast<long>(i));
    } else {
      slot.start += need;
      slot.rows -= need;
    }
    return start;
  }
  // Grow from a trailing free hole when present.
  for (size_t i = 0; i < free_list->size(); ++i) {
    FreeInterval &slot = (*free_list)[i];
    if (slot.start + slot.rows != *high_water) {
      continue;
    }
    const int start = slot.start;
    free_list->erase(free_list->begin() + static_cast<long>(i));
    *high_water = start + need;
    return start;
  }
  const int start = *high_water;
  *high_water += need;
  return start;
}

void FreeToFreelist(std::vector<FreeInterval> *free_list, int start, int rows) {
  free_list->push_back(FreeInterval{start, rows});
  std::sort(free_list->begin(), free_list->end(),
            [](const FreeInterval &a, const FreeInterval &b) {
              return a.start < b.start;
            });
  // Coalesce adjacent intervals.
  std::vector<FreeInterval> coalesced;
  for (const FreeInterval &slot : *free_list) {
    if (!coalesced.empty() &&
        coalesced.back().start + coalesced.back().rows == slot.start) {
      coalesced.back().rows += slot.rows;
    } else {
      coalesced.push_back(slot);
    }
  }
  *free_list = std::move(coalesced);
}

class ThreadExtentCollector : public StmtExprVisitor {
public:
  int thread_x_extent{-1};

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>()) {
        if (iv->thread_tag == "threadIdx.x") {
          if (const int64_t *ext = as_const_int(op->value)) {
            thread_x_extent = static_cast<int>(*ext);
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

void PlanWaveRows(const std::vector<Buffer> &scale_buffers,
                  ScaleBufferMetaCollector *meta_collector, Target target,
                  int block_size) {
  for (const Buffer &buf : scale_buffers) {
    ScaleBufferPlan plan;
    plan.buffer = buf;
    auto it = meta_collector->plans.find(buf);
    if (it != meta_collector->plans.end()) {
      plan = it->second;
    }
    if (plan.scale_shape_mn <= 0 || plan.scale_shape_k <= 0) {
      auto shape = ParseScaleShape(buf, plan.scale_k_major);
      plan.scale_shape_mn = shape.first;
      plan.scale_shape_k = shape.second;
    }
    if (plan.scale_shape_mn > 0 && plan.scale_shape_k > 0 && plan.gemm_m > 0 &&
        plan.gemm_n > 0 && plan.gemm_k > 0 && block_size > 0) {
      auto policy = GemmWarpPolicy(plan.gemm_policy);
      hcu::ScaleWarpSeg seg = hcu::ComputeScaleWarpSeg(
          *policy.get(), plan.gemm_m, plan.gemm_n, plan.gemm_k,
          plan.gemm_k_pack, plan.gemm_elem_bits,
          plan.gemm_threads > 0 ? plan.gemm_threads : block_size, target,
          plan.a_from_mls != 0, plan.b_from_mls != 0, plan.a_mls_trans != 0,
          plan.b_mls_trans != 0, plan.min_m_per_warp, plan.min_n_per_warp);
      ICHECK_EQ(seg.k_warp, 1)
          << "allocate_scale_buffer requires k_warp==1 for blockscaled";
      const int mn_warps = (plan.role == 0) ? seg.m_seg : seg.n_seg;
      plan.mn_warps = mn_warps;
    } else {
      // Pass-only callers without producer/consumer warp clues use a single
      // segment. No compact-K search is attempted without a known GEMM K.
      plan.used_fallback = true;
    }
    const int mmac_k = plan.gemm_elem_bits == 8 ? 32 : 64;
    const int atom_mn = plan.scale_format == 5 ? 4
                        : (plan.scale_format == 3 || plan.scale_format == 4)
                            ? 2
                            : 1;
    int atom_k = plan.scale_format == 2 ? 4
                 : (plan.scale_format == 1 || plan.scale_format == 3 ||
                    (plan.scale_format == 0 && mmac_k == 64))
                     ? 2
                     : 1;
    // K-major Identity can copy a K4 atom even when MMAC consumes K2.
    if (plan.scale_format == 0 && plan.op_ctrl > 0) {
      atom_k = std::max(atom_k, 1 << plan.op_ctrl);
    }
    ICHECK_EQ(ComputeRowsMn(plan.scale_shape_mn, plan.granularity_mn) %
                  plan.mn_warps,
              0)
        << "Scale MN rows must divide the consumer wave segments";
    plan.layout = {::tl::hcu::ScaleRowMode::FixedBank,
                   0,
                   ComputeRowsMn(plan.scale_shape_mn, plan.granularity_mn) /
                       plan.mn_warps,
                   plan.scale_shape_k *
                       ((mmac_k == 64 && plan.granularity_k >= 64) ? 2 : 1),
                   atom_mn,
                   atom_k,
                   plan.row_order,
                   atom_k};
    ICHECK(plan.layout.Valid()) << "Invalid scale row layout for " << buf->name;
    plan.reserved_rows = plan.mn_warps * plan.layout.SegmentRows();
    meta_collector->plans[buf] = plan;
  }
}

// Choose the common K phase and per-operand row permutations separately
// from physical allocation. The callback replays the actual lifetime timeline
// for a trial footprint without coupling row algebra to the free-list policy.
void PlanComponentRows(const std::vector<Buffer> &component,
                       ScaleBufferMetaCollector *meta_collector, Target target,
                       int block_size,
                       const std::function<int()> &evaluate_peak) {
  using Mode = ::tl::hcu::ScaleRowMode;
  int k_limit = 0;
  for (size_t node = 0; node < component.size(); ++node)
    k_limit =
        std::max(k_limit, meta_collector->plans.at(component[node]).gemm_k);
  std::vector<::tl::hcu::ScaleRowLayoutDesc> best;
  auto best_score = std::make_tuple(std::numeric_limits<int>::max(),
                                    std::numeric_limits<int>::max(),
                                    std::numeric_limits<int>::max());
  // A common logical-K width must not split either side's reuse range or
  // copy atom. Score RowOrder alternatives against the preceding LDS plan;
  // changing the row order must not rewrite an already resolved source layout.
  for (int width = 32; width <= k_limit; width += 32) {
    std::vector<::tl::hcu::ScaleRowLayoutDesc> candidate;
    bool valid = true;
    int rows = 0;
    int lds_cost = 0;
    for (size_t node = 0; node < component.size(); ++node) {
      const auto &plan = meta_collector->plans.at(component[node]);
      auto layout = plan.layout;
      if (width % plan.granularity_k != 0) {
        valid = false;
        break;
      }
      layout.phase_k_rows =
          width / plan.granularity_k * (layout.physical_k / plan.scale_shape_k);
      layout.mode = Mode::KPhasePacked;
      if (!layout.Valid()) {
        valid = false;
        break;
      }
      if (layout.DensePhaseCompatible())
        layout.mode = Mode::DenseXor;
      // RowOrder changes only the within-phase rank. A common K phase
      // therefore permits independently ranked A/B candidates while
      // preserving the joint per-MMAC bank constraint.
      if (plan.copy_op.defined()) {
        auto current = hcu::ScoreScaleRowLayout(
            plan.copy_op.get(), layout, block_size, target, plan.source_layout);
        if (current.has_value()) {
          auto alternative = layout;
          alternative.row_order ^= 1;
          alternative.mode = alternative.DensePhaseCompatible()
                                 ? Mode::DenseXor
                                 : Mode::KPhasePacked;
          auto other =
              hcu::ScoreScaleRowLayout(plan.copy_op.get(), alternative,
                                       block_size, target, plan.source_layout);
          if (other.has_value() &&
              std::make_pair(alternative.SegmentRows(), other.value()) <
                  std::make_pair(layout.SegmentRows(), current.value())) {
            layout = alternative;
            current = other;
          }
          lds_cost += current.value()[1];
        }
      }
      candidate.push_back(layout);
      rows += plan.mn_warps * layout.SegmentRows();
    }
    if (!valid)
      continue;
    std::vector<int> saved;
    for (size_t pos = 0; pos < component.size(); ++pos) {
      auto &plan = meta_collector->plans.at(component[pos]);
      saved.push_back(plan.reserved_rows);
      plan.reserved_rows = plan.mn_warps * candidate[pos].SegmentRows();
    }
    auto cost = std::make_tuple(evaluate_peak(), rows, lds_cost);
    for (size_t pos = 0; pos < component.size(); ++pos)
      meta_collector->plans.at(component[pos]).reserved_rows = saved[pos];
    if (cost < best_score) {
      best_score = cost;
      best = std::move(candidate);
    }
  }
  bool alternating = false;
  for (const auto &layout : best)
    alternating |= layout.PhaseAtoms(1) > 0;
  if (alternating) {
    // Preserve one common phase even when one operand has no size saving.
    for (size_t pos = 0; pos < component.size(); ++pos) {
      auto &plan = meta_collector->plans.at(component[pos]);
      plan.layout = best[pos];
      plan.reserved_rows = plan.mn_warps * plan.layout.SegmentRows();
    }
  }
}

std::pair<std::unordered_map<Buffer, int, ObjectPtrHash, ObjectPtrEqual>, int>
BuildStartRowMap(const std::vector<Buffer> &scale_buffers,
                 ScaleBufferMetaCollector *meta_collector, const Stmt &body,
                 Target target, int block_size) {
  PlanWaveRows(scale_buffers, meta_collector, target, block_size);

  ScaleLivenessFinder finder;
  finder(body);

  struct Item {
    Buffer buf;
    int gen{0};
    int kill{0};
  };
  std::vector<Item> items;
  items.reserve(scale_buffers.size());
  for (const Buffer &buf : scale_buffers) {
    Item item;
    item.buf = buf;
    auto rit = finder.ranges.find(buf);
    if (rit != finder.ranges.end() && rit->second.gen >= 0) {
      item.gen = rit->second.gen;
      item.kill = rit->second.kill >= 0 ? rit->second.kill : rit->second.gen;
    } else {
      // No touches: allocate at the beginning, never free (conservative).
      item.gen = 0;
      item.kill = std::numeric_limits<int>::max() / 4;
    }
    if (meta_collector->plans.at(buf).resident) {
      // First-stage manual WS support: lexical order is not a happens-before
      // relation between thread groups. Own the whole physical interval for
      // the entire kernel, including every alignment/padding hole.
      item.gen = 0;
      item.kill = std::numeric_limits<int>::max() / 4;
    }
    items.push_back(item);
  }

  // Event-driven pack: allocate at gen, free after kill (lifetime ends once
  // gemm_blockscaled has consumed the buffer).
  struct Event {
    int time{0};
    int kind{0}; // 0=free, 1=alloc (free before alloc at same time)
    size_t idx{0};
  };
  std::vector<Event> events;
  events.reserve(items.size() * 2);
  for (size_t i = 0; i < items.size(); ++i) {
    events.push_back(Event{items[i].gen, 1, i});
    events.push_back(Event{items[i].kill + 1, 0, i});
  }
  std::sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
    if (a.time != b.time) {
      return a.time < b.time;
    }
    return std::tie(a.kind, a.idx) < std::tie(b.kind, b.idx);
  });

  std::unordered_map<Buffer, size_t, ObjectPtrHash, ObjectPtrEqual> indices;
  for (size_t i = 0; i < items.size(); ++i) {
    indices.emplace(items[i].buf, i);
  }
  std::vector<std::pair<size_t, size_t>> edges;
  std::vector<std::vector<size_t>> neighbors(items.size());
  for (const auto &[a, b] : finder.consumers) {
    const size_t ai = indices.at(a), bi = indices.at(b);
    edges.emplace_back(ai, bi);
    neighbors[ai].push_back(bi);
    neighbors[bi].push_back(ai);
  }

  struct Placement {
    std::vector<int> starts;
    int high_water{0};
  };
  // Own a complete physical interval, including padding, for each lifetime.
  auto place = [&](const std::vector<int> &banks) {
    Placement result;
    result.starts.resize(items.size(), -1);
    std::vector<FreeInterval> free_list;
    for (const Event &ev : events) {
      if (banks[ev.idx] < 0)
        continue;
      const int rows =
          meta_collector->plans.at(items[ev.idx].buf).reserved_rows;
      if (ev.kind == 0)
        FreeToFreelist(&free_list, result.starts[ev.idx], rows);
      else
        result.starts[ev.idx] =
            AllocateFromFreelist(&free_list, rows, &result.high_water);
    }
    return result;
  };
  auto score = [&](const std::vector<int> &banks) {
    int conflicts = 0;
    for (const auto &[a, b] : edges)
      conflicts += banks[a] >= 0 && banks[a] == banks[b];
    return conflicts;
  };

  std::vector<int> banks(items.size(), -1), colors(items.size(), -1);
  for (size_t root = 0; root < items.size(); ++root) {
    if (colors[root] >= 0) {
      continue;
    }
    std::vector<size_t> component{root};
    colors[root] = 0;
    bool bipartite = true;
    for (size_t pos = 0; pos < component.size(); ++pos) {
      const size_t node = component[pos];
      for (size_t other : neighbors[node]) {
        if (colors[other] < 0) {
          colors[other] = colors[node] ^ 1;
          component.push_back(other);
        } else if (colors[other] == colors[node]) {
          bipartite = false;
        }
      }
    }
    if (bipartite) {
      auto trial_banks = banks;
      std::vector<Buffer> buffers;
      for (size_t node : component) {
        trial_banks[node] = colors[node];
        buffers.push_back(items[node].buf);
      }
      PlanComponentRows(buffers, meta_collector, target, block_size,
                        [&]() { return place(trial_banks).high_water; });
    }
    if (bipartite) {
      auto flipped = banks;
      for (size_t node : component) {
        banks[node] = colors[node];
        flipped[node] = colors[node] ^ 1;
      }
      if (score(flipped) < score(banks)) {
        banks = std::move(flipped);
      }
    } else {
      // Odd cycles / self edges cannot be separated with one stored copy.
      // Greedily minimize already determined conflicts, then row footprint.
      for (size_t node : component) {
        banks[node] = 0;
        auto alternative = banks;
        alternative[node] = 1;
        if (score(alternative) < score(banks)) {
          banks = std::move(alternative);
        }
      }
    }
  }
  const Placement result = place(banks);
  ICHECK_LE(result.high_water, 512)
      << "Scale-buffer capacity exceeded: rows=" << result.high_water
      << ", capacity=512 rows including layout padding";
  for (const auto &[a, b] : edges) {
    if (banks[a] == banks[b]) {
      LOG(WARNING) << "GEMM scale bank conflict: ScaleA=" << items[a].buf->name
                   << ", ScaleB=" << items[b].buf->name << ", both in bank"
                   << banks[a]
                   << "; retaining one copy with correct row addressing";
    } else {
      const auto &pa = meta_collector->plans.at(items[a].buf);
      const auto &pb = meta_collector->plans.at(items[b].buf);
      const int k_extent = std::min(pa.gemm_k, pb.gemm_k);
      const int mmac_k = pa.gemm_elem_bits == 8 ? 32 : 64;
      // Check actual paired consumption, including unequal MN lengths and
      // different scale granularities. A polarity-only check is insufficient
      // when the operands use different layout modes or RowOrders.
      for (int k = 0; k < k_extent; k += mmac_k) {
        const int ka =
            k / pa.granularity_k * (pa.layout.physical_k / pa.scale_shape_k);
        const int kb =
            k / pb.granularity_k * (pb.layout.physical_k / pb.scale_shape_k);
        ICHECK_LT(ka, pa.layout.physical_k);
        ICHECK_LT(kb, pb.layout.physical_k);
        for (int m = 0; m < pa.layout.rows_mn; ++m) {
          for (int n = 0; n < pb.layout.rows_mn; ++n) {
            ICHECK_NE((pa.layout.RowOffset(m, ka) / 4 % 2) ^ banks[a],
                      (pb.layout.RowOffset(n, kb) / 4 % 2) ^ banks[b])
                << "Inconsistent scale layout for paired MMAC consumption";
          }
        }
      }
    }
  }
  std::unordered_map<Buffer, int, ObjectPtrHash, ObjectPtrEqual> start_rows;
  for (size_t i = 0; i < items.size(); ++i) {
    const int physical_start = result.starts[i];
    ICHECK_EQ(physical_start % 8, 0)
        << "Scale-buffer interval base must be 8-row aligned for m0";
    auto &plan = meta_collector->plans[items[i].buf];
    ICHECK_EQ(plan.layout.SegmentRows() % 8, 0)
        << "Every consumer wave's scale base must remain 8-row aligned";
    plan.layout.polarity = banks[i];
    for (int mn = 0; mn < plan.layout.rows_mn; ++mn) {
      for (int k = 0; k < plan.layout.physical_k; ++k) {
        const int row = plan.layout.RowOffset(mn, k);
        ICHECK_GE(row, 0);
        ICHECK_LT(row, plan.layout.SegmentRows());
        ICHECK_EQ(plan.layout.RawRowFromPhysical(row),
                  plan.layout.RawRow(mn, k))
            << "Scale copy/consumer row maps disagree for "
            << items[i].buf->name;
      }
    }
    if (plan.layout.mode == ::tl::hcu::ScaleRowMode::FixedBank) {
      LOG(WARNING) << "FixedBank fallback: buffer=" << items[i].buf->name
                   << "; no selected compact common-K layout; effective_rows="
                   << plan.mn_warps * plan.layout.rows_mn *
                          plan.layout.physical_k
                   << ", padding_rows="
                   << plan.reserved_rows - plan.mn_warps * plan.layout.rows_mn *
                                               plan.layout.physical_k
                   << ", reserved_rows=" << plan.reserved_rows;
    }
    start_rows.emplace(items[i].buf, physical_start);
    meta_collector->plans[items[i].buf].start_row = physical_start;
  }
  return {start_rows, result.high_water};
}

class AllocateScaleBufferMutator : public StmtExprMutator {
public:
  explicit AllocateScaleBufferMutator(
      const std::unordered_map<Buffer, int, ObjectPtrHash, ObjectPtrEqual>
          &start_rows,
      const ScaleBufferMetaCollector &meta)
      : start_rows_(start_rows) {
    for (const auto &[buf, plan] : meta.plans) {
      const auto &d = plan.layout;
      std::ostringstream type;
      type << "tl::hcu::ScaleRowLayout<" << static_cast<int>(d.mode) << ", "
           << d.polarity << ", " << d.rows_mn << ", " << d.physical_k << ", "
           << d.atom_mn << ", " << d.atom_k << ", " << d.row_order << ", "
           << d.phase_k_rows << ">";
      layout_types_.emplace(buf, StringImm(type.str()));
      segment_rows_.emplace(buf, Integer(d.SegmentRows()));
      row_orders_.emplace(buf, Integer(d.row_order));
    }
  }

  static PrimFunc Substitute(PrimFunc f) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsHCU(target.value())) {
      return f;
    }

    // Must run while copy_scale / gemm tileops still exist (before
    // LowerTileOp).
    ScaleAllocCollector alloc_collector;
    alloc_collector(f->body);
    if (alloc_collector.scale_buffers.empty()) {
      return f;
    }

    ScaleBufferMetaCollector meta_collector;
    meta_collector(f->body);

    ThreadExtentCollector thr;
    thr(f->body);
    ICHECK(thr.thread_x_extent > 0)
        << "AllocateScaleBuffer: cannot find threadIdx.x extent on PrimFunc";
    const int block_size = thr.thread_x_extent;

    auto [start_rows, max_end_row] =
        BuildStartRowMap(alloc_collector.scale_buffers, &meta_collector,
                         f->body, target.value(), block_size);
    const int slots = (max_end_row + kRowsPerSlot - 1) / kRowsPerSlot;
    ICHECK_LE(slots, kMaxScaleSlots)
        << "HCU scale_buffer requires <= " << kMaxScaleSlots << " slots, got "
        << slots << " rows=" << max_end_row;

    AllocateScaleBufferMutator mutator(start_rows, meta_collector);
    Stmt body = mutator(f->body);
    // Carry the HCU-only value inside the device body. AnnotateDeviceRegions
    // will enclose this marker, so the backend-neutral SplitHostDevice moves
    // it with the device region without knowing its meaning.
    body = AttrStmt(Integer(0), attr::kHcuScaleBufferSize, Integer(slots),
                    std::move(body));
    f.CopyOnWrite()->body =
        AttrStmt(Integer(0), tirx::attr::device_scope, 0, std::move(body));
    return f;
  }

private:
  std::unordered_map<Buffer, StringImm, ObjectPtrHash, ObjectPtrEqual>
      layout_types_;
  std::unordered_map<Buffer, Integer, ObjectPtrHash, ObjectPtrEqual>
      segment_rows_;
  std::unordered_map<Buffer, Integer, ObjectPtrHash, ObjectPtrEqual>
      row_orders_;

  void AttachLayout(Map<String, ObjectRef> *annotations, const Buffer &buf,
                    const char *type_key, const char *rows_key) const {
    annotations->Set(type_key, layout_types_.at(buf));
    annotations->Set(rows_key, segment_rows_.at(buf));
  }

  Optional<IntImm> LookupStartRow(const Buffer &buf) const {
    auto it = start_rows_.find(buf);
    if (it == start_rows_.end()) {
      return Optional<IntImm>();
    }
    return IntImm(DataType::Int(32), it->second);
  }

  // Rewrite annotation ObjectRef if it is a scale_row_base_* Var.
  ObjectRef RewriteRowBaseAnn(const ObjectRef &obj) const {
    if (const auto *vnode = obj.as<VarNode>()) {
      const std::string &name = vnode->name_hint;
      constexpr const char *prefix = "scale_row_base_";
      if (name.rfind(prefix, 0) == 0) {
        const std::string buf_name = name.substr(std::strlen(prefix));
        for (const auto &[buf, start] : start_rows_) {
          if (buf->name == buf_name) {
            return IntImm(DataType::Int(32), start);
          }
        }
      }
    }
    return obj;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.as<OpNode>()) {
        Op tir_op = Downcast<Op>(call->op);
        if (tir_op == CopyScale::Get() || tir_op == GemmBlockScaled::Get()) {
          auto annotations = call->annotations;
          bool changed = false;
          for (const auto &kv : call->annotations) {
            ObjectRef rewritten = RewriteRowBaseAnn(kv.second);
            if (!rewritten.same_as(kv.second)) {
              annotations.Set(kv.first, rewritten);
              changed = true;
            }
          }
          // Also rewrite by known keys even if value identity matches.
          if (tir_op == CopyScale::Get()) {
            auto cs = Downcast<CopyScale>(
                ParseOperator(tvm::ffi::GetRef<Call>(call)));
            if (auto start = LookupStartRow(cs->dst)) {
              annotations.Set(attr::kScaleRowBase, start.value());
              AttachLayout(&annotations, cs->dst, "tl.hcu.scale_row_layout",
                           "tl.hcu.scale_segment_rows");
              annotations.Set(attr::kScaleRowOrder, row_orders_.at(cs->dst));
              changed = true;
            }
          } else {
            auto gemm = Downcast<GemmBlockScaled>(
                ParseOperator(tvm::ffi::GetRef<Call>(call)));
            if (gemm->sfaRegion_.defined()) {
              if (auto start = LookupStartRow(gemm->sfaRegion_->buffer)) {
                annotations.Set(attr::kScaleARowBase, start.value());
                AttachLayout(&annotations, gemm->sfaRegion_->buffer,
                             "tl.hcu.scale_a_row_layout",
                             "tl.hcu.scale_a_segment_rows");
                annotations.Set(attr::kScaleARowOrder,
                                row_orders_.at(gemm->sfaRegion_->buffer));
                changed = true;
              }
            }
            if (gemm->sfbRegion_.defined()) {
              if (auto start = LookupStartRow(gemm->sfbRegion_->buffer)) {
                annotations.Set(attr::kScaleBRowBase, start.value());
                AttachLayout(&annotations, gemm->sfbRegion_->buffer,
                             "tl.hcu.scale_b_row_layout",
                             "tl.hcu.scale_b_segment_rows");
                annotations.Set(attr::kScaleBRowOrder,
                                row_orders_.at(gemm->sfbRegion_->buffer));
                changed = true;
              }
            }
          }
          if (changed) {
            Call new_call(call->dtype, call->op, call->args, annotations,
                          call->span);
            return Evaluate(new_call);
          }
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    if (op->buffer.scope() == "shared.scale") {
      return Evaluate(0);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // Drop logical shared.scale views from SBlock; physical rows are attrs + Imm.
  Stmt VisitStmt_(const SBlockNode *op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    const auto *block = stmt.as<SBlockNode>();
    ICHECK(block);
    bool need_filter = false;
    for (const Buffer &buf : block->alloc_buffers) {
      if (buf.scope() == "shared.scale") {
        need_filter = true;
        break;
      }
    }
    if (!need_filter) {
      return stmt;
    }
    Array<Buffer> kept;
    for (const Buffer &buf : block->alloc_buffers) {
      if (buf.scope() != "shared.scale") {
        kept.push_back(buf);
      }
    }
    ObjectPtr<SBlockNode> n = tvm::ffi::make_object<SBlockNode>(*block);
    n->alloc_buffers = kept;
    return SBlock(n);
  }

  std::unordered_map<Buffer, int, ObjectPtrHash, ObjectPtrEqual> start_rows_;
};

} // namespace

tvm::transform::Pass AllocateScaleBuffer() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    (void)m;
    (void)ctx;
    return AllocateScaleBufferMutator::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AllocateScaleBuffer", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AllocateScaleBuffer",
                        AllocateScaleBuffer);
}

} // namespace tl
} // namespace tvm
