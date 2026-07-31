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
 * Fallback when scale metadata is unavailable: treat buffer as (K, MN) and use
 *   alignedRows = align8((shape[-2] * shape[-1] + 15) / 16)
 */

#include "hcu/op/copy_scale.h"
#include "hcu/op/gemm_partition.h"
#include "hcu/target_utils.h"
#include "hcu/utils/scale_gemm_dep.h"
#include "op/builtin.h"
#include "op/gemm.h"
#include "op/operator.h"

#include <tvm/ir/transform.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstring>
#include <limits>
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

int Align8(int rows) { return (rows + 7) / 8 * 8; }

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
  int aligned_rows{0};
  int start_row{0};
  bool used_fallback{false};
};

int ComputeRowsMn(int scale_shape_mn, int granularity_mn) {
  if (granularity_mn >= kMmacMn) {
    return scale_shape_mn;
  }
  ICHECK(kMmacMn % granularity_mn == 0) << "granularity_mn=" << granularity_mn
                                        << " must divide mmac_mn=" << kMmacMn;
  return scale_shape_mn * granularity_mn / kMmacMn;
}

int ComputeAlignedRows(int scale_shape_mn, int scale_shape_k,
                       int granularity_mn, int granularity_k, int mn_warps,
                       int mmac_k = 64) {
  ICHECK(mn_warps >= 1);
  const int rows_mn = ComputeRowsMn(scale_shape_mn, granularity_mn);
  ICHECK_EQ(rows_mn % mn_warps, 0)
      << "rows_mn=" << rows_mn << " not divisible by mn_warps=" << mn_warps;
  const int k_dup = (mmac_k == 64 && granularity_k >= 64) ? 2 : 1;
  const int phys_k = scale_shape_k * k_dup;
  const int logical_per_seg = (rows_mn / mn_warps) * phys_k;
  return mn_warps * Align8(logical_per_seg);
}

int FallbackAlignedRows(const Buffer &buf) {
  ICHECK_EQ(buf->shape.size(), 2u)
      << "allocate_scale_buffer requires 2D scale_buffer, got rank="
      << buf->shape.size();
  const int64_t *dim_k = as_const_int(buf->shape[0]);
  const int64_t *dim_mn = as_const_int(buf->shape[1]);
  ICHECK(dim_k && dim_mn)
      << "allocate_scale_buffer fallback requires static scale buffer shape";
  const int64_t product = (*dim_k) * (*dim_mn);
  return Align8(static_cast<int>((product + 15) / 16));
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
  void RecordFromCopyScale(const CopyScaleNode *cs,
                           const Map<String, ObjectRef> &ann) {
    ScaleBufferPlan plan;
    plan.buffer = cs->dst;
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

  void RecordFromGemm(const GemmNode *gemm, const Map<String, ObjectRef> &ann,
                      bool is_a) {
    BufferRegion region = is_a ? gemm->sfaRegion_ : gemm->sfbRegion_;
    if (!region.defined()) {
      return;
    }
    ScaleBufferPlan plan;
    plan.buffer = region->buffer;
    if (is_a) {
      plan.granularity_mn = GetIntAnn(ann, "sf_a_granularity_m").value_or(1);
      plan.granularity_k = GetIntAnn(ann, "sf_a_granularity_k").value_or(1);
      plan.scale_k_major = GetIntAnn(ann, "a_scale_k_major").value_or(0);
    } else {
      plan.granularity_mn = GetIntAnn(ann, "sf_b_granularity_n").value_or(1);
      plan.granularity_k = GetIntAnn(ann, "sf_b_granularity_k").value_or(1);
      plan.scale_k_major = GetIntAnn(ann, "b_scale_k_major").value_or(0);
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
    if (incoming.gemm_m > 0) {
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
        } else if (tir_op == Gemm::Get()) {
          auto gemm =
              Downcast<Gemm>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
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
        } else if (tir_op == Gemm::Get()) {
          auto gemm =
              Downcast<Gemm>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
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
        if (tir_op == Gemm::Get()) {
          auto gemm =
              Downcast<Gemm>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
          if (gemm->sfaRegion_.defined() && gemm->sfbRegion_.defined()) {
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

void FillAlignedRows(const std::vector<Buffer> &scale_buffers,
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
          plan.gemm_k_pack, plan.gemm_elem_bits, block_size, target,
          plan.a_from_mls != 0, plan.b_from_mls != 0, plan.a_mls_trans != 0,
          plan.b_mls_trans != 0, plan.min_m_per_warp, plan.min_n_per_warp);
      ICHECK_EQ(seg.k_warp, 1)
          << "allocate_scale_buffer requires k_warp==1 for blockscaled";
      const int mn_warps = (plan.role == 0) ? seg.m_seg : seg.n_seg;
      plan.aligned_rows = ComputeAlignedRows(
          plan.scale_shape_mn, plan.scale_shape_k, plan.granularity_mn,
          plan.granularity_k, mn_warps, plan.gemm_elem_bits == 8 ? 32 : 64);
    } else if (plan.scale_shape_mn > 0 && plan.scale_shape_k > 0) {
      // Fallback without warp clues: single segment.
      plan.aligned_rows =
          ComputeAlignedRows(plan.scale_shape_mn, plan.scale_shape_k,
                             plan.granularity_mn, plan.granularity_k, 1);
      plan.used_fallback = true;
    } else {
      plan.used_fallback = true;
      plan.aligned_rows = FallbackAlignedRows(buf);
    }
    meta_collector->plans[buf] = plan;
  }
}

std::pair<std::unordered_map<Buffer, int, ObjectPtrHash, ObjectPtrEqual>, int>
BuildStartRowMap(const std::vector<Buffer> &scale_buffers,
                 ScaleBufferMetaCollector *meta_collector, const Stmt &body,
                 Target target, int block_size) {
  FillAlignedRows(scale_buffers, meta_collector, target, block_size);

  ScaleLivenessFinder finder;
  finder(body);

  struct Item {
    Buffer buf;
    int aligned_rows{0};
    int gen{0};
    int kill{0};
  };
  std::vector<Item> items;
  items.reserve(scale_buffers.size());
  for (const Buffer &buf : scale_buffers) {
    Item item;
    item.buf = buf;
    item.aligned_rows = meta_collector->plans[buf].aligned_rows;
    auto rit = finder.ranges.find(buf);
    if (rit != finder.ranges.end() && rit->second.gen >= 0) {
      item.gen = rit->second.gen;
      item.kill = rit->second.kill >= 0 ? rit->second.kill : rit->second.gen;
    } else {
      // No touches: allocate at the beginning, never free (conservative).
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
    return a.kind < b.kind;
  });

  std::vector<FreeInterval> free_list;
  std::unordered_map<Buffer, int, ObjectPtrHash, ObjectPtrEqual> start_rows;
  int high_water = 0;
  for (const Event &ev : events) {
    const Item &item = items[ev.idx];
    if (ev.kind == 0) {
      auto it = start_rows.find(item.buf);
      if (it != start_rows.end()) {
        FreeToFreelist(&free_list, it->second, item.aligned_rows);
      }
    } else {
      if (start_rows.count(item.buf)) {
        continue; // already allocated (duplicate gen)
      }
      const int start =
          AllocateFromFreelist(&free_list, item.aligned_rows, &high_water);
      start_rows.emplace(item.buf, start);
      meta_collector->plans[item.buf].start_row = start;
    }
  }

  // Buffers with no events still need a placement.
  for (const Item &item : items) {
    if (!start_rows.count(item.buf)) {
      const int start =
          AllocateFromFreelist(&free_list, item.aligned_rows, &high_water);
      start_rows.emplace(item.buf, start);
      meta_collector->plans[item.buf].start_row = start;
    }
  }
  return {start_rows, high_water};
}

class AllocateScaleBufferMutator : public StmtExprMutator {
public:
  explicit AllocateScaleBufferMutator(
      const std::unordered_map<Buffer, int, ObjectPtrHash, ObjectPtrEqual>
          &start_rows)
      : start_rows_(start_rows) {}

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

    AllocateScaleBufferMutator mutator(start_rows);
    f.CopyOnWrite()->body = mutator(f->body);
    return WithAttr(std::move(f), attr::kHcuScaleBufferSize, Integer(slots));
  }

private:
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
        if (tir_op == CopyScale::Get() || tir_op == Gemm::Get()) {
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
              changed = true;
            }
          } else {
            auto gemm =
                Downcast<Gemm>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
            if (gemm->sfaRegion_.defined()) {
              if (auto start = LookupStartRow(gemm->sfaRegion_->buffer)) {
                annotations.Set(attr::kScaleARowBase, start.value());
                changed = true;
              }
            }
            if (gemm->sfbRegion_.defined()) {
              if (auto start = LookupStartRow(gemm->sfbRegion_->buffer)) {
                annotations.Set(attr::kScaleBRowBase, start.value());
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
