/*!
 * \file insert_mls_waitcnt.cc
 * \brief Insert s_waitcnt + syncthreads before MLS async consumers on HCU.
 *
 * Runs after InjectSoftwarePipeline (tile-op IR): match producer/consumer on
 * shared Buffer + Region ranges (like cp.async pipeline planning), not on
 * merged dyn-shmem byte offsets.
 *
 * Producers: MatrixLoad -> shared dst region.
 * Consumers: DsReadFormat -> shared src region; Gemm -> shared A/B regions when
 * annotated as MLS-fed (same as cp.async treating Gemm reads as consumers).
 *
 * Algorithm:
 * 1. Insert count: for each consumer, find matching MatrixLoad producer(s),
 *    count outstanding MLS issues between them, take min vmcnt.
 * 2. Merge: walk consumers in reverse; drop a wait if a stricter wait on the
 *    same shared region appears before its related producer is reached.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hcu/op/ds_read_format.h"
#include "hcu/op/mls.h"
#include "hcu/utils/mls_gemm_dep.h"
#include "op/builtin.h"
#include "op/gemm.h"
#include "op/operator.h"
#include "op/utils.h"
#include "transform/common/pipeline_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using tvm::transform::PassContext;

struct SharedRegion {
  Buffer buffer;
  Array<Range> ranges;
};

namespace {

SharedRegion SharedRegionFromBufferRegion(const BufferRegion &region) {
  return SharedRegion{region->buffer, region->region};
}

bool AnnotationIsTrue(const Map<String, ObjectRef> &annotations,
                      const char *key) {
  if (auto value = annotations.Get(key)) {
    if (const auto *imm = value.value().as<IntImmNode>()) {
      return imm->value != 0;
    }
  }
  return false;
}

} // namespace

// ---------------------------------------------------------------------------
// Timeline IR
// ---------------------------------------------------------------------------

enum class EventKind { kProducer, kConsumer };

struct TimelineEvent {
  EventKind kind;
  const Object *stmt{};
  int index{-1};
  int loop_id{-1}; // innermost enclosing loop, -1 if none
  int issue_count{0};
  std::vector<SharedRegion> regions;
};

struct LoopCtx {
  Var loop_var;
  const VarNode *var_node{};
  int begin{0};
  int end{0};
  int64_t extent{1};
  bool pipelined{false};
};

struct WaitPlan {
  int vmcnt{0};
  std::unordered_set<int> producer_indices;
};

class MlsWaitcntPlanner : public StmtVisitor {
public:
  void Build(const Stmt &body) { VisitStmt(body); }

  void Plan() {
    if (!has_mls_tile_ops_)
      return;
    for (const TimelineEvent &ev : timeline_) {
      if (ev.kind != EventKind::kConsumer)
        continue;
      if (auto plan = PlanConsumer(ev)) {
        plan_by_index_[ev.index] = std::move(plan.value());
      }
    }
    MergePlans();
  }

  std::unordered_map<const Object *, std::vector<WaitPlan>>
  OccurrencePlans() const {
    std::unordered_map<const Object *, std::vector<WaitPlan>> out;
    for (const TimelineEvent &ev : timeline_) {
      if (ev.kind != EventKind::kConsumer)
        continue;
      auto it = plan_by_index_.find(ev.index);
      if (it == plan_by_index_.end())
        continue;
      out[ev.stmt].push_back(it->second);
    }
    return out;
  }

  bool HasPlans() const { return !plan_by_index_.empty(); }
  bool HasMlsTileOps() const { return has_mls_tile_ops_; }

private:
  struct MatchResult {
    int producer_idx{-1};
    int keep{0};
  };

  int CountOutstandingSincePrevConsumer(int consumer_idx,
                                        const LoopCtx *loop) const {
    // Interleaved pipeline (ds_read A -> reload A -> ds_read B): vmcnt is
    // global, so count MLS producers issued after the previous consumer in this
    // loop body.
    int prev_consumer = -1;
    for (int i = consumer_idx - 1; i >= 0; --i) {
      if (timeline_[i].kind == EventKind::kConsumer) {
        prev_consumer = i;
        break;
      }
    }
    if (prev_consumer < 0)
      return 0;
    if (loop != nullptr && prev_consumer < loop->begin)
      return 0;

    int keep = 0;
    for (int i = prev_consumer + 1; i < consumer_idx; ++i) {
      if (timeline_[i].kind != EventKind::kProducer)
        continue;
      if (timeline_[i].issue_count <= 0)
        return 0;
      keep += timeline_[i].issue_count;
    }
    return keep;
  }

  bool ConsumerInsideLoop(const TimelineEvent &consumer,
                          const LoopCtx *loop) const {
    if (loop == nullptr)
      return false;
    return consumer.index >= loop->begin && consumer.index < loop->end;
  }

  int FinalizeKeep(int keep, const TimelineEvent &consumer,
                   const LoopCtx *loop) const {
    if (ConsumerInsideLoop(consumer, loop)) {
      keep = std::max(keep,
                      CountOutstandingSincePrevConsumer(consumer.index, loop));
    }
    if (keep > 63)
      return 0;
    return keep;
  }

  std::optional<WaitPlan> PlanConsumer(const TimelineEvent &consumer) {
    const LoopCtx *loop = LoopForConsumer(consumer);
    WaitPlan plan;
    plan.vmcnt = std::numeric_limits<int>::max();

    for (const SharedRegion &creg : consumer.regions) {
      auto match = MatchProducerForRegion(consumer, creg, loop);
      if (!match)
        return std::nullopt;
      plan.vmcnt =
          std::min(plan.vmcnt, FinalizeKeep(match->keep, consumer, loop));
      plan.producer_indices.insert(match->producer_idx);
    }
    if (plan.vmcnt == std::numeric_limits<int>::max())
      plan.vmcnt = 0;
    return plan;
  }

  std::optional<MatchResult>
  MatchProducerAcrossPhases(const TimelineEvent &consumer,
                            const SharedRegion &creg, const LoopCtx *loop) {
    // One wait is shared by every loop phase at runtime. Take the largest keep
    // so the inserted vmcnt reflects the true outstanding MLS count (phase 0
    // prologue pairing), not the tighter carry-over from later phases.
    std::optional<MatchResult> best;
    for (int64_t phase = 0; phase < loop->extent; ++phase) {
      if (auto m = FindProducer(consumer.index, creg, loop, phase, false)) {
        if (!best || m->keep > best->keep)
          best = m;
      }
      if (phase > 0) {
        if (auto m = FindProducer(consumer.index, creg, loop, phase, true)) {
          if (!best || m->keep > best->keep)
            best = m;
        }
      }
    }
    return best;
  }

  int CountLoopProducers(const LoopCtx *loop) const {
    int count = 0;
    for (int i = loop->begin; i < loop->end; ++i) {
      if (timeline_[i].kind != EventKind::kProducer)
        continue;
      if (timeline_[i].issue_count <= 0)
        return 0;
      count += timeline_[i].issue_count;
    }
    return count;
  }

  int CountOutstandingEpilogue(int prod_idx, int consumer_idx,
                               const LoopCtx *loop, int64_t prod_phase) {
    int keep = CountOutstanding(prod_idx, loop->end);
    int producers_per_iter = CountLoopProducers(loop);
    int64_t remaining_iters = loop->extent - 1 - prod_phase;
    if (remaining_iters > 0 && producers_per_iter > 0) {
      keep += static_cast<int>(remaining_iters * producers_per_iter);
    }
    keep += CountOutstanding(loop->end, consumer_idx);
    if (keep > 63)
      return 0;
    return keep;
  }

  std::optional<MatchResult>
  MatchEpilogueProducer(const TimelineEvent &consumer, const SharedRegion &creg,
                        const LoopCtx *loop) {
    auto matches = [&](const TimelineEvent &prod, int64_t prod_phase) {
      for (const SharedRegion &preg : prod.regions) {
        if (RegionMatchAtPhase(preg, creg, loop, prod_phase, 0))
          return true;
      }
      return false;
    };

    // Prefer the latest loop iteration that still writes this slot (k = last,
    // last-1, ...).
    for (int64_t phase = loop->extent - 1; phase >= 0; --phase) {
      for (int i = loop->end - 1; i >= loop->begin; --i) {
        if (timeline_[i].kind != EventKind::kProducer)
          continue;
        if (!matches(timeline_[i], phase))
          continue;
        return MatchResult{
            i, CountOutstandingEpilogue(i, consumer.index, loop, phase)};
      }
    }
    return std::nullopt;
  }

  std::optional<MatchResult>
  MatchProducerForRegion(const TimelineEvent &consumer,
                         const SharedRegion &creg, const LoopCtx *loop) {
    const bool inside_loop = consumer.loop_id >= 0 && loop != nullptr &&
                             consumer.loop_id < static_cast<int>(loops_.size());

    if (inside_loop && loop != nullptr) {
      if (loop->pipelined) {
        return MatchProducerAcrossPhases(consumer, creg, loop);
      }
      return FindProducer(consumer.index, creg, loop, std::nullopt, false);
    }

    // Epilogue: constant shared regions must pair with the last loop writer,
    // not prologue.
    if (loop != nullptr && loop->extent > 1) {
      if (auto m = MatchEpilogueProducer(consumer, creg, loop))
        return m;
    }

    return FindProducer(consumer.index, creg, nullptr, std::nullopt, false);
  }

  std::optional<MatchResult>
  FindProducer(int consumer_idx, const SharedRegion &creg, const LoopCtx *loop,
               std::optional<int64_t> phase, bool wrap) {
    auto try_producer = [&](int prod_idx, int keep) -> MatchResult {
      return MatchResult{prod_idx, keep};
    };

    auto matches = [&](const TimelineEvent &prod, int64_t prod_phase,
                       int64_t cons_phase) {
      for (const SharedRegion &preg : prod.regions) {
        if (RegionMatchAtPhase(preg, creg, loop, prod_phase, cons_phase))
          return true;
      }
      return false;
    };

    if (!phase.has_value()) {
      for (int i = consumer_idx - 1; i >= 0; --i) {
        if (timeline_[i].kind != EventKind::kProducer)
          continue;
        if (RegionMatchConcrete(timeline_[i].regions, creg)) {
          return try_producer(i, CountOutstanding(i, consumer_idx));
        }
      }
      return std::nullopt;
    }

    ICHECK(loop != nullptr);
    int64_t cons_phase = phase.value();

    if (!wrap) {
      for (int i = consumer_idx - 1; i >= 0; --i) {
        if (timeline_[i].kind != EventKind::kProducer)
          continue;
        if (matches(timeline_[i], cons_phase, cons_phase)) {
          return try_producer(i, CountOutstanding(i, consumer_idx));
        }
      }
      return std::nullopt;
    }

    int64_t prod_phase = (cons_phase + loop->extent - 1) % loop->extent;
    for (int i = loop->end - 1; i >= loop->begin; --i) {
      if (timeline_[i].kind != EventKind::kProducer)
        continue;
      if (matches(timeline_[i], prod_phase, cons_phase)) {
        int keep = 0;
        if (consumer_idx <= loop->end) {
          keep = CountOutstandingLoopCarry(i, consumer_idx, loop);
        } else {
          keep = CountOutstandingEpilogue(i, consumer_idx, loop, prod_phase);
        }
        return try_producer(i, keep);
      }
    }
    return std::nullopt;
  }

  bool RegionMatchConcrete(const std::vector<SharedRegion> &prods,
                           const SharedRegion &cons) {
    for (const SharedRegion &preg : prods) {
      if (RegionMatchAtPhase(preg, cons, nullptr, 0, 0))
        return true;
    }
    return false;
  }

  PrimExpr EvalExprPhase(const PrimExpr &expr, const LoopCtx *loop,
                         int64_t phase) {
    if (loop == nullptr)
      return analyzer_.Simplify(expr);
    Map<Var, PrimExpr> subst;
    subst.Set(loop->loop_var, IntImm(loop->loop_var.dtype(), phase));
    return analyzer_.Simplify(Substitute(expr, subst));
  }

  Range EvalRangePhase(const Range &range, const LoopCtx *loop, int64_t phase) {
    return Range::FromMinExtent(EvalExprPhase(range->min, loop, phase),
                                EvalExprPhase(range->extent, loop, phase));
  }

  bool ExprEqual(const PrimExpr &lhs, const PrimExpr &rhs) {
    PrimExpr a = analyzer_.Simplify(lhs);
    PrimExpr b = analyzer_.Simplify(rhs);
    if (const int64_t *ai = as_const_int(a)) {
      if (const int64_t *bi = as_const_int(b))
        return *ai == *bi;
    }
    return analyzer_.CanProve(analyzer_.Simplify(a == b),
                              arith::ProofStrength::kSymbolicBound);
  }

  bool RangeEqualAtPhase(const Range &lhs, const Range &rhs,
                         const LoopCtx *loop, int64_t lhs_phase,
                         int64_t rhs_phase) {
    Range l = EvalRangePhase(lhs, loop, lhs_phase);
    Range r = EvalRangePhase(rhs, loop, rhs_phase);
    return ExprEqual(l->min, r->min) && ExprEqual(l->extent, r->extent);
  }

  bool RegionMatchAtPhase(const SharedRegion &prod, const SharedRegion &cons,
                          const LoopCtx *loop, int64_t prod_phase,
                          int64_t cons_phase) {
    if (!prod.buffer.same_as(cons.buffer))
      return false;
    if (prod.ranges.size() != cons.ranges.size())
      return false;
    for (size_t i = 0; i < prod.ranges.size(); ++i) {
      if (!RangeEqualAtPhase(prod.ranges[i], cons.ranges[i], loop, prod_phase,
                             cons_phase)) {
        return false;
      }
    }
    return true;
  }

  bool ConsumerRegionsOverlap(const TimelineEvent &lhs,
                              const TimelineEvent &rhs) {
    for (const SharedRegion &lreg : lhs.regions) {
      for (const SharedRegion &rreg : rhs.regions) {
        if (RegionMatchAtPhase(lreg, rreg, nullptr, 0, 0))
          return true;
      }
    }
    return false;
  }

  int CountOutstanding(int from_exclusive, int to_exclusive) {
    int keep = 0;
    for (int i = from_exclusive + 1; i < to_exclusive; ++i) {
      if (timeline_[i].kind != EventKind::kProducer)
        continue;
      if (timeline_[i].issue_count <= 0)
        return 0;
      keep += timeline_[i].issue_count;
      if (keep > 63)
        return 0;
    }
    return keep;
  }

  int CountOutstandingLoopCarry(int prod_idx, int consumer_idx,
                                const LoopCtx *loop) {
    int keep = CountOutstanding(prod_idx, loop->end);
    for (int i = loop->begin; i < consumer_idx; ++i) {
      if (timeline_[i].kind != EventKind::kProducer)
        continue;
      if (timeline_[i].issue_count <= 0)
        return 0;
      keep += timeline_[i].issue_count;
      if (keep > 63)
        return 0;
    }
    return keep;
  }

  void MergePlans() {
    std::vector<int> consumer_idxs;
    for (const TimelineEvent &ev : timeline_) {
      if (ev.kind == EventKind::kConsumer && plan_by_index_.count(ev.index)) {
        consumer_idxs.push_back(ev.index);
      }
    }
    std::vector<int> erase;
    for (auto it = consumer_idxs.rbegin(); it != consumer_idxs.rend(); ++it) {
      const TimelineEvent &cons = timeline_[*it];
      auto cur = plan_by_index_.find(*it);
      if (cur == plan_by_index_.end())
        continue;
      int cur_vmcnt = cur->second.vmcnt;
      bool drop = false;
      for (int i = *it - 1; i >= 0; --i) {
        const TimelineEvent &prev = timeline_[i];
        if (prev.kind == EventKind::kProducer) {
          if (cur->second.producer_indices.count(i))
            break;
          continue;
        }
        if (prev.kind != EventKind::kConsumer)
          continue;
        auto prev_plan = plan_by_index_.find(i);
        if (prev_plan != plan_by_index_.end() &&
            prev_plan->second.vmcnt <= cur_vmcnt &&
            ConsumerRegionsOverlap(prev, cons)) {
          drop = true;
          break;
        }
      }
      if (drop)
        erase.push_back(*it);
    }
    for (int idx : erase)
      plan_by_index_.erase(idx);
  }

  const LoopCtx *LoopForConsumer(const TimelineEvent &consumer) {
    if (consumer.loop_id >= 0 &&
        consumer.loop_id < static_cast<int>(loops_.size())) {
      return &loops_[consumer.loop_id];
    }
    for (auto it = loops_.rbegin(); it != loops_.rend(); ++it) {
      if (it->extent > 1)
        return &(*it);
    }
    return loops_.empty() ? nullptr : &loops_.back();
  }

  void VisitStmt_(const ForNode *op) final {
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent),
                   true);
    int id = static_cast<int>(loops_.size());
    int64_t ext = 1;
    if (const auto *imm = op->extent.as<IntImmNode>())
      ext = imm->value;
    loops_.push_back(LoopCtx{op->loop_var, op->loop_var.get(),
                             static_cast<int>(timeline_.size()), 0, ext,
                             GetPipelineNumStages(op).defined()});
    loop_stack_.push_back(id);
    StmtVisitor::VisitStmt_(op);
    loop_stack_.pop_back();
    loops_[id].end = static_cast<int>(timeline_.size());
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (!call->op.as<OpNode>()) {
        StmtVisitor::VisitStmt_(op);
        return;
      }
      Op tir_op = Downcast<Op>(call->op);
      if (tir_op == MatrixLoad::Get()) {
        has_mls_tile_ops_ = true;
        auto mls =
            Downcast<MatrixLoad>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
        PushProducer(op, SharedRegion{mls->dst, mls->dst_ranges});
      } else if (tir_op == DsReadFormat::Get()) {
        has_mls_tile_ops_ = true;
        auto ds =
            Downcast<DsReadFormat>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
        PushConsumer(op, {SharedRegion{ds->src, ds->src_ranges}});
      } else if (tir_op == Gemm::Get()) {
        auto gemm = Downcast<Gemm>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
        const bool a_from_mls =
            AnnotationIsTrue(call->annotations, attr::kHcuAFromMls);
        const bool b_from_mls =
            AnnotationIsTrue(call->annotations, attr::kHcuBFromMls);
        std::vector<SharedRegion> consumer_regions;
        if (a_from_mls && IsSharedBuffer(gemm->a_)) {
          consumer_regions.push_back(
              SharedRegionFromBufferRegion(gemm->aRegion_));
        }
        if (b_from_mls && IsSharedBuffer(gemm->b_)) {
          consumer_regions.push_back(
              SharedRegionFromBufferRegion(gemm->bRegion_));
        }
        if (!consumer_regions.empty()) {
          has_mls_tile_ops_ = true;
          PushConsumer(op, std::move(consumer_regions));
        }
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  void PushProducer(const EvaluateNode *op, SharedRegion reg) {
    TimelineEvent ev;
    ev.kind = EventKind::kProducer;
    ev.stmt = op;
    ev.index = static_cast<int>(timeline_.size());
    ev.loop_id = loop_stack_.empty() ? -1 : loop_stack_.back();
    ev.issue_count = 1;
    ev.regions = {std::move(reg)};
    timeline_.push_back(ev);
  }

  void PushConsumer(const EvaluateNode *op, std::vector<SharedRegion> regs) {
    TimelineEvent ev;
    ev.kind = EventKind::kConsumer;
    ev.stmt = op;
    ev.index = static_cast<int>(timeline_.size());
    ev.loop_id = loop_stack_.empty() ? -1 : loop_stack_.back();
    ev.regions = std::move(regs);
    timeline_.push_back(ev);
  }

  std::vector<TimelineEvent> timeline_;
  std::vector<LoopCtx> loops_;
  std::vector<int> loop_stack_;
  std::unordered_map<int, WaitPlan> plan_by_index_;
  arith::Analyzer analyzer_;
  bool has_mls_tile_ops_{false};
};

// ---------------------------------------------------------------------------
// Insert waitcnt + syncthreads
// ---------------------------------------------------------------------------

class InsertWaitcntMutator : public StmtMutator {
public:
  explicit InsertWaitcntMutator(
      std::unordered_map<const Object *, std::vector<WaitPlan>>
          occurrence_plans)
      : occurrence_plans_(std::move(occurrence_plans)) {}

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt body = StmtMutator::VisitStmt_(op);
    auto it = occurrence_plans_.find(op);
    if (it == occurrence_plans_.end() || it->second.empty())
      return body;
    int vmcnt = it->second.front().vmcnt;
    it->second.erase(it->second.begin());
    return SeqStmt({MakeWait(vmcnt), MakeSync(), body});
  }

private:
  static int PackVmcntImm(int vmcnt) {
    const int kExpcnt = 7, kLgkmcnt = 15, kUbResearch = 1, kBits1213 = 3;
    return (vmcnt & 0xF) | (kExpcnt << 4) | (kUbResearch << 7) |
           (kLgkmcnt << 8) | (kBits1213 << 12) | ((vmcnt & 0x30) << 10);
  }

  static Stmt MakeWait(int vmcnt) {
    return Evaluate(Call(DataType::Int(32), builtin::call_extern(),
                         {StringImm("__builtin_amdgcn_s_waitcnt"),
                          IntImm(DataType::Int(32), PackVmcntImm(vmcnt))}));
  }

  static Stmt MakeSync() {
    return Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                         {StringImm("shared")}));
  }

  std::unordered_map<const Object *, std::vector<WaitPlan>> occurrence_plans_;
};

PrimFunc InsertMlsWaitcnt(PrimFunc func) {
  MlsWaitcntPlanner planner;
  planner.Build(func->body);
  if (!planner.HasMlsTileOps())
    return func;
  planner.Plan();
  if (!planner.HasPlans())
    return func;

  auto *n = func.CopyOnWrite();
  n->body =
      InsertWaitcntMutator(planner.OccurrencePlans())(std::move(func->body));
  return func;
}

} // namespace tl
} // namespace tvm

namespace tvm {
namespace tl {
namespace transform {

tvm::transform::Pass InsertMlsWaitcntPass() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    bool disable =
        ctx->GetConfig(kDisableThreadStorageSync, Bool(false)).value()->value;
    if (disable)
      return f;
    return tl::InsertMlsWaitcnt(std::move(f));
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.InsertMlsWaitcnt", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InsertMlsWaitcnt", InsertMlsWaitcntPass);
}

} // namespace transform
} // namespace tl
} // namespace tvm
