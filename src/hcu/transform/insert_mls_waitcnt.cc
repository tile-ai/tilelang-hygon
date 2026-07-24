/*!
 * \file insert_mls_waitcnt.cc
 * \brief Insert s_waitcnt + syncthreads before MLS async consumers on HCU.
 *
 * Runs after InjectSoftwarePipeline (tile-op IR): match producer/consumer on
 * shared Buffer + overlapping Region ranges (like cp.async pipeline planning),
 * not on merged dyn-shmem byte offsets.
 *
 * Producers: MatrixLoad -> shared dst region.
 * Consumers: DsReadFormat -> shared src region; Gemm -> shared A/B regions when
 * annotated as MLS-fed (same as cp.async treating Gemm reads as consumers).
 *
 * Thread/wave partition IfThenElse branches carry mutually exclusive ConstrSet
 * constraints; producer/consumer pairs across exclusive branches are never
 * matched and do not contribute to outstanding MLS counts.
 *
 * Algorithm:
 * 1. Insert count: for each consumer, find matching MatrixLoad producer(s),
 *    count outstanding MLS issues between them, take min vmcnt.
 * 2. Merge: walk consumers in reverse; drop a wait if a stricter wait on the
 *    same overlapping shared region appears before its related producer is
 *    reached.
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
#include "transform/common/constr_visitor.h"
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

bool UsesThreadOrWavePartition(
    const PrimExpr &expr,
    const std::unordered_set<const VarNode *> &thread_vars,
    const std::unordered_set<const VarNode *> &thread_alias_vars,
    const std::unordered_set<const VarNode *> &seq_thread_aliases) {
  bool found = false;
  tirx::PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (found) {
      return;
    }
    if (const auto *var = node.as<VarNode>()) {
      if (thread_vars.count(var) || thread_alias_vars.count(var) ||
          seq_thread_aliases.count(var)) {
        found = true;
      }
      return;
    }
    if (const auto *call = node.as<CallNode>()) {
      if (call->op.same_as(get_wave_id()) || call->op.same_as(get_warp_idx()) ||
          call->op.same_as(get_warp_idx_sync()) ||
          call->op.same_as(get_lane_idx()) ||
          call->op.same_as(get_warp_group_idx())) {
        found = true;
      }
    }
  });
  return found;
}

bool IsThreadWavePartitionIf(
    const IfThenElseNode *op,
    const std::unordered_set<const VarNode *> &thread_vars,
    const std::unordered_set<const VarNode *> &thread_alias_vars,
    const std::unordered_set<const VarNode *> &seq_thread_aliases) {
  return UsesThreadOrWavePartition(op->condition, thread_vars,
                                   thread_alias_vars, seq_thread_aliases);
}

void CollectSeqThreadAliases(
    const SeqStmtNode *op,
    const std::unordered_set<const VarNode *> &thread_vars,
    std::unordered_set<const VarNode *> *seq_thread_aliases) {
  for (const Stmt &stmt : op->seq) {
    if (const auto *let = stmt.as<BindNode>()) {
      if (const auto *v = let->value.as<VarNode>()) {
        if (thread_vars.count(v)) {
          seq_thread_aliases->insert(let->var.get());
        }
      }
    }
  }
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
  ConstrSet cset;
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
    const TimelineEvent &consumer = timeline_[consumer_idx];
    int prev_consumer = -1;
    for (int i = consumer_idx - 1; i >= 0; --i) {
      if (timeline_[i].kind == EventKind::kConsumer) {
        if (EventsCompatible(timeline_[i], consumer)) {
          prev_consumer = i;
          break;
        }
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
      if (!EventsCompatible(timeline_[i], consumer))
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
    const TimelineEvent &consumer = timeline_[consumer_idx];
    int keep = CountOutstanding(prod_idx, loop->end, consumer_idx);
    int producers_per_iter = CountLoopProducers(loop);
    int64_t remaining_iters = loop->extent - 1 - prod_phase;
    if (remaining_iters > 0 && producers_per_iter > 0) {
      keep += static_cast<int>(remaining_iters * producers_per_iter);
    }
    keep += CountOutstanding(loop->end, consumer_idx, consumer_idx);
    if (keep > 63)
      return 0;
    return keep;
  }

  std::optional<MatchResult>
  MatchEpilogueProducer(const TimelineEvent &consumer, const SharedRegion &creg,
                        const LoopCtx *loop) {
    auto matches = [&](const TimelineEvent &prod, int64_t prod_phase) {
      if (!EventsCompatible(prod, consumer))
        return false;
      for (const SharedRegion &preg : prod.regions) {
        if (RegionOverlapAtPhase(preg, creg, loop, prod_phase, 0))
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
    const TimelineEvent &consumer = timeline_[consumer_idx];
    auto try_producer = [&](int prod_idx, int keep) -> MatchResult {
      return MatchResult{prod_idx, keep};
    };

    auto matches = [&](const TimelineEvent &prod, int64_t prod_phase,
                       int64_t cons_phase) {
      if (!EventsCompatible(prod, consumer))
        return false;
      for (const SharedRegion &preg : prod.regions) {
        if (RegionOverlapAtPhase(preg, creg, loop, prod_phase, cons_phase))
          return true;
      }
      return false;
    };

    if (!phase.has_value()) {
      for (int i = consumer_idx - 1; i >= 0; --i) {
        if (timeline_[i].kind != EventKind::kProducer)
          continue;
        if (!EventsCompatible(timeline_[i], consumer))
          continue;
        if (RegionMatchConcrete(timeline_[i].regions, creg)) {
          return try_producer(i,
                              CountOutstanding(i, consumer_idx, consumer_idx));
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
          return try_producer(i,
                              CountOutstanding(i, consumer_idx, consumer_idx));
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
      if (RegionOverlapAtPhase(preg, cons, nullptr, 0, 0))
        return true;
    }
    return false;
  }

  bool EventsCompatible(const TimelineEvent &lhs,
                        const TimelineEvent &rhs) const {
    arith::Analyzer analyzer;
    ConstrSet combined;
    combined.Extend(lhs.cset);
    combined.Extend(rhs.cset);
    combined.Populate(analyzer);
    PrimExpr conj =
        tirx::And(lhs.cset.ToConjunction(), rhs.cset.ToConjunction());
    return !analyzer.CanProve(tirx::Not(conj));
  }

  ConstrSet GetConstrSet() const {
    return ConstrSet{.constrs_ = constr_stack_};
  }

  struct ConstrGuard {
    std::vector<Constr> &constrs;
    ~ConstrGuard() { constrs.pop_back(); }
  };

  template <typename... Args> ConstrGuard MakeGuard(Args... args) {
    constr_stack_.push_back(Constr(args...));
    return ConstrGuard{constr_stack_};
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

  bool RangeOverlapAtPhase(const Range &lhs, const Range &rhs,
                           const LoopCtx *loop, int64_t lhs_phase,
                           int64_t rhs_phase) {
    Range l = EvalRangePhase(lhs, loop, lhs_phase);
    Range r = EvalRangePhase(rhs, loop, rhs_phase);
    PrimExpr l_end = l->min + l->extent;
    PrimExpr r_end = r->min + r->extent;
    PrimExpr overlap = tirx::And(l->min < r_end, r->min < l_end);
    if (const int64_t *l_min = as_const_int(analyzer_.Simplify(l->min))) {
      if (const int64_t *l_ext = as_const_int(analyzer_.Simplify(l->extent))) {
        if (const int64_t *r_min = as_const_int(analyzer_.Simplify(r->min))) {
          if (const int64_t *r_ext =
                  as_const_int(analyzer_.Simplify(r->extent))) {
            return *l_min < *r_min + *r_ext && *r_min < *l_min + *l_ext;
          }
        }
      }
    }
    return analyzer_.CanProve(overlap, arith::ProofStrength::kSymbolicBound);
  }

  bool RegionOverlapAtPhase(const SharedRegion &prod, const SharedRegion &cons,
                            const LoopCtx *loop, int64_t prod_phase,
                            int64_t cons_phase) {
    if (!prod.buffer.same_as(cons.buffer))
      return false;
    if (prod.ranges.size() != cons.ranges.size())
      return false;
    for (size_t i = 0; i < prod.ranges.size(); ++i) {
      if (!RangeOverlapAtPhase(prod.ranges[i], cons.ranges[i], loop, prod_phase,
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
        if (RegionOverlapAtPhase(lreg, rreg, nullptr, 0, 0))
          return true;
      }
    }
    return false;
  }

  int CountOutstanding(int from_exclusive, int to_exclusive, int consumer_idx) {
    const TimelineEvent &consumer = timeline_[consumer_idx];
    int keep = 0;
    for (int i = from_exclusive + 1; i < to_exclusive; ++i) {
      if (timeline_[i].kind != EventKind::kProducer)
        continue;
      if (!EventsCompatible(timeline_[i], consumer))
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
    const TimelineEvent &consumer = timeline_[consumer_idx];
    int keep = CountOutstanding(prod_idx, loop->end, consumer_idx);
    for (int i = loop->begin; i < consumer_idx; ++i) {
      if (timeline_[i].kind != EventKind::kProducer)
        continue;
      if (!EventsCompatible(timeline_[i], consumer))
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

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      const std::string &tag = iv->thread_tag;
      if (tag.rfind("threadIdx", 0) == 0) {
        thread_vars_.insert(iv->var.get());
        Range dom =
            Range::FromMinExtent(tirx::make_zero(op->value.dtype()), op->value);
        auto guard = MakeGuard(iv->var, dom);
        StmtVisitor::VisitStmt_(op);
        thread_vars_.erase(iv->var.get());
        return;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BindNode *op) final {
    if (const auto *v = op->value.as<VarNode>()) {
      if (thread_vars_.count(v)) {
        thread_alias_vars_.insert(op->var.get());
      }
    }
    auto guard = MakeGuard(op->var, op->value);
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    if (IsThreadWavePartitionIf(op, thread_vars_, thread_alias_vars_,
                                seq_thread_aliases_)) {
      {
        auto guard = MakeGuard(op->condition);
        VisitStmt(op->then_case);
      }
      if (op->else_case.defined()) {
        auto guard = MakeGuard(tirx::Not(op->condition));
        VisitStmt(op->else_case.value());
      }
      return;
    }
    VisitStmt(op->then_case);
    if (op->else_case.defined()) {
      VisitStmt(op->else_case.value());
    }
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    std::unordered_set<const VarNode *> saved_seq_aliases = seq_thread_aliases_;
    seq_thread_aliases_.clear();
    CollectSeqThreadAliases(op, thread_vars_, &seq_thread_aliases_);
    for (const Stmt &stmt : op->seq) {
      VisitStmt(stmt);
    }
    seq_thread_aliases_ = saved_seq_aliases;
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
    ev.cset = GetConstrSet();
    timeline_.push_back(ev);
  }

  void PushConsumer(const EvaluateNode *op, std::vector<SharedRegion> regs) {
    TimelineEvent ev;
    ev.kind = EventKind::kConsumer;
    ev.stmt = op;
    ev.index = static_cast<int>(timeline_.size());
    ev.loop_id = loop_stack_.empty() ? -1 : loop_stack_.back();
    ev.regions = std::move(regs);
    ev.cset = GetConstrSet();
    timeline_.push_back(ev);
  }

  std::vector<TimelineEvent> timeline_;
  std::vector<LoopCtx> loops_;
  std::vector<int> loop_stack_;
  std::unordered_map<int, WaitPlan> plan_by_index_;
  std::vector<Constr> constr_stack_;
  std::unordered_set<const VarNode *> thread_vars_;
  std::unordered_set<const VarNode *> thread_alias_vars_;
  std::unordered_set<const VarNode *> seq_thread_aliases_;
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
