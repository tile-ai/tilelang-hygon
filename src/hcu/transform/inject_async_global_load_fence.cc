/*!
 * \file inject_async_global_load_fence.cc
 * \brief Lower async wait scopes and insert G2S barriers.
 */

#include "hcu/op/builtin.h"
#include "hcu/target_utils.h"
#include "hcu/transform/gemm_k_loop_utils.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "rocm/op/builtin.h"
#include "transform/common/attr.h"
#include "transform/common/pipeline_utils.h"

#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace {

using namespace tirx;
using namespace ffi;
using ffi::Array;
using ffi::GetRef;

bool IsWaitAttr(const AttrStmtNode *op) {
  return op->attr_key == s_tir::attr::async_wait_queue_scope ||
         op->attr_key == s_tir::attr::async_wait_inflight_count;
}

bool IsCommitAttr(const AttrStmtNode *op) {
  return op->attr_key == s_tir::attr::async_commit_queue_scope;
}

bool IsPtxCommitStmt(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      return call->op.same_as(builtin::ptx_commit_group());
    }
  }
  return false;
}

bool IsPtxWaitGroupStmt(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      return call->op.same_as(builtin::ptx_wait_group());
    }
  }
  return false;
}

bool IsAsyncGldFenceStmt(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      return call->op.same_as(async_gld_fence());
    }
  }
  return false;
}

bool IsSyncWarpStmt(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      return call->op.same_as(sync_warp());
    }
  }
  return false;
}

bool IsSBarrierStmt(const Stmt &stmt) { return IsSyncWarpStmt(stmt); }

Stmt MakePtxWaitGroupStmt(int wait_count) {
  if (wait_count < 0) {
    wait_count = 0;
  }
  return Evaluate(Call(DataType::Handle(), builtin::ptx_wait_group(),
                       {Integer(wait_count)}));
}

Stmt MakeSyncWarpStmt() {
  return Evaluate(Call(DataType::Void(), sync_warp(), {}));
}

void AppendWaveBarrierIfNeeded(Array<Stmt> &result) {
  if (result.empty() || !IsSBarrierStmt(result.back())) {
    result.push_back(MakeSyncWarpStmt());
  }
}

void DropTrailingWaveBarrier(Array<Stmt> &result) {
  if (result.empty() || !IsSBarrierStmt(result.back())) {
    return;
  }
  Array<Stmt> trimmed;
  trimmed.reserve(result.size() - 1);
  for (size_t i = 0; i + 1 < result.size(); ++i) {
    trimmed.push_back(result[i]);
  }
  result = trimmed;
}

void AppendGldWaitSync(Array<Stmt> &result) {
  if (result.empty() || !IsSBarrierStmt(result.back())) {
    result.push_back(MakeSyncWarpStmt());
  }
}

// Reads shared memory and issues no async copy of its own: the consumer side
// of a pipeline stage. Matches plain LDS-read copies; unlike
// IsSharedToLocalCopy it also matches blocks that mix reads with MMA, which is
// what the consumer region of a pipelined GEMM looks like.
bool ReadsSharedWithoutAsyncCopy(const Stmt &stmt) {
  struct Checker : public StmtExprVisitor {
    bool has_shared_src{false};
    bool has_async_copy{false};

    void VisitExpr_(const BufferLoadNode *op) override {
      if (IsSharedBuffer(op->buffer, true)) {
        has_shared_src = true;
      }
      ExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const CallNode *op) override {
      if (op->op.same_as(ds_read_vector()) ||
          op->op.same_as(ds_read_tr16_b64()) ||
          op->op.same_as(ds_read_tr8_b64())) {
        has_shared_src = true;
      }
      if (IsAsyncCopyCall(op)) {
        has_async_copy = true;
      }
      ExprVisitor::VisitExpr_(op);
    }
  };
  Checker checker;
  checker(stmt);
  return checker.has_shared_src && !checker.has_async_copy;
}

bool IsSharedToLocalCopy(const Stmt &stmt) {
  return !StmtContainsMma(stmt) && ReadsSharedWithoutAsyncCopy(stmt);
}

void CollectSharedVarsInExpr(const PrimExpr &expr,
                             std::unordered_set<const VarNode *> *out) {
  PostOrderVisit(expr, [out](const ObjectRef &node) {
    if (const auto *load = node.as<BufferLoadNode>()) {
      if (IsSharedBuffer(load->buffer, true)) {
        out->insert(load->buffer->data.get());
      }
    }
    if (const auto *call = node.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_access_ptr()) &&
          call->args.size() >= 2) {
        if (const auto *var = call->args[1].as<VarNode>()) {
          out->insert(var);
        }
      }
      if (call->op.same_as(access_ptr()) && !call->args.empty()) {
        CollectSharedVarsInExpr(call->args[0], out);
      }
    }
  });
}

void CollectAsyncDstVars(const Stmt &stmt,
                         std::unordered_set<const VarNode *> *out) {
  PostOrderVisit(stmt, [out](const ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    if (call == nullptr || !IsAsyncCopyCall(call) || call->args.empty()) {
      return;
    }
    CollectSharedVarsInExpr(call->args[0], out);
  });
}

void CollectSharedReadVars(const Stmt &stmt,
                           std::unordered_set<const VarNode *> *out) {
  PostOrderVisit(stmt, [out](const ObjectRef &node) {
    if (const auto *load = node.as<BufferLoadNode>()) {
      if (IsSharedBuffer(load->buffer, true)) {
        out->insert(load->buffer->data.get());
      }
    }
    if (const auto *call = node.as<CallNode>()) {
      if (call->op.same_as(ds_read_vector()) ||
          call->op.same_as(ds_read_tr16_b64()) ||
          call->op.same_as(ds_read_tr8_b64())) {
        for (const PrimExpr &arg : call->args) {
          CollectSharedVarsInExpr(arg, out);
        }
      }
    }
  });
}

std::unordered_set<const VarNode *>
CollectPipelinedAsyncBuffers(const Stmt &body) {
  std::unordered_set<const VarNode *> out;
  CollectAsyncDstVars(body, &out);
  return out;
}

bool ReadsPipelinedAsync(const Stmt &stmt,
                         const std::unordered_set<const VarNode *> &pipelined) {
  if (pipelined.empty() || !ReadsSharedWithoutAsyncCopy(stmt)) {
    return false;
  }
  std::unordered_set<const VarNode *> reads;
  CollectSharedReadVars(stmt, &reads);
  for (const VarNode *var : reads) {
    if (pipelined.count(var)) {
      return true;
    }
  }
  return false;
}

// True when the function issues at least one async copy (or commits a group).
bool HasAsyncCopies(const Stmt &root) {
  struct Scanner : public StmtExprVisitor {
    bool found{false};

    void VisitStmt_(const AttrStmtNode *op) override {
      if (IsCommitAttr(op)) {
        found = true;
        return;
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const EvaluateNode *op) override {
      if (IsPtxCommitStmt(GetRef<Evaluate>(op))) {
        found = true;
        return;
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const CallNode *op) override {
      if (IsAsyncCopyCall(op)) {
        found = true;
        return;
      }
      ExprVisitor::VisitExpr_(op);
    }
  };
  Scanner scanner;
  scanner(root);
  return scanner.found;
}

Stmt UnwrapWaitAttrs(Stmt stmt) {
  while (true) {
    const auto *attr = stmt.as<AttrStmtNode>();
    if (attr == nullptr || !IsWaitAttr(attr)) {
      break;
    }
    stmt = attr->body;
  }
  return stmt;
}

bool TryGetWaitInflight(const Stmt &stmt, int *wait_count) {
  const AttrStmtNode *attr = stmt.as<AttrStmtNode>();
  bool found = false;
  while (attr != nullptr && IsWaitAttr(attr)) {
    if (attr->attr_key == s_tir::attr::async_wait_inflight_count) {
      if (const auto *imm = attr->value.as<IntImmNode>()) {
        *wait_count = static_cast<int>(imm->value);
        found = true;
      }
    }
    attr = attr->body.as<AttrStmtNode>();
  }
  return found;
}

bool TryGetPtxWaitCount(const Stmt &stmt, int *wait_count) {
  Stmt inner = UnwrapWaitAttrs(stmt);
  const auto *eval = inner.as<EvaluateNode>();
  if (eval == nullptr) {
    return false;
  }
  const auto *call = eval->value.as<CallNode>();
  if (call == nullptr || !call->op.same_as(builtin::ptx_wait_group()) ||
      call->args.empty()) {
    return false;
  }
  if (const auto *imm = call->args[0].as<IntImmNode>()) {
    *wait_count = static_cast<int>(imm->value);
    return true;
  }
  return false;
}

bool IsWaitPlaceholder(const Stmt &stmt) {
  int unused = 0;
  if (TryGetPtxWaitCount(stmt, &unused)) {
    return true;
  }
  if (!TryGetWaitInflight(stmt, &unused)) {
    return false;
  }
  Stmt inner = UnwrapWaitAttrs(stmt);
  return inner.as<EvaluateNode>() != nullptr;
}

bool IsOutermostCommitStmt(const Stmt &stmt) {
  if (IsPtxCommitStmt(stmt)) {
    return true;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    return IsCommitAttr(attr);
  }
  return false;
}

int CountOutermostCommitsBeforeLds(const Stmt &body, bool stop_at_lds) {
  struct Scanner : public StmtVisitor {
    int commits{0};
    bool seen_lds{false};
    int commit_depth{0};
    bool stop_at_lds{true};

    void NoteCommit() {
      if (commit_depth == 0 && !seen_lds) {
        commits += 1;
      }
    }

    void VisitStmt_(const AttrStmtNode *op) override {
      if (seen_lds) {
        return;
      }
      if (IsWaitAttr(op)) {
        VisitStmt(UnwrapWaitAttrs(GetRef<AttrStmt>(op)));
        return;
      }
      if (IsCommitAttr(op)) {
        NoteCommit();
        ++commit_depth;
        VisitStmt(op->body);
        --commit_depth;
        return;
      }
      StmtVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const SeqStmtNode *op) override {
      if (seen_lds) {
        return;
      }
      Stmt flattened = SeqStmt::Flatten(GetRef<Stmt>(op));
      const auto *seq = flattened.as<SeqStmtNode>();
      if (seq == nullptr) {
        VisitStmt(flattened);
        return;
      }
      for (const Stmt &s : seq->seq) {
        if (seen_lds) {
          break;
        }
        Stmt cur = UnwrapWaitAttrs(s);
        if (IsOutermostCommitStmt(cur)) {
          NoteCommit();
          ++commit_depth;
          VisitStmt(cur);
          --commit_depth;
          continue;
        }
        if (stop_at_lds && IsSharedToLocalCopy(cur)) {
          seen_lds = true;
          break;
        }
        VisitStmt(cur);
      }
    }

    void VisitStmt_(const ForNode *op) override {
      if (!seen_lds) {
        VisitStmt(op->body);
      }
    }

    void VisitStmt_(const EvaluateNode *op) override {
      if (IsPtxCommitStmt(GetRef<Evaluate>(op))) {
        NoteCommit();
      }
    }
  };
  Scanner scanner;
  scanner.stop_at_lds = stop_at_lds;
  scanner(body);
  return scanner.commits;
}

enum class PipelineEventKind { kCommit, kConsumer };

std::vector<PipelineEventKind>
CollectPipelineEvents(const Stmt &body,
                      const std::unordered_set<const VarNode *> &pipelined) {
  struct Collector : public StmtVisitor {
    std::vector<PipelineEventKind> events;
    int commit_depth{0};
    const std::unordered_set<const VarNode *> *pipelined{nullptr};

    bool IsConsumer(const Stmt &stmt) const {
      if (pipelined != nullptr && !pipelined->empty()) {
        return ReadsPipelinedAsync(stmt, *pipelined);
      }
      return ReadsSharedWithoutAsyncCopy(stmt);
    }

    void VisitStmt_(const AttrStmtNode *op) override {
      if (IsWaitAttr(op)) {
        VisitStmt(UnwrapWaitAttrs(GetRef<AttrStmt>(op)));
        return;
      }
      if (IsCommitAttr(op)) {
        if (commit_depth == 0) {
          events.push_back(PipelineEventKind::kCommit);
        }
        ++commit_depth;
        VisitStmt(op->body);
        --commit_depth;
        return;
      }
      StmtVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const SeqStmtNode *op) override {
      Stmt flattened = SeqStmt::Flatten(GetRef<Stmt>(op));
      const auto *seq = flattened.as<SeqStmtNode>();
      if (seq == nullptr) {
        VisitStmt(flattened);
        return;
      }
      for (const Stmt &s : seq->seq) {
        Stmt cur = UnwrapWaitAttrs(s);
        if (commit_depth == 0 && IsOutermostCommitStmt(cur)) {
          events.push_back(PipelineEventKind::kCommit);
          ++commit_depth;
          VisitStmt(cur);
          --commit_depth;
          continue;
        }
        if (commit_depth == 0 && IsConsumer(cur)) {
          events.push_back(PipelineEventKind::kConsumer);
          continue;
        }
        VisitStmt(cur);
      }
    }

    void VisitStmt_(const ForNode *op) override { VisitStmt(op->body); }

    void VisitStmt_(const EvaluateNode *op) override {
      if (commit_depth == 0 && IsPtxCommitStmt(GetRef<Evaluate>(op))) {
        events.push_back(PipelineEventKind::kCommit);
      }
    }
  };
  Collector collector;
  collector.pipelined = &pipelined;
  collector(body);
  return collector.events;
}

int CountCommitRuns(const std::vector<PipelineEventKind> &events) {
  int runs = 0;
  bool in_run = false;
  for (PipelineEventKind kind : events) {
    if (kind == PipelineEventKind::kCommit) {
      if (!in_run) {
        ++runs;
        in_run = true;
      }
    } else {
      in_run = false;
    }
  }
  return runs;
}

struct ConsumerGroupPlan {
  std::vector<int> needs;
  std::vector<int> consumer_counts;
  bool model_ok{true};
};

ConsumerGroupPlan
MakeConsumerGroupPlan(const std::vector<PipelineEventKind> &events) {
  const int n = static_cast<int>(events.size());
  ConsumerGroupPlan plan;
  for (int i = 0; i < n;) {
    if (events[i] != PipelineEventKind::kConsumer) {
      ++i;
      continue;
    }
    int consumers = 0;
    while (i < n && events[i] == PipelineEventKind::kConsumer) {
      ++consumers;
      ++i;
    }
    int need = 0;
    int k = i;
    for (int steps = 0; steps < n; ++steps) {
      if (k >= n) {
        k = 0;
      }
      if (events[k] == PipelineEventKind::kConsumer) {
        break;
      }
      if (events[k] == PipelineEventKind::kCommit) {
        ++need;
      }
      ++k;
    }
    if (need <= 0) {
      DLOG(WARNING) << "consumer group has no producer commit in the K loop; "
                       "the wait values fall back to a drain";
      plan.model_ok = false;
    }
    plan.needs.push_back(need);
    plan.consumer_counts.push_back(consumers);
  }
  if (!plan.needs.empty() &&
      static_cast<int>(plan.needs.size()) != CountCommitRuns(events)) {
    DLOG(WARNING) << "consumer groups do not line up with the commit runs in "
                     "the K loop; the wait values fall back to a drain";
    plan.model_ok = false;
  }
  return plan;
}

struct MainLoopWaitSim {
  std::vector<int> waits;
  int outstanding_end{0};
  bool model_ok{true};
};

MainLoopWaitSim
SimulateMainLoopWaits(const std::vector<PipelineEventKind> &events,
                      const std::vector<int> &needs, int outstanding) {
  MainLoopWaitSim sim;
  size_t g = 0;
  const int n = static_cast<int>(events.size());
  for (int i = 0; i < n;) {
    if (events[i] == PipelineEventKind::kCommit) {
      ++outstanding;
      ++i;
      continue;
    }
    if (g >= needs.size()) {
      sim.model_ok = false;
      break;
    }
    const int need = needs[g++];
    if (need <= 0) {
      sim.model_ok = false;
      break;
    }
    int wait_count = outstanding - need;
    if (wait_count < 0) {
      wait_count = 0;
    }
    if (outstanding != 0 && wait_count >= outstanding) {
      sim.model_ok = false;
      break;
    }
    sim.waits.push_back(wait_count);
    outstanding = wait_count;
    while (i < n && events[i] == PipelineEventKind::kConsumer) {
      ++i;
    }
  }
  if (g != needs.size()) {
    sim.model_ok = false;
  }
  sim.outstanding_end = outstanding;
  return sim;
}

struct SharedPipelineWaitPlan {
  int commits_per_tile{0};
  int main_wait{0};
  int outstanding_after_main{0};
  int num_stages{0};
  bool has_async_copies{false};
  bool register_pipeline{false};
  bool compiler_pipeline{false};
  // False when consumer groups and commit runs in the K loop do not line up,
  // or the prologue does not fill a whole number of pipeline tiles.
  bool model_ok{true};
  // Drain every outstanding commit group instead of using a planned depth.
  bool conservative{false};
  std::vector<int> epilogue_gld_fence_values;
  std::vector<int> main_loop_wait_counts;
  std::vector<int> consumer_group_needs;
  std::vector<int> group_consumer_counts;
  std::unordered_set<const VarNode *> pipelined_buffers;
  // Reuse inflight counts already attached by InjectSoftwarePipeline.
  bool reuse_pipeline_waits{false};
};

// Collect epilogue async_wait_inflight values after the K loop.
std::vector<int> CollectEpilogueWaitCounts(const Stmt &root) {
  struct Collector : public StmtVisitor {
    std::vector<int> counts;
    bool past_main_loop{false};

    void VisitStmt_(const ForNode *op) override {
      StmtVisitor::VisitStmt_(op);
      if (IsGemmKLoop(op)) {
        past_main_loop = true;
      }
    }

    void VisitStmt_(const AttrStmtNode *op) override {
      if (past_main_loop) {
        if (op->attr_key == s_tir::attr::async_wait_queue_scope) {
          const auto *inner = op->body.as<AttrStmtNode>();
          if (inner &&
              inner->attr_key == s_tir::attr::async_wait_inflight_count) {
            if (const auto *imm = inner->value.as<IntImmNode>()) {
              counts.push_back(static_cast<int>(imm->value));
            }
            VisitStmt(inner->body);
            return;
          }
        } else if (op->attr_key == s_tir::attr::async_wait_inflight_count) {
          if (const auto *imm = op->value.as<IntImmNode>()) {
            counts.push_back(static_cast<int>(imm->value));
          }
          VisitStmt(op->body);
          return;
        }
      }
      StmtVisitor::VisitStmt_(op);
    }
  };
  Collector collector;
  collector(root);
  return collector.counts;
}

// Scale iteration-unit waits into commit units when needed.
std::vector<int> ScaleWaitCountsToCommitUnits(const std::vector<int> &counts,
                                              int commits_per_tile,
                                              int main_wait) {
  if (counts.empty() || commits_per_tile <= 1) {
    return counts;
  }
  std::vector<int> scaled = counts;
  if (counts[0] != main_wait && counts[0] * commits_per_tile == main_wait) {
    for (int &v : scaled) {
      v *= commits_per_tile;
    }
  }
  return scaled;
}

// Synthesize the epilogue wait counts from the main-loop plan, used when the
// epilogue carries no explicit wait counts: start at `main_wait` outstanding
// commits and drop by `commits_per_tile` per iteration until none are left.
std::vector<int> SynthesizeEpilogueWaitCounts(int main_wait,
                                              int commits_per_tile) {
  std::vector<int> counts;
  if (commits_per_tile <= 0) {
    return counts;
  }
  int v = main_wait;
  while (true) {
    counts.push_back(v);
    if (v <= 0) {
      break;
    }
    v -= commits_per_tile;
    if (v < 0) {
      v = 0;
    }
  }
  return counts;
}

std::vector<int>
BuildEpilogueGldFenceValues(const std::vector<int> &wait_counts) {
  std::vector<int> shifted;
  if (wait_counts.size() <= 1) {
    return shifted;
  }
  shifted.reserve(wait_counts.size() - 1);
  for (size_t i = 1; i < wait_counts.size(); ++i) {
    shifted.push_back(wait_counts[i]);
  }
  return shifted;
}

int CountPrologueOutermostCommits(const Stmt &root) {
  struct Scanner : public StmtVisitor {
    int commits{0};
    int commit_depth{0};
    bool past_k_loop{false};

    void VisitStmt_(const ForNode *op) override {
      if (past_k_loop) {
        return;
      }
      if (IsGemmKLoop(op)) {
        past_k_loop = true;
        return;
      }
      VisitStmt(op->body);
    }

    void VisitStmt_(const AttrStmtNode *op) override {
      if (past_k_loop) {
        return;
      }
      if (IsWaitAttr(op)) {
        VisitStmt(UnwrapWaitAttrs(GetRef<AttrStmt>(op)));
        return;
      }
      if (IsCommitAttr(op)) {
        if (commit_depth == 0) {
          commits += 1;
        }
        ++commit_depth;
        VisitStmt(op->body);
        --commit_depth;
        return;
      }
      StmtVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const EvaluateNode *op) override {
      if (!past_k_loop && commit_depth == 0 &&
          IsPtxCommitStmt(GetRef<Evaluate>(op))) {
        commits += 1;
      }
    }
  };
  Scanner scanner;
  scanner(root);
  return scanner.commits;
}

SharedPipelineWaitPlan MakeWaitPlan(const Stmt &root) {
  SharedPipelineWaitPlan plan;
  auto loops = CollectGemmKLoops(root);
  if (loops.empty()) {
    return plan;
  }
  plan.has_async_copies = HasAsyncCopies(root);
  const ForNode *k_loop = loops.front();
  // NOTE: the plan is derived from the first GEMM K loop and applies to the
  // whole function. The pipelined kernels this pass serves have a single GEMM
  // K loop (unroll_pipelined_k_loop.cc guards with the same loops.size() != 1
  // assumption); a second K loop would share the first loop's plan.
  plan.register_pipeline = LoopHasAppliedRegisterPipeline(k_loop);
  int cpt = CountOutermostCommitsBeforeLds(k_loop->body, /*stop_at_lds=*/false);
  // A register pipeline materializes its register slots by unrolling whole
  // rotation periods into one iteration. The plan counts commit groups per
  // pipeline tile, which the rotation -- not the unroll -- sets, so report the
  // per-tile count the body carried before it was folded.
  if (auto phases =
          ReadLoopAnnotationAsInt(k_loop, kRegisterPipelinePhaseUnroll)) {
    ICHECK_GT(*phases, 0)
        << "the register-pipeline phase unroll must be positive, got "
        << *phases;
    ICHECK_EQ(cpt % *phases, 0)
        << "the K loop body commits " << cpt << " group(s) but folds "
        << *phases << " pipeline iterations into it";
    cpt /= static_cast<int>(*phases);
  }
  plan.commits_per_tile = cpt;
  plan.pipelined_buffers = CollectPipelinedAsyncBuffers(k_loop->body);

  if (auto ns = GetExplicitPipelinedNumStages(k_loop)) {
    plan.num_stages = static_cast<int>(ns.value().IntValue());
  }
  plan.compiler_pipeline = FindOwnedWaitPlanKLoop(root) != nullptr;
  const int prologue = CountPrologueOutermostCommits(root);
  if (plan.register_pipeline && plan.num_stages >= 2 && cpt > 0) {
    plan.main_wait = (plan.num_stages - 2) * cpt;
  } else if (prologue > 0) {
    if (cpt > 0 && plan.num_stages >= 2) {
      const int filled_minus_one = (plan.num_stages - 1) * cpt;
      const int filled_all = plan.num_stages * cpt;
      plan.model_ok = plan.model_ok &&
                      (prologue == filled_minus_one || prologue == filled_all);
      DLOG_IF(WARNING, !plan.model_ok)
          << "prologue commit count " << prologue
          << " is not a whole number of pipeline tiles (cpt=" << cpt
          << ", num_stages=" << plan.num_stages
          << "); the wait values fall back to a drain";
    }
    plan.main_wait = prologue;
  } else if (plan.num_stages >= 2 && cpt > 0) {
    plan.main_wait = (plan.num_stages - 1) * cpt;
  }
  plan.outstanding_after_main = plan.main_wait;
  if (!plan.register_pipeline) {
    auto events = CollectPipelineEvents(k_loop->body, plan.pipelined_buffers);
    ConsumerGroupPlan groups = MakeConsumerGroupPlan(events);
    plan.consumer_group_needs = groups.needs;
    plan.group_consumer_counts = groups.consumer_counts;
    const std::vector<int> pipeline_waits = CollectEpilogueWaitCounts(root);
    std::vector<int> k_loop_waits;
    {
      struct RootWaitCollector : public StmtVisitor {
        std::vector<int> *counts{nullptr};
        bool in_k{false};
        void VisitStmt_(const ForNode *op) override {
          const bool k = IsGemmKLoop(op);
          const bool prev = in_k;
          if (k) {
            in_k = true;
          }
          VisitStmt(op->body);
          in_k = prev;
        }
        void VisitStmt_(const AttrStmtNode *op) override {
          if (in_k && op->attr_key == s_tir::attr::async_wait_inflight_count) {
            if (const auto *imm = op->value.as<IntImmNode>()) {
              counts->push_back(static_cast<int>(imm->value));
            }
          }
          StmtVisitor::VisitStmt_(op);
        }
        void VisitStmt_(const EvaluateNode *op) override {
          if (!in_k) {
            return;
          }
          int wait_count = 0;
          if (TryGetPtxWaitCount(GetRef<Stmt>(op), &wait_count)) {
            counts->push_back(wait_count);
          }
        }
      };
      RootWaitCollector collector;
      collector.counts = &k_loop_waits;
      collector(root);
    }
    plan.reuse_pipeline_waits =
        plan.compiler_pipeline &&
        (!k_loop_waits.empty() || !pipeline_waits.empty());
    MainLoopWaitSim sim =
        SimulateMainLoopWaits(events, groups.needs, plan.main_wait);
    if (plan.reuse_pipeline_waits && !k_loop_waits.empty()) {
      plan.main_loop_wait_counts = std::move(k_loop_waits);
      plan.outstanding_after_main = plan.main_wait;
    } else {
      plan.main_loop_wait_counts = std::move(sim.waits);
      plan.outstanding_after_main = sim.outstanding_end;
      plan.model_ok = plan.model_ok && groups.model_ok && sim.model_ok;
    }
    plan.conservative =
        !plan.model_ok && plan.compiler_pipeline && plan.has_async_copies &&
        plan.pipelined_buffers.empty() && !plan.reuse_pipeline_waits;
  }
  if (plan.main_wait < 0) {
    plan.main_wait = 0;
  }
  if (plan.register_pipeline) {
    std::vector<int> epi = ScaleWaitCountsToCommitUnits(
        CollectEpilogueWaitCounts(root), plan.commits_per_tile, plan.main_wait);
    if (epi.size() <= 1) {
      epi = SynthesizeEpilogueWaitCounts(plan.main_wait, plan.commits_per_tile);
    }
    plan.epilogue_gld_fence_values = BuildEpilogueGldFenceValues(epi);
  }
  return plan;
}

enum class PipelinePhase { kPrologue, kMainLoop, kEpilogue };

class InjectAsyncGlobalLoadFenceMutator : public StmtMutator {
public:
  explicit InjectAsyncGlobalLoadFenceMutator(const Stmt &root)
      : plan_(MakeWaitPlan(root)), outstanding_(plan_.main_wait) {}

  bool IsPipelinedConsumer(const Stmt &stmt) const {
    if (!plan_.pipelined_buffers.empty()) {
      return ReadsPipelinedAsync(stmt, plan_.pipelined_buffers);
    }
    return ReadsSharedWithoutAsyncCopy(stmt);
  }

  // True when this pass rewrites the kernel's waits (K loop owned and async DMA
  // present). Otherwise the IR is left to the earlier lowering.
  bool OwnsWaitPlan() const {
    return plan_.compiler_pipeline && plan_.has_async_copies;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) override {
    if (IsWaitAttr(op)) {
      if (phase_ == PipelinePhase::kPrologue && !plan_.register_pipeline) {
        return StmtMutator::VisitStmt_(op);
      }
      if (plan_.reuse_pipeline_waits) {
        int wait_count = 0;
        if (TryGetWaitInflight(GetRef<Stmt>(op), &wait_count)) {
          pending_pipeline_waits_.push_back(wait_count);
        }
      }
      const auto *inner = op->body.as<AttrStmtNode>();
      if (inner && IsWaitAttr(inner)) {
        return VisitStmt(inner->body);
      }
      return VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode *op) override {
    const bool is_k = IsGemmKLoop(op);
    const bool lds_for = !is_k && IsSharedToLocalCopy(GetRef<For>(op));
    if (is_k) {
      phase_ = PipelinePhase::kMainLoop;
      outstanding_ = plan_.main_wait;
      k_iter_gld_fence_done_ = false;
      main_loop_wait_idx_ = 0;
    }
    if (lds_for) {
      ++suppress_inner_lds_insert_;
    }
    Stmt body = VisitStmt(op->body);
    if (lds_for) {
      --suppress_inner_lds_insert_;
    }
    if (is_k) {
      if (!plan_.register_pipeline && !plan_.reuse_pipeline_waits &&
          !plan_.conservative && plan_.model_ok &&
          !plan_.main_loop_wait_counts.empty()) {
        ICHECK_EQ(main_loop_wait_idx_,
                  static_cast<int>(plan_.main_loop_wait_counts.size()))
            << "main-loop wait inserts must match the planned wait count";
      }
      phase_ = PipelinePhase::kEpilogue;
      outstanding_ = plan_.outstanding_after_main;
      epi_group_idx_ = 0;
      epi_consumers_in_group_ = 0;
      body = InsertLoopEntryReleaseBarrier(body);
    }
    return For(op->loop_var, op->min, op->extent, op->kind, body,
               op->thread_binding, op->annotations, op->step, op->span);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) override {
    Stmt flattened = SeqStmt::Flatten(GetRef<Stmt>(op));
    const auto *seq_op = flattened.as<SeqStmtNode>();
    if (seq_op == nullptr) {
      return VisitStmt(flattened);
    }

    Array<Stmt> result;
    result.reserve(seq_op->seq.size() + 8);
    bool in_lds_cluster = false;
    int prologue_stage_commits = 0;
    auto append_stmt = [&](const Stmt &stmt) {
      if (const auto *inner = stmt.as<SeqStmtNode>()) {
        for (const Stmt &child : inner->seq) {
          result.push_back(child);
        }
      } else {
        result.push_back(stmt);
      }
    };

    for (size_t i = 0; i < seq_op->seq.size(); ++i) {
      const Stmt &s = seq_op->seq[i];
      Stmt cur = UnwrapWaitAttrs(s);
      // Drop pipeline wait placeholders and re-emit them at the next pipelined
      // consumer, using the inflight count already on the placeholder.
      if (plan_.reuse_pipeline_waits && phase_ != PipelinePhase::kPrologue) {
        int wait_count = 0;
        if (TryGetWaitInflight(s, &wait_count) ||
            TryGetPtxWaitCount(s, &wait_count)) {
          pending_pipeline_waits_.push_back(wait_count);
        }
        if (IsWaitPlaceholder(s)) {
          continue;
        }
      }
      if (IsAsyncGldFenceStmt(cur) || IsPtxWaitGroupStmt(cur)) {
        continue;
      }

      if (cur.as<SeqStmtNode>()) {
        const bool consumer_seq = IsPipelinedConsumer(cur);
        if (!(consumer_seq && !plan_.register_pipeline &&
              (phase_ == PipelinePhase::kMainLoop ||
               phase_ == PipelinePhase::kEpilogue))) {
          in_lds_cluster = false;
          append_stmt(VisitStmt(cur));
          if (plan_.num_stages >= 2 && phase_ == PipelinePhase::kPrologue &&
              i + 1 < seq_op->seq.size() &&
              IsSBarrierStmt(UnwrapWaitAttrs(seq_op->seq[i + 1]))) {
            DropTrailingWaveBarrier(result);
          }
          continue;
        }
      }

      const bool lds = IsSharedToLocalCopy(cur);
      const bool consumer_region = IsPipelinedConsumer(cur);
      const bool mma_region = !lds && consumer_region && StmtContainsMma(cur);
      bool inserted_here = false;
      if (plan_.register_pipeline) {
        if (phase_ == PipelinePhase::kPrologue) {
          if (lds && !in_lds_cluster) {
            MaybeInsertGldFence(result);
            in_lds_cluster = true;
          } else if (!lds) {
            in_lds_cluster = false;
          }
        } else {
          if (in_lds_cluster && !lds) {
            MaybeInsertGldFence(result);
            in_lds_cluster = false;
          }
          if (lds) {
            in_lds_cluster = true;
          }
        }
      } else if (phase_ == PipelinePhase::kMainLoop) {
        if (consumer_region && FlushPendingPipelineWaits(result)) {
          k_iter_gld_fence_done_ = true;
          inserted_here = true;
        } else if (consumer_region && !plan_.reuse_pipeline_waits &&
                   !k_iter_gld_fence_done_) {
          MaybeInsertGldFence(result);
          k_iter_gld_fence_done_ = true;
          inserted_here = true;
        }
      } else if (!plan_.register_pipeline &&
                 phase_ == PipelinePhase::kEpilogue) {
        if (consumer_region && FlushPendingPipelineWaits(result)) {
          inserted_here = true;
        } else if (plan_.conservative) {
          if (consumer_region && suppress_inner_lds_insert_ == 0) {
            MaybeInsertGldFence(result);
            inserted_here = true;
          }
        } else if (consumer_region && suppress_inner_lds_insert_ == 0 &&
                   !plan_.reuse_pipeline_waits) {
          if (!plan_.model_ok || plan_.group_consumer_counts.empty()) {
            MaybeInsertGldFence(result);
            inserted_here = true;
          } else {
            if (epi_consumers_in_group_ == 0) {
              MaybeInsertGldFence(result);
              inserted_here = true;
            }
            ++epi_consumers_in_group_;
            const int g = epi_group_idx_ %
                          static_cast<int>(plan_.group_consumer_counts.size());
            if (epi_consumers_in_group_ >= plan_.group_consumer_counts[g]) {
              ++epi_group_idx_;
              epi_consumers_in_group_ = 0;
            }
          }
        }
      } else if (mma_region || (lds && !in_lds_cluster)) {
        MaybeInsertGldFence(result);
        inserted_here = true;
      }
      if (lds) {
        in_lds_cluster = true;
      } else {
        in_lds_cluster = false;
      }
      // The region the wait was emitted for needs no further wait inside it.
      if (inserted_here) {
        ++suppress_inner_lds_insert_;
      }
      append_stmt(VisitStmt(cur));
      if (inserted_here) {
        --suppress_inner_lds_insert_;
      }
      if (!plan_.register_pipeline && phase_ == PipelinePhase::kMainLoop &&
          IsOutermostCommitStmt(cur)) {
        k_iter_gld_fence_done_ = false;
      }
      if (plan_.num_stages >= 2 && phase_ == PipelinePhase::kPrologue &&
          plan_.commits_per_tile > 0 && IsOutermostCommitStmt(cur)) {
        prologue_stage_commits += 1;
        if (prologue_stage_commits >= plan_.commits_per_tile) {
          bool next_is_sbarrier = false;
          if (i + 1 < seq_op->seq.size()) {
            next_is_sbarrier =
                IsSBarrierStmt(UnwrapWaitAttrs(seq_op->seq[i + 1]));
          }
          if (!next_is_sbarrier) {
            AppendWaveBarrierIfNeeded(result);
          }
          prologue_stage_commits = 0;
        }
      }
    }
    if (plan_.register_pipeline && phase_ != PipelinePhase::kPrologue &&
        in_lds_cluster) {
      MaybeInsertGldFence(result);
    }
    if (result.empty()) {
      return Evaluate(0);
    }
    if (result.size() == 1) {
      return result[0];
    }
    return SeqStmt(result);
  }

private:
  // Every iteration issues the async copies for tile k+1 into the buffer slot
  // the previous iteration read from, so the loop needs a release barrier at
  // its entry: without it a wave can issue its DMA while a peer wave is still
  // reading that slot (write-after-read). The prologue gets the same barrier
  // after its last commit (the commit-stage handling in VisitStmt_(SeqStmt));
  // the fence+barrier pairs before the LDS read clusters only order the
  // producer->consumer direction and do not cover this one. sync_warp lowers
  // to __builtin_amdgcn_s_barrier, which the AMDGPU machine scheduler can see
  // through; that matters for a call site inside the K loop.
  Stmt InsertLoopEntryReleaseBarrier(const Stmt &body) {
    // Only the stage-wise compiler pipeline reuses a slot this way; a register
    // pipeline keeps its own release discipline and a loop without commits per
    // tile has no producer to release.
    if (plan_.register_pipeline || plan_.num_stages < 2 ||
        plan_.commits_per_tile <= 0) {
      return body;
    }
    Stmt flattened = SeqStmt::Flatten(body);
    const auto *seq = flattened.as<SeqStmtNode>();
    if (seq == nullptr) {
      return SeqStmt::Flatten(Array<Stmt>{MakeSyncWarpStmt(), flattened});
    }
    if (!seq->seq.empty() && IsSBarrierStmt(UnwrapWaitAttrs(seq->seq[0]))) {
      return body;
    }
    Array<Stmt> stmts;
    stmts.reserve(seq->seq.size() + 1);
    stmts.push_back(MakeSyncWarpStmt());
    for (const Stmt &s : seq->seq) {
      stmts.push_back(s);
    }
    return SeqStmt(stmts);
  }

  int NextWaitCount() {
    int wait_count = 0;
    if (plan_.conservative) {
      if (phase_ == PipelinePhase::kMainLoop && !plan_.register_pipeline &&
          !plan_.main_loop_wait_counts.empty()) {
        ++main_loop_wait_idx_;
      }
      return wait_count;
    }
    if (!plan_.model_ok && !plan_.register_pipeline &&
        phase_ != PipelinePhase::kPrologue) {
      if (outstanding_ == 0) {
        return -1;
      }
      int need = plan_.commits_per_tile > 0 ? plan_.commits_per_tile : 1;
      wait_count = outstanding_ - need;
      if (wait_count < 0) {
        wait_count = 0;
      }
      outstanding_ = wait_count;
      return wait_count;
    }
    if (phase_ == PipelinePhase::kMainLoop && !plan_.register_pipeline &&
        !plan_.main_loop_wait_counts.empty()) {
      ICHECK_LT(main_loop_wait_idx_,
                static_cast<int>(plan_.main_loop_wait_counts.size()))
          << "main-loop wait insert exceeds the planned wait count";
      wait_count = plan_.main_loop_wait_counts[main_loop_wait_idx_++];
    } else if (phase_ == PipelinePhase::kMainLoop ||
               phase_ == PipelinePhase::kPrologue) {
      wait_count = plan_.main_wait;
    } else if (plan_.register_pipeline) {
      if (epilogue_gld_insert_idx_ >=
          static_cast<int>(plan_.epilogue_gld_fence_values.size())) {
        // The plan shifts the loop-top wait off the collected epilogue
        // schedule, so a pipeline that drains inside the loop (a two-stage
        // one, whose main wait is zero) leaves the schedule empty. That is
        // the same drained state an exhausted schedule with a trailing zero
        // describes, and asking for one more wait there needs no emission.
        ICHECK(plan_.epilogue_gld_fence_values.empty()
                   ? plan_.main_wait == 0
                   : plan_.epilogue_gld_fence_values.back() == 0)
            << "register-pipeline epilogue wait insert exceeds the planned "
               "fence count before drain";
        return -1;
      }
      wait_count = plan_.epilogue_gld_fence_values[epilogue_gld_insert_idx_++];
    } else {
      ICHECK(!plan_.consumer_group_needs.empty())
          << "compiler-pipeline epilogue wait has no consumer-group needs";
      const int g =
          epi_group_idx_ % static_cast<int>(plan_.consumer_group_needs.size());
      const int need = plan_.consumer_group_needs[g];
      ICHECK_GT(need, 0);
      wait_count = outstanding_ - need;
      if (wait_count < 0) {
        wait_count = 0;
      }
      if (outstanding_ == 0) {
        return -1;
      }
      ICHECK(wait_count < outstanding_);
      outstanding_ = wait_count;
    }
    return wait_count;
  }

  void EmitWaitWithBarrier(Array<Stmt> &result, int wait_count) {
    result.push_back(MakePtxWaitGroupStmt(wait_count));
    if (plan_.register_pipeline) {
      AppendGldWaitSync(result);
    } else {
      AppendWaveBarrierIfNeeded(result);
    }
  }

  bool FlushPendingPipelineWaits(Array<Stmt> &result) {
    if (!plan_.reuse_pipeline_waits || pending_pipeline_waits_.empty()) {
      return false;
    }
    if (suppress_inner_lds_insert_ > 0) {
      return false;
    }
    if (phase_ == PipelinePhase::kPrologue && !plan_.register_pipeline) {
      pending_pipeline_waits_.clear();
      return false;
    }
    for (int wait_count : pending_pipeline_waits_) {
      EmitWaitWithBarrier(result, wait_count);
    }
    pending_pipeline_waits_.clear();
    return true;
  }

  void MaybeInsertGldFence(Array<Stmt> &result) {
    if (suppress_inner_lds_insert_ > 0) {
      return;
    }
    if (phase_ == PipelinePhase::kPrologue && !plan_.register_pipeline) {
      return;
    }
    const int wait_count = NextWaitCount();
    if (wait_count < 0) {
      // No planned wait count left for this cluster (see NextWaitCount).
      return;
    }
    // The wait plan counts outstanding commit groups, so the wait must be
    // issued with the group-tracking instruction; a raw s_waitcnt vmcnt(N)
    // would read N as an instruction count and stall until nearly all
    // outstanding DMA has landed.
    EmitWaitWithBarrier(result, wait_count);
  }

  SharedPipelineWaitPlan plan_;
  PipelinePhase phase_{PipelinePhase::kPrologue};
  bool k_iter_gld_fence_done_{false};
  int main_loop_wait_idx_{0};
  int epi_group_idx_{0};
  int epi_consumers_in_group_{0};
  int outstanding_{0};
  int suppress_inner_lds_insert_{0};
  int epilogue_gld_insert_idx_{0};
  std::vector<int> pending_pipeline_waits_;
};

} // namespace

namespace transform {
using namespace tirx::transform;

Pass InjectAsyncGlobalLoadFence() {
  auto pass_func = [](PrimFunc f, IRModule, const PassContext &ctx) {
    bool disable_thread_storage_sync =
        ctx->GetConfig(kDisableThreadStorageSync, Bool(false)).value()->value;
    if (disable_thread_storage_sync) {
      return f;
    }
    Target target = f->GetAttr<Target>(tvm::attr::kTarget).value_or(Target());
    if (!target.defined() || !TargetIsHCU(target)) {
      return f;
    }
    auto *n = f.CopyOnWrite();
    if (FindOwnedWaitPlanKLoop(n->body) == nullptr ||
        !HasAsyncCopies(n->body)) {
      return f;
    }
    InjectAsyncGlobalLoadFenceMutator mutator(n->body);
    if (!mutator.OwnsWaitPlan()) {
      return f;
    }
    n->body = mutator(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectAsyncGlobalLoadFence", {});
}

namespace {

/*!
 * \brief Declares the indexed copies in a scope as asynchronous shared writes.
 *
 * The common synchronization analysis knows the effect from the annotation
 * rather than from the vendor op name. Only the scope owner sets it: the plan
 * pairs every wait it emits with a barrier, so within the scope it can also
 * treat back-to-back copies as writes that need no barrier between them.
 * Elsewhere the copies keep the conservative handling the common pass gives
 * an op it does not know to be asynchronous.
 */
class AsyncSharedWriteAnnotator : public StmtExprMutator {
private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    const auto *call = expr.as<CallNode>();
    if (call == nullptr || !call->op.same_as(hcu_cp_async_idxen()) ||
        call->annotations.Get(attr::kAsyncSharedWrite)) {
      return expr;
    }
    Map<String, ObjectRef> annotations = call->annotations;
    annotations.Set(attr::kAsyncSharedWrite, Integer(1));
    return Call(call->dtype, call->op, call->args, annotations, call->span);
  }
};

/*!
 * \brief Marks the K loop the fence planner rewrites as a sync-owned scope.
 *
 * ThreadSync runs before InjectAsyncGlobalLoadFence and would otherwise place
 * its barriers from the memory conflicts it sees, including the ones the plan
 * covers with its own fence-and-barrier pairs. The marker names the loop whose
 * synchronization the plan owns, so ThreadSync can leave that scope alone and
 * keep handling every other scope on its own.
 */
class OwnedScopeMarker : public StmtExprMutator {
public:
  explicit OwnedScopeMarker(const ForNode *owned_loop)
      : owned_loop_(owned_loop) {}

  int marked{0};

private:
  Stmt VisitStmt_(const ForNode *op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    if (op != owned_loop_) {
      return stmt;
    }
    const auto *rewritten = stmt.as<ForNode>();
    ICHECK(rewritten != nullptr);
    Map<String, Any> annotations = rewritten->annotations;
    annotations.Set(attr::kSyncOwnedScope, Integer(1));
    marked++;
    AsyncSharedWriteAnnotator annotator;
    Stmt body = annotator(std::move(rewritten->body));
    return For(rewritten->loop_var, rewritten->min, rewritten->extent,
               rewritten->kind, body, rewritten->thread_binding, annotations,
               rewritten->step, rewritten->span);
  }

  const ForNode *owned_loop_;
};

} // namespace

Pass MarkFencePlannerOwnership() {
  auto pass_func = [](PrimFunc f, IRModule, const PassContext &ctx) {
    bool disable_thread_storage_sync =
        ctx->GetConfig(kDisableThreadStorageSync, Bool(false)).value()->value;
    if (disable_thread_storage_sync) {
      return f;
    }
    Target target = f->GetAttr<Target>(tvm::attr::kTarget).value_or(Target());
    if (!target.defined() || !TargetIsHCU(target)) {
      return f;
    }
    const ForNode *owned_loop = FindOwnedWaitPlanKLoop(f->body);
    if (owned_loop == nullptr) {
      return f;
    }
    auto *n = f.CopyOnWrite();
    OwnedScopeMarker marker(owned_loop);
    n->body = marker(std::move(n->body));
    ICHECK_EQ(marker.marked, 1)
        << "the owned K loop disappeared while the ownership marker was "
           "being written";
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MarkFencePlannerOwnership", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.transform.InjectAsyncGlobalLoadFence",
           InjectAsyncGlobalLoadFence)
      .def("tl.transform.MarkFencePlannerOwnership", MarkFencePlannerOwnership);
}
} // namespace transform
} // namespace tl
} // namespace tvm
