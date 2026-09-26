/*!
 * \file inject_async_global_load_fence.cc
 * \brief Lower async wait scopes and insert G2S barriers.
 *
 * On HCU, tl::wave_barrier and tl::sync_warp both lower to s_barrier. Prefetch
 * (prologue) inserts run only when T.Pipelined depth is >= 2, so handwritten
 * K loops that already have T.sync_warp are left alone.
 */

#include "common/gemm_k_loop_utils.h"
#include "common/pipeline_utils.h"
#include "op/builtin.h"
#include "op/utils.h"

#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <vector>

namespace tvm {
namespace tl {
namespace {

using namespace tirx;
using ffi::Array;
using ffi::GetRef;

bool IsWaitAttr(const AttrStmtNode *op) {
  return op->attr_key == s_tir::attr::async_wait_queue_scope ||
         op->attr_key == "async_wait_queue_scope" ||
         op->attr_key == s_tir::attr::async_wait_inflight_count ||
         op->attr_key == "async_wait_inflight_count";
}

bool IsCommitAttr(const AttrStmtNode *op) {
  return op->attr_key == s_tir::attr::async_commit_queue_scope ||
         op->attr_key == "async_commit_queue_scope";
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

bool IsWaveBarrierStmt(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      return call->op.same_as(wave_barrier());
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

// HCU: wave_barrier is asm s_barrier; sync_warp is __builtin_amdgcn_s_barrier.
bool IsSBarrierStmt(const Stmt &stmt) {
  return IsWaveBarrierStmt(stmt) || IsSyncWarpStmt(stmt);
}

Stmt MakeAsyncGldFenceStmt(int wait_count) {
  if (wait_count < 0) {
    wait_count = 0;
  }
  return Evaluate(
      Call(DataType::Void(), async_gld_fence(), {Integer(wait_count)}));
}

Stmt MakePtxWaitGroupStmt(int wait_count) {
  if (wait_count < 0) {
    wait_count = 0;
  }
  return Evaluate(
      Call(DataType::Handle(), builtin::ptx_wait_group(), {Integer(wait_count)}));
}

Stmt MakeWaveBarrierStmt() {
  return Evaluate(Call(DataType::Void(), wave_barrier(), {}));
}

Stmt MakeSyncWarpStmt() {
  return Evaluate(Call(DataType::Void(), sync_warp(), {}));
}

void AppendWaveBarrierIfNeeded(Array<Stmt> &result) {
  if (result.empty() || !IsSBarrierStmt(result.back())) {
    result.push_back(MakeWaveBarrierStmt());
  }
}

void DropTrailingWaveBarrier(Array<Stmt> &result) {
  if (result.empty() || !IsWaveBarrierStmt(result.back())) {
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

bool IsSharedToLocalCopy(const Stmt &stmt) {
  if (StmtContainsMma(stmt)) {
    return false;
  }
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

struct SharedPipelineWaitPlan {
  int commits_per_tile{0};
  int main_wait{0};
  int num_stages{0};
  bool register_pipeline{false};
  // T.Pipelined / InjectSoftwarePipeline with depth >= 2, or register pipeline.
  bool compiler_pipeline{false};
  std::vector<int> epilogue_gld_fence_values;
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
        if (op->attr_key == s_tir::attr::async_wait_queue_scope ||
            op->attr_key == "async_wait_queue_scope") {
          const auto *inner = op->body.as<AttrStmtNode>();
          if (inner && (inner->attr_key == s_tir::attr::async_wait_inflight_count ||
                        inner->attr_key == "async_wait_inflight_count")) {
            if (const auto *imm = inner->value.as<IntImmNode>()) {
              counts.push_back(static_cast<int>(imm->value));
            }
            VisitStmt(inner->body);
            return;
          }
        } else if (op->attr_key == s_tir::attr::async_wait_inflight_count ||
                   op->attr_key == "async_wait_inflight_count") {
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

std::vector<int> SynthesizeEpilogueWaitCounts(int main_wait, int commits_per_tile) {
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

std::vector<int> BuildEpilogueGldFenceValues(const std::vector<int> &wait_counts) {
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
  const ForNode *k_loop = loops.front();
  plan.register_pipeline = LoopHasRegisterPipeline(k_loop);
  int cpt = CountOutermostCommitsBeforeLds(k_loop->body, /*stop_at_lds=*/true);
  plan.commits_per_tile = cpt;

  if (auto ns = GetExplicitPipelinedNumStages(k_loop)) {
    plan.num_stages = static_cast<int>(ns.value().IntValue());
  }
  plan.compiler_pipeline =
      plan.register_pipeline || plan.num_stages >= 2;
  if (plan.num_stages >= 2 && cpt > 0) {
    if (plan.register_pipeline) {
      plan.main_wait = (plan.num_stages - 2) * cpt;
    } else {
      plan.main_wait = (plan.num_stages - 1) * cpt;
    }
  } else {
    plan.main_wait = CountPrologueOutermostCommits(root);
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

  Stmt VisitStmt_(const AttrStmtNode *op) override {
    if (IsWaitAttr(op)) {
      if (!plan_.compiler_pipeline ||
          (phase_ == PipelinePhase::kPrologue && !plan_.register_pipeline)) {
        return StmtMutator::VisitStmt_(op);
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
    const bool lds_for =
        !is_k && IsSharedToLocalCopy(GetRef<For>(op));
    if (is_k) {
      phase_ = PipelinePhase::kMainLoop;
      outstanding_ = plan_.main_wait;
    }
    if (lds_for) {
      ++suppress_inner_lds_insert_;
    }
    Stmt body = VisitStmt(op->body);
    if (lds_for) {
      --suppress_inner_lds_insert_;
    }
    if (is_k) {
      phase_ = PipelinePhase::kEpilogue;
      outstanding_ = plan_.main_wait;
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
      if (plan_.register_pipeline &&
          (IsAsyncGldFenceStmt(cur) || IsPtxWaitGroupStmt(cur))) {
        continue;
      }

      if (cur.as<SeqStmtNode>()) {
        in_lds_cluster = false;
        append_stmt(VisitStmt(cur));
        if (plan_.num_stages >= 2 && phase_ == PipelinePhase::kPrologue &&
            i + 1 < seq_op->seq.size() &&
            IsSBarrierStmt(UnwrapWaitAttrs(seq_op->seq[i + 1]))) {
          DropTrailingWaveBarrier(result);
        }
        continue;
      }

      const bool lds = IsSharedToLocalCopy(cur);
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
      } else if (lds && !in_lds_cluster) {
        MaybeInsertGldFence(result);
        in_lds_cluster = true;
      } else if (!lds) {
        in_lds_cluster = false;
      }
      append_stmt(VisitStmt(cur));
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
  int NextWaitCount() {
    int wait_count = 0;
    if (phase_ == PipelinePhase::kMainLoop ||
        phase_ == PipelinePhase::kPrologue) {
      wait_count = plan_.main_wait;
    } else if (plan_.register_pipeline) {
      if (epilogue_gld_insert_idx_ >=
          static_cast<int>(plan_.epilogue_gld_fence_values.size())) {
        return -1;
      }
      wait_count = plan_.epilogue_gld_fence_values[epilogue_gld_insert_idx_++];
    } else {
      wait_count = outstanding_ - plan_.commits_per_tile;
      if (wait_count < 0) {
        wait_count = 0;
      }
      outstanding_ = wait_count;
    }
    return wait_count;
  }

  void MaybeInsertGldFence(Array<Stmt> &result) {
    if (!plan_.compiler_pipeline) {
      return;
    }
    if (suppress_inner_lds_insert_ > 0) {
      return;
    }
    if (phase_ == PipelinePhase::kPrologue && !plan_.register_pipeline) {
      return;
    }
    const int wait_count = NextWaitCount();
    if (wait_count < 0) {
      return;
    }
    if (plan_.register_pipeline) {
      result.push_back(MakePtxWaitGroupStmt(wait_count));
      AppendGldWaitSync(result);
    } else {
      result.push_back(MakeAsyncGldFenceStmt(wait_count));
      AppendWaveBarrierIfNeeded(result);
    }
  }

  SharedPipelineWaitPlan plan_;
  PipelinePhase phase_{PipelinePhase::kPrologue};
  int outstanding_{0};
  int suppress_inner_lds_insert_{0};
  int epilogue_gld_insert_idx_{0};
};

} // namespace

namespace transform {
using namespace tirx::transform;

Pass InjectAsyncGlobalLoadFence() {
  auto pass_func = [](PrimFunc f, IRModule, PassContext) {
    auto *n = f.CopyOnWrite();
    InjectAsyncGlobalLoadFenceMutator mutator(n->body);
    n->body = mutator(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectAsyncGlobalLoadFence", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectAsyncGlobalLoadFence",
                        InjectAsyncGlobalLoadFence);
}
} // namespace transform
} // namespace tl
} // namespace tvm
