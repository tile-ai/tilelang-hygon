/*!
 * \file inject_async_global_load_fence.cc
 * \brief Lower async wait scopes to async_gld_fence and insert wave_barrier.
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

Stmt MakeAsyncGldFenceStmt(int wait_count) {
  if (wait_count < 0) {
    wait_count = 0;
  }
  return Evaluate(
      Call(DataType::Void(), async_gld_fence(), {Integer(wait_count)}));
}

Stmt MakeWaveBarrierStmt() {
  return Evaluate(Call(DataType::Void(), wave_barrier(), {}));
}

void AppendWaveBarrierIfNeeded(Array<Stmt> &result) {
  if (result.empty() || !IsWaveBarrierStmt(result.back())) {
    result.push_back(MakeWaveBarrierStmt());
  }
}

bool IsAsyncCopyRootStmt(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      return IsAsyncCopyCall(call);
    }
    return false;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    if (IsWaitAttr(attr) || IsCommitAttr(attr)) {
      return IsAsyncCopyRootStmt(attr->body);
    }
    return false;
  }
  if (const auto *for_op = stmt.as<ForNode>()) {
    return IsAsyncCopyRootStmt(for_op->body);
  }
  if (const auto *ife = stmt.as<IfThenElseNode>()) {
    return IsAsyncCopyRootStmt(ife->then_case);
  }
  return false;
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
  bool register_pipeline{false};
};

bool AnnotationIsTruthy(const ForNode *loop, const char *key) {
  if (auto val = loop->annotations.Get(key)) {
    if (const auto *imm = val.value().as<IntImmNode>()) {
      return imm->value != 0;
    }
  }
  return false;
}

bool LoopHasRegisterPipeline(const ForNode *loop) {
  return AnnotationIsTruthy(loop, "tl_register_pipeline_applied") ||
         AnnotationIsTruthy(loop, kEnableRegisterPipeline);
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

  int num_stages = 0;
  if (auto ns = GetPipelineNumStages(k_loop)) {
    num_stages = static_cast<int>(ns.value().IntValue());
  }
  if (num_stages >= 2 && cpt > 0) {
    plan.main_wait = num_stages * cpt - cpt;
  } else {
    plan.main_wait = CountPrologueOutermostCommits(root);
  }
  if (plan.main_wait < 0) {
    plan.main_wait = 0;
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
      if (phase_ == PipelinePhase::kPrologue) {
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
    int prologue_stage_copies = 0;
    auto append_stmt = [&](const Stmt &stmt) {
      if (const auto *inner = stmt.as<SeqStmtNode>()) {
        for (const Stmt &child : inner->seq) {
          result.push_back(child);
        }
      } else {
        result.push_back(stmt);
      }
    };

    for (const Stmt &s : seq_op->seq) {
      Stmt cur = UnwrapWaitAttrs(s);
      if (IsAsyncGldFenceStmt(cur) ||
          (phase_ != PipelinePhase::kPrologue && IsPtxWaitGroupStmt(cur))) {
        continue;
      }

      if (cur.as<SeqStmtNode>()) {
        in_lds_cluster = false;
        append_stmt(VisitStmt(cur));
        continue;
      }

      const bool lds = IsSharedToLocalCopy(cur);
      if (lds && !in_lds_cluster) {
        MaybeInsertGldFence(result);
        in_lds_cluster = true;
      } else if (!lds) {
        in_lds_cluster = false;
      }
      const bool async_copy = IsAsyncCopyRootStmt(cur);
      append_stmt(VisitStmt(cur));
      if (phase_ == PipelinePhase::kPrologue && async_copy &&
          plan_.commits_per_tile > 0) {
        prologue_stage_copies += 1;
        if (prologue_stage_copies >= plan_.commits_per_tile) {
          AppendWaveBarrierIfNeeded(result);
          prologue_stage_copies = 0;
        }
      }
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
    if (phase_ == PipelinePhase::kMainLoop) {
      wait_count = plan_.main_wait;
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
    if (suppress_inner_lds_insert_ > 0) {
      return;
    }
    if (phase_ == PipelinePhase::kPrologue) {
      return;
    }
    result.push_back(MakeAsyncGldFenceStmt(NextWaitCount()));
    AppendWaveBarrierIfNeeded(result);
  }

  SharedPipelineWaitPlan plan_;
  PipelinePhase phase_{PipelinePhase::kPrologue};
  int outstanding_{0};
  int suppress_inner_lds_insert_{0};
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
