/*!
 * \file inject_async_mma_fence.cc
 * \brief Insert LDS waits before MMA.
 *
 * A wait on lgkmcnt already issued after the shared-memory loads is enough to
 * order them against the following MMA, so no extra wave barrier is emitted in
 * front of the MMA in that case: it would be redundant, and inside a
 * warp-divergent region it would also serialize the G2S and LDS traffic of the
 * participating waves.
 */

#include "common/gemm_k_loop_utils.h"
#include "common/pipeline_utils.h"
#include "hcu/target_utils.h"
#include "op/builtin.h"
#include "op/utils.h"

#include <algorithm>

#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {
using namespace tirx;
using ffi::Array;
using ffi::GetRef;

namespace {

Stmt MakeWaveBarrierStmt() {
  return Evaluate(Call(DataType::Void(), wave_barrier(), {}));
}

Stmt MakeSldFenceStmt(int wait_count) {
  constexpr int kLgkmcntMax = 15;
  if (wait_count < 0) {
    wait_count = 0;
  } else if (wait_count > kLgkmcntMax) {
    wait_count = kLgkmcntMax;
  }
  return Evaluate(
      Call(DataType::Void(), async_gld_sld_fence(), {Integer(wait_count)}));
}

// AMDGCN s_waitcnt immediate: wait only on lgkmcnt, leave vmcnt/expcnt idle.
int PackLgkmcntImm(int cnt) {
  constexpr int kLgkmcntMax = 15;
  constexpr int kIdleVmcnt = 0xF;
  constexpr int kIdleExpcnt = 7;
  constexpr int kWaitcntBit7 = 1;
  constexpr int kWaitcntHi = 3;
  cnt = std::max(0, std::min(cnt, kLgkmcntMax));
  return kIdleVmcnt | (kIdleExpcnt << 4) | (kWaitcntBit7 << 7) | (cnt << 8) |
         (kWaitcntHi << 12) | (kWaitcntHi << 14);
}

Stmt MakeLgkmcntWaitcntStmt(int cnt) {
  return Evaluate(Call(DataType::Int(32), builtin::call_extern(),
                       {StringImm("__builtin_amdgcn_s_waitcnt"),
                        IntImm(DataType::Int(32), PackLgkmcntImm(cnt))}));
}

void AppendRegisterPipelineLdsWait(Array<Stmt> &seq, int lds_count) {
  seq.push_back(MakeLgkmcntWaitcntStmt(lds_count));
}

class LoadCounter : public StmtExprVisitor {
public:
  int total_loads = 0;
  int current_multiplier = 1;

  void VisitStmt_(const ForNode *op) override {
    int64_t extent = 1;
    if (auto imm = op->extent.as<IntImmNode>()) {
      extent = imm->value;
    }
    int prev = current_multiplier;
    current_multiplier *= static_cast<int>(extent);
    StmtVisitor::VisitStmt_(op);
    current_multiplier = prev;
  }

  void VisitExpr_(const BufferLoadNode *op) override {
    if (IsSharedBuffer(op->buffer, true)) {
      total_loads += current_multiplier;
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) override {
    if (op->op.same_as(ds_read_vector()) ||
        op->op.same_as(ds_read_tr16_b64()) ||
        op->op.same_as(ds_read_tr8_b64())) {
      total_loads += current_multiplier;
      return;
    }
    ExprVisitor::VisitExpr_(op);
  }
};

enum class PipelinePhase { kBeforeMainLoop, kInMainLoop, kAfterMainLoop };

bool IsAsyncWaitScopeStmt(const Stmt &stmt) {
  const auto *op = stmt.as<AttrStmtNode>();
  if (op == nullptr) {
    return false;
  }
  return op->attr_key == s_tir::attr::async_wait_queue_scope ||
         op->attr_key == s_tir::attr::async_wait_inflight_count;
}

bool CallProvidesLgkmcntWait(const CallNode *call, int *lgkmcnt) {
  if (call == nullptr) {
    return false;
  }
  if (call->op.same_as(async_gld_sld_fence())) {
    if (call->args.empty()) {
      return false;
    }
    const auto *imm = call->args[0].as<IntImmNode>();
    if (imm == nullptr) {
      return false;
    }
    *lgkmcnt = static_cast<int>(imm->value);
    return true;
  }
  if (!call->op.same_as(builtin::call_extern()) || call->args.size() < 2) {
    return false;
  }
  const auto *name = call->args[0].as<StringImmNode>();
  if (name == nullptr || name->value != "__builtin_amdgcn_s_waitcnt") {
    return false;
  }
  const auto *imm = call->args[1].as<IntImmNode>();
  if (imm == nullptr) {
    return false;
  }
  // s_waitcnt immediate: lgkmcnt lives in bits [11:8]. 15 means "do not wait
  // on lgkmcnt" (vmcnt/expcnt-only waits from T.s_waitcnt).
  constexpr int kIdleLgkmcnt = 15;
  const int cnt = (static_cast<int>(imm->value) >> 8) & 0xF;
  if (cnt == kIdleLgkmcnt) {
    return false;
  }
  *lgkmcnt = cnt;
  return true;
}

// True if `stmt` already waits on lgkmcnt (T.s_waitcnt / async_gld_sld_fence).
// Any such wait is treated as the MMA LDS barrier; do not insert another
// async_gld_sld_fence(0) on top of it.
bool StmtProvidesLgkmcntWait(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&found](const ObjectRef &node) {
    if (const auto *call = node.as<CallNode>()) {
      int lgkmcnt = 0;
      if (CallProvidesLgkmcntWait(call, &lgkmcnt)) {
        found = true;
      }
    }
  });
  return found;
}

struct EpilogueMmaCounts {
  int total{0};
  int with_pending_lds{0};
};

EpilogueMmaCounts CountEpilogueMmas(const Stmt &root) {
  struct Collector : public StmtVisitor {
    EpilogueMmaCounts counts;
    bool past_main_loop{false};
    int pending_load_count{0};

    void VisitStmt_(const ForNode *op) override {
      if (IsGemmKLoop(op)) {
        VisitStmt(op->body);
        past_main_loop = true;
        pending_load_count = 0;
        return;
      }
      StmtVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const SeqStmtNode *op) override {
      Stmt flattened = SeqStmt::Flatten(GetRef<Stmt>(op));
      const auto *seq_op = flattened.as<SeqStmtNode>();
      if (seq_op == nullptr) {
        VisitStmt(flattened);
        return;
      }
      for (const Stmt &s : seq_op->seq) {
        if (past_main_loop && IsMmaCluster(s)) {
          ++counts.total;
          if (pending_load_count > 0) {
            ++counts.with_pending_lds;
          }
          pending_load_count = 0;
          continue;
        }
        if (past_main_loop && !StmtContainsMma(s)) {
          LoadCounter counter;
          counter(s);
          if (IsAsyncWaitScopeStmt(s) && counter.total_loads > 0) {
            pending_load_count = counter.total_loads;
          } else {
            pending_load_count += counter.total_loads;
          }
          if (StmtProvidesLgkmcntWait(s)) {
            pending_load_count = 0;
          }
        }
        VisitStmt(s);
      }
    }
  };
  Collector collector;
  collector(root);
  return collector.counts;
}

class MMABarrierMutator : public StmtExprMutator {
public:
  explicit MMABarrierMutator(const Stmt &root_body)
      : epilogue_counts_(CountEpilogueMmas(root_body)) {
    for (const ForNode *loop : CollectGemmKLoops(root_body)) {
      if (LoopHasRegisterPipeline(loop)) {
        register_pipeline_ = true;
        break;
      }
    }
  }

  Stmt VisitStmt_(const ForNode *op) override {
    bool is_main_loop = IsGemmKLoop(op);
    if (is_main_loop) {
      phase_ = PipelinePhase::kInMainLoop;
      register_pipeline_ = LoopHasRegisterPipeline(op);
    }
    Stmt body = VisitStmt(op->body);
    Stmt for_stmt =
        For(op->loop_var, op->min, op->extent, op->kind, body,
            op->thread_binding, op->annotations, op->step, op->span);
    if (is_main_loop) {
      phase_ = PipelinePhase::kAfterMainLoop;
    }
    return for_stmt;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) override {
    Stmt flattened = SeqStmt::Flatten(GetRef<Stmt>(op));
    const auto *seq_op = flattened.as<SeqStmtNode>();
    if (seq_op == nullptr) {
      return VisitStmt(flattened);
    }

    Array<Stmt> new_seq;
    int pending_load_count = 0;
    auto insert_conservative_sld_sync = [&]() {
      new_seq.push_back(MakeSldFenceStmt(0));
      if (phase_ != PipelinePhase::kAfterMainLoop) {
        new_seq.push_back(MakeWaveBarrierStmt());
      }
    };
    for (const Stmt &stmt : seq_op->seq) {
      if (IsMmaCluster(stmt)) {
        if (in_mma_stmt_ > 0) {
          new_seq.push_back(VisitStmt(stmt));
          continue;
        }
        ++in_mma_stmt_;
        bool last_epilogue_mma = false;
        if (phase_ == PipelinePhase::kAfterMainLoop) {
          ++epilogue_mma_seen_;
          last_epilogue_mma = epilogue_counts_.total > 0 &&
                              epilogue_mma_seen_ >= epilogue_counts_.total;
        }
        if (pending_load_count > 0) {
          if (register_pipeline_) {
            const int wait_n =
                last_epilogue_mma
                    ? 0
                    : ResolveRegisterLgkmcnt(pending_load_count);
            AppendRegisterPipelineLdsWait(new_seq, wait_n);
            last_sld_fence_val_ = wait_n;
          } else {
            insert_conservative_sld_sync();
          }
          pending_load_count = 0;
        } else if (register_pipeline_ && last_epilogue_mma) {
          AppendRegisterPipelineLdsWait(new_seq, 0);
        }
        new_seq.push_back(VisitStmt(stmt));
        --in_mma_stmt_;
      } else if (StmtContainsMma(stmt)) {
        if (pending_load_count > 0) {
          if (register_pipeline_) {
            AppendRegisterPipelineLdsWait(new_seq, 0);
            last_sld_fence_val_ = 0;
          } else {
            insert_conservative_sld_sync();
          }
          pending_load_count = 0;
        }
        new_seq.push_back(VisitStmt(stmt));
      } else {
        LoadCounter counter;
        counter(stmt);
        if (IsAsyncWaitScopeStmt(stmt) && counter.total_loads > 0) {
          pending_load_count = counter.total_loads;
        } else {
          pending_load_count += counter.total_loads;
        }
        if (StmtProvidesLgkmcntWait(stmt)) {
          pending_load_count = 0;
        }
        new_seq.push_back(VisitStmt(stmt));
      }
    }
    if (new_seq.size() == 1) {
      return new_seq[0];
    }
    return SeqStmt(new_seq);
  }

private:
  int ResolveRegisterLgkmcnt(int pending_load_count) {
    if (phase_ == PipelinePhase::kAfterMainLoop && last_sld_fence_val_ > 0) {
      return last_sld_fence_val_;
    }
    return pending_load_count;
  }

  EpilogueMmaCounts epilogue_counts_;
  int epilogue_mma_seen_{0};
  int in_mma_stmt_{0};
  PipelinePhase phase_{PipelinePhase::kBeforeMainLoop};
  bool register_pipeline_{false};
  int last_sld_fence_val_{0};
};

} // namespace

namespace transform {

tirx::transform::Pass InjectAsyncMmaFence() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, IRModule, PassContext) {
    Target target = f->GetAttr<Target>(tvm::attr::kTarget).value_or(Target());
    if (!target.defined() || !TargetIsHCU(target)) {
      return f;
    }
    auto *n = f.CopyOnWrite();
    MMABarrierMutator mutator(n->body);
    n->body = mutator(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectAsyncMmaFence", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectAsyncMmaFence", InjectAsyncMmaFence);
}

} // namespace transform
} // namespace tl
} // namespace tvm
