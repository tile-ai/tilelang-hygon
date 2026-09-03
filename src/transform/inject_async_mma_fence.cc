/*!
 * \file inject_async_mma_fence.cc
 * \brief Insert LDS waits before MMA. Register pipeline uses packed
 *        s_waitcnt(lgkmcnt=N) where N is the LDS op count of the preceding
 *        S2R cluster (same as T.s_waitcnt(N, "lgkmcnt")).
 */

#include "common/gemm_k_loop_utils.h"
#include "common/pipeline_utils.h"
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

// Same packing as T.s_waitcnt(cnt, "lgkmcnt") in tilelang/language/builtin.py.
int PackLgkmcntImm(int cnt) {
  constexpr int kLgkmcntMax = 15;
  cnt = std::max(0, std::min(cnt, kLgkmcntMax));
  return 0xF | (7 << 4) | (1 << 7) | (cnt << 8) | (3 << 12) | (3 << 14);
}

Stmt MakeLgkmcntWaitcntStmt(int cnt) {
  return Evaluate(Call(DataType::Int(32), builtin::call_extern(),
                       {StringImm("__builtin_amdgcn_s_waitcnt"),
                        IntImm(DataType::Int(32), PackLgkmcntImm(cnt))}));
}

Stmt MakeSchedBarrierStmt() {
  return Evaluate(Call(DataType::Void(), builtin::call_extern(),
                       {StringImm("__builtin_amdgcn_sched_barrier"),
                        IntImm(DataType::Int(32), 0)}));
}

void AppendRegisterPipelineLdsWait(Array<Stmt> &seq, int lds_count) {
  seq.push_back(MakeLgkmcntWaitcntStmt(lds_count));
  seq.push_back(MakeSchedBarrierStmt());
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
         op->attr_key == "async_wait_queue_scope" ||
         op->attr_key == s_tir::attr::async_wait_inflight_count ||
         op->attr_key == "async_wait_inflight_count";
}

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

Stmt ComputeEpilogueLastMmaStmt(const Stmt &root) {
  struct Collector : public StmtVisitor {
    Stmt last_mma;
    bool past_main_loop{false};

    void VisitStmt_(const ForNode *op) override {
      StmtVisitor::VisitStmt_(op);
      if (IsGemmKLoop(op)) {
        past_main_loop = true;
      }
    }

    void VisitStmt_(const SeqStmtNode *op) override {
      for (const Stmt &s : op->seq) {
        if (past_main_loop && StmtContainsMma(s)) {
          last_mma = s;
        }
        VisitStmt(s);
      }
    }

    void VisitStmt_(const IfThenElseNode *op) override {
      if (past_main_loop) {
        if (StmtContainsMma(op->then_case)) {
          last_mma = op->then_case;
        }
        if (op->else_case.defined() &&
            StmtContainsMma(op->else_case.value())) {
          last_mma = op->else_case.value();
        }
      }
      StmtVisitor::VisitStmt_(op);
    }
  };
  Collector collector;
  collector(root);
  return collector.last_mma;
}

class MMABarrierMutator : public StmtExprMutator {
public:
  explicit MMABarrierMutator(const Stmt &root_body)
      : epilogue_last_mma_(ComputeEpilogueLastMmaStmt(root_body)) {}

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
    for (const Stmt &stmt : seq_op->seq) {
      if (StmtContainsMma(stmt)) {
        if (pending_load_count > 0) {
          if (register_pipeline_) {
            AppendRegisterPipelineLdsWait(new_seq, pending_load_count);
          } else {
            new_seq.push_back(MakeSldFenceStmt(0));
            if (phase_ != PipelinePhase::kAfterMainLoop) {
              new_seq.push_back(MakeWaveBarrierStmt());
            }
          }
          pending_load_count = 0;
        } else if (register_pipeline_ &&
                   phase_ == PipelinePhase::kAfterMainLoop &&
                   epilogue_last_mma_.defined() &&
                   stmt.same_as(epilogue_last_mma_)) {
          AppendRegisterPipelineLdsWait(new_seq, 0);
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
        new_seq.push_back(VisitStmt(stmt));
      }
    }
    if (new_seq.size() == 1) {
      return new_seq[0];
    }
    return SeqStmt(new_seq);
  }

private:
  Stmt epilogue_last_mma_;
  PipelinePhase phase_{PipelinePhase::kBeforeMainLoop};
  bool register_pipeline_{false};
};

} // namespace

namespace transform {

tirx::transform::Pass InjectAsyncMmaFence() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, IRModule, PassContext) {
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
