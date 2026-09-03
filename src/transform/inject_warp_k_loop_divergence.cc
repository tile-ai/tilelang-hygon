#include "common/gemm_k_loop_utils.h"
#include "hcu/target_utils.h"
#include "op/builtin.h"
#include "tir/transforms/ir_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Array;

namespace {

bool HasUseWarpDivergenceAttr(const PrimFunc &func) {
  if (func->GetAttr<Bool>(attr::kUseWarpDivergence).value_or(Bool(false))) {
    return true;
  }
  class Checker : public StmtVisitor {
  public:
    bool found{false};

  private:
    void VisitStmt_(const AttrStmtNode *op) final {
      if (op->attr_key == attr::kUseWarpDivergence) {
        if (const auto *imm = op->value.as<IntImmNode>()) {
          found = found || imm->value != 0;
        }
      }
      StmtVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const SBlockNode *op) final {
      auto it = op->annotations.find(attr::kUseWarpDivergence);
      if (it != op->annotations.end()) {
        if (const auto *imm = (*it).second.as<IntImmNode>()) {
          found = found || imm->value != 0;
        }
      }
      StmtVisitor::VisitStmt_(op);
    }
  };
  Checker checker;
  checker(func->body);
  return checker.found;
}

class StripUseWarpDivergenceMutator : public StmtMutator {
  Stmt VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == attr::kUseWarpDivergence) {
      return VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }
};

std::optional<int64_t> GetConstIntValue(const PrimExpr &expr) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    return imm->value;
  }
  return std::nullopt;
}

bool IsThreadIdxXAttr(const AttrStmtNode *op, const IterVarNode **out_iv = nullptr) {
  if (op->attr_key != tirx::attr::thread_extent) {
    return false;
  }
  const auto *iv = op->node.as<IterVarNode>();
  if (iv == nullptr || iv->thread_tag != "threadIdx.x") {
    return false;
  }
  if (out_iv != nullptr) {
    *out_iv = iv;
  }
  return true;
}

bool IsWarpIdxInit(const PrimExpr &expr) {
  if (const auto *call = expr.as<CallNode>()) {
    return call->op.same_as(get_warp_idx_sync()) ||
           call->op.same_as(get_warp_idx());
  }
  return false;
}

Stmt MakeAsyncGldSldFenceStmt(int wait_count) {
  return Evaluate(
      Call(DataType::Void(), async_gld_sld_fence(), {Integer(wait_count)}));
}

bool IsAsyncGldSldFenceStmt(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      return call->op.same_as(async_gld_sld_fence());
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

Array<Stmt> FlattenStmts(const Stmt &stmt) {
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    Array<Stmt> result;
    for (const Stmt &s : seq->seq) {
      for (const Stmt &inner : FlattenStmts(s)) {
        result.push_back(inner);
      }
    }
    return result;
  }
  return {stmt};
}

Stmt RebuildFlattened(const Array<Stmt> &stmts) {
  if (stmts.empty()) {
    return Evaluate(0);
  }
  if (stmts.size() == 1) {
    return stmts[0];
  }
  return SeqStmt::Flatten(stmts);
}

Array<Stmt> ReorderMmacBeforeGldFence(const Array<Stmt> &stmts) {
  Array<Stmt> result;
  size_t i = 0;
  while (i < stmts.size()) {
    if (i + 2 < stmts.size() && IsAsyncGldSldFenceStmt(stmts[i]) &&
        IsAsyncGldFenceStmt(stmts[i + 1]) && IsWaveBarrierStmt(stmts[i + 2])) {
      result.push_back(stmts[i]);
      size_t j = i + 3;
      while (j < stmts.size() && StmtContainsMma(stmts[j])) {
        result.push_back(stmts[j]);
        ++j;
      }
      result.push_back(stmts[i + 1]);
      result.push_back(stmts[i + 2]);
      i = j;
    } else {
      result.push_back(stmts[i]);
      ++i;
    }
  }
  return result;
}

class InfoCollector : public StmtExprVisitor {
public:
  Var warp_idx_var;
  PrimExpr thread_extent;
  bool found_k_loop{false};

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (IsThreadIdxXAttr(op)) {
      thread_extent = op->value;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kThreadBinding && op->thread_binding.defined() &&
        op->thread_binding.value()->thread_tag == "threadIdx.x") {
      thread_extent = op->extent;
    }
    if (IsGemmKLoop(op)) {
      found_k_loop = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BindNode *op) final {
    if (IsWarpIdxInit(op->value)) {
      warp_idx_var = op->var;
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

class InjectWarpIdxMutator : public StmtMutator {
public:
  explicit InjectWarpIdxMutator(Var *out_var) : out_var_(out_var) {}

  bool injected() const { return injected_; }

private:
  Stmt PrependWarpIdxBind(Stmt body) {
    injected_ = true;
    *out_var_ = Var("warp_idx", DataType::Int(32));
    Array<Stmt> seq;
    seq.push_back(
        Bind(*out_var_, Call(DataType::Int(32), get_warp_idx_sync(), {})));
    if (const auto *s = body.as<SeqStmtNode>()) {
      for (const Stmt &child : s->seq) {
        seq.push_back(child);
      }
    } else {
      seq.push_back(std::move(body));
    }
    return SeqStmt(seq);
  }

  Stmt VisitStmt_(const ForNode *op) override {
    Stmt body = VisitStmt(op->body);
    if (op->kind == ForKind::kThreadBinding && op->thread_binding.defined() &&
        op->thread_binding.value()->thread_tag == "threadIdx.x" &&
        !injected_) {
      body = PrependWarpIdxBind(std::move(body));
    }
    return For(op->loop_var, op->min, op->extent, op->kind, body,
               op->thread_binding, op->annotations, op->step, op->span);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) override {
    Stmt body = VisitStmt(op->body);
    if (IsThreadIdxXAttr(op) && !injected_) {
      body = PrependWarpIdxBind(std::move(body));
    }
    return AttrStmt(op->node, op->attr_key, op->value, std::move(body),
                    op->span);
  }

  Var *out_var_;
  bool injected_{false};
};

enum class PipelinePhase { kBeforeMainLoop, kInMainLoop, kAfterMainLoop };

class WarpKLoopDivergenceMutator : public StmtMutator {
public:
  WarpKLoopDivergenceMutator(Var warp_idx_var, int warp_half)
      : warp_idx_var_(std::move(warp_idx_var)), warp_half_(warp_half) {}

private:
  Stmt VisitStmt_(const ForNode *op) override {
    if (IsGemmKLoop(op)) {
      phase_ = PipelinePhase::kInMainLoop;
      For orig_loop(op->loop_var, op->min, op->extent, op->kind, op->body,
                    op->thread_binding, op->annotations, op->step, op->span);
      Stmt alt_body =
          RebuildFlattened(ReorderMmacBeforeGldFence(FlattenStmts(op->body)));
      For alt_loop(op->loop_var, op->min, op->extent, op->kind, alt_body,
                   op->thread_binding, op->annotations, op->step, op->span);
      PrimExpr cond = warp_idx_var_ < make_const(DataType::Int(32), warp_half_);
      phase_ = PipelinePhase::kAfterMainLoop;
      return IfThenElse(cond, orig_loop, alt_loop);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) override {
    if (phase_ == PipelinePhase::kBeforeMainLoop) {
      if (const auto *call = op->value.as<CallNode>()) {
        if (call->op.same_as(async_gld_sld_fence())) {
          return MakeAsyncGldSldFenceStmt(0);
        }
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Var warp_idx_var_;
  int warp_half_;
  PipelinePhase phase_{PipelinePhase::kBeforeMainLoop};
};

std::optional<int> ComputeWarpHalf(const PrimExpr &thread_extent) {
  if (auto extent = GetConstIntValue(thread_extent)) {
    if (*extent <= 0 || (*extent % 64) != 0) {
      return std::nullopt;
    }
    int warp_num = static_cast<int>(*extent / 64);
    if (warp_num < 2 || (warp_num % 2) != 0) {
      return std::nullopt;
    }
    return warp_num / 2;
  }
  return std::nullopt;
}

} // namespace

namespace transform {

tirx::transform::Pass InjectWarpKLoopDivergence() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &, const PassContext &) {
    Target target = f->GetAttr<Target>(tvm::attr::kTarget).value_or(Target());
    if (target.defined() && !TargetIsHCU(target)) {
      return f;
    }
    InfoCollector collector;
    collector(f->body);
    bool enabled = HasUseWarpDivergenceAttr(f);
    if (!enabled) {
      if (auto extent = GetConstIntValue(collector.thread_extent)) {
        enabled = *extent >= 512;
      }
    }
    if (!enabled || !collector.found_k_loop ||
        !collector.thread_extent.defined()) {
      return f;
    }
    Var warp_idx = collector.warp_idx_var;
    auto *n = f.CopyOnWrite();
    n->body = StripUseWarpDivergenceMutator()(std::move(n->body));
    if (!warp_idx.defined()) {
      InjectWarpIdxMutator injector(&warp_idx);
      n->body = injector(std::move(n->body));
      if (!injector.injected() || !warp_idx.defined()) {
        return f;
      }
    }
    auto warp_half = ComputeWarpHalf(collector.thread_extent);
    if (!warp_half.has_value()) {
      return f;
    }
    WarpKLoopDivergenceMutator mutator(warp_idx, warp_half.value());
    n->body = mutator(std::move(n->body));
    n->body = ConvertSSA(std::move(n->body));
    return WithoutAttr(std::move(f), attr::kUseWarpDivergence);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectWarpKLoopDivergence", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectWarpKLoopDivergence",
                        InjectWarpKLoopDivergence);
}

} // namespace transform
} // namespace tl
} // namespace tvm
