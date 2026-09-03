/*!
 * \file inject_warp_divergence.cc
 * \brief Split a register-pipelined K loop across two warp groups.
 */

#include "common/gemm_k_loop_utils.h"
#include "common/pipeline_utils.h"
#include "hcu/target_utils.h"
#include "op/builtin.h"
#include "tir/transforms/ir_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Array;
using ffi::GetRef;

namespace {

bool LoopEnabled(const ForNode *loop) {
  return LoopHasRegisterPipeline(loop) && LoopHasWarpDivergence(loop);
}

std::optional<int64_t> GetConstIntValue(const PrimExpr &expr) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    return imm->value;
  }
  return std::nullopt;
}

bool IsThreadIdxXAttr(const AttrStmtNode *op) {
  if (op->attr_key != tirx::attr::thread_extent) {
    return false;
  }
  const auto *iv = op->node.as<IterVarNode>();
  return iv != nullptr && iv->thread_tag == "threadIdx.x";
}

bool IsPtxWaitGroupStmt(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      return call->op.same_as(builtin::ptx_wait_group());
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

bool IsSchedBarrierStmt(const Stmt &stmt) {
  const auto *eval = stmt.as<EvaluateNode>();
  if (eval == nullptr) {
    return false;
  }
  const auto *call = eval->value.as<CallNode>();
  if (call == nullptr || !call->op.same_as(builtin::call_extern()) ||
      call->args.empty()) {
    return false;
  }
  if (const auto *s = call->args[0].as<StringImmNode>()) {
    return s->value == "__builtin_amdgcn_sched_barrier";
  }
  return false;
}

bool StmtContainsAsyncCopy(const Stmt &stmt) {
  struct Checker : public StmtExprVisitor {
    bool found{false};
    void VisitExpr_(const CallNode *op) override {
      if (IsAsyncCopyCall(op)) {
        found = true;
        return;
      }
      ExprVisitor::VisitExpr_(op);
    }
  };
  Checker checker;
  checker(stmt);
  return checker.found;
}

bool IsMmaCluster(const Stmt &stmt) {
  if (!StmtContainsMma(stmt)) {
    return false;
  }
  if (const auto *for_op = stmt.as<ForNode>()) {
    if (IsGemmKLoop(for_op)) {
      return false;
    }
  }
  return !StmtContainsAsyncCopy(stmt);
}

Stmt MakeSyncWarpStmt() {
  return Evaluate(Call(DataType::Void(), sync_warp(), {}));
}

PrimExpr MakeWarpIndexExpr() {
  return Call(DataType::Int(32), get_warp_idx(), {});
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

Array<Stmt> RewriteAltLoopBody(const Array<Stmt> &stmts) {
  Array<Stmt> result;
  size_t i = 0;
  while (i < stmts.size()) {
    if (IsPtxWaitGroupStmt(stmts[i]) && i + 1 < stmts.size() &&
        IsSyncWarpStmt(stmts[i + 1])) {
      result.push_back(stmts[i]);
      i += 2;
      while (i < stmts.size() && !IsMmaCluster(stmts[i]) &&
             !IsSchedBarrierStmt(stmts[i])) {
        result.push_back(stmts[i]);
        ++i;
      }
      while (i < stmts.size() && IsSchedBarrierStmt(stmts[i])) {
        result.push_back(stmts[i]);
        ++i;
      }
      if (i < stmts.size() && IsMmaCluster(stmts[i])) {
        result.push_back(stmts[i]);
        ++i;
        while (i < stmts.size() && IsSchedBarrierStmt(stmts[i])) {
          result.push_back(stmts[i]);
          ++i;
        }
      }
      if (result.empty() || !IsSyncWarpStmt(result.back())) {
        result.push_back(MakeSyncWarpStmt());
      }
      continue;
    }
    result.push_back(stmts[i]);
    ++i;
  }
  return result;
}

class InfoCollector : public StmtExprVisitor {
public:
  PrimExpr thread_extent;
  bool found_enabled_loop{false};

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
    if (IsGemmKLoop(op) && LoopEnabled(op)) {
      found_enabled_loop = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

class WarpDivergenceMutator : public StmtMutator {
public:
  explicit WarpDivergenceMutator(int split_index) : split_index_(split_index) {}

private:
  Stmt VisitStmt_(const ForNode *op) override {
    if (IsGemmKLoop(op) && LoopEnabled(op)) {
      For then_loop(op->loop_var, op->min, op->extent, op->kind, op->body,
                    op->thread_binding, op->annotations, op->step, op->span);
      Stmt else_body =
          RebuildFlattened(RewriteAltLoopBody(FlattenStmts(op->body)));
      For else_loop(op->loop_var, op->min, op->extent, op->kind, else_body,
                    op->thread_binding, op->annotations, op->step, op->span);
      PrimExpr cond =
          MakeWarpIndexExpr() < make_const(DataType::Int(32), split_index_);
      return IfThenElse(cond, then_loop, else_loop);
    }
    return StmtMutator::VisitStmt_(op);
  }

  int split_index_;
};

std::optional<int> ComputeSplitIndex(const PrimExpr &thread_extent,
                                     int warp_size) {
  if (warp_size <= 0) {
    return std::nullopt;
  }
  auto extent = GetConstIntValue(thread_extent);
  if (!extent.has_value() || *extent <= 0 || (*extent % warp_size) != 0) {
    return std::nullopt;
  }
  int warp_num = static_cast<int>(*extent / warp_size);
  if (warp_num < 2 || (warp_num % 2) != 0) {
    return std::nullopt;
  }
  return warp_num / 2;
}

} // namespace

namespace transform {

tirx::transform::Pass InjectWarpDivergence() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &, const PassContext &) {
    Target target = f->GetAttr<Target>(tvm::attr::kTarget).value_or(Target());
    if (!target.defined() || !TargetIsHCU(target)) {
      return f;
    }
    InfoCollector collector;
    collector(f->body);
    if (!collector.found_enabled_loop || !collector.thread_extent.defined()) {
      return f;
    }
    const int warp_size = TargetHcuGetWarpSize(target);
    auto split_index = ComputeSplitIndex(collector.thread_extent, warp_size);
    if (!split_index.has_value()) {
      return f;
    }
    auto *n = f.CopyOnWrite();
    WarpDivergenceMutator mutator(split_index.value());
    n->body = mutator(std::move(n->body));
    n->body = ConvertSSA(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectWarpDivergence", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectWarpDivergence",
                        InjectWarpDivergence);
}

} // namespace transform
} // namespace tl
} // namespace tvm
