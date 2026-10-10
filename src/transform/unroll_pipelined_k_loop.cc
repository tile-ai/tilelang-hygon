/*!
 * \file unroll_pipelined_k_loop.cc
 * \brief Unroll a pipelined GEMM K loop and fold stage indices.
 */

#include "common/gemm_k_loop_utils.h"
#include "common/pipeline_utils.h"
#include "hcu/target_utils.h"
#include "op/builtin.h"
#include "tir/transforms/ir_utils.h"

#include <tvm/arith/pattern.h>
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

class PipelineIndexFolder : public StmtExprMutator {
public:
  explicit PipelineIndexFolder(const Var &loop_var) : loop_var_(loop_var) {}

private:
  static int64_t FloorModI64(int64_t a, int64_t b) {
    int64_t r = a % b;
    if (r < 0) {
      r += (b > 0 ? b : -b);
    }
    return r;
  }

  std::optional<int64_t> MatchDivisor(const PrimExpr &rhs) const {
    if (auto val = GetConstIntValue(rhs)) {
      int64_t v = *val;
      if (v > 0 && (v & (v - 1)) == 0) {
        return v;
      }
    }
    return std::nullopt;
  }

  std::optional<int64_t> MatchAndMaskDivisor(const PrimExpr &rhs) const {
    if (auto val = GetConstIntValue(rhs)) {
      int64_t mask = *val;
      int64_t divisor = mask + 1;
      if (mask > 0 && (divisor & (divisor - 1)) == 0) {
        return divisor;
      }
    }
    return std::nullopt;
  }

  std::optional<PrimExpr> TryFoldMod(const PrimExpr &lhs,
                                     int64_t divisor) const {
    if (divisor <= 0) {
      return std::nullopt;
    }
    if (auto lhs_val = GetConstIntValue(lhs)) {
      return make_const(loop_var_.dtype(), FloorModI64(*lhs_val, divisor));
    }
    ffi::Array<PrimExpr> linear = arith::DetectLinearEquation(lhs, {loop_var_});
    if (linear.size() != 2) {
      return std::nullopt;
    }
    auto coeff = GetConstIntValue(linear[0]);
    if (!coeff.has_value()) {
      return std::nullopt;
    }
    if (*coeff == 0) {
      if (auto c = GetConstIntValue(linear[1])) {
        return make_const(loop_var_.dtype(), FloorModI64(*c, divisor));
      }
      return std::nullopt;
    }
    if ((*coeff % divisor) != 0) {
      return std::nullopt;
    }
    PrimExpr folded =
        floormod(linear[1], make_const(loop_var_.dtype(), divisor));
    if (auto folded_val = GetConstIntValue(folded)) {
      return make_const(loop_var_.dtype(), *folded_val);
    }
    return std::nullopt;
  }

  PrimExpr VisitExpr_(const FloorModNode *op) final {
    PrimExpr lhs = VisitExpr(op->a);
    PrimExpr rhs = VisitExpr(op->b);
    if (auto divisor = MatchDivisor(rhs)) {
      if (auto folded = TryFoldMod(lhs, *divisor)) {
        return *folded;
      }
    }
    if (!lhs.same_as(op->a) || !rhs.same_as(op->b)) {
      return FloorMod(lhs, rhs);
    }
    return GetRef<PrimExpr>(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::bitwise_and()) && op->args.size() == 2) {
      PrimExpr lhs = VisitExpr(op->args[0]);
      PrimExpr rhs = VisitExpr(op->args[1]);
      if (auto divisor = MatchAndMaskDivisor(rhs)) {
        if (auto folded = TryFoldMod(lhs, *divisor)) {
          return *folded;
        }
      }
      if (!lhs.same_as(op->args[0]) || !rhs.same_as(op->args[1])) {
        return Call(op->dtype, op->op, {lhs, rhs});
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Var loop_var_;
};

Stmt FoldPipelineIndices(const Stmt &stmt, const Var &loop_var) {
  return PipelineIndexFolder(loop_var)(stmt);
}

Stmt BuildUnrolledBody(const Stmt &body, const Var &loop_var, int factor) {
  if (factor == 1) {
    return body;
  }
  Array<Stmt> parts;
  parts.reserve(factor);
  for (int i = 0; i < factor; ++i) {
    PrimExpr replaced = loop_var * factor + make_const(loop_var.dtype(), i);
    Stmt substituted = Substitute(body, {{loop_var, replaced}});
    parts.push_back(FoldPipelineIndices(substituted, loop_var));
  }
  return SeqStmt::Flatten(parts);
}

class PipelinedKLoopUnrollMutator : public StmtMutator {
public:
  PipelinedKLoopUnrollMutator(int factor, bool *transformed)
      : factor_(factor), transformed_(transformed) {}

private:
  Stmt VisitStmt_(const ForNode *op) final {
    if (!IsGemmKLoop(op) || *transformed_) {
      return StmtMutator::VisitStmt_(op);
    }
    auto extent_opt = GetConstIntValue(op->extent);
    if (!extent_opt.has_value() || *extent_opt <= 0 || factor_ <= 1) {
      return StmtMutator::VisitStmt_(op);
    }
    const int64_t old_extent = *extent_opt;
    const int64_t main_extent = old_extent / factor_;
    const int64_t remainder = old_extent % factor_;
    Stmt body = op->body;
    Stmt main_body = BuildUnrolledBody(body, op->loop_var, factor_);
    Array<Stmt> result;
    if (main_extent > 0) {
      result.push_back(For(op->loop_var, op->min,
                           make_const(op->extent.dtype(), main_extent),
                           op->kind, std::move(main_body), op->thread_binding,
                           op->annotations, op->step, op->span));
    }
    if (remainder > 0) {
      Var k_epi = op->loop_var.copy_with_suffix("_epi");
      PrimExpr epilogue_base =
          make_const(op->loop_var.dtype(), main_extent * factor_);
      Stmt epilogue_body =
          Substitute(body, {{op->loop_var, epilogue_base + k_epi}});
      if (remainder == 1) {
        epilogue_body =
            Substitute(epilogue_body, {{k_epi, make_zero(k_epi.dtype())}});
      }
      epilogue_body = FoldPipelineIndices(epilogue_body, k_epi);
      result.push_back(For(
          k_epi, op->min, make_const(op->extent.dtype(), remainder), op->kind,
          std::move(epilogue_body), std::nullopt, {}, op->step, op->span));
    }
    *transformed_ = true;
    if (result.size() == 1) {
      return result[0];
    }
    return SeqStmt::Flatten(result);
  }

  int factor_;
  bool *transformed_;
};

int GetUnrollFactor(const tvm::transform::PassContext &ctx) {
  if (ctx.defined()) {
    auto val = ctx->GetConfig<Integer>(kPipelinedKUnrollFactor, Integer(4));
    if (val.defined()) {
      return static_cast<int>(val.value()->value);
    }
  }
  return 4;
}

} // namespace

tirx::transform::Pass UnrollPipelinedKLoop() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &, const PassContext &ctx) {
    Target target = f->GetAttr<Target>(tvm::attr::kTarget).value_or(Target());
    if (!target.defined() || !TargetIsHCU(target)) {
      return f;
    }
    auto loops = CollectGemmKLoops(f->body);
    if (loops.size() != 1) {
      return f;
    }
    if (!LoopHasRegisterPipeline(loops.front())) {
      return f;
    }
    int factor = GetUnrollFactor(ctx);
    if (factor <= 1) {
      return f;
    }
    bool transformed = false;
    auto *n = f.CopyOnWrite();
    n->body =
        PipelinedKLoopUnrollMutator(factor, &transformed)(std::move(n->body));
    if (transformed) {
      n->body = ConvertSSA(std::move(n->body));
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.UnrollPipelinedKLoop", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.UnrollPipelinedKLoop",
                        UnrollPipelinedKLoop);
}

} // namespace tl
} // namespace tvm
