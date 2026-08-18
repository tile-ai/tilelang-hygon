/*!
 * \file inject_hcu_copy_idxen.cc
 * \brief Late HCU rewrite of annotated async copies for wrap and idxen addressing.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <vector>

#include "hcu/utils/gemm_a_lds_strategy.h"
#include "hcu/utils/gemm_b_lds_strategy.h"
#include "layout/layout.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {
using namespace tirx;
using namespace ffi;

namespace {

int64_t LargestPow2LessThanOrEqual(int64_t value) {
  if (value <= 1)
    return 1;
  int64_t result = 1;
  while ((result << 1) <= value)
    result <<= 1;
  return result;
}

int64_t TileWidthFromK(int64_t k_dim) {
  return std::min<int64_t>(LargestPow2LessThanOrEqual(k_dim), 4096);
}

int StructBitFromTileWidth(int64_t tile_width) {
  int bit = 0;
  while ((int64_t{1} << bit) < tile_width)
    ++bit;
  return bit;
}

Optional<PrimExpr> ExtractRowExpr(const PrimExpr &expr, int64_t k_dim) {
  if (const auto *mul = expr.as<MulNode>()) {
    if (const auto *imm = mul->b.as<IntImmNode>()) {
      if (imm->value == k_dim)
        return mul->a;
    }
    if (const auto *imm = mul->a.as<IntImmNode>()) {
      if (imm->value == k_dim)
        return mul->b;
    }
  }
  if (const auto *add = expr.as<AddNode>()) {
    if (auto lhs = ExtractRowExpr(add->a, k_dim))
      return lhs;
    if (auto rhs = ExtractRowExpr(add->b, k_dim))
      return rhs;
  }
  if (const auto *sub = expr.as<SubNode>()) {
    if (auto lhs = ExtractRowExpr(sub->a, k_dim))
      return lhs;
    if (auto rhs = ExtractRowExpr(sub->b, k_dim))
      return rhs;
  }
  return Optional<PrimExpr>();
}

bool IsRowOnlyLayout(const Layout &layout) {
  if (!layout.defined() || layout->InputDim() != 2 ||
      layout->OutputDim() != 2) {
    return false;
  }
  Var row("layout_row");
  Var col("layout_col");
  Array<PrimExpr> forward = layout->Forward({row, col});
  if (forward.size() != 2 || !StructuralEqual()(forward[1], col))
    return false;
  return !UsesVar(forward[0],
                  [&](const VarNode *var) { return var == col.get(); });
}

std::vector<Var> CollectVars(const PrimExpr &expr) {
  std::vector<Var> result;
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (const auto *var = node.as<VarNode>()) {
      Var current = ffi::GetRef<Var>(var);
      for (const Var &existing : result) {
        if (existing.same_as(current))
          return;
      }
      result.push_back(current);
    }
  });
  return result;
}

Optional<Var> FindSingleCopyLoopVar(const PrimExpr &expr) {
  std::vector<Var> vars = CollectVars(expr);
  if (vars.size() != 1U)
    return Optional<Var>();
  return vars[0];
}

class HcuCopyIdxenRewriter : public StmtExprMutator {
public:
  explicit HcuCopyIdxenRewriter(const PrimFunc &func) {
    for (const auto &kv : func->buffer_map)
      buffer_data_to_buffer_.Set(kv.second->data, kv.second);
  }

private:
  Stmt VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers)
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    for (const MatchBufferRegion &match_buffer : op->match_buffers) {
      buffer_data_to_buffer_.Set(match_buffer->buffer->data,
                                 match_buffer->buffer);
    }

    bool pushed_layout = false;
    if (auto value = op->annotations.Get(attr::kLayoutMap)) {
      if (auto layout_map = value.value().as<Map<Var, Layout>>()) {
        layout_map_stack_.push_back(layout_map.value());
        pushed_layout = true;
      }
    }
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    if (pushed_layout)
      layout_map_stack_.pop_back();
    return stmt;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!call->op.same_as(tl::ptx_cp_async()))
      return call;

    auto use_idxen_attr = call->annotations.Get("use_idxen");
    const auto *use_idxen_imm =
        use_idxen_attr ? use_idxen_attr.value().as<IntImmNode>() : nullptr;
    bool use_idxen = use_idxen_imm && use_idxen_imm->value != 0;
    auto wrap_offset = call->annotations.Get("wrap_offset");
    auto wrap_idx_mask = call->annotations.Get("wrap_idx_mask");
    const auto *wrap_offset_imm =
        wrap_offset ? wrap_offset.value().as<IntImmNode>() : nullptr;
    const auto *wrap_idx_mask_imm =
        wrap_idx_mask ? wrap_idx_mask.value().as<IntImmNode>() : nullptr;
    bool has_wrap = wrap_offset_imm && wrap_offset_imm->value > 0 &&
                    wrap_idx_mask_imm && wrap_idx_mask_imm->value > 0;
    bool has_gemm_lds_strategy =
        call->annotations.Get(attr::kHcuGemmALdsStrategy).has_value() ||
        call->annotations.Get(attr::kHcuGemmBLdsStrategy).has_value();
    ICHECK_EQ(wrap_offset_imm != nullptr, wrap_idx_mask_imm != nullptr)
        << "wrap_offset and wrap_idx_mask must be specified together";
    if (!use_idxen && !has_wrap)
      return call;
    ICHECK(call->args.size() == 3U || call->args.size() == 4U)
        << "wrapped tl.ptx_cp_async must have 3 or 4 args";

    const auto *src_call = call->args[1].as<CallNode>();
    if (!src_call || !src_call->op.same_as(builtin::tvm_access_ptr()))
      return call;
    const auto *src_var = src_call->args[1].as<VarNode>();
    if (!src_var)
      return call;

    const auto *dst_call = call->args[0].as<CallNode>();
    ICHECK(dst_call && dst_call->op.same_as(builtin::tvm_access_ptr()))
        << "annotated HCU async copy expects dst tvm_access_ptr";
    Optional<Buffer> dst_buffer = LookupBuffer(dst_call->args[1]);
    if (dst_buffer) {
      Optional<Layout> layout = LookupLayout(dst_buffer.value());
      if (layout && !IsRowOnlyLayout(layout.value())) {
        ICHECK(has_wrap)
            << "non row-only wrapped shared layout requires wrap hints";
      }
    }

    PrimExpr new_dst = call->args[0];
    if (has_wrap && !has_gemm_lds_strategy) {
      PrimExpr old_dst_offset = dst_call->args[2];
      Optional<Var> copy_loop_var = FindSingleCopyLoopVar(old_dst_offset);
      ICHECK(copy_loop_var)
          << "wrapped async-copy dst compensation requires one copy loop var";
      PrimExpr base_offset = analyzer_.Simplify(Substitute(
          old_dst_offset,
          {{copy_loop_var.value(),
            IntImm(copy_loop_var.value()->dtype, 0)}}));
      PrimExpr contiguous_offset = analyzer_.Simplify(
          base_offset + copy_loop_var.value() *
                            Cast(copy_loop_var.value()->dtype, call->args[2]));
      Array<PrimExpr> args = dst_call->args;
      args.Set(2, contiguous_offset);
      new_dst = Call(call->args[0].dtype(), builtin::tvm_access_ptr(), args,
                     dst_call->annotations, dst_call->span);
    }

    PrimExpr predicate =
        call->args.size() == 4U ? call->args[3] : const_true();
    if (!use_idxen) {
      return Call(call->dtype, tl::ptx_cp_async(),
                  {new_dst, call->args[1], call->args[2], predicate},
                  call->annotations, call->span);
    }

    Optional<Buffer> src_buffer = LookupBuffer(src_call->args[1]);
    if (!src_buffer || src_buffer.value()->shape.size() != 2)
      return call;
    const auto *k_imm = src_buffer.value()->shape[1].as<IntImmNode>();
    if (!k_imm)
      return call;
    int64_t k_dim = k_imm->value;
    int64_t tile_width = TileWidthFromK(k_dim);
    int struct_bit = StructBitFromTileWidth(tile_width);

    PrimExpr old_offset = src_call->args[2];
    Optional<PrimExpr> row_expr = ExtractRowExpr(old_offset, k_dim);
    if (!row_expr)
      return call;
    PrimExpr idxen = analyzer_.Simplify(row_expr.value());
    PrimExpr new_offset = analyzer_.Simplify(
        old_offset - row_expr.value() * make_const(old_offset.dtype(), k_dim) +
        idxen * make_const(old_offset.dtype(), k_dim - tile_width));
    Array<PrimExpr> src_args = src_call->args;
    src_args.Set(2, new_offset);
    PrimExpr new_src = Call(call->args[1].dtype(), builtin::tvm_access_ptr(),
                            src_args, src_call->annotations, src_call->span);
    return Call(call->dtype, tl::hcu_cp_async_idxen(),
                {new_dst, new_src, call->args[2], predicate, idxen,
                 IntImm(DataType::Int(32), struct_bit)},
                call->annotations, call->span);
  }

  Optional<Buffer> LookupBuffer(const PrimExpr &expr) const {
    if (const auto *var = expr.as<VarNode>()) {
      auto it = buffer_data_to_buffer_.find(ffi::GetRef<Var>(var));
      if (it != buffer_data_to_buffer_.end())
        return (*it).second;
    }
    return Optional<Buffer>();
  }

  Optional<Layout> LookupLayout(const Buffer &buffer) const {
    for (auto it = layout_map_stack_.rbegin(); it != layout_map_stack_.rend();
         ++it) {
      if ((*it).count(buffer->data))
        return (*it)[buffer->data];
    }
    return Optional<Layout>();
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::vector<Map<Var, Layout>> layout_map_stack_;
  arith::Analyzer analyzer_;
};

PrimFunc InjectHcuCopyIdxenPrimFunc(PrimFunc func) {
  if (!func.defined() || !func->body.defined())
    return func;
  PrimFuncNode *node = func.CopyOnWrite();
  HcuCopyIdxenRewriter rewriter(func);
  node->body = rewriter(std::move(node->body));
  return func;
}

} // namespace

namespace transform {

tvm::transform::Pass InjectHcuCopyIdxen() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) {
    (void)mod;
    (void)ctx;
    return InjectHcuCopyIdxenPrimFunc(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.InjectHcuCopyIdxen", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectHcuCopyIdxen",
                        InjectHcuCopyIdxen);
}

} // namespace transform
} // namespace tl
} // namespace tvm
