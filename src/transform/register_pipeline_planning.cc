#include "support/check.h"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "common/pipeline_utils.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "op/operator.h"
#include "op/region.h"
#include "op/utils.h"

#include <tvm/tirx/buffer.h>

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;

namespace {

inline bool IsSharedLike(const Buffer &buffer) {
  return IsSharedBuffer(buffer, true);
}

class RegisterPipelineClassifier : public StmtExprVisitor {
public:
  static bool IsSharedToLocalCopy(const Stmt &stmt) {
    RegisterPipelineClassifier c;
    c(stmt);
    return c.has_local_store_ && c.reads_shared_;
  }

  static bool HasMmaCompute(const Stmt &stmt) {
    RegisterPipelineClassifier c;
    c(stmt);
    return c.has_mma_compute_;
  }

  static bool HasAnyLocalAccess(const Stmt &stmt) {
    RegisterPipelineClassifier c;
    c(stmt);
    return c.reads_local_ || c.has_local_store_;
  }

private:
  void HandleTileOp(const TileOperator &tile_op) {
    if (const auto *copy = tile_op.as<CopyNode>()) {
      if (IsRegisterPipelineBuffer(copy->dst) && IsSharedLike(copy->src)) {
        has_local_store_ = true;
        reads_shared_ = true;
      }
      if (IsRegisterPipelineBuffer(copy->src) ||
          IsRegisterPipelineBuffer(copy->dst)) {
        reads_local_ = true;
      }
    }
    if (tile_op.as<GemmNode>()) {
      has_mma_compute_ = true;
    }
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.as<OpNode>()) {
        auto tile_op = ParseOperator(GetRef<Call>(call));
        if (tile_op.defined()) {
          HandleTileOp(tile_op);
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (IsRegisterPipelineBuffer(op->buffer)) {
      has_local_store_ = true;
      bool old = in_local_store_value_;
      in_local_store_value_ = true;
      VisitExpr(op->value);
      in_local_store_value_ = old;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (in_local_store_value_ && IsSharedLike(op->buffer)) {
      reads_shared_ = true;
    } else if (IsRegisterPipelineBuffer(op->buffer)) {
      reads_local_ = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.as<OpNode>()) {
      auto tile_op = ParseOperator(GetRef<Call>(op));
      if (tile_op.defined()) {
        HandleTileOp(tile_op);
      }
    }
    if (const auto *op_node = op->op.as<OpNode>()) {
      std::string name = op_node->name;
      if (name.find("mmac") != std::string::npos ||
          name.find("mma") != std::string::npos) {
        has_mma_compute_ = true;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  bool in_local_store_value_ = false;
  bool has_local_store_ = false;
  bool reads_shared_ = false;
  bool reads_local_ = false;
  bool has_mma_compute_ = false;
};

const SeqStmtNode *ExtractSplittableInnerSeq(const Stmt &stmt,
                                             bool *has_unsupported_mma_loop) {
  const auto *br = stmt.as<SBlockRealizeNode>();
  Stmt current = stmt;
  if (br && is_one(br->predicate)) {
    current = br->block->body;
  }
  if (!RegisterPipelineClassifier::HasMmaCompute(current) &&
      !RegisterPipelineClassifier::HasMmaCompute(stmt)) {
    return nullptr;
  }
  while (true) {
    if (const auto *seq = current.as<SeqStmtNode>()) {
      return seq;
    }
    if (const auto *inner_br = current.as<SBlockRealizeNode>()) {
      current = inner_br->block->body;
      continue;
    }
    if (const auto *attr = current.as<AttrStmtNode>()) {
      current = attr->body;
      continue;
    }
    if (const auto *for_stmt = current.as<ForNode>()) {
      if (is_one(for_stmt->extent)) {
        current = for_stmt->body;
        continue;
      }
      if (has_unsupported_mma_loop != nullptr &&
          RegisterPipelineClassifier::HasMmaCompute(for_stmt->body)) {
        *has_unsupported_mma_loop = true;
      }
      return nullptr;
    }
    if (const auto *if_then_else = current.as<IfThenElseNode>()) {
      if (!if_then_else->else_case.defined()) {
        current = if_then_else->then_case;
        continue;
      }
    }
    return nullptr;
  }
}

const SeqStmtNode *GetPipelineBodySeq(const Stmt &stmt) {
  Stmt current = stmt;
  while (true) {
    if (const auto *seq = current.as<SeqStmtNode>()) {
      return seq;
    }
    if (const auto *br = current.as<SBlockRealizeNode>()) {
      current = br->block->body;
      continue;
    }
    if (const auto *attr = current.as<AttrStmtNode>()) {
      current = attr->body;
      continue;
    }
    if (const auto *if_then_else = current.as<IfThenElseNode>()) {
      if (!if_then_else->else_case.defined()) {
        current = if_then_else->then_case;
        continue;
      }
    }
    return nullptr;
  }
}

Stmt ReplaceInnermostSeq(const Stmt &stmt, Array<Stmt> components) {
  Stmt inner = MakePipelineBody(components);
  if (stmt.as<SeqStmtNode>()) {
    return inner;
  }
  if (const auto *br = stmt.as<SBlockRealizeNode>()) {
    SBlock block = br->block;
    block.CopyOnWrite()->body = ReplaceInnermostSeq(block->body, components);
    return SBlockRealize(br->iter_values, br->predicate, block, br->span);
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    return AttrStmt(attr->node, attr->attr_key, attr->value,
                    ReplaceInnermostSeq(attr->body, components), attr->span);
  }
  return inner;
}

PrimExpr MakeTileRegionExpr(const BufferRegion &region, int access_mask) {
  Array<PrimExpr> mins;
  Array<PrimExpr> args;
  mins.reserve(region->region.size());
  for (const Range &range : region->region) {
    mins.push_back(range->min);
  }
  args.push_back(BufferLoad(region->buffer, mins));
  args.push_back(Integer(access_mask));
  for (const Range &range : region->region) {
    args.push_back(range->extent);
  }
  return Call(DataType::Handle(), RegionOp::Get(), args);
}

BufferRegion MakeFragmentRegion(const Buffer &buffer,
                                const BufferRegion &src_region) {
  Array<Range> ranges;
  ranges.reserve(src_region->region.size());
  for (const Range &range : src_region->region) {
    ranges.push_back(Range::FromMinExtent(
        make_const(range->extent.dtype(), 0), range->extent));
  }
  return BufferRegion(buffer, ranges);
}

Stmt MakeCopyEvaluate(const BufferRegion &src, const BufferRegion &dst) {
  static const Op &copy_op = Op::Get("tl.tileop.copy");
  PrimExpr call = Call(DataType::Handle(), copy_op,
                       {MakeTileRegionExpr(src, /*read*/ 1),
                        MakeTileRegionExpr(dst, /*write*/ 2)});
  return Evaluate(call);
}

Call RewriteGemmSharedOperands(const Call &call, const BufferRegion &a_region,
                               const BufferRegion &b_region) {
  Array<PrimExpr> args = call->args;
  ICHECK_GE(args.size(), 3);
  args.Set(0, MakeTileRegionExpr(a_region, /*read*/ 1));
  args.Set(1, MakeTileRegionExpr(b_region, /*read*/ 1));
  return Call(call->dtype, call->op, args, call->annotations, call->span);
}

int ResolveNumRegisterStages(const ForNode *op) {
  int num_register_stages = 0;
  if (auto num_reg_stages_anno = op->annotations.Get(kNumRegisterStages)) {
    if (const auto *imm = num_reg_stages_anno.value().as<IntImmNode>()) {
      num_register_stages = static_cast<int>(imm->value);
    }
  } else if (auto enable_anno =
                 op->annotations.Get(kEnableRegisterPipeline)) {
    if (const auto *imm = enable_anno.value().as<IntImmNode>()) {
      if (imm->value != 0) {
        num_register_stages = kDefaultNumRegisterStages;
      }
    }
  }
  return num_register_stages;
}

class RegisterPipelinePlanner : public StmtExprMutator {
public:
  Stmt VisitStmt_(const SBlockNode *op) final {
    Stmt body = VisitStmt(op->body);
    Array<Buffer> allocs = op->alloc_buffers;
    if (!pending_fragment_allocs_.empty()) {
      for (const Buffer &buffer : pending_fragment_allocs_) {
        allocs.push_back(buffer);
      }
      pending_fragment_allocs_.clear();
    }
    if (body.same_as(op->body) && allocs.size() == op->alloc_buffers.size()) {
      return GetRef<Stmt>(op);
    }
    SBlock block = GetRef<SBlock>(op);
    auto *n = block.CopyOnWrite();
    n->body = std::move(body);
    n->alloc_buffers = std::move(allocs);
    return block;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    bool has_shared_pipeline_anno =
        op->annotations.count(s_tir::attr::software_pipeline_stage) &&
        op->annotations.count(s_tir::attr::software_pipeline_order);
    if (!has_shared_pipeline_anno) {
      return for_node;
    }

    int num_register_stages = ResolveNumRegisterStages(op);
    int num_shared_stages = 0;
    if (auto num_stages_anno = op->annotations.Get("num_stages")) {
      if (const auto *imm = num_stages_anno->as<IntImmNode>()) {
        num_shared_stages = static_cast<int>(imm->value);
      }
    } else if (auto num_stages_anno =
                   op->annotations.Get("tl_pipelined_num_stages")) {
      if (const auto *imm = num_stages_anno->as<IntImmNode>()) {
        num_shared_stages = static_cast<int>(imm->value);
      }
    }
    if (num_register_stages <= 1 || num_shared_stages <= 1) {
      return for_node;
    }
    if (for_node->kind != ForKind::kSerial) {
      return for_node;
    }
    for_node = SplitSharedGemmOperands(for_node);
    const SeqStmtNode *seq = GetPipelineBodySeq(for_node->body);
    if (seq == nullptr) {
      return for_node;
    }

    Array<Stmt> decls;
    Array<Stmt> components;
    for (const Stmt &child : seq->seq) {
      if (IsPipelineDeclarationStmt(child)) {
        decls.push_back(child);
        continue;
      }
      bool has_unsupported_mma_loop = false;
      if (const auto *inner_seq =
              ExtractSplittableInnerSeq(child, &has_unsupported_mma_loop)) {
        for (const Stmt &inner : inner_seq->seq) {
          components.push_back(inner);
        }
      } else {
        if (has_unsupported_mma_loop) {
          return for_node;
        }
        components.push_back(child);
      }
    }

    const int n = static_cast<int>(components.size());
    if (n == 0) {
      return for_node;
    }

    std::vector<bool> is_shared_to_local(n, false);
    std::vector<bool> has_mma_compute(n, false);
    std::vector<bool> has_local_access(n, false);
    int first_register_producer_idx = -1;
    int first_compute_idx = -1;

    for (int i = 0; i < n; ++i) {
      const Stmt &s = components[i];
      if (RegisterPipelineClassifier::IsSharedToLocalCopy(s)) {
        is_shared_to_local[i] = true;
        if (first_register_producer_idx == -1) {
          first_register_producer_idx = i;
        }
      }
      if (RegisterPipelineClassifier::HasMmaCompute(s)) {
        has_mma_compute[i] = true;
        if (first_compute_idx == -1) {
          first_compute_idx = i;
        }
      }
      has_local_access[i] = RegisterPipelineClassifier::HasAnyLocalAccess(s);
    }
    if (first_register_producer_idx == -1 || first_compute_idx == -1 ||
        first_register_producer_idx >= first_compute_idx) {
      return for_node;
    }

    int compute_stage = 1;
    if (auto stage_anno = op->annotations.Get(kRegisterPipelineStage)) {
      if (auto old_stages = stage_anno.value().try_cast<Array<Integer>>()) {
        for (const Integer &stage : old_stages.value()) {
          compute_stage =
              std::max(compute_stage, static_cast<int>(stage->value));
        }
      }
    } else if (auto stage_anno =
                   op->annotations.Get(s_tir::attr::software_pipeline_stage)) {
      if (auto old_stages = stage_anno.value().try_cast<Array<Integer>>()) {
        for (const Integer &stage : old_stages.value()) {
          compute_stage =
              std::max(compute_stage, static_cast<int>(stage->value));
        }
      }
    }
    if (num_shared_stages > 0) {
      compute_stage = std::max(compute_stage, num_shared_stages);
    }
    int register_stage = std::max(0, compute_stage - 1);
    if (register_stage <= 0) {
      return for_node;
    }

    std::vector<Integer> orders(n, Integer(-1));
    std::vector<Integer> stages(n, Integer(compute_stage));
    bool user_provided_register_orders = false;
    if (auto order_anno = op->annotations.Get(kRegisterPipelineOrder)) {
      if (auto old_orders = order_anno.value().try_cast<Array<Integer>>()) {
        if (static_cast<int>(old_orders.value().size()) == n) {
          for (int i = 0; i < n; ++i) {
            orders[i] = old_orders.value()[i];
          }
          user_provided_register_orders = true;
        }
      }
    }

    for (int i = 0; i < n; ++i) {
      if (i < first_register_producer_idx) {
        stages[i] = Integer(0);
        continue;
      }
      if (i < first_compute_idx) {
        stages[i] = Integer(register_stage);
        continue;
      }
      if (has_mma_compute[i]) {
        stages[i] = Integer(compute_stage);
      } else if (is_shared_to_local[i]) {
        stages[i] = Integer(register_stage);
      } else if (has_local_access[i] && i < first_compute_idx) {
        stages[i] = Integer(register_stage);
      } else {
        stages[i] = Integer(compute_stage);
      }
    }

    if (!user_provided_register_orders) {
      for (int i = 0; i < n; ++i) {
        orders[i] = Integer(i);
      }
    }

    Map<String, Any> annotations;
    for (const auto &kv : for_node->annotations) {
      const String &key = kv.first;
      if (key == kRegisterPipelineStage || key == kRegisterPipelineOrder ||
          key == kRegisterPipelineAsyncStages) {
        continue;
      }
      annotations.Set(key, kv.second);
    }
    Array<Integer> stage_arr(stages);
    Array<Integer> order_arr(orders);
    annotations.Set(kRegisterPipelineStage, stage_arr);
    annotations.Set(kRegisterPipelineOrder, order_arr);

    if (auto sw_stage_anno =
            op->annotations.Get(s_tir::attr::software_pipeline_stage)) {
      if (auto sw_stages = sw_stage_anno.value().try_cast<Array<Integer>>()) {
        int sw_max = 0;
        for (const Integer &st : sw_stages.value()) {
          sw_max = std::max(sw_max, static_cast<int>(st->value));
        }
        if (sw_max < compute_stage) {
          Array<Integer> updated_sw;
          for (const Integer &st : sw_stages.value()) {
            if (st->value == sw_max && st->value > 0) {
              updated_sw.push_back(Integer(compute_stage));
            } else {
              updated_sw.push_back(st);
            }
          }
          annotations.Set(s_tir::attr::software_pipeline_stage, updated_sw);
        }
      }
    }

    if (auto async_stages = op->annotations.Get(kRegisterPipelineAsyncStages)) {
      annotations.Set(kRegisterPipelineAsyncStages, async_stages.value());
    } else if (auto sw_async = op->annotations.Get(
                   s_tir::attr::software_pipeline_async_stages)) {
      annotations.Set(kRegisterPipelineAsyncStages, sw_async.value());
    }

    Array<Stmt> rebuilt = decls;
    for (const Stmt &s : components) {
      rebuilt.push_back(s);
    }
    Stmt new_body = ReplaceInnermostSeq(for_node->body, rebuilt);
    return For(for_node->loop_var, for_node->min, for_node->extent,
               for_node->kind, new_body, for_node->thread_binding,
               std::move(annotations), for_node->step, for_node->span);
  }

private:
  Buffer GetOrCreateFragmentBuffer(const BufferRegion &src_region,
                                   const std::string &suffix) {
    const Buffer &src = src_region->buffer;
    if (auto existing = shared_to_fragment_.Get(src)) {
      return existing.value();
    }
    Array<PrimExpr> shape;
    shape.reserve(src_region->region.size());
    for (const Range &range : src_region->region) {
      shape.push_back(range->extent);
    }
    Buffer frag = decl_buffer(shape, src->dtype, src->name + suffix,
                              "local.fragment");
    shared_to_fragment_.Set(src, frag);
    pending_fragment_allocs_.push_back(frag);
    return frag;
  }

  Stmt RewriteGemmStmt(const Stmt &stmt) {
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *call = eval->value.as<CallNode>()) {
        auto tile_op = ParseOperator(GetRef<Call>(call));
        const auto *gemm = tile_op.as<GemmNode>();
        if (gemm == nullptr) {
          return stmt;
        }
        const bool a_shared = IsSharedLike(gemm->a_);
        const bool b_shared = IsSharedLike(gemm->b_);
        if (!a_shared && !b_shared) {
          return stmt;
        }
        BufferRegion a_region = gemm->aRegion_;
        BufferRegion b_region = gemm->bRegion_;
        Array<Stmt> copies;
        if (a_shared) {
          Buffer frag = GetOrCreateFragmentBuffer(
              a_region, kRegisterPipelineBufferSuffix);
          BufferRegion frag_region = MakeFragmentRegion(frag, a_region);
          copies.push_back(MakeCopyEvaluate(a_region, frag_region));
          a_region = frag_region;
        }
        if (b_shared) {
          Buffer frag = GetOrCreateFragmentBuffer(
              b_region, kRegisterPipelineBufferSuffix);
          BufferRegion frag_region = MakeFragmentRegion(frag, b_region);
          copies.push_back(MakeCopyEvaluate(b_region, frag_region));
          b_region = frag_region;
        }
        copies.push_back(Evaluate(RewriteGemmSharedOperands(
            GetRef<Call>(call), a_region, b_region)));
        return MakePipelineBody(copies);
      }
      return stmt;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> items;
      items.reserve(seq->seq.size());
      bool changed = false;
      for (const Stmt &child : seq->seq) {
        Stmt next = RewriteGemmStmt(child);
        changed = changed || !next.same_as(child);
        items.push_back(next);
      }
      if (!changed) {
        return stmt;
      }
      return MakePipelineBody(items);
    }
    if (const auto *ite = stmt.as<IfThenElseNode>()) {
      Stmt then_case = RewriteGemmStmt(ite->then_case);
      Stmt else_case = stmt;
      bool changed = !then_case.same_as(ite->then_case);
      if (ite->else_case.defined()) {
        else_case = RewriteGemmStmt(ite->else_case.value());
        changed = changed || !else_case.same_as(ite->else_case.value());
      }
      if (!changed) {
        return stmt;
      }
      return IfThenElse(ite->condition, then_case,
                        ite->else_case.defined()
                            ? Optional<Stmt>(else_case)
                            : Optional<Stmt>(),
                        ite->span);
    }
    if (const auto *br = stmt.as<SBlockRealizeNode>()) {
      SBlock block = br->block;
      Stmt new_body = RewriteGemmStmt(block->body);
      if (new_body.same_as(block->body)) {
        return stmt;
      }
      block.CopyOnWrite()->body = new_body;
      return SBlockRealize(br->iter_values, br->predicate, block, br->span);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      Stmt new_body = RewriteGemmStmt(attr->body);
      if (new_body.same_as(attr->body)) {
        return stmt;
      }
      return AttrStmt(attr->node, attr->attr_key, attr->value, new_body,
                      attr->span);
    }
    return stmt;
  }

  For SplitSharedGemmOperands(For for_node) {
    Array<Stmt> stmts = NormalizePipelineBody(for_node->body);
    Array<Stmt> rewritten;
    rewritten.reserve(stmts.size());
    bool changed = false;
    for (const Stmt &stmt : stmts) {
      Stmt next = RewriteGemmStmt(stmt);
      if (!next.same_as(stmt)) {
        changed = true;
      }
      rewritten.push_back(next);
    }
    if (!changed) {
      return for_node;
    }
    auto *n = for_node.CopyOnWrite();
    n->body = MakePipelineBody(rewritten);
    return for_node;
  }

  Array<Buffer> pending_fragment_allocs_;
  Map<Buffer, Buffer> shared_to_fragment_;
};

} // namespace

tirx::transform::Pass RegisterPipelinePlanning() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &, const PassContext &) {
    auto *fptr = f.CopyOnWrite();
    fptr->body = RegisterPipelinePlanner()(fptr->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.RegisterPipelinePlanning", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RegisterPipelinePlanning",
                        RegisterPipelinePlanning);
}

} // namespace tl
} // namespace tvm
