// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file plan_scale_buffer_layout.cc
 * \brief Select the internal scale-buffer row order after async-copy planning.
 */

#include "hcu/op/copy_scale.h"
#include "hcu/target_utils.h"
#include "hcu/utils/scale_gemm_dep.h"
#include "hcu/utils/scale_lds_layout.h"
#include "op/copy.h"
#include "op/gemm_blockscaled.h"
#include "op/operator.h"
#include "op/utils.h"

#include <tvm/ir/transform.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

bool GetBoolAnnotation(const Map<String, ObjectRef> &annotations,
                       const char *key) {
  if (auto value = annotations.Get(key)) {
    if (const auto *imm = value->as<IntImmNode>()) {
      return imm->value != 0;
    }
  }
  return false;
}

bool IsAsyncCopyCall(const CallNode *call, const Op &tir_op) {
  static const Op &async_copy = Op::Get("tl.tileop.async_copy");
  if (tir_op.same_as(async_copy)) {
    return true;
  }
  if (!tir_op.same_as(Copy::Get())) {
    return false;
  }
  return GetBoolAnnotation(call->annotations, "is_async_copy") ||
         GetBoolAnnotation(call->annotations, "force_cp_async") ||
         GetBoolAnnotation(call->annotations,
                           attr::kAsyncCopyNoImplicitCommitWait);
}

struct CopyScaleSite {
  Call call;
  CopyScale op;
};

class ScaleLayoutCollector : public StmtExprVisitor {
public:
  std::vector<CopyScaleSite> copy_scale_sites;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> async_lds_outputs;
  int thread_x_extent{-1};

private:
  static bool IsZeroValue(const PrimExpr &value) {
    if (const auto *broadcast = value.as<BroadcastNode>()) {
      return IsZeroValue(broadcast->value);
    }
    if (const auto *float_imm = value.as<FloatImmNode>()) {
      return float_imm->value == 0.0;
    }
    if (const auto *int_imm = value.as<IntImmNode>()) {
      return int_imm->value == 0;
    }
    return false;
  }

  static const BufferLoadNode *MatchPreferredAsyncLoad(const PrimExpr &value) {
    if (const auto *load = value.as<BufferLoadNode>()) {
      return IsGlobalBuffer(load->buffer) ? load : nullptr;
    }
    const auto *call = value.as<CallNode>();
    if (call == nullptr || !call->op.same_as(tirx::builtin::if_then_else()) ||
        call->args.size() != 3 || !IsZeroValue(call->args[2])) {
      return nullptr;
    }
    return MatchPreferredAsyncLoad(call->args[1]);
  }

  void VisitStmt_(const ForNode *op) final {
    bool preferred_async = false;
    if (auto value = op->annotations.Get(attr::kLoopPreferAsync)) {
      if (auto flag = value.value().try_cast<Bool>()) {
        preferred_async = flag.value()->value;
      }
    }
    const int previous_depth = preferred_async_parallel_depth_;
    if (preferred_async) {
      ++preferred_async_parallel_depth_;
    }
    StmtExprVisitor::VisitStmt_(op);
    preferred_async_parallel_depth_ = previous_depth;
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (preferred_async_parallel_depth_ > 0 && IsSharedBuffer(op->buffer) &&
        MatchPreferredAsyncLoad(op->value) != nullptr) {
      // T.Parallel(prefer_async=True) is a mandatory async-copy request: HCU
      // lowering rejects the loop if none of its global->shared stores can be
      // rewritten. Anchor every matching LDS destination to the same linear
      // producer protocol as an explicit T.async_copy.
      async_lds_outputs.insert(op->buffer->data);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>()) {
        if (iv->thread_tag == "threadIdx.x") {
          if (const int64_t *extent = as_const_int(op->value)) {
            thread_x_extent = static_cast<int>(*extent);
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.as<OpNode>()) {
        Op tir_op = Downcast<Op>(call->op);
        if (tir_op.same_as(CopyScale::Get())) {
          copy_scale_sites.push_back(
              {GetRef<Call>(call),
               Downcast<CopyScale>(ParseOperator(GetRef<Call>(call)))});
        } else if (tir_op.same_as(Copy::Get()) ||
                   tir_op.same_as(Op::Get("tl.tileop.async_copy"))) {
          Copy copy = Downcast<Copy>(ParseOperator(GetRef<Call>(call)));
          if (IsAsyncCopyCall(call, tir_op) && IsSharedBuffer(copy->dst)) {
            async_lds_outputs.insert(copy->dst->data);
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  int preferred_async_parallel_depth_{0};
};

struct ScaleLayoutPlan {
  ScaleBufferRowOrder row_order{ScaleBufferRowOrder::kKInner};
  bool from_async_copy{false};
};

class PlanScaleBufferLayoutMutator : public StmtExprMutator {
public:
  explicit PlanScaleBufferLayoutMutator(
      const std::unordered_map<Buffer, ScaleLayoutPlan, ObjectPtrHash,
                               ObjectPtrEqual> &plans)
      : plans_(plans) {}

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr || !call->op.as<OpNode>()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Op tir_op = Downcast<Op>(call->op);
    auto annotations = call->annotations;
    bool changed = false;
    if (tir_op.same_as(CopyScale::Get())) {
      CopyScale copy = Downcast<CopyScale>(ParseOperator(GetRef<Call>(call)));
      auto it = plans_.find(copy->dst);
      ICHECK(it != plans_.end())
          << "PlanScaleBufferLayout: copy_scale destination has no plan: "
          << copy->dst;
      annotations.Set(
          attr::kScaleRowOrder,
          IntImm(DataType::Int(32), static_cast<int>(it->second.row_order)));
      annotations.Set(
          attr::kScaleLdsFromAsync,
          IntImm(DataType::Int(32), it->second.from_async_copy ? 1 : 0));
      changed = true;
    } else if (tir_op.same_as(GemmBlockScaled::Get())) {
      GemmBlockScaled gemm =
          Downcast<GemmBlockScaled>(ParseOperator(GetRef<Call>(call)));
      if (gemm->sfaRegion_.defined()) {
        auto it = plans_.find(gemm->sfaRegion_->buffer);
        if (it != plans_.end()) {
          annotations.Set(attr::kScaleARowOrder,
                          IntImm(DataType::Int(32),
                                 static_cast<int>(it->second.row_order)));
          changed = true;
        }
      }
      if (gemm->sfbRegion_.defined()) {
        auto it = plans_.find(gemm->sfbRegion_->buffer);
        if (it != plans_.end()) {
          annotations.Set(attr::kScaleBRowOrder,
                          IntImm(DataType::Int(32),
                                 static_cast<int>(it->second.row_order)));
          changed = true;
        }
      }
    }
    if (!changed) {
      return StmtExprMutator::VisitStmt_(op);
    }
    return Evaluate(
        Call(call->dtype, call->op, call->args, annotations, call->span));
  }

  const std::unordered_map<Buffer, ScaleLayoutPlan, ObjectPtrHash,
                           ObjectPtrEqual> &plans_;
};

PrimFunc PlanScaleBufferLayoutImpl(PrimFunc f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.defined() || !TargetIsHCU(target.value())) {
    return f;
  }
  ScaleLayoutCollector collector;
  collector(f->body);
  if (collector.copy_scale_sites.empty()) {
    return f;
  }
  ICHECK_GT(collector.thread_x_extent, 0)
      << "PlanScaleBufferLayout requires static threadIdx.x extent";

  std::unordered_map<Buffer, ScaleLayoutPlan, ObjectPtrHash, ObjectPtrEqual>
      plans;
  for (const CopyScaleSite &site : collector.copy_scale_sites) {
    const bool from_async =
        collector.async_lds_outputs.count(site.op->src->data) != 0;
    ScaleBufferRowOrder order = ScaleBufferRowOrder::kKInner;
    if (from_async) {
      order = hcu::SelectScaleBufferRowOrder(
          site.op.get(), collector.thread_x_extent, target.value());
    }
    ScaleLayoutPlan plan{order, from_async};
    auto [it, inserted] = plans.emplace(site.op->dst, plan);
    if (!inserted) {
      ICHECK_EQ(static_cast<int>(it->second.row_order),
                static_cast<int>(plan.row_order))
          << "shared scale buffer has incompatible row-order plans: "
          << site.op->dst;
      ICHECK_EQ(it->second.from_async_copy, plan.from_async_copy)
          << "shared scale buffer is produced from incompatible LDS producer "
             "kinds: "
          << site.op->dst;
    }
  }

  PlanScaleBufferLayoutMutator mutator(plans);
  PrimFuncNode *fn = f.CopyOnWrite();
  fn->body = mutator(f->body);
  return f;
}

} // namespace

tvm::transform::Pass PlanScaleBufferLayout() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    return PlanScaleBufferLayoutImpl(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PlanScaleBufferLayout", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.PlanScaleBufferLayout",
                        PlanScaleBufferLayout);
}

} // namespace tl
} // namespace tvm
