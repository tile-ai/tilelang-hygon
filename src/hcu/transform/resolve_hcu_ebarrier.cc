// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file resolve_hcu_ebarrier.cc
 * \brief Resolve deferred HCU AllReduce barrier policies.
 */
#include "arith/ir_mutator_with_analyzer.h"
#include "hcu/target_utils.h"
#include "hcu/utils/auto_ebarrier.h"
#include "op/builtin.h"
#include "transform/common/thread_sync_types.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

class HcuEBarrierResolver : public arith::IRMutatorWithAnalyzer {
public:
  static Stmt Rewrite(Stmt stmt, int warp_size) {
    class UsedBarrierCollector : public StmtExprVisitor {
    public:
      void VisitExpr_(const CallNode *op) final {
        const IntImmNode *id = nullptr;
        if ((op->op.same_as(ebarrier_sync()) ||
             op->op.same_as(ebarrier_sync_cnt()) ||
             op->op.same_as(ebarrier_arrive())) &&
            !op->args.empty()) {
          id = op->args[0].as<IntImmNode>();
        } else if (op->op.same_as(builtin::tvm_storage_sync()) &&
                   op->args.size() == 3) {
          id = op->args[1].as<IntImmNode>();
        }
        if (id != nullptr) {
          ICHECK_GE(id->value, 0);
          ICHECK_LT(id->value, 16);
          used_ids.insert(static_cast<size_t>(id->value));
        }
        StmtExprVisitor::VisitExpr_(op);
      }

      std::unordered_set<size_t> used_ids;
    } collector;
    collector(stmt);

    arith::Analyzer analyzer;
    HcuEBarrierResolver resolver(&analyzer, warp_size,
                                 std::move(collector.used_ids));
    return resolver(std::move(stmt));
  }

private:
  HcuEBarrierResolver(arith::Analyzer *analyzer, int warp_size,
                      std::unordered_set<size_t> used_ids)
      : IRMutatorWithAnalyzer(analyzer), warp_size_(warp_size),
        used_barrier_ids_(std::move(used_ids)) {}

  PrimExpr VisitExpr_(const CallNode *op) final {
    PrimExpr visited = IRMutatorWithAnalyzer::VisitExpr_(op);
    const auto *call = visited.as<CallNode>();
    if (call == nullptr || !call->op.same_as(builtin::call_extern()) ||
        call->args.empty()) {
      return visited;
    }

    const auto *name_node = call->args[0].as<StringImmNode>();
    if (name_node == nullptr) {
      return visited;
    }
    std::string name = name_node->value;
    const std::string marker = hcu::kAutoEBarrierPolicyMarker;
    size_t marker_pos = name.find(marker);
    if (marker_pos == std::string::npos) {
      return visited;
    }

    auto bound_tx = analyzer_->const_int_bound(tx_);
    auto bound_ty = analyzer_->const_int_bound(ty_);
    auto bound_tz = analyzer_->const_int_bound(tz_);
    std::string policy;
    if (IsFullThreadExtent(tx_, bound_tx) &&
        IsFullThreadExtent(ty_, bound_ty) &&
        IsFullThreadExtent(tz_, bound_tz)) {
      policy = "tl::SyncThreadsBarrier";
    } else {
      size_t thread_count = CalculateThreadExtent(tx_, bound_tx) *
                            CalculateThreadExtent(ty_, bound_ty) *
                            CalculateThreadExtent(tz_, bound_tz);
      ICHECK_EQ(thread_count % warp_size_, 0)
          << "HCU partial AllReduce thread count must be a multiple of the "
             "target warp size";
      size_t barrier_id = GetOrCreateBarrier(
          MakeKey(bound_tx, bound_ty, bound_tz), thread_count);
      policy = "tl::EBarrier<" + std::to_string(barrier_id) + ", " +
               std::to_string(thread_count / warp_size_) + ">";
    }

    name.replace(marker_pos, marker.size(), policy);
    Array<PrimExpr> args = call->args;
    args.Set(0, StringImm(name));
    return Call(call->dtype, call->op, args, call->annotations, call->span);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call != nullptr && call->op.same_as(builtin::tvm_storage_sync()) &&
        call->args.size() == 3) {
      const auto *id = call->args[1].as<IntImmNode>();
      const auto *thread_count = call->args[2].as<IntImmNode>();
      ICHECK(id != nullptr && thread_count != nullptr);
      auto key = MakeKey(analyzer_->const_int_bound(tx_),
                         analyzer_->const_int_bound(ty_),
                         analyzer_->const_int_bound(tz_));
      auto it = barrier_id_map_.find(key);
      if (it == barrier_id_map_.end()) {
        barrier_id_map_[key] = static_cast<size_t>(id->value);
        thread_count_map_[key] = static_cast<size_t>(thread_count->value);
      } else {
        ICHECK_EQ(it->second, static_cast<size_t>(id->value));
        ICHECK_EQ(thread_count_map_[key],
                  static_cast<size_t>(thread_count->value));
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  ThreadBoundKey MakeKey(const arith::ConstIntBound &bound_tx,
                         const arith::ConstIntBound &bound_ty,
                         const arith::ConstIntBound &bound_tz) const {
    return ThreadBoundKey{bound_tx->min_value, bound_tx->max_value,
                          bound_ty->min_value, bound_ty->max_value,
                          bound_tz->min_value, bound_tz->max_value};
  }

  size_t GetOrCreateBarrier(const ThreadBoundKey &key, size_t thread_count) {
    if (barrier_id_map_.count(key)) {
      ICHECK_EQ(thread_count_map_[key], thread_count);
      return barrier_id_map_[key];
    }
    while (used_barrier_ids_.count(next_barrier_id_)) {
      ++next_barrier_id_;
    }
    ICHECK_LT(next_barrier_id_, 16)
        << "HCU EBarrier allocation exhausted hardware IDs [0, 15]";
    size_t barrier_id = next_barrier_id_++;
    used_barrier_ids_.insert(barrier_id);
    barrier_id_map_[key] = barrier_id;
    thread_count_map_[key] = thread_count;
    return barrier_id;
  }

  size_t CalculateThreadExtent(const IterVar &iv,
                               const arith::ConstIntBound &bound) {
    if (!analyzer_->const_int_bound.IsBound(iv->var)) {
      return 1;
    }
    auto extent = *as_const_int(iv->dom->extent);
    int64_t count = analyzer_->z3_prover.CountSatisfyingValues(iv->var, extent);
    if (count > 0) {
      return static_cast<size_t>(count);
    }
    return static_cast<size_t>(bound->max_value - bound->min_value + 1);
  }

  bool IsFullThreadExtent(const IterVar &iv,
                          const arith::ConstIntBound &bound) const {
    if (!analyzer_->const_int_bound.IsBound(iv->var) || !iv->dom.defined()) {
      return true;
    }
    int64_t min = *as_const_int(iv->dom->min);
    int64_t extent = *as_const_int(iv->dom->extent);
    return min == bound->min_value && min + extent - 1 == bound->max_value;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tvm::tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        tx_ = iv;
      } else if (iv->thread_tag == "threadIdx.y") {
        ty_ = iv;
      } else if (iv->thread_tag == "threadIdx.z") {
        tz_ = iv;
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  int warp_size_;
  size_t next_barrier_id_{0};
  std::unordered_set<size_t> used_barrier_ids_;
  std::unordered_map<ThreadBoundKey, size_t> barrier_id_map_;
  std::unordered_map<ThreadBoundKey, size_t> thread_count_map_;
  IterVar tx_ =
      IterVar(Range::FromMinExtent(0, 1), Var("tx"), IterVarType::kDataPar);
  IterVar ty_ =
      IterVar(Range::FromMinExtent(0, 1), Var("ty"), IterVarType::kDataPar);
  IterVar tz_ =
      IterVar(Range::FromMinExtent(0, 1), Var("tz"), IterVarType::kDataPar);
};

tvm::transform::Pass ResolveHcuEBarrier() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target || !TargetIsHCU(target.value())) {
      return f;
    }
    int warp_size = TargetHcuGetWarpSize(target.value());
    auto *n = f.CopyOnWrite();
    n->body = HcuEBarrierResolver::Rewrite(n->body, warp_size);
    return f;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.ResolveHcuEBarrier", {});
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ResolveHcuEBarrier", ResolveHcuEBarrier);
}

} // namespace tl
} // namespace tvm
