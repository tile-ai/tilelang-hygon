/*!
 * \file annotate_mls_gemm_dep.cc
 * \brief Pre-layout pass: annotate matrix_load / ds_read_format / gemm with MLS
 * consumer GEMM facts collected via PropagationTirCollector.
 */

#include "hcu/op/ds_read_format.h"
#include "hcu/op/mls.h"
#include "hcu/target_utils.h"
#include "hcu/utils/gemm_a_lds_strategy.h"
#include "hcu/utils/gemm_b_lds_strategy.h"
#include "hcu/utils/mls_gemm_dep.h"
#include "hcu/utils/propagation_tir_collector.h"
#include "hcu/utils/propagation_util.h"
#include "layout/layout.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "op/operator.h"
#include "op/utils.h"

#include <tvm/ir/transform.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

Map<Var, Buffer> BuildBufferMap(const PrimFunc &f) {
  Map<Var, Buffer> buffer_map;
  for (const auto &[var, buffer] : f->buffer_map) {
    buffer_map.Set(var, buffer);
  }
  return buffer_map;
}

/*! \brief Fast scan for tile-level MLS ops; avoids PropagationTirCollector on
 * non-MLS kernels. */
class MlsTileOpPresenceChecker : public StmtExprVisitor {
public:
  bool found() const { return found_; }

private:
  void VisitStmt(const Stmt &stmt) final {
    if (found_) {
      return;
    }
    StmtExprVisitor::VisitStmt(stmt);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (found_) {
      return;
    }
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.as<OpNode>()) {
        Op tir_op = Downcast<Op>(call->op);
        static const Op &async_copy = Op::Get("tl.tileop.async_copy");
        if (tir_op == MatrixLoad::Get() || tir_op == DsReadFormat::Get() ||
            tir_op == Copy::Get() || tir_op == async_copy ||
            tir_op == Gemm::Get()) {
          found_ = true;
          return;
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool found_{false};
};

bool ContainsMlsTileOps(const Stmt &body) {
  MlsTileOpPresenceChecker checker;
  checker(body);
  return checker.found();
}

bool IsStaticMultipleOf(const PrimExpr &expr, int64_t value) {
  const auto *imm = expr.as<IntImmNode>();
  return imm && value != 0 && imm->value % value == 0;
}

bool IsBLinearDsReadCopyCandidate(const CopyNode *copy,
                                  const GemmWithInput &consumer) {
  if (!copy || !IsSharedBuffer(copy->src) ||
      copy->dst.scope() != "local.fragment")
    return false;
  const GemmNode *gemm = consumer.gemm.get();
  if (!gemm || !consumer.input.same_as(gemm->b_) || gemm->transB_ ||
      gemm->k_ % 16 != 0 || gemm->kPack_ != 1)
    return false;
  if (!(copy->src->dtype.is_float16() || copy->src->dtype.is_bfloat16()) ||
      copy->src->dtype != copy->dst->dtype)
    return false;
  if (copy->src_range.size() < 2 || copy->dst_range.size() < 2)
    return false;
  size_t sr = copy->src_range.size();
  size_t dr = copy->dst_range.size();
  return IsStaticMultipleOf(copy->src_range[sr - 2]->extent, 16) &&
         IsStaticMultipleOf(copy->src_range[sr - 1]->extent, 32) &&
         IsStaticMultipleOf(copy->dst_range[dr - 2]->extent, 16) &&
         IsStaticMultipleOf(copy->dst_range[dr - 1]->extent, 32);
}

bool IsBLinearDirectGemmCandidate(const GemmNode *gemm) {
  return gemm && !gemm->transB_ && gemm->k_ % 16 == 0 &&
         gemm->kPack_ == 1 &&
         (gemm->b_->dtype.is_float16() || gemm->b_->dtype.is_bfloat16());
}

class ThreadExtentCollector : public StmtExprVisitor {
public:
  static int Collect(const Stmt &body) {
    ThreadExtentCollector collector;
    collector(body);
    if (collector.thread_x_extent_ <= 1 || collector.thread_y_extent_ != 1 ||
        collector.thread_z_extent_ != 1) {
      return -1;
    }
    return collector.thread_x_extent_;
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iter_var = op->node.as<IterVarNode>()) {
        if (const int64_t *extent = as_const_int(op->value)) {
          if (iter_var->thread_tag == "threadIdx.x") {
            thread_x_extent_ = static_cast<int>(*extent);
          } else if (iter_var->thread_tag == "threadIdx.y") {
            thread_y_extent_ = static_cast<int>(*extent);
          } else if (iter_var->thread_tag == "threadIdx.z") {
            thread_z_extent_ = static_cast<int>(*extent);
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  int thread_x_extent_{-1};
  int thread_y_extent_{1};
  int thread_z_extent_{1};
};

Optional<int> GetIntAnnotation(const Map<String, ObjectRef> &annotations,
                               const char *key) {
  if (auto value = annotations.Get(key)) {
    if (const auto *imm = value.value().as<IntImmNode>()) {
      return static_cast<int>(imm->value);
    }
    LOG(FATAL) << "Annotation `" << key << "` expects IntImm, got "
               << value.value()->GetTypeKey();
  }
  return std::nullopt;
}

bool HaveSameGemmAStrategyParameters(const HcuGemmALdsStrategy &lhs,
                                     const HcuGemmALdsStrategy &rhs) {
  return lhs->strategy_version == rhs->strategy_version &&
         lhs->block_m == rhs->block_m && lhs->block_k == rhs->block_k &&
         lhs->block_threads == rhs->block_threads &&
         lhs->warp_size == rhs->warp_size &&
         lhs->warp_m_count == rhs->warp_m_count &&
         lhs->bank_num == rhs->bank_num &&
         lhs->bank_width_bytes == rhs->bank_width_bytes &&
         lhs->element_bytes == rhs->element_bytes &&
         lhs->copy_bytes_per_lane == rhs->copy_bytes_per_lane &&
         lhs->copy_transaction_bytes == rhs->copy_transaction_bytes &&
         lhs->copy_transactions_per_lane ==
             rhs->copy_transactions_per_lane &&
         lhs->read_bytes_per_lane == rhs->read_bytes_per_lane &&
         lhs->row_period == rhs->row_period &&
         lhs->row_bank_stride == rhs->row_bank_stride &&
         lhs->segment_shift == rhs->segment_shift &&
         lhs->rows_per_copy_wave == rhs->rows_per_copy_wave &&
         lhs->row_slab_count == rhs->row_slab_count &&
         lhs->warp_tile_m == rhs->warp_tile_m &&
         lhs->wrap_offset == rhs->wrap_offset &&
         lhs->wrap_idx_mask == rhs->wrap_idx_mask;
}

bool HaveSameGemmBStrategyParameters(const HcuGemmBLdsStrategy &lhs,
                                     const HcuGemmBLdsStrategy &rhs) {
  return lhs->strategy_version == rhs->strategy_version &&
         lhs->block_k == rhs->block_k && lhs->block_n == rhs->block_n &&
         lhs->block_threads == rhs->block_threads &&
         lhs->warp_size == rhs->warp_size &&
         lhs->bank_num == rhs->bank_num &&
         lhs->bank_width_bytes == rhs->bank_width_bytes &&
         lhs->element_bytes == rhs->element_bytes &&
         lhs->copy_bytes_per_lane == rhs->copy_bytes_per_lane &&
         lhs->copy_transaction_bytes == rhs->copy_transaction_bytes &&
         lhs->copy_transactions_per_lane ==
             rhs->copy_transactions_per_lane &&
         lhs->read_bytes_per_lane == rhs->read_bytes_per_lane &&
         lhs->phase_bytes == rhs->phase_bytes &&
         lhs->panel_n == rhs->panel_n &&
         lhs->wrap_offset == rhs->wrap_offset &&
         lhs->wrap_idx_mask == rhs->wrap_idx_mask;
}

MlsGemmDepMeta BuildCopyToGemmBMeta(
    const GemmWithInput &consumer,
    const PropagationTirCollector *collector) {
  const GemmNode *gemm = consumer.gemm.get();
  auto node = tvm::ffi::make_object<MlsGemmDepMetaNode>();
  node->feeds_slot = 1;
  node->trans = gemm->transB_;
  node->gemm_m = gemm->m_;
  node->gemm_n = gemm->n_;
  node->gemm_k = gemm->k_;
  node->gemm_k_pack = gemm->kPack_;
  node->gemm_trans_a = gemm->transA_;
  node->gemm_trans_b = gemm->transB_;
  if (gemm->policy_.defined())
    node->gemm_policy = gemm->policy_->policy_type;
  node->a_from_mls = collector && IsFromMls(gemm->a_, collector);
  node->b_from_mls = true;
  return MlsGemmDepMeta(std::move(node));
}

} // namespace

class AnnotateMlsGemmDepMutator : public StmtExprMutator {
public:
  AnnotateMlsGemmDepMutator(PropagationTirCollector *collector, Target target,
                            int block_threads)
      : collector_(collector), target_(std::move(target)),
        block_threads_(block_threads) {}

  static PrimFunc Substitute(PrimFunc f) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsHCU(target.value())) {
      return f;
    }
    if (!ContainsMlsTileOps(f->body)) {
      return f;
    }
    Map<Var, Buffer> buffer_map = BuildBufferMap(f);
    PropagationTirCollector collector(buffer_map);
    collector.Collect(f->body);
    int block_threads = ThreadExtentCollector::Collect(f->body);
    AnnotateMlsGemmDepMutator mutator(&collector, target.value(),
                                      block_threads);
    PrimFuncNode *fn = f.CopyOnWrite();
    fn->body = mutator(f->body);
    mutator.ValidateAutoLayoutsAttached();
    return f;
  }

private:
  bool IsCopyLikeOp(const Op &op) const {
    static const Op &async_copy = Op::Get("tl.tileop.async_copy");
    return op.same_as(Copy::Get()) || op.same_as(async_copy);
  }

  bool IsAsyncCopyOp(const Op &op) const {
    static const Op &async_copy = Op::Get("tl.tileop.async_copy");
    return op.same_as(async_copy);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    Map<Var, Layout> block_layout_map;
    if (auto value = op->annotations.Get(attr::kLayoutMap)) {
      if (auto layout_map = value.value().as<Map<Var, Layout>>()) {
        block_layout_map = layout_map.value();
      }
    }
    annotated_layout_maps_stack_.push_back(block_layout_map);

    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    bool layout_map_changed = false;
    // Materialize auto layouts at the shared buffer's allocation scope, just
    // like an explicit T.annotate_layout entry.
    for (const Buffer &buffer : block->alloc_buffers) {
      auto a_strategy = auto_a_strategies_.find(buffer->data);
      auto b_strategy = auto_b_strategies_.find(buffer->data);
      ICHECK(a_strategy == auto_a_strategies_.end() ||
             b_strategy == auto_b_strategies_.end())
          << "Shared buffer `" << buffer->name
          << "` cannot use both HCU GEMM A and B LDS strategies";
      if (a_strategy == auto_a_strategies_.end() &&
          b_strategy == auto_b_strategies_.end()) {
        continue;
      }

      if (Optional<Layout> existing = FindAnnotatedLayout(buffer)) {
        if (a_strategy != auto_a_strategies_.end()) {
          ValidateHcuGemmAStorageLayout(existing.value(), a_strategy->second);
        } else {
          ValidateHcuGemmBStorageLayout(existing.value(), b_strategy->second);
        }
      } else {
        const Layout &storage_layout =
            a_strategy != auto_a_strategies_.end()
                ? a_strategy->second->storage_layout
                : b_strategy->second->storage_layout;
        block_layout_map.Set(buffer->data, storage_layout);
        annotated_layout_maps_stack_.back().Set(buffer->data, storage_layout);
        layout_map_changed = true;
      }
      auto_layout_attached_vars_.insert(buffer->data);
    }

    if (layout_map_changed) {
      SBlockNode *block_ptr = block.CopyOnWrite();
      block_ptr->annotations.Set(attr::kLayoutMap, block_layout_map);
    }
    annotated_layout_maps_stack_.pop_back();
    return block;
  }

  bool HasAnnotatedLayout(const Buffer &buffer) const {
    return FindAnnotatedLayout(buffer).defined();
  }

  Optional<Layout> FindAnnotatedLayout(const Buffer &buffer) const {
    for (auto it = annotated_layout_maps_stack_.rbegin();
         it != annotated_layout_maps_stack_.rend(); ++it) {
      if (Optional<Layout> layout = it->Get(buffer->data)) {
        return layout;
      }
    }
    return std::nullopt;
  }

  void ValidateAutoLayoutsAttached() const {
    for (const auto &[var, strategy] : auto_a_strategies_) {
      (void)strategy;
      ICHECK(auto_layout_attached_vars_.count(var))
          << "Cannot attach compiler-derived HCU GEMM A LDS layout to the "
             "SBlock that allocates buffer `"
          << var->name_hint << "`";
    }
    for (const auto &[var, strategy] : auto_b_strategies_) {
      (void)strategy;
      ICHECK(auto_layout_attached_vars_.count(var))
          << "Cannot attach compiler-derived HCU GEMM B LDS layout to the "
             "SBlock that allocates buffer `"
          << var->name_hint << "`";
    }
  }

  void ValidateOrSetIntAnnotation(Map<String, ObjectRef> *annotations,
                                  const char *operand, const char *key,
                                  int expected) const {
    if (Optional<int> actual = GetIntAnnotation(*annotations, key)) {
      ICHECK_EQ(actual.value(), expected)
          << "HCU GEMM " << operand << " auto LDS strategy requires `" << key
          << "`="
          << expected << ", but got " << actual.value();
      return;
    }
    annotations->Set(key, IntImm(DataType::Int(32), expected));
  }

  Optional<HcuGemmALdsStrategy>
  TryDeriveGemmAStrategy(const CopyNode &copy,
                         const GemmWithInput &consumer) const {
    const GemmNode *gemm = consumer.gemm.get();
    if (gemm == nullptr || !consumer.input.same_as(gemm->a_)) {
      return std::nullopt;
    }
    return DeriveHcuGemmALdsStrategy(copy, *gemm, block_threads_, target_);
  }

  Optional<HcuGemmBLdsStrategy>
  TryDeriveGemmBStrategy(const CopyNode &copy,
                         const GemmWithInput &consumer) const {
    const GemmNode *gemm = consumer.gemm.get();
    if (gemm == nullptr || !consumer.input.same_as(gemm->b_)) {
      return std::nullopt;
    }
    return DeriveHcuGemmBLdsStrategy(copy, *gemm, block_threads_, target_);
  }

  bool LookupSharedMlsTrans(const Buffer &dst, bool *out_trans) {
    auto it = shared_mls_trans_.find(dst);
    if (it != shared_mls_trans_.end()) {
      *out_trans = it->second;
      return true;
    }
    bool trans = true;
    ComputeSharedDstMlsTrans(dst, collector_, target_, &trans);
    shared_mls_trans_[dst] = trans;
    *out_trans = trans;
    return true;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    auto call = op->value.as<CallNode>();
    if (call == nullptr || !call->op.as<OpNode>()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Op tir_op = Downcast<Op>(call->op);

    if (tir_op == MatrixLoad::Get()) {
      auto mls =
          Downcast<MatrixLoad>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
      bool trans = true;
      LookupSharedMlsTrans(mls->dst, &trans);
      auto annotations = call->annotations;
      annotations.Set(attr::kMlsTrans,
                      IntImm(DataType::Int(32), trans ? 1 : 0));
      Call new_call(call->dtype, call->op, call->args, annotations, call->span);
      return Evaluate(new_call);
    }

    if (tir_op == DsReadFormat::Get()) {
      auto ds =
          Downcast<DsReadFormat>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
      auto gemm_with_input = PropagateToFindGemmConsumerOpWithInputAfterCall(
          ds->dst, collector_, call);
      if (gemm_with_input) {
        auto meta = BuildMlsGemmDepMeta(*gemm_with_input, collector_);
        auto annotations = call->annotations;
        annotations.Set(attr::kMlsGemmDep, meta);
        Call new_call(call->dtype, call->op, call->args, annotations,
                      call->span);
        return Evaluate(new_call);
      }
      return StmtExprMutator::VisitStmt_(op);
    }

    if (IsCopyLikeOp(tir_op)) {
      auto copy = Downcast<Copy>(ParseOperator(ffi::GetRef<Call>(call)));
      auto consumer = PropagateToFindGemmConsumerOpWithInputAfterCall(
          copy->dst, collector_, call);
      if (consumer &&
          IsBLinearDsReadCopyCandidate(copy.get(), consumer.value())) {
        auto annotations = call->annotations;
        annotations.Set(attr::kMlsGemmDep,
                        BuildCopyToGemmBMeta(consumer.value(), collector_));
        annotations.Set(attr::kHcuBLinearDsRead,
                        IntImm(DataType::Int(32), 1));
        if (HasAnnotatedLayout(copy->src) ||
            auto_b_strategies_.count(copy->src->data)) {
          annotations.Set(attr::kHcuBLayoutDsRead,
                          IntImm(DataType::Int(32), 1));
        }
        copy_ds_read_b_outputs_.insert(copy->dst);
        return Evaluate(Call(call->dtype, DsReadFormat::Get(), call->args,
                             annotations, call->span));
      }
      if (IsAsyncCopyOp(tir_op) && consumer) {
        Optional<HcuGemmALdsStrategy> strategy =
            TryDeriveGemmAStrategy(*copy.get(), consumer.value());
        if (strategy.defined()) {
          auto annotations = call->annotations;
          if (auto loop_layout = annotations.Get(attr::kParallelLoopLayout)) {
            Fragment actual = Downcast<Fragment>(loop_layout.value());
            ValidateHcuGemmACopyLayout(actual, strategy.value());
          } else {
            annotations.Set(attr::kParallelLoopLayout,
                            strategy.value()->copy_loop_layout);
          }
          ValidateOrSetIntAnnotation(&annotations, "A", "use_idxen", 1);
          ValidateOrSetIntAnnotation(&annotations, "A", "wrap_offset",
                                     strategy.value()->wrap_offset);
          ValidateOrSetIntAnnotation(&annotations, "A", "wrap_idx_mask",
                                     strategy.value()->wrap_idx_mask);
          annotations.Set(attr::kHcuGemmALdsStrategy, strategy.value());
          auto [it, inserted] =
              auto_a_strategies_.emplace(copy->dst->data, strategy.value());
          if (!inserted) {
            const HcuGemmALdsStrategy &existing = it->second;
            const HcuGemmALdsStrategy &current = strategy.value();
            ICHECK(HaveSameGemmAStrategyParameters(existing, current))
                << "Conflicting HCU GEMM A strategies for shared buffer "
                << copy->dst->name;
          }
          return Evaluate(Call(call->dtype, call->op, call->args, annotations,
                               call->span));
        }
        Optional<HcuGemmBLdsStrategy> b_strategy;
        auto existing_b_strategy = auto_b_strategies_.find(copy->dst->data);
        if (existing_b_strategy != auto_b_strategies_.end()) {
          b_strategy = existing_b_strategy->second;
        } else {
          b_strategy = TryDeriveGemmBStrategy(*copy.get(), consumer.value());
        }
        if (b_strategy.defined()) {
          auto annotations = call->annotations;
          if (auto loop_layout = annotations.Get(attr::kParallelLoopLayout)) {
            Fragment actual = Downcast<Fragment>(loop_layout.value());
            auto validated = validated_b_copy_layouts_.find(copy->dst->data);
            if (validated == validated_b_copy_layouts_.end() ||
                !validated->second.same_as(actual)) {
              ValidateHcuGemmBCopyLayout(actual, b_strategy.value());
              validated_b_copy_layouts_[copy->dst->data] = actual;
            }
          } else {
            annotations.Set(attr::kParallelLoopLayout,
                            b_strategy.value()->copy_loop_layout);
          }
          ValidateOrSetIntAnnotation(&annotations, "B", "wrap_offset",
                                     b_strategy.value()->wrap_offset);
          ValidateOrSetIntAnnotation(&annotations, "B", "wrap_idx_mask",
                                     b_strategy.value()->wrap_idx_mask);
          annotations.Set(attr::kHcuGemmBLdsStrategy, b_strategy.value());
          auto [it, inserted] =
              auto_b_strategies_.emplace(copy->dst->data, b_strategy.value());
          if (!inserted) {
            const HcuGemmBLdsStrategy &existing = it->second;
            const HcuGemmBLdsStrategy &current = b_strategy.value();
            ICHECK(HaveSameGemmBStrategyParameters(existing, current))
                << "Conflicting HCU GEMM B strategies for shared buffer "
                << copy->dst->name;
          }
          return Evaluate(Call(call->dtype, call->op, call->args, annotations,
                               call->span));
        }
      }
      if (IsAsyncCopyOp(tir_op) && IsSharedBuffer(copy->dst) &&
          !HasAnnotatedLayout(copy->dst)) {
        async_copy_linear_outputs_.insert(copy->dst->data);
      }
      return StmtExprMutator::VisitStmt_(op);
    }

    if (tir_op.same_as(Gemm::Get())) {
      auto gemm = Downcast<Gemm>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
      auto annotations =
          AnnotateGemmHcuMlsFlags(call->annotations, gemm.get(), collector_);
      if (copy_ds_read_b_outputs_.count(gemm->b_)) {
        annotations.Set(attr::kHcuBFromMls, IntImm(DataType::Int(32), 1));
      }
      auto auto_a_strategy = auto_a_strategies_.find(gemm->a_->data);
      if (IsSharedBuffer(gemm->a_) &&
          auto_a_strategy != auto_a_strategies_.end()) {
        annotations.Set(attr::kHcuGemmALdsStrategy, auto_a_strategy->second);
      } else if (IsSharedBuffer(gemm->a_) && HasAnnotatedLayout(gemm->a_)) {
        annotations.Set(attr::kHcuARespectLayoutMap,
                        IntImm(DataType::Int(32), 1));
      } else if (IsSharedBuffer(gemm->a_) &&
                 async_copy_linear_outputs_.count(gemm->a_->data)) {
        annotations.Set(attr::kHcuAFromAsyncCopyLinear,
                        IntImm(DataType::Int(32), 1));
      }
      auto auto_b_strategy = auto_b_strategies_.find(gemm->b_->data);
      if (IsSharedBuffer(gemm->b_) &&
          auto_b_strategy != auto_b_strategies_.end()) {
        annotations.Set(attr::kHcuGemmBLdsStrategy, auto_b_strategy->second);
        annotations.Set(attr::kHcuBRespectLayoutMap,
                        IntImm(DataType::Int(32), 1));
      } else if (IsSharedBuffer(gemm->b_) && HasAnnotatedLayout(gemm->b_)) {
        annotations.Set(attr::kHcuBRespectLayoutMap,
                        IntImm(DataType::Int(32), 1));
      } else if (IsSharedBuffer(gemm->b_) &&
                 async_copy_linear_outputs_.count(gemm->b_->data) &&
                 IsBLinearDirectGemmCandidate(gemm.get())) {
        annotations.Set(attr::kHcuBFromAsyncCopyLinear,
                        IntImm(DataType::Int(32), 1));
      }
      Call new_call(call->dtype, call->op, call->args, annotations, call->span);
      return Evaluate(new_call);
    }

    return StmtExprMutator::VisitStmt_(op);
  }

  PropagationTirCollector *collector_;
  Target target_;
  int block_threads_{-1};
  std::unordered_map<Buffer, bool, ObjectPtrHash, ObjectPtrEqual>
      shared_mls_trans_;
  std::vector<Map<Var, Layout>> annotated_layout_maps_stack_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>
      copy_ds_read_b_outputs_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>
      async_copy_linear_outputs_;
  std::unordered_map<Var, HcuGemmALdsStrategy, ObjectPtrHash, ObjectPtrEqual>
      auto_a_strategies_;
  std::unordered_map<Var, HcuGemmBLdsStrategy, ObjectPtrHash, ObjectPtrEqual>
      auto_b_strategies_;
  std::unordered_map<Var, Fragment, ObjectPtrHash, ObjectPtrEqual>
      validated_b_copy_layouts_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>
      auto_layout_attached_vars_;
};

tvm::transform::Pass AnnotateMlsGemmDep() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    (void)m;
    (void)ctx;
    return AnnotateMlsGemmDepMutator::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateMlsGemmDep", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnnotateMlsGemmDep", AnnotateMlsGemmDep);
}

} // namespace tl
} // namespace tvm
