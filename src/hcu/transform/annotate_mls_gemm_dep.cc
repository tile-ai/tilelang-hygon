/*!
 * \file annotate_mls_gemm_dep.cc
 * \brief Pre-layout pass: annotate matrix_load / ds_read_format / gemm with MLS
 * consumer GEMM facts collected via PropagationTirCollector.
 */

#include "hcu/op/ds_read_format.h"
#include "hcu/op/mls.h"
#include "hcu/target_utils.h"
#include "hcu/utils/gemm_an_bt_lds_strategy.h"
#include "hcu/utils/gemm_at_bn_lds_strategy.h"
#include "hcu/utils/gemm_lds_access.h"
#include "hcu/utils/gemm_lds_strategy_utils.h"
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
        if (tir_op == MatrixLoad::Get() || tir_op == DsReadFormat::Get() ||
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

bool IsAnBtLinearDsReadCopyCandidate(const CopyNode *copy,
                                     const GemmWithInput &consumer) {
  if (!copy || !IsSharedBuffer(copy->src) ||
      copy->dst.scope() != "local.fragment")
    return false;
  const GemmNode *gemm = consumer.gemm.get();
  if (!gemm || gemm->k_ % 16 != 0 || gemm->kPack_ != 1)
    return false;
  const bool feeds_a = consumer.input.same_as(gemm->a_);
  if (!(feeds_a || consumer.input.same_as(gemm->b_)) ||
      (feeds_a ? !gemm->transA_ : gemm->transB_))
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
  return gemm && !gemm->transB_ && gemm->k_ % 16 == 0 && gemm->kPack_ == 1 &&
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

bool HaveSameAtBnStrategyParameters(const HcuGemmAtBnLdsStrategy &lhs,
                                    const HcuGemmAtBnLdsStrategy &rhs) {
  return lhs->strategy_version == rhs->strategy_version &&
         lhs->block_mn == rhs->block_mn && lhs->block_k == rhs->block_k &&
         lhs->block_threads == rhs->block_threads &&
         lhs->warp_size == rhs->warp_size &&
         lhs->warp_mn_count == rhs->warp_mn_count &&
         lhs->bank_num == rhs->bank_num &&
         lhs->bank_width_bytes == rhs->bank_width_bytes &&
         lhs->element_bytes == rhs->element_bytes &&
         lhs->copy_bytes_per_lane == rhs->copy_bytes_per_lane &&
         lhs->copy_transaction_bytes == rhs->copy_transaction_bytes &&
         lhs->copy_transactions_per_lane == rhs->copy_transactions_per_lane &&
         lhs->read_bytes_per_lane == rhs->read_bytes_per_lane &&
         lhs->row_period == rhs->row_period &&
         lhs->row_bank_stride == rhs->row_bank_stride &&
         lhs->rows_per_copy_wave == rhs->rows_per_copy_wave &&
         lhs->wrap_offset == rhs->wrap_offset &&
         lhs->wrap_idx_mask == rhs->wrap_idx_mask;
}

bool HaveSameAnBtStrategyParameters(const HcuGemmAnBtLdsStrategy &lhs,
                                    const HcuGemmAnBtLdsStrategy &rhs) {
  return lhs->strategy_version == rhs->strategy_version &&
         lhs->block_k == rhs->block_k && lhs->block_mn == rhs->block_mn &&
         lhs->block_threads == rhs->block_threads &&
         lhs->warp_size == rhs->warp_size && lhs->bank_num == rhs->bank_num &&
         lhs->bank_width_bytes == rhs->bank_width_bytes &&
         lhs->element_bytes == rhs->element_bytes &&
         lhs->copy_bytes_per_lane == rhs->copy_bytes_per_lane &&
         lhs->copy_transaction_bytes == rhs->copy_transaction_bytes &&
         lhs->copy_transactions_per_lane == rhs->copy_transactions_per_lane &&
         lhs->read_bytes_per_lane == rhs->read_bytes_per_lane &&
         lhs->phase_bytes == rhs->phase_bytes &&
         lhs->panel_mn == rhs->panel_mn &&
         lhs->wrap_offset == rhs->wrap_offset &&
         lhs->wrap_idx_mask == rhs->wrap_idx_mask;
}

bool HaveSameCopyStrategyParameters(const HcuGemmLdsCopyStrategy &lhs,
                                    const HcuGemmLdsCopyStrategy &rhs) {
  return lhs->strategy_version == rhs->strategy_version &&
         lhs->use_idxen == rhs->use_idxen &&
         lhs->copy_bytes_per_lane == rhs->copy_bytes_per_lane &&
         lhs->copy_transaction_bytes == rhs->copy_transaction_bytes &&
         lhs->block_threads == rhs->block_threads &&
         lhs->inner_extent == rhs->inner_extent &&
         lhs->wrap_offset == rhs->wrap_offset &&
         lhs->wrap_idx_mask == rhs->wrap_idx_mask &&
         lhs->storage_layout->IsEqual(rhs->storage_layout.get()) &&
         lhs->copy_loop_layout->IsEqual(rhs->copy_loop_layout.get());
}

MlsGemmDepMeta BuildCopyToGemmLinearMeta(const GemmWithInput &consumer,
                                         bool a_from_mls, bool b_from_mls) {
  const GemmNode *gemm = consumer.gemm.get();
  const bool feeds_a = consumer.input.same_as(gemm->a_);
  ICHECK(feeds_a || consumer.input.same_as(gemm->b_));
  auto node = tvm::ffi::make_object<MlsGemmDepMetaNode>();
  node->feeds_slot = feeds_a ? 0 : 1;
  node->trans = feeds_a ? !gemm->transA_ : gemm->transB_;
  node->gemm_m = gemm->m_;
  node->gemm_n = gemm->n_;
  node->gemm_k = gemm->k_;
  node->gemm_k_pack = gemm->kPack_;
  node->gemm_trans_a = gemm->transA_;
  node->gemm_trans_b = gemm->transB_;
  if (gemm->policy_.defined())
    node->gemm_policy = gemm->policy_->policy_type;
  node->a_from_mls = a_from_mls;
  node->b_from_mls = b_from_mls;
  return MlsGemmDepMeta(std::move(node));
}

bool IsAtBnAccess(const GemmWithInput &consumer) {
  const GemmNode *gemm = consumer.gemm.get();
  ICHECK(gemm != nullptr);
  const bool feeds_a = consumer.input.same_as(gemm->a_);
  ICHECK(feeds_a || consumer.input.same_as(gemm->b_));
  return GetHcuGemmLdsAccessKind(*gemm, feeds_a) == HcuGemmLdsAccessKind::kAtBn;
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
      auto copy_strategy = auto_copy_strategies_.find(buffer->data);
      if (copy_strategy == auto_copy_strategies_.end()) {
        continue;
      }
      auto at_bn_strategy = auto_at_bn_strategies_.find(buffer->data);
      auto an_bt_strategy = auto_an_bt_strategies_.find(buffer->data);

      if (Optional<Layout> existing = FindAnnotatedLayout(buffer)) {
        if (an_bt_strategy != auto_an_bt_strategies_.end() &&
            copy_strategy->second->storage_layout.same_as(
                an_bt_strategy->second->storage_layout)) {
          ValidateHcuGemmAnBtStorageLayout(existing.value(),
                                           an_bt_strategy->second);
        } else {
          ICHECK(at_bn_strategy != auto_at_bn_strategies_.end());
          ValidateHcuGemmAtBnStorageLayout(existing.value(),
                                           at_bn_strategy->second);
        }
      } else {
        const Layout &storage_layout = copy_strategy->second->storage_layout;
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
    for (const auto &[var, strategy] : auto_copy_strategies_) {
      (void)strategy;
      ICHECK(auto_layout_attached_vars_.count(var))
          << "Cannot attach compiler-derived HCU GEMM LDS layout to the "
             "SBlock that allocates buffer `"
          << var->name_hint << "`";
    }
  }

  Optional<HcuGemmAtBnLdsStrategy>
  TryDeriveAtBnStrategy(const CopyNode &copy,
                        const GemmWithInput &consumer) const {
    const GemmNode *gemm = consumer.gemm.get();
    if (gemm == nullptr) {
      return std::nullopt;
    }
    const bool feeds_a = consumer.input.same_as(gemm->a_);
    ICHECK(feeds_a || consumer.input.same_as(gemm->b_));
    return DeriveHcuGemmAtBnLdsStrategy(copy, *gemm, feeds_a, block_threads_,
                                        target_);
  }

  Optional<HcuGemmAnBtLdsStrategy>
  TryDeriveAnBtStrategy(const CopyNode &copy,
                        const GemmWithInput &consumer) const {
    const GemmNode *gemm = consumer.gemm.get();
    if (gemm == nullptr) {
      return std::nullopt;
    }
    const bool feeds_a = consumer.input.same_as(gemm->a_);
    ICHECK(feeds_a || consumer.input.same_as(gemm->b_));
    return DeriveHcuGemmAnBtLdsStrategy(copy, *gemm, feeds_a, block_threads_,
                                        target_);
  }

  Optional<HcuGemmAnBtLdsStrategy>
  TryDeriveAnBtCommonWrapStrategy(const CopyNode &copy,
                                  const GemmWithInput &consumer,
                                  int wrap_count) const {
    const GemmNode *gemm = consumer.gemm.get();
    if (gemm == nullptr) {
      return std::nullopt;
    }
    const bool feeds_a = consumer.input.same_as(gemm->a_);
    ICHECK(feeds_a || consumer.input.same_as(gemm->b_));
    return DeriveHcuGemmAnBtLdsStrategyWith64ByteWrap(
        copy, *gemm, feeds_a, block_threads_, target_, wrap_count);
  }

  std::pair<Optional<HcuGemmAtBnLdsStrategy>, Optional<HcuGemmAnBtLdsStrategy>>
  SelectAutoStrategy(const CopyNode &copy,
                     const std::vector<GemmWithInput> &consumers) const {
    Optional<HcuGemmAtBnLdsStrategy> at_bn_strategy;
    Optional<HcuGemmAnBtLdsStrategy> an_bt_strategy;
    std::optional<GemmWithInput> an_bt_consumer;
    for (const GemmWithInput &consumer : consumers) {
      Optional<HcuGemmAtBnLdsStrategy> current_at_bn =
          TryDeriveAtBnStrategy(copy, consumer);
      Optional<HcuGemmAnBtLdsStrategy> current_an_bt =
          TryDeriveAnBtStrategy(copy, consumer);
      ICHECK(!(current_at_bn.defined() && current_an_bt.defined()));
      if (!current_at_bn.defined() && !current_an_bt.defined()) {
        // Auto-layout is optional. If any consumer is unsupported, preserve
        // the ordinary layout inference path for the shared buffer.
        return {std::nullopt, std::nullopt};
      }
      if (current_at_bn.defined()) {
        if (at_bn_strategy.defined()) {
          ICHECK(HaveSameAtBnStrategyParameters(at_bn_strategy.value(),
                                                current_at_bn.value()))
              << "Conflicting AT/BN LDS strategies for shared buffer `"
              << copy.dst->name << "` across GEMM consumers";
        } else {
          at_bn_strategy = current_at_bn;
        }
      } else if (an_bt_strategy.defined()) {
        ICHECK(HaveSameAnBtStrategyParameters(an_bt_strategy.value(),
                                              current_an_bt.value()))
            << "Conflicting AN/BT LDS strategies for shared buffer `"
            << copy.dst->name << "` across GEMM consumers";
      } else {
        an_bt_strategy = current_an_bt;
        an_bt_consumer = consumer;
      }
    }

    const int at_bn_required_wrap_count =
        at_bn_strategy.defined()
            ? GetHcuGemmAtBnRequiredWrapCount(at_bn_strategy.value())
            : 1;
    if (at_bn_strategy.defined() && an_bt_strategy.defined() &&
        an_bt_strategy.value()->wrap_offset == 0 &&
        an_bt_strategy.value()->wrap_idx_mask == 0 &&
        at_bn_required_wrap_count > 1) {
      const int bank_ring_bytes = an_bt_strategy.value()->bank_num *
                                  an_bt_strategy.value()->bank_width_bytes;
      constexpr int kCommonWrapStepBytes = 64;
      constexpr int kCommonWrapCount = 2;
      const int common_wrap_count_limit =
          bank_ring_bytes / (2 * kCommonWrapStepBytes);
      if (common_wrap_count_limit >= kCommonWrapCount &&
          at_bn_required_wrap_count <= kCommonWrapCount &&
          an_bt_consumer.has_value()) {
        Optional<HcuGemmAnBtLdsStrategy> common_strategy =
            TryDeriveAnBtCommonWrapStrategy(copy, *an_bt_consumer,
                                            kCommonWrapCount);
        if (common_strategy.defined()) {
          an_bt_strategy = common_strategy;
        }
      }
    }
    return {at_bn_strategy, an_bt_strategy};
  }

  HcuGemmLdsCopyStrategy SelectCopyStrategy(
      const CopyNode &copy,
      const Optional<HcuGemmAtBnLdsStrategy> &at_bn_strategy,
      const Optional<HcuGemmAnBtLdsStrategy> &an_bt_strategy) const {
    ICHECK(at_bn_strategy.defined() || an_bt_strategy.defined());
    if (an_bt_strategy.defined()) {
      const HcuGemmAnBtLdsStrategy &an_bt = an_bt_strategy.value();
      if (at_bn_strategy.defined()) {
        const HcuGemmAtBnLdsStrategy &at_bn = at_bn_strategy.value();
        ICHECK_EQ(at_bn->block_mn, an_bt->block_k);
        ICHECK_EQ(at_bn->block_k, an_bt->block_mn);
        ICHECK_EQ(at_bn->block_threads, an_bt->block_threads);
        ICHECK_EQ(at_bn->copy_bytes_per_lane, an_bt->copy_bytes_per_lane);
        ICHECK_EQ(at_bn->copy_transaction_bytes, an_bt->copy_transaction_bytes)
            << "Mixed HCU GEMM consumers of shared buffer `" << copy.dst->name
            << "` require incompatible async-copy transactions";
        DLOG(WARNING)
            << "Shared buffer `" << copy.dst->name
            << "` has both AT/BN and AN/BT GEMM consumers; selecting the "
               "AN/BT-safe common LDS strategy";
      }
      return MakeHcuGemmLdsCopyStrategy(
          /*use_idxen=*/false, an_bt->copy_bytes_per_lane,
          an_bt->copy_transaction_bytes, an_bt->block_threads, an_bt->block_mn,
          an_bt->wrap_offset, an_bt->wrap_idx_mask, an_bt->storage_layout,
          an_bt->copy_loop_layout);
    }

    const HcuGemmAtBnLdsStrategy &at_bn = at_bn_strategy.value();
    return MakeHcuGemmLdsCopyStrategy(
        /*use_idxen=*/true, at_bn->copy_bytes_per_lane,
        at_bn->copy_transaction_bytes, at_bn->block_threads, at_bn->block_k,
        at_bn->wrap_offset, at_bn->wrap_idx_mask, at_bn->storage_layout,
        at_bn->copy_loop_layout);
  }

  bool IsLinearDsReadGemmInput(const Gemm &gemm, const Buffer &input) const {
    GemmWithInput consumer{gemm, input};
    for (const ProducerRecord &record : collector_->GetProducerRecords(input)) {
      if (record.call == nullptr) {
        continue;
      }
      Optional<TileOperator> producer =
          ParseOperator(ffi::GetRef<Call>(record.call));
      if (!producer.defined()) {
        continue;
      }
      if (const auto *copy = producer.as<CopyNode>()) {
        if (IsAnBtLinearDsReadCopyCandidate(copy, consumer)) {
          return true;
        }
      }
    }
    return false;
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
      if (IsMatrixLoadPreferredCopy(*copy.get())) {
        bool trans = true;
        LookupSharedMlsTrans(copy->dst, &trans);
        auto annotations = call->annotations;
        annotations.Set(attr::kMlsTrans,
                        IntImm(DataType::Int(32), trans ? 1 : 0));
        return Evaluate(
            Call(call->dtype, call->op, call->args, annotations, call->span));
      }
      auto consumer = PropagateToFindGemmConsumerOpWithInputAfterCall(
          copy->dst, collector_, call);
      if (consumer &&
          IsAnBtLinearDsReadCopyCandidate(copy.get(), consumer.value())) {
        const GemmNode *consumer_gemm = consumer->gemm.get();
        const bool feeds_a = consumer->input.same_as(consumer_gemm->a_);
        const bool has_layout = HasAnnotatedLayout(copy->src) ||
                                auto_an_bt_strategies_.count(copy->src->data);
        // The fallback template is B-specific. AN must have an explicit
        // logical-to-physical map before it can use the generic vector read.
        if (feeds_a && !has_layout) {
          return StmtExprMutator::VisitStmt_(op);
        }
        auto annotations = call->annotations;
        const Gemm &gemm = consumer->gemm;
        const bool a_from_mls = IsFromMls(gemm->a_, collector_) ||
                                IsLinearDsReadGemmInput(gemm, gemm->a_);
        const bool b_from_mls = IsFromMls(gemm->b_, collector_) ||
                                IsLinearDsReadGemmInput(gemm, gemm->b_);
        annotations.Set(attr::kMlsGemmDep,
                        BuildCopyToGemmLinearMeta(consumer.value(), a_from_mls,
                                                  b_from_mls));
        annotations.Set(attr::kHcuLinearDsRead, IntImm(DataType::Int(32), 1));
        if (has_layout) {
          annotations.Set(attr::kHcuLayoutDsRead, IntImm(DataType::Int(32), 1));
        }
        copy_ds_read_outputs_.insert(copy->dst);
        return Evaluate(Call(call->dtype, DsReadFormat::Get(), call->args,
                             annotations, call->span));
      }
      if (IsAsyncCopyOp(tir_op) && consumer) {
        std::vector<GemmWithInput> consumers =
            PropagateToFindAllGemmConsumersAfterCall(copy->dst, collector_,
                                                     call);
        auto [at_bn_strategy, an_bt_strategy] =
            SelectAutoStrategy(*copy.get(), consumers);
        if (at_bn_strategy.defined() || an_bt_strategy.defined()) {
          auto annotations = call->annotations;
          HcuGemmLdsCopyStrategy copy_strategy =
              SelectCopyStrategy(*copy.get(), at_bn_strategy, an_bt_strategy);
          if (auto loop_layout = annotations.Get(attr::kParallelLoopLayout)) {
            Fragment actual = Downcast<Fragment>(loop_layout.value());
            if (an_bt_strategy.defined() &&
                copy_strategy->copy_loop_layout.same_as(
                    an_bt_strategy.value()->copy_loop_layout)) {
              ValidateHcuGemmAnBtCopyLayout(actual, an_bt_strategy.value());
            } else {
              ValidateHcuGemmAtBnCopyLayout(actual, at_bn_strategy.value());
            }
          } else {
            annotations.Set(attr::kParallelLoopLayout,
                            copy_strategy->copy_loop_layout);
          }
          if (at_bn_strategy.defined()) {
            annotations.Set(attr::kHcuGemmAtBnLdsStrategy,
                            at_bn_strategy.value());
            auto [it, inserted] = auto_at_bn_strategies_.emplace(
                copy->dst->data, at_bn_strategy.value());
            if (!inserted) {
              ICHECK(HaveSameAtBnStrategyParameters(it->second,
                                                    at_bn_strategy.value()))
                  << "Conflicting HCU GEMM AT/BN strategies for shared buffer "
                  << copy->dst->name;
            }
          }
          if (an_bt_strategy.defined()) {
            annotations.Set(attr::kHcuGemmAnBtLdsStrategy,
                            an_bt_strategy.value());
            auto [it, inserted] = auto_an_bt_strategies_.emplace(
                copy->dst->data, an_bt_strategy.value());
            if (!inserted) {
              ICHECK(HaveSameAnBtStrategyParameters(it->second,
                                                    an_bt_strategy.value()))
                  << "Conflicting HCU GEMM AN/BT strategies for shared buffer "
                  << copy->dst->name;
            }
          }
          annotations.Set(attr::kHcuGemmLdsCopyStrategy, copy_strategy);
          auto [copy_it, copy_inserted] =
              auto_copy_strategies_.emplace(copy->dst->data, copy_strategy);
          if (!copy_inserted) {
            ICHECK(
                HaveSameCopyStrategyParameters(copy_it->second, copy_strategy))
                << "Conflicting HCU GEMM copy strategies for shared buffer "
                << copy->dst->name;
          }
          return Evaluate(
              Call(call->dtype, call->op, call->args, annotations, call->span));
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
      if (copy_ds_read_outputs_.count(gemm->a_)) {
        annotations.Set(attr::kHcuAFromMls, IntImm(DataType::Int(32), 1));
      }
      if (copy_ds_read_outputs_.count(gemm->b_)) {
        annotations.Set(attr::kHcuBFromMls, IntImm(DataType::Int(32), 1));
      }
      auto a_at_bn_strategy = auto_at_bn_strategies_.find(gemm->a_->data);
      auto a_an_bt_strategy = auto_an_bt_strategies_.find(gemm->a_->data);
      const bool a_uses_at_bn = IsAtBnAccess({gemm, gemm->a_});
      if (IsSharedBuffer(gemm->a_) && a_uses_at_bn &&
          a_at_bn_strategy != auto_at_bn_strategies_.end()) {
        annotations.Set(attr::kHcuGemmAtBnLdsStrategy,
                        a_at_bn_strategy->second);
      } else if (IsSharedBuffer(gemm->a_) && !a_uses_at_bn &&
                 a_an_bt_strategy != auto_an_bt_strategies_.end()) {
        annotations.Set(attr::kHcuARespectLayoutMap,
                        IntImm(DataType::Int(32), 1));
      } else if (IsSharedBuffer(gemm->a_) && HasAnnotatedLayout(gemm->a_)) {
        annotations.Set(attr::kHcuARespectLayoutMap,
                        IntImm(DataType::Int(32), 1));
      } else if (IsSharedBuffer(gemm->a_) &&
                 async_copy_linear_outputs_.count(gemm->a_->data)) {
        annotations.Set(attr::kHcuAFromAsyncCopyLinear,
                        IntImm(DataType::Int(32), 1));
      }
      auto b_an_bt_strategy = auto_an_bt_strategies_.find(gemm->b_->data);
      auto b_at_bn_strategy = auto_at_bn_strategies_.find(gemm->b_->data);
      const bool b_uses_at_bn = IsAtBnAccess({gemm, gemm->b_});
      if (IsSharedBuffer(gemm->b_) && !b_uses_at_bn &&
          b_an_bt_strategy != auto_an_bt_strategies_.end()) {
        annotations.Set(attr::kHcuGemmAnBtLdsStrategy,
                        b_an_bt_strategy->second);
        annotations.Set(attr::kHcuBRespectLayoutMap,
                        IntImm(DataType::Int(32), 1));
      } else if (IsSharedBuffer(gemm->b_) && b_uses_at_bn &&
                 b_at_bn_strategy != auto_at_bn_strategies_.end()) {
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
      copy_ds_read_outputs_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>
      async_copy_linear_outputs_;
  std::unordered_map<Var, HcuGemmAtBnLdsStrategy, ObjectPtrHash, ObjectPtrEqual>
      auto_at_bn_strategies_;
  std::unordered_map<Var, HcuGemmAnBtLdsStrategy, ObjectPtrHash, ObjectPtrEqual>
      auto_an_bt_strategies_;
  std::unordered_map<Var, HcuGemmLdsCopyStrategy, ObjectPtrHash, ObjectPtrEqual>
      auto_copy_strategies_;
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
