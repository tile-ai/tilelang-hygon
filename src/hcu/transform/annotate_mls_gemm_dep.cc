/*!
 * \file annotate_mls_gemm_dep.cc
 * \brief Pre-layout pass: annotate matrix_load / ds_read_format / gemm with MLS
 * consumer GEMM facts collected via PropagationTirCollector.
 */

#include "hcu/op/builtin.h"
#include "hcu/op/ds_read_format.h"
#include "hcu/op/gemm_partition.h"
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
#include "op/copy.h"
#include "op/gemm.h"
#include "op/gemm_blockscaled.h"
#include "op/operator.h"
#include "op/utils.h"
#include "transform/common/pipeline_utils.h"

#include "arith/ir_mutator_with_analyzer.h"

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

constexpr const char *kPendingGemmLdsCopyStrategy =
    "tl.hcu_pending_gemm_lds_copy_strategy";
constexpr const char *kPendingGemmAtBnLdsStrategy =
    "tl.hcu_pending_gemm_at_bn_lds_strategy";
constexpr const char *kPendingGemmAnBtLdsStrategy =
    "tl.hcu_pending_gemm_an_bt_lds_strategy";

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
            tir_op == Gemm::Get() || tir_op == GemmBlockScaled::Get()) {
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

class PipelineCopyCallCollector : public StmtExprVisitor {
public:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(Copy::Get()) ||
        op->op.same_as(Op::Get("tl.tileop.async_copy"))) {
      calls.insert(GetRef<Call>(op));
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  std::unordered_set<Call, ObjectPtrHash, ObjectPtrEqual> calls;
};

class PipelineAsyncProducerCollector : public StmtExprVisitor {
public:
  static std::unordered_set<Call, ObjectPtrHash, ObjectPtrEqual>
  Collect(const Stmt &body) {
    PipelineAsyncProducerCollector collector;
    collector(body);
    return std::move(collector.selected_calls_);
  }

private:
  void VisitStmt_(const ForNode *op) final {
    auto async_producers = op->annotations.Get(kPipelineAsyncProducers);
    if (!async_producers) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }

    Array<Integer> flags = Downcast<Array<Integer>>(async_producers.value());
    Optional<Array<Integer>> replayable_mask;
    if (auto mask = op->annotations.Get(kPipelineReplayableScalarBinds)) {
      replayable_mask = Downcast<Array<Integer>>(mask.value());
    }

    Array<Stmt> statements = NormalizePipelineBody(op->body);
    size_t executable_index = 0;
    size_t flag_index = 0;
    for (const Stmt &stmt : statements) {
      if (IsPipelineDeclarationStmt(stmt)) {
        continue;
      }
      if (replayable_mask.defined()) {
        ICHECK_LT(executable_index, replayable_mask.value().size());
      }
      bool replayable = replayable_mask.defined() &&
                        !is_zero(replayable_mask.value()[executable_index]);
      ++executable_index;
      if (replayable) {
        continue;
      }
      ICHECK_LT(flag_index, flags.size());
      if (!is_zero(flags[flag_index])) {
        PipelineCopyCallCollector calls;
        calls(stmt);
        selected_calls_.insert(calls.calls.begin(), calls.calls.end());
      }
      ++flag_index;
    }
    ICHECK_EQ(flag_index, flags.size());
    if (replayable_mask.defined()) {
      ICHECK_EQ(executable_index, replayable_mask.value().size());
    }

    StmtExprVisitor::VisitStmt_(op);
  }

  std::unordered_set<Call, ObjectPtrHash, ObjectPtrEqual> selected_calls_;
};

bool HasAsyncCopySemantics(const Call &call) {
  if (!call->op.same_as(Copy::Get()))
    return true;
  for (const char *key : {"is_async_copy", "force_cp_async",
                          attr::kAsyncCopyNoImplicitCommitWait}) {
    if (auto value = call->annotations.Get(key)) {
      if (const auto *imm = value.value().as<IntImmNode>(); !imm || imm->value)
        return true;
    }
  }
  return false;
}

class PlainCopyWriterGate : public StmtExprVisitor {
public:
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> blocked;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> blocked_data;

private:
  void VisitStmt_(const BufferStoreNode *op) final {
    blocked.insert(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    // Region bridge loads are metadata and are handled below. Scalar reads
    // are outside the supported GEMM-only consumer set.
    blocked.insert(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(region()))
      return;
    if (async_loop_depth_ && op->op.same_as(Copy::Get())) {
      auto copy = Downcast<Copy>(ParseOperator(GetRef<Call>(op)));
      blocked.insert(copy->dst);
    }
    for (const PrimExpr &arg : op->args) {
      if (const auto *var = arg.as<VarNode>(); var && var->dtype.is_handle())
        blocked_data.insert(GetRef<Var>(var));
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    bool async_loop = false;
    for (const char *key :
         {attr::kLoopPreferAsync, attr::kParallelAsyncWithoutAsyncCommitWait}) {
      if (auto value = op->annotations.Get(key)) {
        if (const auto *imm = value.value().as<IntImmNode>())
          async_loop |= imm->value != 0;
      }
    }
    async_loop_depth_ += async_loop;
    StmtExprVisitor::VisitStmt_(op);
    async_loop_depth_ -= async_loop;
  }

  void VisitStmt_(const SBlockNode *op) final {
    for (const auto &match : op->match_buffers)
      blocked.insert(match->source->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  int async_loop_depth_{0};
};

PrimExpr MakePlainFragmentRegion(const Buffer &buffer, int access_mask) {
  Array<PrimExpr> zeros;
  for (const PrimExpr &extent : buffer->shape)
    zeros.push_back(make_zero(extent.dtype()));
  Array<PrimExpr> args{BufferLoad(buffer, zeros), Integer(access_mask)};
  for (const PrimExpr &extent : buffer->shape)
    args.push_back(extent);
  return Call(DataType::Handle(), region(), args);
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

struct ThreadExtentInfo {
  int block_threads{-1};
  IterVar thread_var;
};

class ThreadExtentCollector : public StmtExprVisitor {
public:
  static ThreadExtentInfo Collect(const Stmt &body) {
    ThreadExtentCollector collector;
    collector(body);
    if (collector.thread_x_extent_ <= 1 || collector.thread_y_extent_ != 1 ||
        collector.thread_z_extent_ != 1) {
      return {};
    }
    return {collector.thread_x_extent_, collector.thread_x_var_};
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iter_var = op->node.as<IterVarNode>()) {
        if (const int64_t *extent = as_const_int(op->value)) {
          if (iter_var->thread_tag == "threadIdx.x") {
            thread_x_extent_ = static_cast<int>(*extent);
            thread_x_var_ = GetRef<IterVar>(iter_var);
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
  IterVar thread_x_var_;
};

bool HaveCompatibleAtBnStrategyParameters(const HcuGemmAtBnLdsStrategy &lhs,
                                          const HcuGemmAtBnLdsStrategy &rhs) {
  // warp_mn_count is consumer-local partition metadata. It validates strategy
  // derivation but does not affect the physical LDS or copy-loop layouts.
  return lhs->strategy_version == rhs->strategy_version &&
         lhs->block_mn == rhs->block_mn && lhs->block_k == rhs->block_k &&
         lhs->block_threads == rhs->block_threads &&
         lhs->thread_offset == rhs->thread_offset &&
         lhs->warp_size == rhs->warp_size && lhs->bank_num == rhs->bank_num &&
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

bool HaveCompatibleAnBtStrategyParameters(const HcuGemmAnBtLdsStrategy &lhs,
                                          const HcuGemmAnBtLdsStrategy &rhs) {
  return lhs->strategy_version == rhs->strategy_version &&
         lhs->block_k == rhs->block_k && lhs->block_mn == rhs->block_mn &&
         lhs->block_threads == rhs->block_threads &&
         lhs->thread_offset == rhs->thread_offset &&
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
         lhs->thread_offset == rhs->thread_offset &&
         lhs->inner_extent == rhs->inner_extent &&
         lhs->warp_size == rhs->warp_size &&
         lhs->wrap_offset == rhs->wrap_offset &&
         lhs->wrap_idx_mask == rhs->wrap_idx_mask &&
         lhs->wrap_uses_copy_transaction == rhs->wrap_uses_copy_transaction &&
         lhs->storage_layout->IsEqual(rhs->storage_layout.get()) &&
         lhs->copy_loop_layout->IsEqual(rhs->copy_loop_layout.get());
}

MlsGemmDepMeta BuildCopyToGemmLinearMeta(const GemmWithInput &consumer,
                                         bool a_from_mls, bool b_from_mls,
                                         Target target) {
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
  node->gemm_mmac_k = hcu::ResolveHcuMmacModeForGemm(gemm, target).mmac_k;
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

bool IsGlobalLikeBuffer(const Buffer &buffer) {
  return IsGlobalBuffer(buffer) || (buffer.defined() && buffer.scope().empty());
}

using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;

std::optional<GemmWithInput> GemmWithInputFromReaderCall(const CallNode *call,
                                                         const Buffer &input) {
  if (!IsGemmTileOpCall(call)) {
    return std::nullopt;
  }
  auto parsed = ParseOperator(tvm::ffi::GetRef<Call>(call));
  if (!parsed.defined()) {
    return std::nullopt;
  }
  auto gemm = parsed.as<GemmNode>();
  if (gemm == nullptr) {
    return std::nullopt;
  }
  const bool feeds_a = gemm->a_.same_as(input);
  const bool feeds_b = gemm->b_.same_as(input);
  if (!feeds_a && !feeds_b) {
    return std::nullopt;
  }
  GemmWithInput result;
  result.gemm = GetRef<Gemm>(gemm);
  result.input = input;
  return result;
}

} // namespace

class AnnotateMlsGemmDepMutator : public arith::IRMutatorWithAnalyzer {
public:
  enum class Mode { kCollectCandidates, kMaterialize };

  AnnotateMlsGemmDepMutator(
      PropagationTirCollector *collector, Target target, int block_threads,
      IterVar thread_var, arith::Analyzer *analyzer, Mode mode,
      std::unordered_set<Call, ObjectPtrHash, ObjectPtrEqual> selected_calls =
          {})
      : arith::IRMutatorWithAnalyzer(analyzer), collector_(collector),
        target_(std::move(target)), block_threads_(block_threads),
        thread_var_(std::move(thread_var)), mode_(mode),
        selected_calls_(std::move(selected_calls)) {}

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
    ThreadExtentInfo thread = ThreadExtentCollector::Collect(f->body);
    arith::Analyzer analyzer;
    AnnotateMlsGemmDepMutator mutator(&collector, target.value(),
                                      thread.block_threads, thread.thread_var,
                                      &analyzer, Mode::kCollectCandidates);
    PrimFuncNode *fn = f.CopyOnWrite();
    fn->body = mutator(f->body);
    return f;
  }

  static PrimFunc Materialize(PrimFunc f) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsHCU(target.value()) ||
        !ContainsMlsTileOps(f->body)) {
      return f;
    }
    Map<Var, Buffer> buffer_map = BuildBufferMap(f);
    PropagationTirCollector collector(buffer_map);
    collector.Collect(f->body);
    ThreadExtentInfo thread = ThreadExtentCollector::Collect(f->body);
    arith::Analyzer analyzer;
    auto selected_calls = PipelineAsyncProducerCollector::Collect(f->body);
    AnnotateMlsGemmDepMutator mutator(
        &collector, target.value(), thread.block_threads, thread.thread_var,
        &analyzer, Mode::kMaterialize, std::move(selected_calls));
    PlainCopyWriterGate writer_gate;
    writer_gate(f->body);
    mutator.blocked_plain_buffers_ = std::move(writer_gate.blocked);
    mutator.blocked_plain_data_ = std::move(writer_gate.blocked_data);
    PrimFuncNode *fn = f.CopyOnWrite();
    fn->body = mutator(f->body);
    mutator.ValidateAutoLayoutsAttached();
    return f;
  }

private:
  struct ParallelGatherCopy {
    Copy copy;
  };

  std::optional<ParallelGatherCopy>
  MatchPreferredAsyncParallelGather(const ForNode *op) {
    auto prefer_async = op->annotations.Get(attr::kLoopPreferAsync);
    if (!prefer_async.has_value()) {
      return std::nullopt;
    }
    auto prefer_async_bool = prefer_async.value().try_cast<Bool>();
    if (!prefer_async_bool.has_value() || !prefer_async_bool.value()->value) {
      return std::nullopt;
    }

    std::vector<Var> loop_vars;
    std::vector<PrimExpr> loop_extents;
    const ForNode *loop = op;
    while (loop != nullptr && loop->kind == ForKind::kParallel) {
      loop_vars.push_back(loop->loop_var);
      loop_extents.push_back(loop->extent);
      loop = loop->body.as<ForNode>();
    }
    if (loop_vars.empty()) {
      return std::nullopt;
    }

    const BufferStoreNode *candidate_store = nullptr;
    const BufferLoadNode *candidate_load = nullptr;
    bool ambiguous = false;
    PostOrderVisit(GetRef<For>(op), [&](const ObjectRef &obj) {
      const auto *store = obj.as<BufferStoreNode>();
      if (store == nullptr || !IsSharedBuffer(store->buffer)) {
        return;
      }
      const auto *load = store->value.as<BufferLoadNode>();
      if (load == nullptr || !IsGlobalBuffer(load->buffer) ||
          load->buffer->dtype != store->buffer->dtype) {
        ambiguous = true;
        return;
      }
      if (candidate_store != nullptr) {
        ambiguous = true;
        return;
      }
      candidate_store = store;
      candidate_load = load;
    });
    if (ambiguous || candidate_store == nullptr || candidate_load == nullptr) {
      return std::nullopt;
    }

    Map<Var, PrimExpr> loop_zeros;
    for (const Var &var : loop_vars) {
      loop_zeros.Set(var, make_zero(var->dtype));
    }
    Array<Range> dst_range;
    for (const PrimExpr &index : candidate_store->indices) {
      PrimExpr min = analyzer_->Simplify(tirx::Substitute(index, loop_zeros));
      PrimExpr residual = analyzer_->Simplify(index - min);
      PrimExpr extent = Integer(1);
      bool matched_loop = false;
      for (size_t i = 0; i < loop_vars.size(); ++i) {
        if (analyzer_->CanProveEqual(residual, loop_vars[i])) {
          extent = loop_extents[i];
          matched_loop = true;
          break;
        }
      }
      if (!matched_loop && UsesVar(index, [&](const VarNode *var) {
            for (const Var &loop_var : loop_vars) {
              if (var == loop_var.get()) {
                return true;
              }
            }
            return false;
          })) {
        return std::nullopt;
      }
      dst_range.push_back(Range::FromMinExtent(min, extent));
    }

    Array<Range> src_range;
    for (const PrimExpr &index : candidate_load->indices) {
      src_range.push_back(Range::FromMinExtent(index, Integer(1)));
    }
    auto copy_node = make_object<CopyNode>();
    copy_node->src = candidate_load->buffer;
    copy_node->dst = candidate_store->buffer;
    copy_node->src_range = src_range;
    copy_node->dst_range = dst_range;
    copy_node->annotations = {};
    return ParallelGatherCopy{Copy(copy_node)};
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (mode_ != Mode::kCollectCandidates) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    std::optional<ParallelGatherCopy> gather =
        MatchPreferredAsyncParallelGather(op);
    if (!gather.has_value()) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    const Copy &copy = gather.value().copy;
    auto consumers = GetExclusiveGemmConsumersAfterCall(copy->dst, nullptr);
    if (!consumers.has_value()) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    auto pending = TryBuildPendingGemmCopyAnnotations(
        *copy.get(), Map<String, ObjectRef>(), *consumers);
    if (!pending.defined()) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    Map<String, ObjectRef> strategy_annotations =
        MaterializeGemmCopyAnnotations(*copy.get(), pending.value());
    Map<String, Any> annotations = op->annotations;
    for (const auto &[key, value] : strategy_annotations) {
      annotations.Set(key, value);
    }
    For updated(op->loop_var, op->min, op->extent, op->kind, op->body,
                op->thread_binding, annotations, op->step, op->span);
    // Bypass this override for the newly annotated root.  Calling VisitStmt
    // here dispatches back into VisitStmt_(ForNode*) and can materialize the
    // same gather strategy repeatedly, duplicating surrounding declarations.
    return IRMutatorWithAnalyzer::VisitStmt_(updated.get());
  }

  bool IsFullPlainRegion(const BufferRegion &region) const {
    const Buffer &buffer = region->buffer;
    if (region->region.size() != buffer->shape.size() ||
        !buffer->strides.empty())
      return false;
    for (size_t i = 0; i < buffer->shape.size(); ++i) {
      if (!is_zero(region->region[i]->min) ||
          !analyzer_->CanProveEqual(region->region[i]->extent,
                                    buffer->shape[i]))
        return false;
    }
    return true;
  }

  Optional<Layout> TryMakePlainLayout(const Buffer &buffer) const {
    if (block_threads_ < TargetHcuGetWarpSize(target_) ||
        block_threads_ % TargetHcuGetWarpSize(target_) != 0 ||
        !IsSharedBuffer(buffer) || HasAnnotatedLayout(buffer) ||
        blocked_plain_buffers_.count(buffer) ||
        blocked_plain_data_.count(buffer->data) ||
        !(buffer->dtype.is_float16() || buffer->dtype.is_bfloat16()) ||
        buffer->dtype.lanes() != 1 || !buffer->strides.empty() ||
        buffer->shape.size() != 2 || !is_zero(buffer->elem_offset))
      return std::nullopt;
    auto writers = collector_->GetProducerRecords(buffer);
    if (writers.empty())
      return std::nullopt;
    bool has_fragment_writer = false;
    for (const auto &writer : writers) {
      if (!writer.call)
        return std::nullopt;
      Call call = GetRef<Call>(writer.call);
      if (HasAsyncCopySemantics(call) || selected_calls_.count(call))
        return std::nullopt;
      auto parsed = ParseOperator(call);
      const auto *copy = parsed.as<CopyNode>();
      if (!copy || IsMatrixLoadPreferredCopy(*copy) ||
          copy->src->dtype != buffer->dtype ||
          !IsFullPlainRegion(BufferRegion(buffer, copy->dst_range)))
        return std::nullopt;
      if (IsGlobalLikeBuffer(copy->src)) {
        if (copy->src_range.size() < 2 ||
            !analyzer_->CanProveEqual(floormod(copy->src_range.back()->min, 8),
                                      0))
          return std::nullopt;
      } else if (copy->src.scope() == "local.fragment") {
        if (!copy->dst.same_as(buffer) || !copy->src->strides.empty() ||
            !is_zero(copy->src->elem_offset) ||
            !StructuralEqual()(copy->src->shape, buffer->shape) ||
            !IsFullPlainRegion(BufferRegion(copy->src, copy->src_range)))
          return std::nullopt;
        has_fragment_writer = true;
      } else {
        return std::nullopt;
      }
    }
    auto consumers = GetExclusiveGemmConsumersAfterCall(buffer, nullptr);
    if (!consumers || consumers->empty())
      return std::nullopt;
    std::vector<HcuGemmLdsReadPattern> reads;
    for (const GemmWithInput &consumer : *consumers) {
      const Gemm &gemm = consumer.gemm;
      if (gemm->a_.same_as(gemm->b_) ||
          gemm->IsInstance<GemmBlockScaledNode>() ||
          gemm->k_ % (16 * gemm->kPack_) != 0 ||
          gemm->a_->dtype != buffer->dtype || gemm->b_->dtype != buffer->dtype)
        return std::nullopt;
      bool feeds_a = consumer.input.same_as(gemm->a_);
      HcuGemmLdsAccessKind kind = GetHcuGemmLdsAccessKind(*gemm.get(), feeds_a);
      const BufferRegion &input = feeds_a ? gemm->aRegion_ : gemm->bRegion_;
      // First-stage scope is full compact tiles, including identity relays.
      if (!IsFullPlainRegion(input))
        return std::nullopt;
      if (!consumer.input.same_as(buffer)) {
        for (const auto &relay : collector_->GetReaderCalls(buffer)) {
          if (relay.call && GemmWithInputFromReaderCall(relay.call, buffer))
            continue;
          if (!relay.call || !relay.call->op.same_as(Copy::Get()))
            return std::nullopt;
          auto copy = Downcast<Copy>(ParseOperator(GetRef<Call>(relay.call)));
          if (has_fragment_writer &&
              (HasAsyncCopySemantics(GetRef<Call>(relay.call)) ||
               selected_calls_.count(GetRef<Call>(relay.call)) ||
               !copy->src.same_as(buffer) ||
               copy->dst->dtype != buffer->dtype ||
               !copy->dst->strides.empty() || !is_zero(copy->dst->elem_offset)))
            return std::nullopt;
          if (!IsFullPlainRegion(BufferRegion(copy->src, copy->src_range)) ||
              !IsFullPlainRegion(BufferRegion(copy->dst, copy->dst_range)) ||
              !StructuralEqual()(copy->src->shape, copy->dst->shape) ||
              copy->dst.scope() != "local.fragment")
            return std::nullopt;
        }
      }
      if (kind == HcuGemmLdsAccessKind::kAnBt) {
        if (gemm->kPack_ != 1 ||
            GetHcuGemmLdsMnExtent(*gemm.get(), feeds_a) % 32)
          return std::nullopt;
        GemmWarpPolicy policy(gemm->policy_->policy_type);
        hcu::ComputeWarpPartitionHCU(
            *policy.get(), gemm->m_, gemm->n_, gemm->k_, gemm->kPack_,
            buffer->dtype.bits(), block_threads_, target_,
            IsFromMls(gemm->a_, collector_), IsFromMls(gemm->b_, collector_),
            !gemm->transA_, gemm->transB_);
        int warp_mn = feeds_a ? policy->m_warp : policy->n_warp;
        int mn = GetHcuGemmLdsMnExtent(*gemm.get(), feeds_a);
        if (policy->k_warp != 1 || mn % warp_mn || (mn / warp_mn) % 32)
          return std::nullopt;
      }
      reads.push_back({16, 16, 32, 8, 16, kind, gemm->kPack_});
    }
    return MakeHcuPlainGemmLdsLayout(
        buffer->shape, TargetHcuGetLdsBankCount(target_),
        TargetHcuGetLdsBankWidthBytes(target_), reads);
  }

  Range CurrentStrategyThreadBounds() const {
    if (thread_var_.defined()) {
      Range bounds = ComputeThreadBounds(thread_var_, *analyzer_);
      if (const int64_t *extent = as_const_int(bounds->extent)) {
        if (*extent > 1) {
          return bounds;
        }
      }
    }
    return Range::FromMinExtent(Integer(0), Integer(block_threads_));
  }

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
    bool plain_layout_map_changed = false;
    annotated_layout_maps_stack_.push_back(block_layout_map);
    if (mode_ == Mode::kMaterialize) {
      for (const Buffer &buffer : op->alloc_buffers) {
        if (Optional<Layout> layout = TryMakePlainLayout(buffer)) {
          // Ordinary global and fragment producers share the same storage
          // map and layout-aware GEMM reads. Their copies remain synchronous.
          plain_layouts_.emplace(buffer->data, layout.value());
          block_layout_map.Set(buffer->data, layout.value());
          annotated_layout_maps_stack_.back().Set(buffer->data, layout.value());
          plain_layout_map_changed = true;
        }
      }
    }

    SBlock block = Downcast<SBlock>(IRMutatorWithAnalyzer::VisitStmt_(op));
    bool layout_map_changed = plain_layout_map_changed;
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
    return DeriveHcuGemmAtBnLdsStrategy(copy, *gemm, feeds_a,
                                        CurrentStrategyThreadBounds(), target_);
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
    return DeriveHcuGemmAnBtLdsStrategy(copy, *gemm, feeds_a,
                                        CurrentStrategyThreadBounds(), target_);
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
        copy, *gemm, feeds_a, CurrentStrategyThreadBounds(), target_,
        wrap_count);
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
          if (!HaveCompatibleAtBnStrategyParameters(at_bn_strategy.value(),
                                                    current_at_bn.value())) {
            return {std::nullopt, std::nullopt};
          }
        } else {
          at_bn_strategy = current_at_bn;
        }
      } else if (an_bt_strategy.defined()) {
        if (!HaveCompatibleAnBtStrategyParameters(an_bt_strategy.value(),
                                                  current_an_bt.value())) {
          return {std::nullopt, std::nullopt};
        }
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
      if (common_wrap_count_limit < kCommonWrapCount ||
          at_bn_required_wrap_count > kCommonWrapCount ||
          !an_bt_consumer.has_value()) {
        return {std::nullopt, std::nullopt};
      }
      Optional<HcuGemmAnBtLdsStrategy> common_strategy =
          TryDeriveAnBtCommonWrapStrategy(copy, *an_bt_consumer,
                                          kCommonWrapCount);
      if (!common_strategy.defined()) {
        return {std::nullopt, std::nullopt};
      }
      an_bt_strategy = common_strategy;
    }
    return {at_bn_strategy, an_bt_strategy};
  }

  Optional<HcuGemmLdsCopyStrategy> TrySelectCopyStrategy(
      const CopyNode &copy,
      const Optional<HcuGemmAtBnLdsStrategy> &at_bn_strategy,
      const Optional<HcuGemmAnBtLdsStrategy> &an_bt_strategy) const {
    ICHECK(at_bn_strategy.defined() || an_bt_strategy.defined());
    if (an_bt_strategy.defined()) {
      const HcuGemmAnBtLdsStrategy &an_bt = an_bt_strategy.value();
      if (at_bn_strategy.defined()) {
        const HcuGemmAtBnLdsStrategy &at_bn = at_bn_strategy.value();
        if (at_bn->block_mn != an_bt->block_k ||
            at_bn->block_k != an_bt->block_mn ||
            at_bn->block_threads != an_bt->block_threads ||
            at_bn->thread_offset != an_bt->thread_offset ||
            at_bn->copy_bytes_per_lane != an_bt->copy_bytes_per_lane ||
            at_bn->copy_transaction_bytes != an_bt->copy_transaction_bytes) {
          return std::nullopt;
        }
        DLOG(WARNING)
            << "Shared buffer `" << copy.dst->name
            << "` has both AT/BN and AN/BT GEMM consumers; selecting the "
               "AN/BT-safe common LDS strategy";
      }
      return MakeHcuGemmLdsCopyStrategy(
          /*use_idxen=*/false, an_bt->copy_bytes_per_lane,
          an_bt->copy_transaction_bytes, an_bt->block_threads,
          an_bt->thread_offset, an_bt->block_mn, an_bt->warp_size,
          an_bt->wrap_offset, an_bt->wrap_idx_mask,
          /*wrap_uses_copy_transaction=*/false, an_bt->storage_layout,
          an_bt->copy_loop_layout);
    }

    const HcuGemmAtBnLdsStrategy &at_bn = at_bn_strategy.value();
    const int copy_warp_count = at_bn->block_threads / at_bn->warp_size;
    const bool wrap_uses_copy_transaction =
        at_bn->wrap_idx_mask + 1 > copy_warp_count;
    return MakeHcuGemmLdsCopyStrategy(
        /*use_idxen=*/true, at_bn->copy_bytes_per_lane,
        at_bn->copy_transaction_bytes, at_bn->block_threads,
        at_bn->thread_offset, at_bn->block_k, at_bn->warp_size,
        at_bn->wrap_offset, at_bn->wrap_idx_mask, wrap_uses_copy_transaction,
        at_bn->storage_layout, at_bn->copy_loop_layout);
  }

  int GetDsReadAlt(const Buffer &input) const {
    int alt = 1;
    bool found = false;
    for (const ProducerRecord &record : collector_->GetProducerRecords(input)) {
      if (record.call == nullptr) {
        continue;
      }
      Optional<TileOperator> producer =
          ParseOperator(ffi::GetRef<Call>(record.call));
      const auto *ds_read = producer.as<DsReadFormatNode>();
      if (ds_read == nullptr) {
        continue;
      }
      ICHECK(!found || alt == ds_read->alt_)
          << "GEMM B fragment has ds_read_format producers with conflicting "
             "Alt values";
      alt = ds_read->alt_;
      found = true;
    }
    return alt;
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

  bool CollectExclusiveGemmConsumersAfterCall(
      const Buffer &buffer, int after_order, BufferSet *visited_buffers,
      std::unordered_set<const CallNode *, CallNodePtrHash, CallNodePtrEqual>
          *seen_calls,
      std::vector<GemmWithInput> *result) const {
    if (collector_ == nullptr || !visited_buffers->insert(buffer).second) {
      return true;
    }
    bool has_reader = false;
    for (const ReaderCallRecord &reader : collector_->GetReaderCalls(buffer)) {
      if (reader.call == nullptr || reader.stmt_order <= after_order) {
        continue;
      }
      has_reader = true;
      if (auto gemm = GemmWithInputFromReaderCall(reader.call, buffer)) {
        if (seen_calls->insert(reader.call).second) {
          result->push_back(*gemm);
        }
        continue;
      }
      Optional<TileOperator> reader_op =
          ParseOperator(ffi::GetRef<Call>(reader.call));
      if (!reader_op.defined()) {
        return false;
      }
      const auto *relay_copy = reader_op.as<CopyNode>();
      if (relay_copy == nullptr || IsSharedBuffer(reader.write) ||
          !reader.write.defined()) {
        return false;
      }
      if (!CollectExclusiveGemmConsumersAfterCall(
              reader.write, after_order, visited_buffers, seen_calls, result)) {
        return false;
      }
    }
    return has_reader;
  }

  std::optional<std::vector<GemmWithInput>>
  GetExclusiveGemmConsumersAfterCall(const Buffer &buffer,
                                     const CallNode *after_site_call) const {
    int after_order = -1;
    if (collector_ != nullptr && after_site_call != nullptr) {
      after_order = collector_->GetCallStmtOrder(after_site_call);
    }
    BufferSet visited_buffers;
    std::unordered_set<const CallNode *, CallNodePtrHash, CallNodePtrEqual>
        seen_calls;
    std::vector<GemmWithInput> result;
    if (!CollectExclusiveGemmConsumersAfterCall(
            buffer, after_order, &visited_buffers, &seen_calls, &result)) {
      return std::nullopt;
    }
    return result;
  }

  Optional<Map<String, ObjectRef>> TryBuildPendingGemmCopyAnnotations(
      const CopyNode &copy, const Map<String, ObjectRef> &base_annotations,
      const std::vector<GemmWithInput> &consumers) {
    auto [at_bn_strategy, an_bt_strategy] = SelectAutoStrategy(copy, consumers);
    if (!at_bn_strategy.defined() && !an_bt_strategy.defined()) {
      return std::nullopt;
    }
    auto annotations = base_annotations;
    Optional<HcuGemmLdsCopyStrategy> copy_strategy =
        TrySelectCopyStrategy(copy, at_bn_strategy, an_bt_strategy);
    if (!copy_strategy.defined()) {
      return std::nullopt;
    }
    if (at_bn_strategy.defined()) {
      annotations.Set(kPendingGemmAtBnLdsStrategy, at_bn_strategy.value());
    }
    if (an_bt_strategy.defined()) {
      annotations.Set(kPendingGemmAnBtLdsStrategy, an_bt_strategy.value());
    }
    annotations.Set(kPendingGemmLdsCopyStrategy, copy_strategy.value());
    return annotations;
  }

  Map<String, ObjectRef>
  MaterializeGemmCopyAnnotations(const CopyNode &copy,
                                 Map<String, ObjectRef> annotations) {
    auto pending_copy = annotations.Get(kPendingGemmLdsCopyStrategy);
    ICHECK(pending_copy);
    HcuGemmLdsCopyStrategy copy_strategy =
        Downcast<HcuGemmLdsCopyStrategy>(pending_copy.value());
    Optional<HcuGemmAtBnLdsStrategy> at_bn_strategy;
    Optional<HcuGemmAnBtLdsStrategy> an_bt_strategy;
    if (auto pending = annotations.Get(kPendingGemmAtBnLdsStrategy)) {
      at_bn_strategy = Downcast<HcuGemmAtBnLdsStrategy>(pending.value());
    }
    if (auto pending = annotations.Get(kPendingGemmAnBtLdsStrategy)) {
      an_bt_strategy = Downcast<HcuGemmAnBtLdsStrategy>(pending.value());
    }

    if (auto loop_layout = annotations.Get(attr::kParallelLoopLayout)) {
      Fragment actual = Downcast<Fragment>(loop_layout.value());
      if (an_bt_strategy.defined() &&
          copy_strategy->copy_loop_layout.same_as(
              an_bt_strategy.value()->copy_loop_layout)) {
        ValidateHcuGemmAnBtCopyLayout(actual, an_bt_strategy.value());
      } else {
        ICHECK(at_bn_strategy.defined());
        ValidateHcuGemmAtBnCopyLayout(actual, at_bn_strategy.value());
      }
    } else {
      annotations.Set(attr::kParallelLoopLayout,
                      copy_strategy->copy_loop_layout);
    }
    if (at_bn_strategy.defined()) {
      annotations.Set(attr::kHcuGemmAtBnLdsStrategy, at_bn_strategy.value());
      auto [it, inserted] = auto_at_bn_strategies_.emplace(
          copy.dst->data, at_bn_strategy.value());
      if (!inserted) {
        ICHECK(HaveCompatibleAtBnStrategyParameters(it->second,
                                                    at_bn_strategy.value()))
            << "Conflicting HCU GEMM AT/BN strategies for shared buffer "
            << copy.dst->name;
      }
    }
    if (an_bt_strategy.defined()) {
      annotations.Set(attr::kHcuGemmAnBtLdsStrategy, an_bt_strategy.value());
      auto [it, inserted] = auto_an_bt_strategies_.emplace(
          copy.dst->data, an_bt_strategy.value());
      if (!inserted) {
        ICHECK(HaveCompatibleAnBtStrategyParameters(it->second,
                                                    an_bt_strategy.value()))
            << "Conflicting HCU GEMM AN/BT strategies for shared buffer "
            << copy.dst->name;
      }
    }
    annotations.Set(attr::kHcuGemmLdsCopyStrategy, copy_strategy);
    auto [copy_it, copy_inserted] =
        auto_copy_strategies_.emplace(copy.dst->data, copy_strategy);
    if (!copy_inserted) {
      ICHECK(HaveSameCopyStrategyParameters(copy_it->second, copy_strategy))
          << "Conflicting HCU GEMM copy strategies for shared buffer "
          << copy.dst->name;
    }
    annotations.erase(kPendingGemmLdsCopyStrategy);
    annotations.erase(kPendingGemmAtBnLdsStrategy);
    annotations.erase(kPendingGemmAnBtLdsStrategy);
    return annotations;
  }

  Stmt MaterializePlainGemm(const Call &call, const Gemm &gemm,
                            Map<String, ObjectRef> annotations) {
    Array<PrimExpr> args = call->args;
    Array<Buffer> fragments;
    Array<Stmt> statements;
    bool convert_a =
        plain_layouts_.count(gemm->a_->data) && !IsAtBnAccess({gemm, gemm->a_});
    bool convert_b =
        plain_layouts_.count(gemm->b_->data) && !IsAtBnAccess({gemm, gemm->b_});
    for (int slot = 0; slot < 2; ++slot) {
      const Buffer &input = slot == 0 ? gemm->a_ : gemm->b_;
      if (!plain_layouts_.count(input->data))
        continue;
      annotations.Set(slot == 0 ? attr::kHcuARespectLayoutMap
                                : attr::kHcuBRespectLayoutMap,
                      Integer(1));
      if (!(slot == 0 ? convert_a : convert_b))
        continue;
      Buffer fragment =
          decl_buffer(input->shape, input->dtype,
                      input->name + "_plain_fragment", "local.fragment");
      fragments.push_back(fragment);
      Map<String, ObjectRef> read_annotations;
      read_annotations.Set(attr::kHcuLinearDsRead, Integer(1));
      read_annotations.Set(attr::kHcuLayoutDsRead, Integer(1));
      read_annotations.Set(attr::kMlsGemmDep,
                           BuildCopyToGemmLinearMeta(
                               {gemm, input}, IsFromMls(gemm->a_, collector_),
                               IsFromMls(gemm->b_, collector_), target_));
      statements.push_back(Evaluate(
          Call(call->dtype, DsReadFormat::Get(),
               {args[slot], MakePlainFragmentRegion(fragment, kAccessWrite)},
               read_annotations)));
      args.Set(slot, MakePlainFragmentRegion(fragment, kAccessRead));
    }
    statements.push_back(
        Evaluate(Call(call->dtype, call->op, args, annotations, call->span)));
    if (fragments.empty())
      return statements[0];
    // Preserve a single planner statement while giving temporary fragments a
    // lexical allocation scope; pipeline rewriting sees all source accesses.
    SBlock block({}, {gemm->aRegion_, gemm->bRegion_}, {gemm->cRegion_},
                 "plain_gemm_read", SeqStmt(statements), std::nullopt,
                 fragments);
    return SBlockRealize({}, const_true(), block);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    auto call = op->value.as<CallNode>();
    if (call == nullptr || !call->op.as<OpNode>()) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    Op tir_op = Downcast<Op>(call->op);

    if (tir_op == MatrixStore::Get()) {
      auto store = Downcast<MatrixStore>(ParseOperator(GetRef<Call>(call)));
      Buffer source = store->src;
      int before = collector_->GetCallStmtOrder(call);
      std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> visited;
      while (IsFragmentBuffer(source) && visited.insert(source).second) {
        auto records = collector_->GetProducerRecords(source);
        const ProducerRecord *last = nullptr;
        for (const auto &record : records) {
          if (record.stmt_order < before &&
              (!last || record.stmt_order > last->stmt_order))
            last = &record;
        }
        if (!last || !last->call)
          break;
        auto producer = last->call;
        if (producer->op.same_as(Gemm::Get())) {
          auto gemm = Downcast<Gemm>(ParseOperator(GetRef<Call>(producer)));
          auto meta =
              BuildMlsGemmDepMeta({gemm, gemm->b_}, collector_, target_);
          auto annotations = call->annotations;
          annotations.Set(attr::kMlsGemmDep, meta);
          annotations.Set(
              "tl.store_gemm_element_bits",
              Integer(hcu::ResolveHcuMmacModeForGemm(gemm.get(), target_)
                          .element_bits));
          return Evaluate(
              Call(call->dtype, call->op, call->args, annotations, call->span));
        }
        if (!producer->op.same_as(Copy::Get()))
          break;
        auto copy = Downcast<Copy>(ParseOperator(GetRef<Call>(producer)));
        // Only whole-fragment identity copies preserve GEMM warp ownership.
        if (!IsFragmentBuffer(copy->src) ||
            !StructuralEqual()(copy->src->shape, copy->dst->shape))
          break;
        bool full_copy = copy->src_range.size() == copy->src->shape.size() &&
                         copy->dst_range.size() == copy->dst->shape.size();
        for (size_t axis = 0; full_copy && axis < copy->src_range.size();
             ++axis) {
          full_copy = is_zero(copy->src_range[axis]->min) &&
                      is_zero(copy->dst_range[axis]->min) &&
                      analyzer_->CanProveEqual(copy->src_range[axis]->extent,
                                               copy->src->shape[axis]) &&
                      analyzer_->CanProveEqual(copy->dst_range[axis]->extent,
                                               copy->dst->shape[axis]);
        }
        if (!full_copy)
          break;
        source = copy->src;
        before = last->stmt_order;
      }
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

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
      if (ds->gemm_dep_.defined()) {
        return IRMutatorWithAnalyzer::VisitStmt_(op);
      }
      auto gemm_with_input = PropagateToFindGemmConsumerOpWithInputAfterCall(
          ds->dst, collector_, call);
      if (gemm_with_input) {
        auto meta = BuildMlsGemmDepMeta(*gemm_with_input, collector_, target_);
        auto annotations = call->annotations;
        annotations.Set(attr::kMlsGemmDep, meta);
        Call new_call(call->dtype, call->op, call->args, annotations,
                      call->span);
        return Evaluate(new_call);
      }
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    if (IsCopyLikeOp(tir_op)) {
      auto copy = Downcast<Copy>(ParseOperator(ffi::GetRef<Call>(call)));
      if (mode_ == Mode::kMaterialize &&
          plain_layouts_.count(copy->dst->data)) {
        auto annotations = call->annotations;
        annotations.erase(kPendingGemmLdsCopyStrategy);
        annotations.erase(kPendingGemmAtBnLdsStrategy);
        annotations.erase(kPendingGemmAnBtLdsStrategy);
        annotations.erase(attr::kHcuCopyAsyncPromotable);
        annotations.Set(attr::kHcuPlainGemmLdsCopy, Integer(1));
        return Evaluate(
            Call(call->dtype, call->op, call->args, annotations, call->span));
      }
      if (mode_ == Mode::kMaterialize &&
          call->annotations.count(kPendingGemmLdsCopyStrategy)) {
        const bool selected =
            IsAsyncCopyOp(tir_op) || selected_calls_.count(GetRef<Call>(call));
        auto annotations = call->annotations;
        if (selected) {
          annotations =
              MaterializeGemmCopyAnnotations(*copy.get(), annotations);
        } else {
          annotations.erase(kPendingGemmLdsCopyStrategy);
          annotations.erase(kPendingGemmAtBnLdsStrategy);
          annotations.erase(kPendingGemmAnBtLdsStrategy);
        }
        annotations.erase(attr::kHcuCopyAsyncPromotable);
        return Evaluate(
            Call(call->dtype, call->op, call->args, annotations, call->span));
      }
      if (IsMatrixLoadPreferredCopy(*copy.get())) {
        bool trans = true;
        LookupSharedMlsTrans(copy->dst, &trans);
        auto annotations = call->annotations;
        annotations.Set(attr::kMlsTrans,
                        IntImm(DataType::Int(32), trans ? 1 : 0));
        if (mode_ == Mode::kCollectCandidates) {
          // An explicitly selected MatrixLoad is an HCU async producer by
          // construction.  Tell the backend-neutral pipeline planner that
          // this copy has passed the HCU promotion gate so it can retain the
          // original producer grouping and wait placement.
          annotations.Set(attr::kHcuCopyAsyncPromotable,
                          IntImm(DataType::Int(32), 1));
        } else {
          // This marker is only an input to PipelinePlanning.  Do not leak it
          // into the materialized TileOp/codegen annotations.
          annotations.erase(attr::kHcuCopyAsyncPromotable);
        }
        return Evaluate(
            Call(call->dtype, call->op, call->args, annotations, call->span));
      }
      auto consumer = PropagateToFindGemmConsumerOpWithInputAfterCall(
          copy->dst, collector_, call);
      const bool can_materialize_ds_read =
          mode_ == Mode::kMaterialize || HasAnnotatedLayout(copy->src) ||
          auto_copy_strategies_.count(copy->src->data) ||
          async_copy_linear_outputs_.count(copy->src->data);
      if (can_materialize_ds_read && consumer &&
          IsAnBtLinearDsReadCopyCandidate(copy.get(), consumer.value())) {
        const GemmNode *consumer_gemm = consumer->gemm.get();
        const bool feeds_a = consumer->input.same_as(consumer_gemm->a_);
        const bool has_layout = HasAnnotatedLayout(copy->src) ||
                                auto_an_bt_strategies_.count(copy->src->data);
        // The fallback template is B-specific. AN must have an explicit
        // logical-to-physical map before it can use the generic vector read.
        if (feeds_a && !has_layout) {
          return IRMutatorWithAnalyzer::VisitStmt_(op);
        }
        auto annotations = call->annotations;
        const Gemm &gemm = consumer->gemm;
        const bool a_from_mls = IsFromMls(gemm->a_, collector_);
        const bool b_from_mls = IsFromMls(gemm->b_, collector_);
        MlsGemmDepMeta meta = BuildCopyToGemmLinearMeta(
            consumer.value(), a_from_mls, b_from_mls, target_);
        GemmWarpPolicy policy(meta->gemm_policy);
        hcu::ComputeWarpPartitionHCU(
            *policy.get(), meta->gemm_m, meta->gemm_n, meta->gemm_k,
            meta->gemm_k_pack, copy->dst->dtype.bits(), block_threads_, target_,
            a_from_mls, b_from_mls, !meta->gemm_trans_a, meta->gemm_trans_b,
            meta->min_m_per_warp, meta->min_n_per_warp, meta->gemm_mmac_k);
        const int warp_mn = feeds_a ? policy->m_warp : policy->n_warp;
        const int mn = GetHcuGemmLdsMnExtent(*gemm.get(), feeds_a);
        if (policy->k_warp != 1 || mn % warp_mn || (mn / warp_mn) % 32)
          return IRMutatorWithAnalyzer::VisitStmt_(op);
        annotations.Set(attr::kMlsGemmDep, meta);
        annotations.Set(attr::kHcuLinearDsRead, IntImm(DataType::Int(32), 1));
        if (has_layout) {
          annotations.Set(attr::kHcuLayoutDsRead, IntImm(DataType::Int(32), 1));
        }
        return Evaluate(Call(call->dtype, DsReadFormat::Get(), call->args,
                             annotations, call->span));
      }
      if (mode_ == Mode::kCollectCandidates && IsAsyncCopyOp(tir_op) &&
          consumer) {
        std::vector<GemmWithInput> consumers =
            PropagateToFindAllGemmConsumersAfterCall(copy->dst, collector_,
                                                     call);
        if (auto annotations = TryBuildPendingGemmCopyAnnotations(
                *copy.get(), call->annotations, consumers)) {
          Map<String, ObjectRef> active_annotations =
              MaterializeGemmCopyAnnotations(*copy.get(), annotations.value());
          return Evaluate(Call(call->dtype, call->op, call->args,
                               active_annotations, call->span));
        }
      }
      if (mode_ == Mode::kCollectCandidates && !IsAsyncCopyOp(tir_op) &&
          consumer && IsGlobalLikeBuffer(copy->src) &&
          IsSharedBuffer(copy->dst)) {
        if (auto consumers =
                GetExclusiveGemmConsumersAfterCall(copy->dst, call)) {
          if (auto annotations = TryBuildPendingGemmCopyAnnotations(
                  *copy.get(), call->annotations, *consumers)) {
            Map<String, ObjectRef> updated_annotations = annotations.value();
            updated_annotations.Set(attr::kHcuCopyAsyncPromotable,
                                    IntImm(DataType::Int(32), 1));
            return Evaluate(Call(call->dtype, call->op, call->args,
                                 updated_annotations, call->span));
          }
        }
      }
      if (IsAsyncCopyOp(tir_op) && IsSharedBuffer(copy->dst) &&
          !HasAnnotatedLayout(copy->dst)) {
        async_copy_linear_outputs_.insert(copy->dst->data);
      }
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    if (tir_op.same_as(Gemm::Get()) || tir_op.same_as(GemmBlockScaled::Get())) {
      auto gemm = Downcast<Gemm>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
      auto annotations =
          AnnotateGemmHcuMlsFlags(call->annotations, gemm.get(), collector_);
      const int b_ds_read_alt = GetDsReadAlt(gemm->b_);
      if (b_ds_read_alt != 1) {
        annotations.Set(attr::kHcuBDsReadAlt,
                        IntImm(DataType::Int(32), b_ds_read_alt));
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
      if (mode_ == Mode::kMaterialize &&
          (plain_layouts_.count(gemm->a_->data) ||
           plain_layouts_.count(gemm->b_->data))) {
        return MaterializePlainGemm(GetRef<Call>(call), gemm, annotations);
      }
      Call new_call(call->dtype, call->op, call->args, annotations, call->span);
      return Evaluate(new_call);
    }

    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>
      blocked_plain_buffers_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> blocked_plain_data_;
  std::unordered_map<Var, Layout, ObjectPtrHash, ObjectPtrEqual> plain_layouts_;
  PropagationTirCollector *collector_;
  Target target_;
  int block_threads_{-1};
  IterVar thread_var_;
  Mode mode_;
  std::unordered_set<Call, ObjectPtrHash, ObjectPtrEqual> selected_calls_;
  std::unordered_map<Buffer, bool, ObjectPtrHash, ObjectPtrEqual>
      shared_mls_trans_;
  std::vector<Map<Var, Layout>> annotated_layout_maps_stack_;
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

tvm::transform::Pass MaterializeHcuGemmLdsStrategy() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    (void)m;
    (void)ctx;
    return AnnotateMlsGemmDepMutator::Materialize(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MaterializeHcuGemmLdsStrategy",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnnotateMlsGemmDep", AnnotateMlsGemmDep);
  refl::GlobalDef().def("tl.transform.MaterializeHcuGemmLdsStrategy",
                        MaterializeHcuGemmLdsStrategy);
}

} // namespace tl
} // namespace tvm
