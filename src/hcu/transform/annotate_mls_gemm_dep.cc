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
#include "layout/utils.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "op/gemm_blockscaled.h"
#include "op/operator.h"
#include "op/utils.h"
#include "transform/common/pipeline_utils.h"

#include "arith/ir_mutator_with_analyzer.h"

#include <climits>
#include <tvm/ffi/function.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <bitset>
#include <map>
#include <optional>
#include <tuple>
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

bool IsVectorReadPreferred(const CopyNode *copy) {
  auto value = copy->annotations.Get("prefer_instruction");
  if (!value)
    return false;
  auto text = value.value().as<StringImmNode>();
  return text && text->value == "ds_read_vector";
}

bool IsAnBtLinearDsReadCopyCandidate(const CopyNode *copy,
                                     const GemmWithInput &consumer) {
  if (!copy || !IsSharedBuffer(copy->src) ||
      copy->dst.scope() != "local.fragment")
    return false;
  const GemmNode *gemm = consumer.gemm.get();
  const int read_tile_k = copy->src->dtype.bits() == 8 ? 32 : 16;
  if (!gemm || gemm->k_ % read_tile_k != 0)
    return false;
  const bool feeds_a = consumer.input.same_as(gemm->a_);
  if (!(feeds_a || consumer.input.same_as(gemm->b_)) ||
      (feeds_a ? !gemm->transA_ : gemm->transB_))
    return false;
  if (!(copy->src->dtype.is_float16() || copy->src->dtype.is_bfloat16() ||
        copy->src->dtype.is_float8_e4m3fn() ||
        copy->src->dtype.is_float8_e5m2()) ||
      copy->src->dtype != copy->dst->dtype)
    return false;
  if (copy->src_range.size() < 2 || copy->dst_range.size() < 2)
    return false;
  size_t sr = copy->src_range.size();
  size_t dr = copy->dst_range.size();
  return IsStaticMultipleOf(copy->src_range[sr - 2]->extent, read_tile_k) &&
         IsStaticMultipleOf(copy->src_range[sr - 1]->extent, 32) &&
         IsStaticMultipleOf(copy->dst_range[dr - 2]->extent, read_tile_k) &&
         IsStaticMultipleOf(copy->dst_range[dr - 1]->extent, 32);
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
         lhs->storage_layout->IsEqual(rhs->storage_layout.get()) &&
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
                                         Target target, int warp_m = 0,
                                         int warp_n = 0, int warp_k = 0,
                                         int min_n = 0) {
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
  node->plain_warp_m = warp_m;
  node->plain_warp_n = warp_n;
  node->plain_warp_k = warp_k;
  node->plain_min_n = min_n;
  return MlsGemmDepMeta(std::move(node));
}

bool IsAtBnAccess(const GemmWithInput &consumer) {
  const GemmNode *gemm = consumer.gemm.get();
  ICHECK(gemm != nullptr);
  const bool feeds_a = consumer.input.same_as(gemm->a_);
  ICHECK(feeds_a || consumer.input.same_as(gemm->b_));
  return GetHcuGemmLdsAccessKind(*gemm, feeds_a) == HcuGemmLdsAccessKind::kAtBn;
}

// This certificate applies only to native AN/BT storage selected together with
// a compatible AT/BN wrap plan. It inspects static metadata, never IR proofs.
bool CanProveMixedWrapAtMostTwoWay(const CopyNode &copy,
                                   const std::vector<GemmWithInput> &consumers,
                                   const HcuGemmAtBnLdsStrategy &at_bn,
                                   const HcuGemmAnBtLdsStrategy &an_bt) {
  auto is_full_static_region = [](const BufferRegion &region) {
    const Buffer &buffer = region->buffer;
    if (buffer->shape.size() != 2 || region->region.size() != 2 ||
        !buffer->strides.empty() || !is_zero(buffer->elem_offset))
      return false;
    for (size_t dim = 0; dim < 2; ++dim) {
      const auto &range = region->region[dim];
      const int64_t *extent = as_const_int(range->extent);
      const int64_t *shape = as_const_int(buffer->shape[dim]);
      if (!is_zero(range->min) || !extent || !shape || *extent != *shape)
        return false;
    }
    return true;
  };
  if (!is_full_static_region(BufferRegion(copy.dst, copy.dst_range)))
    return false;
  for (const GemmWithInput &consumer : consumers) {
    const Gemm &gemm = consumer.gemm;
    const bool feeds_a = consumer.input.same_as(gemm->a_);
    if (!consumer.input.same_as(copy.dst) ||
        !is_full_static_region(feeds_a ? gemm->aRegion_ : gemm->bRegion_))
      return false;
  }

  const int banks = an_bt->bank_num;
  const int element_bytes = an_bt->element_bytes;
  const int transaction_bytes = an_bt->copy_transaction_bytes;
  const int wrap_count = an_bt->wrap_idx_mask + 1;
  const int64_t row_bytes = int64_t{an_bt->block_mn} * element_bytes;
  const int64_t wave_bytes = int64_t{64} * transaction_bytes;
  if ((banks != 32 && banks != 64) || an_bt->bank_width_bytes != 4 ||
      an_bt->warp_size != 64 || (element_bytes != 1 && element_bytes != 2) ||
      an_bt->read_bytes_per_lane != 16 || an_bt->panel_mn != 32 ||
      (transaction_bytes != 4 && transaction_bytes != 8 &&
       transaction_bytes != 16) ||
      row_bytes <= 0 || row_bytes > wave_bytes || wave_bytes % row_bytes ||
      wave_bytes % (banks * 4) || wrap_count <= 0 || wrap_count > 16 ||
      (wrap_count & (wrap_count - 1)) ||
      (wrap_count == 1 ? an_bt->wrap_offset != 0
                       : an_bt->wrap_offset <= 0 || an_bt->wrap_offset % 4))
    return false;

  // A copy wave contains whole rows and a whole number of bank rings. After
  // row permutation and wrap, its bank word is
  //   (row / wrap_count) * row_words + (row % wrap_count) * wrap_offset
  //   + column_word (mod banks).
  // Hence the coefficients of row bits 0..3 are available without evaluating
  // the storage Layout or enumerating lanes, addresses, or banks.
  auto row_coefficient = [&](int bit) -> int64_t {
    return bit < wrap_count ? int64_t{bit} * an_bt->wrap_offset
                            : (bit / wrap_count) * (row_bytes / 4);
  };
  auto low_bit = [banks](int64_t coefficient) -> uint64_t {
    uint64_t value = static_cast<uint64_t>(coefficient) & (banks - 1);
    return value & (~value + 1);
  };
  auto at_most_two_way = [banks](uint64_t mask) {
    // Choose one binary variable per distinct lowest coefficient bit. These
    // form triangular parity constraints, even in the presence of carries.
    // Each phase has banks dword requests, so its multiplicity is bounded by
    // banks / 2^popcount(mask). Coincident low bits give a conservative bound.
    return (uint64_t{1} << std::bitset<64>(mask).count()) >=
           static_cast<uint64_t>(banks / 2);
  };
  const int64_t r0 = row_coefficient(1), r1 = row_coefficient(2);
  const int64_t r2 = row_coefficient(4), r3 = row_coefficient(8);
  auto paired_phases_at_most_two_way = [&](uint64_t mask, int64_t row,
                                           int column) {
    return at_most_two_way(mask | low_bit(row + column)) &&
           at_most_two_way(mask | low_bit(row - column));
  };

  // Use the derived physical read width, independently of the consumer's
  // arithmetic dtype or kpack. An 8B read has consecutive row IDs; on 64
  // banks it also varies the adjacent 8B column. A 16B read's phase couples
  // row bit 2 to the N16 selector.
  uint64_t row_mask = low_bit(r0) | low_bit(r1);
  bool native_ok;
  if (at_bn->read_bytes_per_lane == 8) {
    native_ok = at_most_two_way((banks == 64 ? 3 : 1) | row_mask | low_bit(r2) |
                                low_bit(r3));
  } else if (at_bn->read_bytes_per_lane == 16) {
    native_ok = paired_phases_at_most_two_way(
        3 | row_mask | (banks == 64 ? low_bit(r3) : 0), r2, 4);
  } else {
    return false;
  }
  if (!native_ok)
    return false;

  // Matrix phases: b16 couples row bit 2 with column word bit 3; 32-bank b8
  // couples row bit 3 with column word bit 2. On 64-bank b8, N16 is fixed.
  if (element_bytes == 2)
    return paired_phases_at_most_two_way(
        7 | low_bit(r0) | (banks == 64 ? low_bit(r1) : 0), r2, 8);
  if (banks == 32)
    return paired_phases_at_most_two_way(3 | row_mask, r3, 4);
  return at_most_two_way(3 | row_mask | low_bit(r2) | low_bit(r3));
}

bool IsGlobalLikeBuffer(const Buffer &buffer) {
  return IsGlobalBuffer(buffer) || (buffer.defined() && buffer.scope().empty());
}

using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;

// Proof summaries only: graph traversal remains in propagation_util and uses
// PropagationTirCollector. Unknown summaries veto automatic promotion, never
// explicit ds_read_format or ds_read_vector dependency propagation.
class AutomaticRegisterAccessProof : public StmtExprVisitor {
public:
  std::unordered_map<Buffer, BufferSet, ObjectPtrHash, ObjectPtrEqual> sources;
  BufferSet invalid;
  std::vector<Var> loop_vars;

  void VisitStmt_(const ForNode *op) final {
    loop_vars.push_back(op->loop_var);
    StmtExprVisitor::VisitStmt_(op);
    loop_vars.pop_back();
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    if (!IsFragmentBuffer(op->buffer) && op->buffer.scope() != "local")
      return;
    bool valid = SideEffect(op->value) <= CallEffectKind::kReadState;
    BufferSet inputs;
    PostOrderVisit(op->value, [&](const ObjectRef &node) {
      if (const auto *load = node.as<BufferLoadNode>()) {
        // Recurrences over an axis absent from the output require reduction
        // inference. This proof is used only by automatic read selection.
        if (load->buffer.same_as(op->buffer)) {
          for (const Var &axis : loop_vars) {
            bool output_uses_axis = false;
            for (const auto &index : op->indices)
              output_uses_axis |= UsesVar(
                  index, [&](const VarNode *v) { return v == axis.get(); });
            if (!output_uses_axis)
              valid = false;
          }
        }
        if (!StructuralEqual()(load->buffer->shape, op->buffer->shape))
          return;
        if ((!IsFragmentBuffer(load->buffer) &&
             load->buffer.scope() != "local") ||
            !StructuralEqual()(load->indices, op->indices)) {
          valid = false;
        } else if (!load->buffer.same_as(op->buffer)) {
          inputs.insert(load->buffer);
        }
      }
      if (const auto *call = node.as<CallNode>()) {
        static auto vectorizable = Op::GetAttrMap<Bool>("TVectorizable");
        const auto *callee = call->op.as<OpNode>();
        // TIR's vectorizer handles these structural elementwise operations
        // directly; they do not carry the TVectorizable attribute.
        const bool structural = call->op.same_as(builtin::reinterpret()) ||
                                call->op.same_as(builtin::if_then_else());
        if (!callee ||
            (!structural && !vectorizable.get(GetRef<Op>(callee), Bool(false))))
          valid = false;
      }
    });
    if (!valid)
      invalid.insert(op->buffer);
    sources[op->buffer].insert(inputs.begin(), inputs.end());
  }

  bool Preserves(const Buffer &input, const Buffer &output) const {
    auto it = sources.find(output);
    return !invalid.count(output) && it != sources.end() &&
           it->second.count(input);
  }
};

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
  result.site = GetRef<Call>(call);
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
    mutator.register_access_proof_(f->body);
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
    mutator.register_access_proof_(f->body);
    writer_gate(f->body);
    mutator.blocked_plain_buffers_ = std::move(writer_gate.blocked);
    mutator.blocked_plain_data_ = std::move(writer_gate.blocked_data);
    PrimFuncNode *fn = f.CopyOnWrite();
    fn->body = mutator(f->body);
    mutator.ValidateAutoLayoutsAttached();
    return f;
  }

private:
  struct PlainReadContract {
    int warp_m, warp_n, warp_k, min_n;
    Fragment layout;
  };

  using ContractKey = std::tuple<const BufferNode *, const GemmNode *, bool>;
  // Keep the GEMM object alive with the entry: parsed operators are temporary
  // objects, so caching a bare node address alone would permit address reuse.
  mutable std::map<ContractKey,
                   std::pair<Gemm, std::optional<PlainReadContract>>>
      contracts_;

  // Consumer facts are independent of the upstream source dtype. Query Python
  // at most once per GEMM operand in this pass, keeping the key owner alive.
  mutable std::map<std::pair<const Object *, bool>,
                   std::pair<ObjectRef, Array<ObjectRef>>>
      consumer_layouts_;

  Array<ObjectRef> QueryConsumerLayout(const GemmWithInput &consumer) const {
    const auto &gemm = consumer.gemm;
    const bool feeds_a = consumer.input.same_as(gemm->a_);
    ObjectRef owner = consumer.site.defined() ? ObjectRef(consumer.site.value())
                                              : ObjectRef(gemm);
    const auto key = std::make_pair(owner.get(), feeds_a);
    auto cached = consumer_layouts_.find(key);
    if (cached != consumer_layouts_.end())
      return cached->second.second;
    auto node = tvm::ffi::make_object<GemmNode>(*gemm.get());
    node->annotations_ =
        AnnotateGemmHcuMlsFlags(gemm->annotations_, gemm.get(), collector_);
    node->policy_ = GemmWarpPolicy(gemm->policy_->policy_type);
    const auto query =
        tvm::ffi::Function::GetGlobal("tl.hcu.query_gemm_operand_layout");
    ICHECK(query.has_value())
        << "HCU Python operand-layout query is not registered";
    auto result = Downcast<Array<ObjectRef>>(
        (*query)(Gemm(node), target_, block_threads_, feeds_a));
    ICHECK_EQ(result.size(), 7U);
    consumer_layouts_.emplace(key, std::make_pair(owner, result));
    return result;
  }

  std::optional<PlainReadContract>
  PlainReadLayoutContract(const Buffer &source, const GemmWithInput &consumer,
                          bool explicit_read = false) const {
    ContractKey key{source.get(), consumer.gemm.get(), explicit_read};
    auto cached = contracts_.find(key);
    if (cached != contracts_.end())
      return cached->second.second;
    auto result = InferPlainReadLayoutContract(source, consumer, explicit_read);
    contracts_.emplace(key, std::make_pair(consumer.gemm, result));
    return result;
  }

  std::optional<PlainReadContract>
  InferPlainReadLayoutContract(const Buffer &source,
                               const GemmWithInput &consumer,
                               bool explicit_read) const {
    const auto &gemm = consumer.gemm;
    const bool feeds_a = consumer.input.same_as(gemm->a_);
    if (!explicit_read && gemm->a_.same_as(gemm->b_))
      return std::nullopt;
    if ((!feeds_a && !consumer.input.same_as(gemm->b_)) ||
        GetHcuGemmLdsAccessKind(*gemm.get(), feeds_a) !=
            HcuGemmLdsAccessKind::kAnBt ||
        gemm->IsInstance<GemmBlockScaledNode>() ||
        (source->dtype.bits() != 8 && source->dtype.bits() != 16))
      return std::nullopt;
    // Verify every register-side consumer, stopping at LDS. Layout
    // compatibility cannot make a path through a second shared-memory buffer
    // eligible.
    if (!explicit_read) {
      auto downstream = GetExclusiveGemmConsumersAfterCall(source, nullptr);
      if (!downstream || downstream->empty())
        return std::nullopt;
    }

    // ParseOperator may create several GemmNodes for one call. Use its stable
    // call site so different source candidates share the consumer query.
    const auto consumer_layout = QueryConsumerLayout(consumer);
    int wm = Downcast<Integer>(consumer_layout[0])->value;
    int wn = Downcast<Integer>(consumer_layout[1])->value;
    int wk = Downcast<Integer>(consumer_layout[2])->value;
    int min_n = Downcast<Integer>(consumer_layout[4])->value;
    const int mn = feeds_a ? gemm->m_ : gemm->n_;
    const int warp_mn = feeds_a ? wm : wn;
    const int native_k = 256 / source->dtype.bits();
    // Do not change the consumer partition or introduce extra MN replication
    // to make a matrix read fit. The instruction owns MN32 per wave.
    if (wk != 1 || warp_mn <= 0 || mn % warp_mn || mn / warp_mn < 32 ||
        (mn / warp_mn) % 32 || gemm->k_ % native_k)
      return std::nullopt;
    Fragment producer =
        feeds_a ? makeGemmFragmentAHCU(gemm->m_, gemm->n_, gemm->k_, wm, wn, wk,
                                       source->dtype.bits(), 1, gemm->transA_)
                : makeGemmFragmentBHCU(gemm->m_, gemm->n_, gemm->k_, wm, wn, wk,
                                       source->dtype.bits(), 1, gemm->transB_,
                                       min_n);
    if (explicit_read)
      return PlainReadContract{wm, wn, wk, min_n, producer};
    Fragment required = Downcast<Fragment>(consumer_layout[5]);
    if (!StructuralEqual()(producer->InputShape(), required->InputShape()))
      return std::nullopt;
    Array<PrimExpr> indices;
    arith::Analyzer analyzer;
    for (int i = 0; i < producer->InputDim(); ++i) {
      Var index("plain_read_index");
      analyzer.Bind(index, Range(0, producer->InputShape()[i]));
      indices.push_back(index);
    }
    if (!ProveFragmentContains(required, producer, indices, indices, analyzer))
      return std::nullopt;
    return PlainReadContract{wm, wn, wk, min_n, producer};
  }

  std::optional<PlainReadContract>
  CommonPlainReadLayoutContract(const Buffer &source,
                                const CallNode *after_call) const {
    auto consumers = GetExclusiveGemmConsumersAfterCall(source, after_call);
    if (!consumers || consumers->empty())
      return std::nullopt;
    std::optional<PlainReadContract> common;
    for (const auto &consumer : *consumers) {
      auto next = PlainReadLayoutContract(source, consumer);
      if (!next ||
          (common &&
           (common->warp_m != next->warp_m || common->warp_n != next->warp_n ||
            common->warp_k != next->warp_k || common->min_n != next->min_n ||
            !common->layout->IsEqual(next->layout.get()))))
        return std::nullopt;
      common = next;
    }
    return common;
  }

  bool IsPreferredMatrixConsumer(const Buffer &source,
                                 const GemmWithInput &consumer) const {
    for (const auto &reader : collector_->GetReaderCalls(source)) {
      if (!reader.call || !reader.call->op.same_as(Copy::Get()))
        continue;
      auto copy = Downcast<Copy>(ParseOperator(GetRef<Call>(reader.call)));
      if (!IsVectorReadPreferred(copy.get()))
        continue;
      auto next = PropagateToFindGemmConsumerOpWithInputAfterCall(
          copy->dst, collector_, reader.call);
      if (next && next->site.defined() && consumer.site.defined() &&
          next->site.value().same_as(consumer.site.value()) &&
          next->input.same_as(consumer.input))
        return true;
    }
    return false;
  }

  std::optional<PlainReadContract>
  MatrixReaderContract(const Buffer &source,
                       const GemmWithInput &consumer) const {
    return PlainReadLayoutContract(source, consumer,
                                   IsPreferredMatrixConsumer(source, consumer));
  }

  // A failed AN/BT matrix candidate vetoes this buffer's whole joint plan.
  // Pure AT/BN consumers retain the existing strategy path.
  bool
  MatrixReadersEligible(const Buffer &source,
                        const std::vector<GemmWithInput> &consumers) const {
    for (const auto &consumer : consumers) {
      if (!IsAtBnAccess(consumer) && !MatrixReaderContract(source, consumer))
        return false;
    }
    return true;
  }

  std::optional<HcuGemmLdsReadPattern>
  MatrixReadPattern(const CopyNode &copy, const GemmWithInput &consumer) const {
    if (IsAtBnAccess(consumer) || !MatrixReaderContract(copy.dst, consumer) ||
        copy.dst_range.size() < 2)
      return std::nullopt;
    const auto &gemm = consumer.gemm;
    const bool feeds_a = consumer.input.same_as(gemm->a_);
    const size_t rank = copy.dst_range.size();
    if (!analyzer_->CanProveEqual(copy.dst_range[rank - 2]->extent,
                                  Integer(gemm->k_)) ||
        !analyzer_->CanProveEqual(copy.dst_range[rank - 1]->extent,
                                  Integer(feeds_a ? gemm->m_ : gemm->n_)))
      return std::nullopt;
    const int bits = copy.dst->dtype.bits();
    return HcuGemmLdsReadPattern{bits, 256 / bits, 32, 128 / bits,
                                 HcuGemmLdsAccessKind::kAnBt};
  }

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
    // Gather sources remain scalar expressions in src_range. Keep the actual
    // loop for vector-width proof instead of synthesizing a rectangular copy.
    copy_node->par_op_ = ParallelOp(GetRef<For>(op));
    return ParallelGatherCopy{Copy(copy_node)};
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (mode_ == Mode::kMaterialize) {
      if (auto value = op->annotations.Get(attr::kHcuGemmAnBtLdsStrategy)) {
        HcuGemmAnBtLdsStrategy strategy =
            Downcast<HcuGemmAnBtLdsStrategy>(value.value());
        auto gather = MatchPreferredAsyncParallelGather(op);
        ICHECK(gather.has_value());
        const Var &data = gather->copy->dst->data;
        auto_an_bt_strategies_.insert_or_assign(data, strategy);
        auto_copy_strategies_.insert_or_assign(
            data, Downcast<HcuGemmLdsCopyStrategy>(
                      op->annotations.at(attr::kHcuGemmLdsCopyStrategy)));
      }
    }
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

  // A single stage/batch slice of a compact LDS buffer. Unlike a full-buffer
  // region, its leading coordinates may be dynamic and have unit extents.
  bool IsFullLdsTileRegion(const BufferRegion &region) const {
    const Buffer &buffer = region->buffer;
    const size_t rank = buffer->shape.size();
    if (rank < 2 || region->region.size() != rank || !buffer->strides.empty())
      return false;
    for (size_t i = 0; i < rank; ++i) {
      if (i + 2 < rank) {
        if (!analyzer_->CanProveEqual(region->region[i]->extent, 1))
          return false;
      } else if (!is_zero(region->region[i]->min) ||
                 !analyzer_->CanProveEqual(region->region[i]->extent,
                                           buffer->shape[i])) {
        return false;
      }
    }
    return true;
  }

  Array<PrimExpr> LdsTileShape(const Buffer &buffer) const {
    const size_t rank = buffer->shape.size();
    ICHECK_GE(rank, 2U);
    return {buffer->shape[rank - 2], buffer->shape[rank - 1]};
  }

  Optional<Layout> TryMakePlainLayout(const Buffer &buffer) const {
    // Explicit reader preference does not authorize replacing an async
    // producer's pending/active storage and ownership plan with a plain one.
    // Reader preference is consumed by MatrixReaderContract; it never owns a
    // separate storage-layout selection path.
    for (const auto &writer : collector_->GetProducerRecords(buffer)) {
      if (writer.call) {
        Call call = GetRef<Call>(writer.call);
        if (HasAsyncCopySemantics(call) || selected_calls_.count(call))
          return std::nullopt;
      }
    }
    if (block_threads_ < TargetHcuGetWarpSize(target_) ||
        block_threads_ % TargetHcuGetWarpSize(target_) != 0 ||
        !IsSharedBuffer(buffer) || HasAnnotatedLayout(buffer) ||
        blocked_plain_buffers_.count(buffer) ||
        blocked_plain_data_.count(buffer->data) ||
        !(buffer->dtype.is_float16() || buffer->dtype.is_bfloat16() ||
          buffer->dtype.is_float8_e4m3fn() || buffer->dtype.is_float8_e5m2()) ||
        buffer->dtype.lanes() != 1 || !buffer->strides.empty() ||
        buffer->shape.size() < 2 || !is_zero(buffer->elem_offset))
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
          !IsFullLdsTileRegion(BufferRegion(buffer, copy->dst_range)))
        return std::nullopt;
      if (IsGlobalLikeBuffer(copy->src)) {
        // Ordinary global copies own their vectorization and padding. A
        // global slice's alignment is not a constraint on the LDS read atom.
      } else if (copy->src.scope() == "local.fragment") {
        if (!copy->dst.same_as(buffer) || !copy->src->strides.empty() ||
            !is_zero(copy->src->elem_offset) ||
            !StructuralEqual()(copy->src->shape, LdsTileShape(buffer)) ||
            !IsFullPlainRegion(BufferRegion(copy->src, copy->src_range)))
          return std::nullopt;
        has_fragment_writer = true;
      } else {
        return std::nullopt;
      }
    }
    auto consumers = GetStrategyGemmConsumers(buffer, nullptr);
    if (!consumers || consumers->empty())
      return std::nullopt;
    return TryMakeCommonGemmLayout(buffer, *consumers, has_fragment_writer);
  }

  using OrdinaryReadKey = std::tuple<const BufferNode *, const Object *, bool>;
  mutable std::map<
      OrdinaryReadKey,
      std::tuple<Buffer, ObjectRef, std::optional<HcuOrdinaryLdsReadInfo>>>
      ordinary_readers_;

  std::optional<HcuOrdinaryLdsReadInfo>
  PrepareOrdinaryLdsRead(const Buffer &buffer,
                         const GemmWithInput &consumer) const {
    const auto &gemm = consumer.gemm;
    const bool feeds_a = consumer.input.same_as(gemm->a_);
    ObjectRef owner = consumer.site.defined() ? ObjectRef(consumer.site.value())
                                              : ObjectRef(gemm);
    const OrdinaryReadKey key{buffer.get(), owner.get(), feeds_a};
    auto cached = ordinary_readers_.find(key);
    if (cached != ordinary_readers_.end())
      return std::get<2>(cached->second);
    auto result = InferOrdinaryLdsRead(buffer, consumer);
    ordinary_readers_.emplace(key, std::make_tuple(buffer, owner, result));
    return result;
  }

  std::optional<HcuOrdinaryLdsReadInfo>
  InferOrdinaryLdsRead(const Buffer &buffer,
                       const GemmWithInput &consumer) const {
    const auto &gemm = consumer.gemm;
    if (!IsAtBnAccess(consumer) || buffer->shape.size() < 2 ||
        (buffer->dtype.bits() != 8 && buffer->dtype.bits() != 16 &&
         buffer->dtype.bits() != 32) ||
        gemm->a_.same_as(gemm->b_) || gemm->IsInstance<GemmBlockScaledNode>())
      return std::nullopt;
    // Use the existing same-coordinate propagation, including its LDS stop.
    // Unknown indexing/reductions cannot supply a native-wrap read contract.
    auto consumers = GetExclusiveGemmConsumersAfterCall(buffer, nullptr);
    if (!consumers || consumers->empty())
      return std::nullopt;
    const bool feeds_a = consumer.input.same_as(gemm->a_);
    const auto facts = QueryConsumerLayout(consumer);
    const int wm = Downcast<Integer>(facts[0])->value;
    const int wn = Downcast<Integer>(facts[1])->value;
    const int wk = Downcast<Integer>(facts[2])->value;
    const int warp_size = TargetHcuGetWarpSize(target_);
    const int mn_atom = Downcast<Integer>(facts[feeds_a ? 3 : 4])->value;
    const int mmac_k = Downcast<Integer>(facts[6])->value;
    const int mn = feeds_a ? gemm->m_ : gemm->n_;
    const auto shape = LdsTileShape(buffer);
    const auto *block_mn = as_const_int(shape[0]);
    const auto *block_k = as_const_int(shape[1]);
    if (wk != 1 || wm <= 0 || wn <= 0 || warp_size <= 0 ||
        block_threads_ % warp_size || wm * wn != block_threads_ / warp_size ||
        !block_mn || !block_k || mn <= 0 || *block_mn < mn || *block_mn % mn ||
        *block_k != gemm->k_ || mn_atom <= 0 || mmac_k <= 0 ||
        gemm->kPack_ <= 0)
      return std::nullopt;
    // Preserve the original MMAC element grouping. Only its storage bytes
    // change across a same-coordinate, one-to-one conversion chain.
    const int64_t packed_elements = int64_t(mn_atom) * mmac_k * gemm->kPack_;
    if (packed_elements % warp_size || packed_elements / warp_size > INT_MAX)
      return std::nullopt;
    const int elements = packed_elements / warp_size;
    const int bits = buffer->dtype.bits();
    return HcuOrdinaryLdsReadInfo{
        {bits, 256 / bits, 32, elements, HcuGemmLdsAccessKind::kAtBn},
        mn,
        feeds_a ? wm : wn,
        mn_atom};
  }

  Optional<Layout>
  TryMakeCommonGemmLayout(const Buffer &buffer,
                          const std::vector<GemmWithInput> &consumers,
                          bool has_fragment_writer = false) const {
    if (!MatrixReadersEligible(buffer, consumers))
      return std::nullopt;
    // Pure ordinary readers retain the existing AT/BN strategy. Do not query
    // Python when there is no matrix reader
    // with which to share the storage layout.
    if (std::none_of(
            consumers.begin(), consumers.end(),
            [](const auto &consumer) { return !IsAtBnAccess(consumer); }))
      return std::nullopt;
    const int element_bits = buffer->dtype.bits();
    const int read_tile_k = element_bits == 8 ? 32 : 16;
    std::vector<HcuGemmLdsReadPattern> reads;
    std::vector<HcuOrdinaryLdsReadInfo> ordinary_readers;
    for (const GemmWithInput &consumer : consumers) {
      const Gemm &gemm = consumer.gemm;
      const bool feeds_a = consumer.input.same_as(gemm->a_);
      const auto kind = GetHcuGemmLdsAccessKind(*gemm.get(), feeds_a);
      const bool matrix_read = kind == HcuGemmLdsAccessKind::kAnBt;
      const auto contract =
          matrix_read ? MatrixReaderContract(buffer, consumer) : std::nullopt;
      if (gemm->a_.same_as(gemm->b_) ||
          gemm->IsInstance<GemmBlockScaledNode>() ||
          (matrix_read && (gemm->k_ % read_tile_k || !contract)))
        return std::nullopt;
      if (!matrix_read) {
        auto reader = PrepareOrdinaryLdsRead(buffer, consumer);
        if (!reader)
          return std::nullopt;
        ordinary_readers.push_back(*reader);
      }
      const BufferRegion &input = feeds_a ? gemm->aRegion_ : gemm->bRegion_;
      // First-stage scope is full compact tiles, including identity relays.
      if (!(IsSharedBuffer(input->buffer) ? IsFullLdsTileRegion(input)
                                          : IsFullPlainRegion(input)))
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
          if (!IsFullLdsTileRegion(BufferRegion(copy->src, copy->src_range)) ||
              !IsFullPlainRegion(BufferRegion(copy->dst, copy->dst_range)) ||
              !StructuralEqual()(LdsTileShape(copy->src), copy->dst->shape) ||
              copy->dst.scope() != "local.fragment")
            return std::nullopt;
        }
      }
      reads.push_back(matrix_read
                          ? HcuGemmLdsReadPattern{element_bits, read_tile_k, 32,
                                                  128 / element_bits, kind}
                          : ordinary_readers.back().read);
    }
    // Supply actual ordinary-read widths to the existing strategy. No new
    // candidate generation, bank scoring, or layout ranking takes place here.
    auto layout = MakeHcuPlainGemmLdsLayout(
        buffer->shape, TargetHcuGetLdsBankCount(target_),
        TargetHcuGetLdsBankWidthBytes(target_), reads);
    if (!layout)
      return std::nullopt;
    return layout;
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
    auto reader = PrepareOrdinaryLdsRead(copy.dst, consumer);
    if (!reader)
      return std::nullopt;
    return DeriveHcuGemmAtBnLdsStrategy(copy, *reader,
                                        CurrentStrategyThreadBounds(), target_);
  }

  Optional<HcuGemmAnBtLdsStrategy>
  TryDeriveAnBtStrategy(const CopyNode &copy, const GemmWithInput &consumer,
                        Optional<Layout> storage_layout = std::nullopt) const {
    auto read = MatrixReadPattern(copy, consumer);
    if (!read) {
      return std::nullopt;
    }
    return DeriveHcuGemmAnBtLdsStrategy(copy, *read,
                                        CurrentStrategyThreadBounds(), target_,
                                        std::move(storage_layout));
  }

  Optional<HcuGemmAnBtLdsStrategy>
  TryDeriveAnBtCommonWrapStrategy(const CopyNode &copy,
                                  const GemmWithInput &consumer,
                                  int wrap_count) const {
    auto read = MatrixReadPattern(copy, consumer);
    if (!read) {
      return std::nullopt;
    }
    return DeriveHcuGemmAnBtLdsStrategyWith64ByteWrap(
        copy, *read, CurrentStrategyThreadBounds(), target_, wrap_count);
  }

  Optional<HcuGemmAnBtLdsStrategy> TrySelectCommonAnBtStrategy(
      const CopyNode &copy, const std::vector<GemmWithInput> &consumers) const {
    if (IsFullLdsTileRegion(BufferRegion(copy.dst, copy.dst_range))) {
      if (Optional<Layout> layout =
              TryMakeCommonGemmLayout(copy.dst, consumers)) {
        for (const GemmWithInput &consumer : consumers) {
          if (!IsAtBnAccess(consumer)) {
            if (auto strategy =
                    TryDeriveAnBtStrategy(copy, consumer, layout.value())) {
              return strategy;
            }
          }
        }
      }
    }
    return std::nullopt;
  }

  std::pair<Optional<HcuGemmAtBnLdsStrategy>, Optional<HcuGemmAnBtLdsStrategy>>
  SelectWrapStrategy(const CopyNode &copy,
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

  std::optional<std::vector<GemmWithInput>>
  GetExclusiveGemmConsumersAfterCall(const Buffer &buffer,
                                     const CallNode *after_site_call) const {
    const int order =
        after_site_call ? collector_->GetCallStmtOrder(after_site_call) : -1;
    const auto key = std::make_pair(buffer.get(), after_site_call);
    auto cached = consumer_queries_.find(key);
    if (cached != consumer_queries_.end())
      return cached->second;
    auto result = QueryGemmConsumers(
        buffer, collector_, order, true,
        [&](const Buffer &input, const Buffer &output, const CallNode *call) {
          if (!call)
            return register_access_proof_.Preserves(input, output);
          auto parsed = ParseOperator(GetRef<Call>(call));
          const auto *copy = parsed.as<CopyNode>();
          if (copy && IsSharedBuffer(input) && input->shape.size() > 2 &&
              output->shape.size() == 2 && IsFragmentBuffer(output)) {
            return StructuralEqual()(LdsTileShape(input), output->shape) &&
                   IsFullLdsTileRegion(BufferRegion(input, copy->src_range)) &&
                   IsFullPlainRegion(BufferRegion(output, copy->dst_range));
          }
          return copy && IsFragmentBuffer(output) &&
                 StructuralEqual()(input->shape, output->shape) &&
                 IsFullPlainRegion(BufferRegion(input, copy->src_range)) &&
                 IsFullPlainRegion(BufferRegion(output, copy->dst_range));
        });
    consumer_queries_.emplace(key, result);
    return result;
  }

  // Explicit matrix readers contribute their fixed instruction geometry even
  // when their later register chain is not an automatic-selection candidate.
  // Unknown ordinary readers are never silently omitted from a joint plan.
  std::optional<std::vector<GemmWithInput>>
  GetStrategyGemmConsumers(const Buffer &buffer,
                           const CallNode *after_call) const {
    if (auto consumers = GetExclusiveGemmConsumersAfterCall(buffer, after_call))
      return consumers;
    auto readers = collector_->GetReaderCalls(buffer);
    if (readers.empty())
      return std::nullopt;
    for (const auto &reader : readers) {
      if (!reader.call || !reader.call->op.same_as(Copy::Get()))
        return std::nullopt;
      auto copy = Downcast<Copy>(ParseOperator(GetRef<Call>(reader.call)));
      if (!IsVectorReadPreferred(copy.get()) || !IsFragmentBuffer(copy->dst))
        return std::nullopt;
    }
    auto consumers = QueryGemmConsumers(
        buffer, collector_,
        after_call ? collector_->GetCallStmtOrder(after_call) : -1, true);
    if (!consumers || consumers->empty())
      return std::nullopt;
    for (const auto &consumer : *consumers) {
      if (IsAtBnAccess(consumer) ||
          !IsPreferredMatrixConsumer(buffer, consumer))
        return std::nullopt;
    }
    return consumers;
  }

  Optional<Map<String, ObjectRef>> TryBuildPendingGemmCopyAnnotations(
      const CopyNode &copy, const Map<String, ObjectRef> &base_annotations,
      const std::vector<GemmWithInput> &consumers) {
    if (!MatrixReadersEligible(copy.dst, consumers))
      return std::nullopt;
    // Prefer a legal mixed wrap when both read kinds are proven <=2-way.
    // Otherwise try common; an unproven bound is not a measured conflict.
    // Pure AN/BT keeps its existing policy for insufficient wrap phases.
    // Retain the legal native plan if common cannot be built or would
    // contradict an explicitly annotated storage layout.
    auto [at_bn_strategy, an_bt_strategy] = SelectWrapStrategy(copy, consumers);
    Optional<HcuGemmLdsCopyStrategy> copy_strategy;
    if (at_bn_strategy.defined() || an_bt_strategy.defined()) {
      copy_strategy =
          TrySelectCopyStrategy(copy, at_bn_strategy, an_bt_strategy);
    }
    if (copy_strategy.defined() && an_bt_strategy.defined() &&
        (at_bn_strategy.defined()
             ? !CanProveMixedWrapAtMostTwoWay(copy, consumers,
                                              at_bn_strategy.value(),
                                              an_bt_strategy.value())
             : HasPartialHcuGemmAnBtWrap(an_bt_strategy.value()))) {
      Optional<HcuGemmAnBtLdsStrategy> common_an_bt =
          TrySelectCommonAnBtStrategy(copy, consumers);
      if (common_an_bt.defined()) {
        Optional<HcuGemmLdsCopyStrategy> common_copy =
            TrySelectCopyStrategy(copy, std::nullopt, common_an_bt);
        Optional<Layout> existing = FindAnnotatedLayout(copy.dst);
        if (common_copy.defined() &&
            (!existing.defined() ||
             existing.value()->IsEqual(
                 common_an_bt.value()->storage_layout.get()))) {
          at_bn_strategy = std::nullopt;
          an_bt_strategy = common_an_bt;
          copy_strategy = common_copy;
        }
      }
    }
    if (!copy_strategy.defined()) {
      // Also fall back when the consumers cannot share one wrap copy plan.
      at_bn_strategy = std::nullopt;
      an_bt_strategy = TrySelectCommonAnBtStrategy(copy, consumers);
      if (!an_bt_strategy.defined()) {
        return std::nullopt;
      }
      copy_strategy =
          TrySelectCopyStrategy(copy, at_bn_strategy, an_bt_strategy);
      if (!copy_strategy.defined()) {
        return std::nullopt;
      }
    }
    auto annotations = base_annotations;
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

  bool HasLayoutAwareGemmInput(const Buffer &buffer) const {
    return plain_layouts_.count(buffer->data) ||
           auto_an_bt_strategies_.count(buffer->data);
  }

  Stmt MaterializeLayoutAwareGemm(const Call &call, const Gemm &gemm,
                                  Map<String, ObjectRef> annotations) {
    Array<PrimExpr> args = call->args;
    Array<Buffer> fragments;
    Array<Stmt> statements;
    bool convert_a =
        HasLayoutAwareGemmInput(gemm->a_) && !IsAtBnAccess({gemm, gemm->a_});
    bool convert_b =
        HasLayoutAwareGemmInput(gemm->b_) && !IsAtBnAccess({gemm, gemm->b_});
    for (int slot = 0; slot < 2; ++slot) {
      const Buffer &input = slot == 0 ? gemm->a_ : gemm->b_;
      if (!HasLayoutAwareGemmInput(input))
        continue;
      annotations.Set(slot == 0 ? attr::kHcuARespectLayoutMap
                                : attr::kHcuBRespectLayoutMap,
                      Integer(1));
      if (!(slot == 0 ? convert_a : convert_b))
        continue;
      const Array<Range> &read_region =
          slot == 0 ? gemm->aRegion_->region : gemm->bRegion_->region;
      Array<PrimExpr> fragment_shape = {
          read_region[read_region.size() - 2]->extent,
          read_region.back()->extent};
      Buffer fragment =
          decl_buffer(fragment_shape, input->dtype,
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
                 "layout_aware_gemm_read", SeqStmt(statements), std::nullopt,
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
      // Explicit async strategies can already be materialized by collection.
      // Restore their per-buffer state before dispatching GEMM reads.
      if (mode_ == Mode::kMaterialize) {
        if (auto strategy =
                call->annotations.Get(attr::kHcuGemmAnBtLdsStrategy)) {
          auto_an_bt_strategies_.insert_or_assign(
              copy->dst->data,
              Downcast<HcuGemmAnBtLdsStrategy>(strategy.value()));
          auto copy_strategy =
              call->annotations.Get(attr::kHcuGemmLdsCopyStrategy);
          ICHECK(copy_strategy);
          auto_copy_strategies_.insert_or_assign(
              copy->dst->data,
              Downcast<HcuGemmLdsCopyStrategy>(copy_strategy.value()));
        }
      }
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
      const bool explicit_read = IsVectorReadPreferred(copy.get());
      if (explicit_read) {
        ICHECK(consumer.has_value())
            << "ds_read_vector requires a register-side GEMM consumer";
        ICHECK(IsAnBtLinearDsReadCopyCandidate(copy.get(), *consumer))
            << "ds_read_vector requires a supported MN-major LDS-to-fragment "
               "tile";
        if (mode_ == Mode::kCollectCandidates)
          return IRMutatorWithAnalyzer::VisitStmt_(op);
      }
      const bool can_materialize_ds_read =
          explicit_read || mode_ == Mode::kMaterialize ||
          HasAnnotatedLayout(copy->src) ||
          auto_copy_strategies_.count(copy->src->data) ||
          async_copy_linear_outputs_.count(copy->src->data);
      auto contract =
          can_materialize_ds_read && consumer &&
                  IsAnBtLinearDsReadCopyCandidate(copy.get(), consumer.value())
              ? (explicit_read
                     ? PlainReadLayoutContract(copy->dst, *consumer, true)
                     : CommonPlainReadLayoutContract(copy->dst, call))
              : std::nullopt;
      ICHECK(!explicit_read || contract.has_value())
          << "ds_read_vector tile is incompatible with the GEMM warp partition "
             "(MN32 required)";
      if (contract) {
        const GemmNode *consumer_gemm = consumer->gemm.get();
        const bool feeds_a = consumer->input.same_as(consumer_gemm->a_);
        const bool has_layout = HasAnnotatedLayout(copy->src) ||
                                auto_an_bt_strategies_.count(copy->src->data);
        // The fallback template is B-specific. AN must have an explicit
        // logical-to-physical map before it can use the generic vector read.
        if ((feeds_a || copy->src->dtype.bits() == 8) && !has_layout) {
          ICHECK(!explicit_read)
              << "ds_read_vector requires a legal MN-major LDS storage layout";
          return IRMutatorWithAnalyzer::VisitStmt_(op);
        }
        auto annotations = call->annotations;
        const Gemm &gemm = consumer->gemm;
        const bool a_from_mls = IsFromMls(gemm->a_, collector_);
        const bool b_from_mls = IsFromMls(gemm->b_, collector_);
        MlsGemmDepMeta meta = BuildCopyToGemmLinearMeta(
            consumer.value(), a_from_mls, b_from_mls, target_, contract->warp_m,
            contract->warp_n, contract->warp_k, contract->min_n);
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
        if (auto consumers = GetStrategyGemmConsumers(copy->dst, call)) {
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
      }
      if (mode_ == Mode::kMaterialize && (HasLayoutAwareGemmInput(gemm->a_) ||
                                          HasLayoutAwareGemmInput(gemm->b_))) {
        return MaterializeLayoutAwareGemm(GetRef<Call>(call), gemm,
                                          annotations);
      }
      Call new_call(call->dtype, call->op, call->args, annotations, call->span);
      return Evaluate(new_call);
    }

    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>
      blocked_plain_buffers_;
  AutomaticRegisterAccessProof register_access_proof_;
  mutable std::map<std::pair<const BufferNode *, const CallNode *>,
                   std::optional<std::vector<GemmWithInput>>>
      consumer_queries_;
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
