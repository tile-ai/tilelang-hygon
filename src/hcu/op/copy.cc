/*!
 * \file hcu/op/copy.cc
 * \brief HCU implementation for tl.copy lowering.
 */

#include "op/copy.h"
#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt_functor.h>

#include "hcu/op/mls.h"
#include "hcu/target_utils.h"
#include "hcu/transform/async_copy_injector.h"
#include "hcu/utils/gemm_lds_strategy_utils.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/transform.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace hcu {

namespace {

constexpr int64_t kWaveSize = 64;
constexpr int64_t kNaturallyDistributedBytes = 256;
constexpr int64_t kMinCacheSwizzleStrideBytes = 64;
constexpr int64_t kMaxCacheSwizzleStrideBytes = 8192;

PrimExpr LinearizeAccess(const Buffer &buffer, Array<PrimExpr> indices) {
  ICHECK_EQ(indices.size(), buffer->shape.size());
  PrimExpr offset = make_zero(indices[0].dtype());
  if (!buffer->strides.empty()) {
    ICHECK_EQ(buffer->strides.size(), indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      PrimExpr index = indices[i];
      if (const auto *ramp = index.as<RampNode>())
        index = ramp->base;
      offset += index * buffer->strides[i];
    }
    return offset;
  }
  PrimExpr stride = make_const(indices[0].dtype(), 1);
  for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
    PrimExpr index = indices[i];
    if (const auto *ramp = index.as<RampNode>())
      index = ramp->base;
    offset += index * stride;
    stride *= buffer->shape[i];
  }
  return offset;
}

// Prove that every group of lanes needed to cover 256 bytes accesses one
// contiguous 256-byte segment.  This uses one symbolic equality check instead
// of enumerating lanes or normalizing the complete (potentially padded)
// address mapping.
bool IsNaturallyDistributed(
    const Buffer &buffer, Array<PrimExpr> indices, DataType access_dtype,
    const Var &thread_var, const Range &thread_bounds,
    const std::vector<std::pair<Var, int64_t>> &vector_loops,
    arith::Analyzer *analyzer) {
  const int64_t element_bits = buffer->dtype.bits() * buffer->dtype.lanes();
  const int64_t scalar_access_bits = access_dtype.bits() * access_dtype.lanes();
  if (element_bits <= 0 || scalar_access_bits <= 0 || element_bits % 8 != 0 ||
      scalar_access_bits % 8 != 0 || scalar_access_bits > 128 ||
      scalar_access_bits % element_bits != 0)
    return false;

  int64_t payload_elements = scalar_access_bits / element_bits;
  PrimExpr base = analyzer->Simplify(LinearizeAccess(buffer, indices));

  // At this lowering point vectorization is still represented by
  // ForKind::kVectorized loops, not necessarily by Ramp/vector dtypes.  First
  // prove that their flattened iteration order is contiguous, then fold their
  // extents into the actual per-thread payload.
  if (!vector_loops.empty()) {
    arith::Analyzer vector_analyzer;
    Map<Var, PrimExpr> zero_subst;
    for (const auto &[var, extent] : vector_loops) {
      if (extent <= 0)
        return false;
      PrimExpr extent_expr = make_const(var.dtype(), extent);
      vector_analyzer.Bind(
          var, Range::FromMinExtent(make_zero(var.dtype()), extent_expr));
      zero_subst.Set(var, make_zero(var.dtype()));
    }
    PrimExpr vector_base =
        vector_analyzer.Simplify(Substitute(base, zero_subst));
    PrimExpr expected = vector_base;
    int64_t stride = payload_elements;
    for (auto it = vector_loops.rbegin(); it != vector_loops.rend(); ++it) {
      expected += it->first * make_const(it->first.dtype(), stride);
      stride *= it->second;
    }
    if (!vector_analyzer.CanProveEqual(base, expected))
      return false;
    base = std::move(vector_base);
    payload_elements = stride;
  }

  const int64_t access_bits = payload_elements * element_bits;
  if (access_bits > 128)
    return false;
  const int64_t access_bytes = access_bits / 8;
  if (kNaturallyDistributedBytes % access_bytes != 0)
    return false;
  const int64_t lanes_per_segment = kNaturallyDistributedBytes / access_bytes;
  if (lanes_per_segment <= 0 || lanes_per_segment > kWaveSize)
    return false;

  const int64_t *thread_extent =
      as_const_int(analyzer->Simplify(thread_bounds->extent));
  if (!thread_extent || *thread_extent < lanes_per_segment ||
      *thread_extent % lanes_per_segment != 0)
    return false;

  DataType index_dtype = thread_var.dtype();
  Var segment("cache_swizzle_segment", index_dtype);
  Var lane("cache_swizzle_lane", index_dtype);
  PrimExpr lanes_per_segment_expr = make_const(index_dtype, lanes_per_segment);
  PrimExpr segment_count_expr =
      make_const(index_dtype, *thread_extent / lanes_per_segment);
  PrimExpr expanded_thread =
      thread_bounds->min + segment * lanes_per_segment_expr + lane;

  arith::Analyzer proof_analyzer;
  proof_analyzer.Bind(lane, Range::FromMinExtent(make_zero(index_dtype),
                                                 lanes_per_segment_expr));
  proof_analyzer.Bind(segment, Range::FromMinExtent(make_zero(index_dtype),
                                                    segment_count_expr));
  PrimExpr expanded = proof_analyzer.Simplify(
      Substitute(base, {{thread_var, expanded_thread}}));
  PrimExpr segment_base = proof_analyzer.Simplify(
      Substitute(expanded, {{lane, make_zero(index_dtype)}}));
  PrimExpr expected =
      segment_base + lane * make_const(index_dtype, payload_elements);
  return proof_analyzer.CanProveEqual(expanded, expected);
}

class GlobalCopyAccessAnalyzer : public StmtExprVisitor {
public:
  GlobalCopyAccessAnalyzer(Var thread_var, Range thread_bounds,
                           arith::Analyzer *analyzer)
      : thread_var_(std::move(thread_var)),
        thread_bounds_(std::move(thread_bounds)), analyzer_(analyzer) {}

  bool NeedsCacheSwizzle(const Stmt &stmt) {
    operator()(stmt);
    return saw_global_access_ && needs_cache_swizzle_;
  }

private:
  void Analyze(const Buffer &buffer, const Array<PrimExpr> &indices,
               DataType access_dtype) {
    if (!IsGlobalBuffer(buffer))
      return;
    if (saw_global_access_)
      return;
    saw_global_access_ = true;
    needs_cache_swizzle_ =
        !IsNaturallyDistributed(buffer, indices, access_dtype, thread_var_,
                                thread_bounds_, vector_loops_, analyzer_);
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind != ForKind::kVectorized) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    const int64_t *extent = as_const_int(analyzer_->Simplify(op->extent));
    vector_loops_.emplace_back(op->loop_var, extent ? *extent : 0);
    StmtExprVisitor::VisitStmt_(op);
    vector_loops_.pop_back();
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    Analyze(op->buffer, op->indices, op->value.dtype());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Analyze(op->buffer, op->indices, op->dtype);
    StmtExprVisitor::VisitExpr_(op);
  }

  Var thread_var_;
  Range thread_bounds_;
  arith::Analyzer *analyzer_;
  std::vector<std::pair<Var, int64_t>> vector_loops_;
  bool saw_global_access_{false};
  bool needs_cache_swizzle_{false};
};

bool NeedsCacheSwizzle(const Stmt &stmt, const LowerArgs &lower_args,
                       arith::Analyzer *analyzer) {
  return GlobalCopyAccessAnalyzer(lower_args.thread_var,
                                  lower_args.thread_bounds, analyzer)
      .NeedsCacheSwizzle(stmt);
}

std::optional<int64_t> GetPhysicalRowStrideBytes(const CopyNode &op,
                                                 arith::Analyzer *analyzer) {
  Buffer buffer;
  Array<Range> ranges;
  if (IsGlobalBuffer(op.dst)) {
    buffer = op.dst;
    ranges = op.dst_range;
  } else if (IsGlobalBuffer(op.src)) {
    buffer = op.src;
    ranges = op.src_range;
  } else {
    return std::nullopt;
  }

  // Conservatively support a two-dimensional copy region only. The inner
  // varying dimension is the column, and the outer one is the physical row.
  int row_dim = -1;
  int varying_dims = 0;
  for (int i = static_cast<int>(ranges.size()) - 1; i >= 0; --i) {
    const int64_t *extent = as_const_int(analyzer->Simplify(ranges[i]->extent));
    if (!extent)
      return std::nullopt;
    if (*extent > 1 && ++varying_dims == 2)
      row_dim = i;
  }
  if (varying_dims != 2 || row_dim < 0)
    return std::nullopt;

  Array<PrimExpr> base_indices;
  Array<PrimExpr> next_row_indices;
  for (int i = 0; i < static_cast<int>(ranges.size()); ++i) {
    base_indices.push_back(ranges[i]->min);
    next_row_indices.push_back(
        ranges[i]->min + make_const(ranges[i]->min.dtype(), i == row_dim));
  }
  Array<PrimExpr> base_offset = buffer.OffsetOf(base_indices);
  Array<PrimExpr> next_row_offset = buffer.OffsetOf(next_row_indices);
  if (base_offset.size() != 1 || next_row_offset.size() != 1)
    return std::nullopt;

  const int64_t *stride_elements =
      as_const_int(analyzer->Simplify(next_row_offset[0] - base_offset[0]));
  const int64_t element_bits = buffer->dtype.bits() * buffer->dtype.lanes();
  if (!stride_elements || *stride_elements <= 0 || element_bits <= 0 ||
      element_bits % 8 != 0)
    return std::nullopt;
  const int64_t stride_bytes = *stride_elements * (element_bits / 8);
  if (stride_bytes < kMinCacheSwizzleStrideBytes ||
      stride_bytes > kMaxCacheSwizzleStrideBytes ||
      (stride_bytes & (stride_bytes - 1)) != 0)
    return std::nullopt;
  return stride_bytes;
}

std::optional<int64_t> GetCacheSwizzleStrideBytes(const CopyNode &op,
                                                  const Stmt &stmt,
                                                  const LowerArgs &lower_args,
                                                  arith::Analyzer *analyzer) {
  if (!NeedsCacheSwizzle(stmt, lower_args, analyzer))
    return std::nullopt;
  return GetPhysicalRowStrideBytes(op, analyzer);
}

Stmt AnnotateCacheSwizzle(const CopyNode &op, Stmt stmt,
                          const LowerArgs &lower_args,
                          arith::Analyzer *analyzer) {
  auto stride = GetCacheSwizzleStrideBytes(op, stmt, lower_args, analyzer);
  if (!stride.has_value())
    return stmt;
  return AttrStmt(lower_args.thread_var, attr::kHcuBufferCacheSwizzleStride,
                  Integer(stride.value()), std::move(stmt));
}

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto val = op.annotations.Get(key)) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value != 0;
    }
  }
  return false;
}

bool GetIsAsyncCopy(const CopyNode &op) {
  if (GetBoolAnnotation(op, "is_async_copy")) {
    return true;
  }
  return GetBoolAnnotation(op, "force_cp_async");
}

bool GetNoImplicitAsyncCommitWait(const CopyNode &op) {
  return GetBoolAnnotation(op, attr::kAsyncCopyNoImplicitCommitWait);
}

Map<String, ObjectRef> GetHCUAsyncCopyAnnotations(const CopyNode &op) {
  Map<String, ObjectRef> result;
  if (auto value = op.annotations.Get(attr::kHcuGemmLdsCopyStrategy)) {
    result.Set(attr::kHcuGemmLdsCopyStrategy, value.value());
  }
  return result;
}

} // namespace

enum class CopyInst : uint8_t {
  kNormal = 0,
  kCPAsync = 1,
  kMatrixLoad = 2,
};

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    if (SelectInst(op, layout_args.target, layout_args.layout_map,
                   layout_args.analyzer) == CopyInst::kMatrixLoad) {
      return {};
    }
    return op.InferSIMTLayout(layout_args, level);
  }

  static CopyInst SelectInst(const CopyNode &op, Target target,
                             const LayoutMap &layout_map,
                             arith::Analyzer *analyzer) {
    if (IsMatrixLoadPreferredCopy(op)) {
      return CopyInst::kMatrixLoad;
    }
    if (GetIsAsyncCopy(op) || GetNoImplicitAsyncCommitWait(op)) {
      bool cp_async_supported =
          CheckCPAsyncCopy(op, target, layout_map, analyzer);
      ICHECK(cp_async_supported)
          << "Explicit async copy semantics require HCU async copy lowering, "
             "but constraints were not satisfied. Got src="
          << op.src->name << " (scope=" << op.src.scope()
          << ", dtype=" << op.src->dtype << "), dst=" << op.dst->name
          << " (scope=" << op.dst.scope() << ", dtype=" << op.dst->dtype
          << ").";
      return CopyInst::kCPAsync;
    }
    return CopyInst::kNormal;
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    auto copy_inst =
        SelectInst(op, lower_args.target, lower_args.layout_map, analyzer);
    if (copy_inst == CopyInst::kCPAsync) {
      return LowerCPAsync(op, lower_args, analyzer);
    }
    if (copy_inst == CopyInst::kMatrixLoad) {
      return MakeMatrixLoadFromCopy(op)->Lower(lower_args, analyzer);
    }
    if (copy_inst == CopyInst::kNormal) {
      return AnnotateCacheSwizzle(op, LowerNormalCopy(op, lower_args, analyzer),
                                  lower_args, analyzer);
    }
    LOG(FATAL) << "Unsupported HCU copy inst " << static_cast<int>(copy_inst);
  }

private:
  static Stmt LowerCPAsync(const CopyNode &op, const LowerArgs &lower_args,
                           arith::Analyzer *analyzer) {
    using namespace tvm::transform;

    PassContext pass_ctx = PassContext::Current();
    bool enable_async_copy =
        pass_ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(true)).value();
    bool no_implicit_commit_wait = GetNoImplicitAsyncCommitWait(op);
    bool explicit_async_semantics =
        no_implicit_commit_wait || GetIsAsyncCopy(op);
    if (!enable_async_copy && !explicit_async_semantics) {
      return AnnotateCacheSwizzle(op, LowerNormalCopy(op, lower_args, analyzer),
                                  lower_args, analyzer);
    }

    auto simt_loop = op.MakeSIMTLoop(analyzer);
    auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
    auto par_op = ParallelOp(fused_loop);

    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout({lower_args.target,
                           lower_args.thread_bounds,
                           lower_args.layout_map,
                           analyzer,
                           false,
                           lower_args.buffer_remap,
                           {}},
                          level);
    }
    auto loop_layout = par_op->GetLoopLayout();
    Stmt lowered_loop = LowerParallelLoop(
        par_op->GetRoot(), loop_layout, lower_args.thread_var, analyzer,
        lower_args.layout_map, par_op->GetPredicate(lower_args.thread_var),
        /*parallel_loop=*/true,
        /*should_vectorize=*/true, par_op->LoopLayoutRequiresPaddingGuard());
    auto annotations = GetHCUAsyncCopyAnnotations(op);
    if (auto stride = GetCacheSwizzleStrideBytes(op, lowered_loop, lower_args,
                                                 analyzer)) {
      annotations.Set(attr::kHcuBufferCacheSwizzleStride,
                      Integer(stride.value()));
    }
    auto inject_result = InjectHCUAsyncCopy(
        lowered_loop, /*async_without_async_commit_wait=*/
        no_implicit_commit_wait || GetIsAsyncCopy(op), annotations,
        lower_args.thread_var, lower_args.buffer_remap);
    Stmt async_copy_loop = inject_result.stmt;
    if (!inject_result.injected_hcu_async_copy) {
      DLOG(WARNING) << "HCU async-copy rewrite miss for copy src="
                    << op.src->name << " (scope=" << op.src.scope()
                    << ", dtype=" << op.src->dtype << "), dst=" << op.dst->name
                    << " (scope=" << op.dst.scope()
                    << ", dtype=" << op.dst->dtype
                    << "), no_implicit_async_commit_wait="
                    << no_implicit_commit_wait
                    << ", is_async_copy=" << GetIsAsyncCopy(op);
      if (no_implicit_commit_wait) {
        DLOG(WARNING)
            << "Pipeline-managed async copy fallback to normal copy because "
               "HCU async-copy rewrite found no eligible global->shared "
               "store.";
        return lowered_loop;
      }
      if (explicit_async_semantics) {
        LOG(FATAL) << "Explicit async copy semantics require HCU async-copy "
                      "lowering, "
                      "but no eligible global->shared store was rewritten.";
      }
      DLOG(WARNING) << "Fallback to normal copy because HCU async-copy rewrite "
                       "found no eligible global->shared store.";
      return AnnotateCacheSwizzle(op, LowerNormalCopy(op, lower_args, analyzer),
                                  lower_args, analyzer);
    }
    if (no_implicit_commit_wait) {
      return async_copy_loop;
    }
    if (GetIsAsyncCopy(op)) {
      Stmt commit_group =
          Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
      return SeqStmt({async_copy_loop, commit_group});
    }
    return async_copy_loop;
  }

  static bool CheckCPAsyncCopyPreconditions(const CopyNode &op) {
    if (!IsGlobalBuffer(op.src) || !IsSharedBuffer(op.dst)) {
      return false;
    }
    if (op.src->dtype != op.dst->dtype) {
      return false;
    }
    return true;
  }

  static bool CheckCPAsyncCopy(const CopyNode &op, Target target,
                               const LayoutMap &layout_map,
                               arith::Analyzer *analyzer) {
    if (!TargetHcuHasAsyncCopy(target)) {
      return false;
    }
    return CheckCPAsyncCopyPreconditions(op);
  }
};

} // namespace hcu

namespace {

bool MatchHCUCopyTarget(Target target) { return TargetIsHCU(target); }

bool RegisterHCUCopy() {
  RegisterCopyImpl(CopyImpl{
      "hcu.Copy",
      MatchHCUCopyTarget,
      100,
      hcu::Copy::InferLayout,
      hcu::Copy::Lower,
  });
  return true;
}

const bool hcu_copy_registered = RegisterHCUCopy();

} // namespace

} // namespace tl
} // namespace tvm
