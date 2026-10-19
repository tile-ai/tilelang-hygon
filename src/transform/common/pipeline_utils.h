/*!
 * \file pipeline_utils.h
 * \brief Shared utilities for software-pipeline and warp-specialization passes.
 *
 * Provides:
 *  - Pipeline annotation attribute keys
 *  - GetPipelineNumStages()  — extract num_stages from loop annotations
 *  - IsPipelineDeclarationStmt() — identify non-stage buffer declarations
 *  - ComputeThreadBounds()  — derive thread bounds from an analyzer + IterVar
 */
#ifndef TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_

#include "support/check.h"
#include <optional>
#include <string>
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

using namespace tirx;

// ---------------------------------------------------------------------------
// Pipeline annotation attribute keys
// ---------------------------------------------------------------------------

/*! Per-statement TMA copy flag (1 = TMA eligible, 0 = not). */
static constexpr const char *kPipelineTmaCopies =
    "software_pipeline_tma_copies";
/*! Per-statement async producer flag (1 = async copy producer, 0 = not). */
static constexpr const char *kPipelineAsyncProducers =
    "software_pipeline_async_producers";
/*! Per-statement async producer group id (-1 = not an async producer). */
static constexpr const char *kPipelineAsyncProducerGroups =
    "software_pipeline_async_producer_groups";
/*! Per-original-statement replayable scalar Bind flag (1 = replayable). */
static constexpr const char *kPipelineReplayableScalarBinds =
    "software_pipeline_replayable_scalar_binds";

static constexpr const char *kRegisterPipelineStage =
    "tl_register_pipeline_stage";
static constexpr const char *kRegisterPipelineOrder =
    "tl_register_pipeline_order";
static constexpr const char *kRegisterPipelineAsyncStages =
    "tl_register_pipeline_async_stages";
static constexpr const char *kEnableRegisterPipeline =
    "enable_register_pipeline";
static constexpr const char *kNumRegisterStages = "num_register_stages";
static constexpr const char *kEnableWarpDivergence = "enable_warp_divergence";
static constexpr const char *kRegisterPipelineApplied =
    "tl_register_pipeline_applied";
/*! Loop annotation recording the local buffers the injector versioned, as a
 * map from buffer name to version count. */
static constexpr const char *kRegisterPipelineVersionedLocals =
    "tl_register_pipeline_versioned_locals";
/*! Loop annotation recording how many pipeline iterations the stage
 * materializer folded into one, for passes that count per-iteration work in
 * the rewritten body. */
static constexpr const char *kRegisterPipelinePhaseUnroll =
    "tl_register_pipeline_phase_unroll";
static constexpr int kDefaultNumRegisterStages = 2;
static constexpr const char *kRegisterPipelineBufferSuffix = "_reg_pipe";

/*! \brief Read an integer loop annotation, accepting both storage shapes.
 *
 * A loop annotation reaches a pass in one of two shapes: the compiler writes
 * its own bookkeeping as an `IntImm` object, while an entry of a user
 * `annotations` dict passed to `T.Pipelined` arrives as a primitive integer or
 * boolean from the Python frontend. Accepting both keeps the vendor
 * annotations readable no matter which side wrote them.
 */
inline std::optional<int64_t> ReadLoopAnnotationAsInt(const ForNode *loop,
                                                      const char *key) {
  if (loop == nullptr) {
    return std::nullopt;
  }
  auto val = loop->annotations.Get(key);
  if (!val.has_value()) {
    return std::nullopt;
  }
  const ffi::Any &any = val.value();
  if (const auto *imm = any.as<IntImmNode>()) {
    return static_cast<int64_t>(imm->value);
  }
  return any.try_cast<int64_t>();
}

inline bool LoopAnnotationIsTruthy(const ForNode *loop, const char *key) {
  auto value = ReadLoopAnnotationAsInt(loop, key);
  return value.has_value() && value.value() != 0;
}

/*!
 * \brief Whether the loop asks for a register-resident pipeline.
 *
 * The request is what the frontend wrote: num_register_stages, or the
 * enable_register_pipeline flag asking for the default depth.
 */
inline bool LoopRequestsRegisterPipeline(const ForNode *loop) {
  if (auto num_register_stages = ReadLoopAnnotationAsInt(loop, kNumRegisterStages)) {
    return num_register_stages.value() > 1;
  }
  return LoopAnnotationIsTruthy(loop, kEnableRegisterPipeline);
}

/*!
 * \brief Whether the register injector actually rewrote this loop.
 *
 * Passes after the injector must key off the applied marker, never off the
 * request: a loop that asked for a register pipeline but was not planned for
 * one keeps its request annotation, while its body is an ordinary
 * shared-memory pipeline that these passes would then misread.
 */
inline bool LoopHasAppliedRegisterPipeline(const ForNode *loop) {
  return LoopAnnotationIsTruthy(loop, kRegisterPipelineApplied);
}

inline bool LoopHasWarpDivergence(const ForNode *loop) {
  return LoopAnnotationIsTruthy(loop, kEnableWarpDivergence);
}

inline bool IsRegisterPipelineBufferScope(const ffi::String &scope) {
  std::string s = scope;
  return s == "local" || (s.size() > 6 && s.compare(0, 6, "local.") == 0);
}

inline bool IsRegisterPipelineBuffer(const Buffer &buffer) {
  return buffer.defined() && IsRegisterPipelineBufferScope(buffer.scope());
}

/*! \brief Whether a flat TIRX statement declares pipeline-local buffer storage.
 *
 * Flat TIRX represents buffer allocations/declarations as standalone
 * statements. They must stay in the loop body so later rewrites can preserve
 * storage, but they are declarations rather than executable pipeline stages.
 */
inline bool IsPipelineDeclarationStmt(const Stmt &stmt) {
  return stmt.as<AllocBufferNode>() != nullptr ||
         stmt.as<DeclBufferNode>() != nullptr;
}

/*! \brief Return the top-level statement stream for a pipeline loop body.
 *
 * IfStmtBinding may preserve an if-body that starts with Bind as a single
 * IfThenElse, so a valid pipeline body is not necessarily a SeqStmt. Keep the
 * downstream planning code Array-based and only reconstruct SeqStmt when there
 * are multiple children.
 */
inline ffi::Array<Stmt> NormalizePipelineBody(const Stmt &body) {
  if (const auto *seq = body.as<SeqStmtNode>()) {
    return seq->seq;
  }
  return ffi::Array<Stmt>{body};
}

/*! \brief Build a valid TIRX statement from a pipeline statement stream. */
inline Stmt MakePipelineBody(const ffi::Array<Stmt> &stmts) {
  ICHECK(!stmts.empty());
  return stmts.size() == 1 ? stmts[0] : SeqStmt(stmts);
}

// ---------------------------------------------------------------------------
// GetPipelineNumStages
// ---------------------------------------------------------------------------

/*!
 * \brief Extract the pipeline stage count from a For loop's annotations.
 *
 * Checks (in order):
 *   1. "num_stages" — user-provided stage count
 *   2. "tl_pipelined_num_stages" — set by InjectSoftwarePipeline
 *   3. s_tir::attr::software_pipeline_stage — max(stage) + 1
 *
 * \return The stage count, or nullopt if the loop is not pipelined.
 */
inline ffi::Optional<Integer> GetPipelineNumStages(const ForNode *loop) {
  if (auto num_stages = loop->annotations.Get("num_stages")) {
    if (const auto *imm = num_stages->as<IntImmNode>()) {
      return Integer(static_cast<int>(imm->value));
    }
  }
  if (auto num_stages = loop->annotations.Get("tl_pipelined_num_stages")) {
    if (const auto *imm = num_stages->as<IntImmNode>()) {
      return Integer(static_cast<int>(imm->value));
    }
  }
  if (auto stages_anno =
          loop->annotations.Get(s_tir::attr::software_pipeline_stage)) {
    auto stages = Downcast<ffi::Array<Integer>>(stages_anno.value());
    int max_stage = -1;
    for (const auto &stage : stages) {
      max_stage = std::max(max_stage, static_cast<int>(stage->value));
    }
    if (max_stage >= 0) {
      return Integer(max_stage + 1);
    }
  }
  return ffi::Optional<Integer>();
}

/*!
 * \brief Stage count from T.Pipelined / InjectSoftwarePipeline only.
 *
 * Reads num_stages / tl_pipelined_num_stages; does not use
 * software_pipeline_stage.
 */
inline ffi::Optional<Integer>
GetExplicitPipelinedNumStages(const ForNode *loop) {
  if (auto num_stages = loop->annotations.Get("num_stages")) {
    if (const auto *imm = num_stages->as<IntImmNode>()) {
      return Integer(static_cast<int>(imm->value));
    }
  }
  if (auto num_stages = loop->annotations.Get("tl_pipelined_num_stages")) {
    if (const auto *imm = num_stages->as<IntImmNode>()) {
      return Integer(static_cast<int>(imm->value));
    }
  }
  return ffi::Optional<Integer>();
}

// ---------------------------------------------------------------------------
// ComputeThreadBounds
// ---------------------------------------------------------------------------

/*!
 * \brief Compute the thread index bounds from an IterVar and an analyzer.
 *
 * \return Range covering the thread index, or [0, 1) if no bound is known.
 */
inline Range ComputeThreadBounds(const IterVar &thread_var,
                                 const arith::Analyzer &analyzer) {
  if (thread_var.defined() &&
      analyzer.const_int_bound.IsBound(thread_var->var)) {
    auto const_int_bound = analyzer.const_int_bound(thread_var);
    auto min_value = const_int_bound->min_value;
    auto max_value = const_int_bound->max_value;
    auto extent = max_value - min_value + 1;
    auto dtype = thread_var->var.dtype();
    return Range::FromMinExtent(IntImm(dtype, min_value),
                                IntImm(dtype, extent));
  }
  return Range::FromMinExtent(0, 1);
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_
