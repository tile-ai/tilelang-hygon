/*!
 * \file inject_pipeline.h
 * \brief Software-pipeline injector and the flavor description it consumes.
 */
#ifndef TVM_TL_TRANSFORM_INJECT_PIPELINE_H_
#define TVM_TL_TRANSFORM_INJECT_PIPELINE_H_

#include <tvm/tirx/transform.h>
#include <vector>

namespace tvm {
namespace tl {
namespace software_pipeline {

/*!
 * \brief Schedule description of one software-pipeline flavor.
 *
 * The injector is flavor-agnostic. It reads the stage/order annotations the
 * flavor names, multi-versions the buffer classes the flavor declares, stamps
 * the flavor's applied marker on the rewritten loop and drops the annotations
 * the flavor consumed. A backend whose pipeline cannot be expressed with the
 * shared schedule (for example a K loop that keeps its operands in registers)
 * describes it here and creates its pass with CreateInjectPipelinePass, instead
 * of adding branches to the shared injector.
 */
struct PipelineFlavor {
  /*! Loop annotation holding this flavor's per-statement stage. */
  const char *stage_key{nullptr};
  /*! Loop annotation holding this flavor's per-statement order. */
  const char *order_key{nullptr};
  /*! Loop annotation holding this flavor's async stage set; may be null. */
  const char *async_stage_key{nullptr};
  /*! Annotation stamped on the rewritten loop to mark the flavor applied. */
  const char *applied_key{nullptr};
  /*!
   * \brief Applied marker of another flavor that owns the loop.
   *
   * A loop already stamped with this key carries a complete schedule from that
   * flavor, so this one must leave it alone. Null when no other flavor can
   * claim a loop this one would inject.
   */
  const char *foreign_applied_key{nullptr};
  /*!
   * \brief Whether the flavor rotates local buffers inside the loop.
   *
   * A flavor that writes and reads a register-resident buffer in the same
   * iteration needs a second version of it, and the loop's own rotation is a
   * reason to keep a version even when no stage distance asks for one. A flavor
   * that only prefetches shared memory leaves local buffers alone.
   */
  bool rotates_local_buffers{false};
  /*!
   * \brief Loop annotation raising the version floor of a rotated buffer.
   *
   * Null to use local_min_versions.
   */
  const char *local_min_versions_key{nullptr};
  /*! Version floor of a rotated local buffer; only read with
   * rotates_local_buffers. */
  int local_min_versions{0};
  /*!
   * \brief Loop annotation under which the injector publishes the local
   * buffers it versioned.
   *
   * A flavor whose consumers must resolve the rotated slots statically (a
   * register pipeline that materializes one buffer per slot) can name a key
   * here. The injector then records a map from buffer name to version count on
   * the rewritten loop, so those consumers need no heuristic to tell a
   * versioned local from a user's own multi-dimensional fragment. Null to
   * publish nothing.
   */
  const char *versioned_locals_key{nullptr};
  /*! Annotations the flavor consumes; dropped from the loop after the rewrite.
   */
  std::vector<const char *> consumed_keys;
  /*!
   * \brief Annotations carried over verbatim, even if the flavor also lists
   * them as consumed.
   *
   * A flavor that reads an annotation to parameterize its own injection (a
   * stage count, say) keeps it on the rewritten loop for the passes that run
   * after the injector.
   */
  std::vector<const char *> preserved_keys;
  /*! Whether shared-memory buffers are multi-versioned by this flavor. */
  bool versions_shared{true};
  /*! Whether local (register) buffers are multi-versioned by this flavor. */
  bool versions_local{true};
  /*!
   * \brief Whether buffers of any other scope are multi-versioned.
   *
   * A flavor that keeps its operands in registers only versions the shared
   * memory it prefetches and the registers it rotates, leaving the rest alone.
   */
  bool versions_other{true};
  /*!
   * \brief Whether a single stage/order annotation is an error.
   *
   * The shared schedule is written by PipelinePlanning, which always emits both
   * keys; a flavor whose schedule is derived elsewhere may legitimately see
   * only one and declines the loop instead.
   */
  bool fatal_on_partial_schedule{true};
  /*! Whether this flavor manages pipeline mbarrier/TMA barrier buffers. */
  bool owns_barrier_buffers{true};
};

/*!
 * \brief Create a PrimFunc pass running the injector with \p flavor.
 *
 * \param pass_name Name of the created pass, for the pass context and logs.
 * \param flavor The schedule description the injector should follow.
 */
tirx::transform::Pass CreateInjectPipelinePass(const char *pass_name,
                                               const PipelineFlavor &flavor);

} // namespace software_pipeline
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_INJECT_PIPELINE_H_
