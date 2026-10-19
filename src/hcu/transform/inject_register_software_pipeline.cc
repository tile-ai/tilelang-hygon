/*!
 * \file inject_register_software_pipeline.cc
 * \brief Software pipeline of the HCU register-resident K loop.
 */

#include "transform/common/pipeline_utils.h"
#include "transform/inject_pipeline.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {
namespace transform {

/*!
 * \brief The register-resident K loop.
 *
 * RegisterPipelinePlanning schedules the whole K loop body — the global→shared
 * prefetches, the shared→register loads and the MMA — and publishes it under
 * the tl_register_pipeline_* keys. Only the shared memory it prefetches and the
 * registers it rotates are multi-versioned: a local buffer that is written and
 * read inside the loop keeps at least num_register_stages versions so the
 * consumer of one stage can rotate to the next while the producer fills it.
 * The loop keeps its own schedule annotations, which the passes after the
 * injector read back to place fences and warp divergence.
 */
const software_pipeline::PipelineFlavor kRegisterPipelineFlavor = [] {
  software_pipeline::PipelineFlavor flavor;
  flavor.stage_key = kRegisterPipelineStage;
  flavor.order_key = kRegisterPipelineOrder;
  flavor.async_stage_key = kRegisterPipelineAsyncStages;
  flavor.applied_key = kRegisterPipelineApplied;
  flavor.rotates_local_buffers = true;
  flavor.local_min_versions_key = kNumRegisterStages;
  flavor.local_min_versions = kDefaultNumRegisterStages;
  flavor.versioned_locals_key = kRegisterPipelineVersionedLocals;
  flavor.consumed_keys = {kRegisterPipelineStage, kRegisterPipelineOrder,
                          kRegisterPipelineAsyncStages};
  flavor.preserved_keys = {"num_stages", kEnableRegisterPipeline,
                           kNumRegisterStages, kEnableWarpDivergence};
  flavor.versions_other = false;
  flavor.fatal_on_partial_schedule = false;
  flavor.owns_barrier_buffers = false;
  return flavor;
}();

tirx::transform::Pass InjectRegisterSoftwarePipeline() {
  return software_pipeline::CreateInjectPipelinePass(
      "tl.InjectRegisterSoftwarePipeline", kRegisterPipelineFlavor);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectRegisterSoftwarePipeline",
                        InjectRegisterSoftwarePipeline);
}

} // namespace transform
} // namespace tl
} // namespace tvm
