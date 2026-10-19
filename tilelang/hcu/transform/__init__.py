"""HCU-only TIR transformation frontends."""
# pylint: disable=invalid-name

from tilelang.transform import _ffi_api


def AnnotateMlsGemmDep():
    """Annotate MLS producers and their GEMM consumers."""
    return _ffi_api.AnnotateMlsGemmDep()  # type: ignore


def MaterializeHcuGemmLdsStrategy():
    """Activate compiler-derived HCU GEMM LDS strategies."""
    return _ffi_api.MaterializeHcuGemmLdsStrategy()  # type: ignore


def InjectHcuCopyIdxen():
    """Rewrite annotated HCU async copies to idxen/wrap addressing."""
    return _ffi_api.InjectHcuCopyIdxen()  # type: ignore


def AnnotateScaleGemmDep():
    """Bind copy_scale producers to block-scaled GEMM consumers."""
    return _ffi_api.AnnotateScaleGemmDep()  # type: ignore


def PlanScaleBufferLayout():
    """Select the internal copy_scale/MMAC row order after pipeline planning."""
    return _ffi_api.PlanScaleBufferLayout()  # type: ignore


def InsertScaleBufferSync():
    """Insert synchronization between copy_scale and its GEMM consumer."""
    return _ffi_api.InsertScaleBufferSync()  # type: ignore


def AllocateScaleBuffer():
    """Plan and rewrite HCU scale-buffer row allocations."""
    return _ffi_api.AllocateScaleBuffer()  # type: ignore


def PrepareMlsSharedMemoryAllocation():
    """Prepare MLS shared-memory metadata for merge planning."""
    return _ffi_api.PrepareMlsSharedMemoryAllocation()  # type: ignore


def InsertMlsWaitcnt():
    """Insert conservative waitcnts before MLS LDS consumers."""
    return _ffi_api.InsertMlsWaitcnt()  # type: ignore


def HoistMlsResource():
    """Hoist HCU MLS resource setup before codegen."""
    return _ffi_api.HoistMlsResource()  # type: ignore


def ResolveHcuEBarrier():
    """Resolve HCU partial-sync and all-reduce ebarrier policies."""
    return _ffi_api.ResolveHcuEBarrier()  # type: ignore


def LowerAndValidateHcuWdra():
    """Lower tx-based warp specialization to HCU WDRA form and validate."""
    return _ffi_api.LowerAndValidateHcuWdra()  # type: ignore


def LowerHcuBlockAnnotations():
    """Materialize HCU-owned block annotations as lexical codegen attrs."""
    return _ffi_api.LowerHcuBlockAnnotations()  # type: ignore


def MaterializeHcuDeviceAttrs():
    """Extract HCU body markers onto split device PrimFuncs."""
    return _ffi_api.MaterializeHcuDeviceAttrs()  # type: ignore


def RegisterPipelinePlanning():
    """Plan register-resident fragments and rewrite GEMMs for the register pipeline."""
    return _ffi_api.RegisterPipelinePlanning()  # type: ignore


def InjectRegisterSoftwarePipeline():
    """Inject the software pipeline of the register-resident K loop."""
    return _ffi_api.InjectRegisterSoftwarePipeline()  # type: ignore


def MarkFencePlannerOwnership():
    """Record whether the fence planner will own this kernel's async waits."""
    return _ffi_api.MarkFencePlannerOwnership()  # type: ignore


def InjectAsyncGlobalLoadFence():
    """Lower async wait scopes and insert G2S barriers."""
    return _ffi_api.InjectAsyncGlobalLoadFence()  # type: ignore


def LowerAsyncCommitWait():
    """Lower async commit/wait attributes to concrete intrinsics."""
    return _ffi_api.LowerAsyncCommitWait()  # type: ignore


def MaterializeRegisterPipelineStages():
    """Give every rotated register-pipeline slot its own same-shape buffer."""
    return _ffi_api.MaterializeRegisterPipelineStages()  # type: ignore


def InjectAsyncMmaFence():
    """Insert LDS waits before MMA."""
    return _ffi_api.InjectAsyncMmaFence()  # type: ignore


def InjectWarpDivergence():
    """Split a register-pipelined K loop across two warp groups."""
    return _ffi_api.InjectWarpDivergence()  # type: ignore


def InjectRegisterPipelineSchedBarrier():
    """Wrap MMA clusters with sched_barrier when the register pipeline is on."""
    return _ffi_api.InjectRegisterPipelineSchedBarrier()  # type: ignore


def VerifyPipelineSync():
    """Verify ownership and wait/barrier structure of fence-planned K loops."""
    return _ffi_api.VerifyPipelineSync()  # type: ignore


__all__ = (
    "AnnotateMlsGemmDep",
    "MaterializeHcuGemmLdsStrategy",
    "InjectHcuCopyIdxen",
    "AnnotateScaleGemmDep",
    "PlanScaleBufferLayout",
    "InsertScaleBufferSync",
    "AllocateScaleBuffer",
    "PrepareMlsSharedMemoryAllocation",
    "InsertMlsWaitcnt",
    "HoistMlsResource",
    "ResolveHcuEBarrier",
    "LowerAndValidateHcuWdra",
    "LowerHcuBlockAnnotations",
    "MaterializeHcuDeviceAttrs",
    "RegisterPipelinePlanning",
    "InjectRegisterSoftwarePipeline",
    "MarkFencePlannerOwnership",
    "InjectAsyncGlobalLoadFence",
    "LowerAsyncCommitWait",
    "MaterializeRegisterPipelineStages",
    "InjectAsyncMmaFence",
    "InjectWarpDivergence",
    "InjectRegisterPipelineSchedBarrier",
    "VerifyPipelineSync",
)
