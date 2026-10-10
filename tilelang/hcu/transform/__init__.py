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


__all__ = (
    "AnnotateMlsGemmDep",
    "MaterializeHcuGemmLdsStrategy",
    "InjectHcuCopyIdxen",
    "AnnotateScaleGemmDep",
    "InsertScaleBufferSync",
    "AllocateScaleBuffer",
    "PrepareMlsSharedMemoryAllocation",
    "InsertMlsWaitcnt",
    "HoistMlsResource",
    "ResolveHcuEBarrier",
    "LowerAndValidateHcuWdra",
    "LowerHcuBlockAnnotations",
    "MaterializeHcuDeviceAttrs",
)
