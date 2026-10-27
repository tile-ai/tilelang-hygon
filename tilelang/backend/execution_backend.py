"""Execution backend policy value type."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tvm.target import Target

TargetPredicate = Callable[[Target], bool]
AvailabilityCheck = Callable[[], bool]


def _always_available() -> bool:
    return True


@dataclass(frozen=True, slots=True)
class ExecutionBackendSpec:
    name: str
    is_available: AvailabilityCheck = _always_available
    supports_target: TargetPredicate | None = None
    enable_host_codegen: bool = False
    enable_device_compile: bool = False
    # Declares that this backend's host codegen lowers the TVM-FFI
    # callee-allocated-output result slot and that its runtime provides an
    # environment tensor allocator, so kernels with out_idx may allocate
    # their outputs inside the generated host function.
    supports_callee_allocated_outputs: bool = False
    # Declares the runtime hooks needed by artifacts that actually use
    # distributed metadata: binding/initialization and lifecycle checks.
    # Backends without these hooks must reject such artifacts
    # instead of launching them without the required initialization.
    supports_distributed_runtime: bool = True

    def matches(self, target: Target) -> bool:
        return True if self.supports_target is None else self.supports_target(target)
