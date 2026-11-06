"""Backend-neutral host APIs for TileLang distributed kernels."""

from .allocator import (
    DistributedAllocator,
    get_distributed_allocator,
    get_distributed_backend_registration,
    list_distributed_backends,
    register_distributed_backend,
)

# Deterministic builtin registration; no special dispatch/lazy import in factory.
from . import backends as _backends  # noqa: F401 -- register builtin backends
from .tensor import release_tensor, tensor

__all__ = [
    "DistributedAllocator",
    "get_distributed_allocator",
    "get_distributed_backend_registration",
    "list_distributed_backends",
    "register_distributed_backend",
    "release_tensor",
    "tensor",
]
