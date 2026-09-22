"""Backend-neutral host APIs for TileLang distributed kernels."""

from .allocator import DistributedAllocator, get_distributed_allocator

__all__ = ["DistributedAllocator", "get_distributed_allocator"]
