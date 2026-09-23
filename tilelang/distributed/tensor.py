"""Non-owning PyTorch views over distributed allocator allocations."""

from __future__ import annotations

from math import prod
from typing import Sequence

from .allocator import Allocation, DistributedAllocator


def _contiguous_stride(shape: Sequence[int]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for extent in reversed(shape):
        stride.append(running)
        running *= extent
    return tuple(reversed(stride))


def tensor(
    shape: Sequence[int],
    dtype,
    *,
    allocator: DistributedAllocator,
    allocation: Allocation | None = None,
):
    """Create a non-owning contiguous torch tensor in an IPC allocation.

    The allocator remains the sole owner of the HIP allocation: callers must
    release tensor views before calling ``allocator.close()``.  ``allocation``
    may be supplied to share a deterministic arena slice across wrappers; when
    omitted, the required slice is allocated from ``allocator``.
    """
    import torch

    normalized_shape = tuple(int(extent) for extent in shape)
    if any(extent <= 0 for extent in normalized_shape):
        raise ValueError("tensor shape extents must be positive")
    if not hasattr(allocator, "device"):
        raise TypeError("distributed tensor views require a device allocator")
    element_size = torch.empty((), dtype=dtype).element_size()
    nbytes = prod(normalized_shape) * element_size
    if allocation is None:
        allocation = allocator.allocate(nbytes)
    if allocation.generation != allocator.generation:
        raise RuntimeError("allocation generation is stale")
    if allocation.nbytes < nbytes:
        raise ValueError("allocation is smaller than requested tensor shape")

    device = torch.device(f"cuda:{allocator.device}")
    storage = torch._C._construct_storage_from_data_pointer(
        allocation.ptr, device, allocation.nbytes
    )
    metadata = {
        "nbytes": allocation.nbytes,
        "data_ptr": allocation.ptr,
        "size": normalized_shape,
        "stride": _contiguous_stride(normalized_shape),
        "dtype": dtype,
        "device": device,
        "storage_offset": 0,
    }
    return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(metadata, storage)
