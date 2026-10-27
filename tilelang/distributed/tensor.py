"""Public array-interface tensor views that retain their backing arena."""

from __future__ import annotations

from math import prod
from collections.abc import Sequence
import weakref

from .allocator import Allocation, DistributedAllocator


class _StorageOwner:
    """Retained by Torch storage, including slices, detach and storage aliases."""

    def __init__(self, allocation: Allocation, nbytes: int, allocator: DistributedAllocator) -> None:
        self.allocation = allocation
        self._import_started = False
        token = allocator._retain_storage(allocation, nbytes)
        self._finalizer = weakref.finalize(self, allocator._release_storage, token)
        # Live storage must not be released early during interpreter teardown.
        self._finalizer.atexit = False
        self.__cuda_array_interface__ = {
            "shape": (nbytes,), "typestr": "|u1", "data": (allocation.ptr, False),
            "strides": None, "version": 3, "stream": 1,
        }


def _view_from_pointer(shape, dtype, storage: _StorageOwner):
    """Zero-copy public import; byte reinterpretation also supports BF16."""
    import torch

    device = torch.device(f"cuda:{storage.allocation.device}")
    with torch.cuda.device(device):
        storage.__cuda_array_interface__["stream"] = torch.cuda.current_stream(device).cuda_stream or 1
        storage._import_started = True
        result = torch.as_tensor(storage, device=device).view(dtype).reshape(shape).detach()
    if result.data_ptr() != storage.allocation.ptr or result.device != device:
        raise RuntimeError("public array-interface import did not preserve the arena pointer/device")
    return result


def release_tensor(view) -> None:
    """Empty this tensor handle in place; aliases keep the backing arena alive.

    Idempotent. This is a deliberate release of the supplied handle, not an
    invalidation of its aliases. Release or delete aliases before close too.
    close() still performs device synchronization before freeing the arena.
    """
    import torch

    if not isinstance(view, torch.Tensor):
        raise TypeError("release_tensor requires a torch.Tensor")
    with torch.no_grad():
        view.set_(torch.empty((0,), dtype=view.dtype, device=view.device))


def tensor(
    shape: Sequence[int],
    dtype,
    *,
    allocator: DistributedAllocator,
    allocation: Allocation | None = None,
):
    """Create a contiguous torch tensor that retains its backing arena.

    close() refuses while any imported storage alias is alive. Delete views
    or explicitly empty their handles with ``release_tensor`` before close.
    ``allocation`` must have been issued by this exact allocator instance.
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
    storage = _StorageOwner(allocation, nbytes, allocator)
    try:
        return _view_from_pointer(normalized_shape, dtype, storage)
    except Exception:
        # Before Torch sees the owner, no imported storage can retain it.
        # After that point its finalizer must follow actual storage lifetime,
        # including storage retained by an exception traceback.
        if not storage._import_started:
            storage._finalizer()
        raise
