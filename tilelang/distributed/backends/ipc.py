"""HIP IPC arena and peer mapping lifecycle for HCU distributed kernels."""

from __future__ import annotations

from typing import Any, Protocol

from ..allocator import Allocation, DistributedAllocator


class _IpcRuntime(Protocol):
    def malloc(self, nbytes: int) -> int: ...
    def free(self, ptr: int) -> None: ...
    def create_handle(self, ptr: int) -> bytes: ...
    def open_handle(self, handle: bytes) -> int: ...
    def close_handle(self, ptr: int) -> None: ...
    def can_access_peer(self, device: int, peer_device: int) -> bool: ...


class _FfiIpcRuntime:
    def __init__(self) -> None:
        from tilelang import tvm

        get = tvm.ffi.get_global_func
        self._malloc = get("tl.hcu.ipc.malloc")
        self._free = get("tl.hcu.ipc.free")
        self._create_handle = get("tl.hcu.ipc.create_handle")
        self._open_handle = get("tl.hcu.ipc.open_handle")
        self._close_handle = get("tl.hcu.ipc.close_handle")
        self._can_access_peer = get("tl.hcu.ipc.can_access_peer")

    def malloc(self, nbytes: int) -> int: return int(self._malloc(nbytes))
    def free(self, ptr: int) -> None: self._free(ptr)
    def create_handle(self, ptr: int) -> bytes: return bytes(self._create_handle(ptr))
    def open_handle(self, handle: bytes) -> int: return int(self._open_handle(handle))
    def close_handle(self, ptr: int) -> None: self._close_handle(ptr)
    def can_access_peer(self, device: int, peer_device: int) -> bool: return bool(self._can_access_peer(device, peer_device))


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class IpcAllocator(DistributedAllocator):
    """One local HIP allocation plus rank-specific IPC mappings.

    Allocation offsets are deterministic: every rank must call ``allocate`` in
    the same order, with identical sizes and alignments.  Peer virtual addresses
    are deliberately stored per rank because HIP IPC does not promise equal VA
    mappings in different processes.
    """

    def __init__(self, size: int, *, rank: int, world_size: int, device: int,
                 group: Any = None, alignment: int = 256, runtime: _IpcRuntime | None = None) -> None:
        if size <= 0 or alignment <= 0:
            raise ValueError("size and alignment must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.size, self.rank, self.world_size, self.device = int(size), int(rank), int(world_size), int(device)
        self.group, self.alignment = group, int(alignment)
        self._runtime = runtime or _FfiIpcRuntime()
        self._base = 0
        self._peer_bases: list[int] = []
        self._offset = 0
        self._generation = 0
        self._closed = False

    @property
    def generation(self) -> int: return self._generation

    @property
    def metadata(self) -> tuple[int, ...]:
        self._require_open()
        return (self.rank, self.world_size, *self._peer_bases)

    @property
    def base_ptr(self) -> int:
        self._require_open()
        return self._base

    def initialize(self, handles: list[bytes] | None = None, device_ids: list[int] | None = None) -> None:
        if self._base:
            return
        self._base = self._runtime.malloc(self.size)
        try:
            handles = handles or self._exchange(self._runtime.create_handle(self._base))
            device_ids = device_ids or self._exchange(self.device)
            if len(handles) != self.world_size or len(device_ids) != self.world_size:
                raise RuntimeError("IPC collective exchange returned an unexpected world size")
            self._peer_bases = [0] * self.world_size
            for peer, (handle, peer_device) in enumerate(zip(handles, device_ids)):
                if peer == self.rank:
                    self._peer_bases[peer] = self._base
                else:
                    if not self._runtime.can_access_peer(self.device, int(peer_device)):
                        raise RuntimeError(f"HCU device {self.device} cannot access peer rank {peer} device {peer_device}")
                    self._peer_bases[peer] = self._runtime.open_handle(handle)
            self._generation += 1
        except Exception:
            self.close()
            raise

    def allocate(self, nbytes: int, alignment: int | None = None) -> Allocation:
        self._require_open()
        if nbytes <= 0:
            raise ValueError("allocation size must be positive")
        offset = _align_up(self._offset, alignment or self.alignment)
        end = offset + nbytes
        if end > self.size:
            raise MemoryError(f"IPC arena exhausted: requested {nbytes} bytes, {self.size - offset} remain")
        self._offset = end
        return Allocation(self._base + offset, offset, nbytes, self._generation)

    def contains(self, ptr: int, nbytes: int) -> bool:
        return self._base <= ptr and nbytes >= 0 and ptr + nbytes <= self._base + self.size

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for peer, ptr in enumerate(self._peer_bases):
            if peer != self.rank and ptr:
                try: self._runtime.close_handle(ptr)
                except Exception: pass
        self._peer_bases = []
        if self._base:
            try: self._runtime.free(self._base)
            finally: self._base = 0
        self._generation += 1

    def _exchange(self, value: Any) -> list[Any]:
        if self.world_size == 1:
            return [value]
        if self.group is None or not hasattr(self.group, "all_gather_object"):
            raise RuntimeError("world_size > 1 requires a group with all_gather_object")
        return list(self.group.all_gather_object(value))

    def _require_open(self) -> None:
        if self._closed or not self._base:
            raise RuntimeError("IPC allocator is not initialized or has been closed")
