"""HIP IPC arena and peer mapping lifecycle for HCU distributed kernels."""

from __future__ import annotations

from threading import RLock
from functools import cache
from typing import Any, Protocol
from collections.abc import Callable

from ..allocator import (
    Allocation, DistributedAllocator,
    get_distributed_backend_registration, register_distributed_backend,
)


@cache
def ipc_metadata_layout() -> dict[str, int]:
    """Common envelope plus derived IPC-private offsets (all from native FFI)."""
    import json
    from tilelang import tvm
    from ..metadata import metadata_layout

    layout = dict(metadata_layout())
    private = json.loads(str(tvm.ffi.get_global_func("tl.hcu.ipc.payload_layout")()))
    layout["arena_size"] = layout["payload"] + private["arena_size"]
    layout["peer_base"] = layout["payload"] + private["peer_bases"]
    layout["private_header_size"] = private["header_size"]
    layout["max_world_size"] = private["max_world_size"]
    for key in ("invalid_source_count", "invalid_source_span", "invalid_source_arena"):
        layout[key] = private[key]
    return layout


def ipc_max_world_size() -> int:
    return ipc_metadata_layout()["max_world_size"]


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

    def malloc(self, nbytes: int) -> int:
        return int(self._malloc(nbytes))

    def free(self, ptr: int) -> None:
        self._free(ptr)

    def create_handle(self, ptr: int) -> bytes:
        return bytes(self._create_handle(ptr))

    def open_handle(self, handle: bytes) -> int:
        return int(self._open_handle(handle))

    def close_handle(self, ptr: int) -> None:
        self._close_handle(ptr)

    def can_access_peer(self, device: int, peer_device: int) -> bool:
        return bool(self._can_access_peer(device, peer_device))


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class IpcAllocator(DistributedAllocator):
    """One local HIP allocation plus rank-specific IPC mappings.

    Allocation offsets are deterministic: every rank must call ``allocate`` in
    the same order, with identical sizes and alignments.  Peer virtual addresses
    are deliberately stored per rank because HIP IPC does not promise equal VA
    mappings in different processes.
    """

    backend_name = "ipc"

    def __init__(
        self,
        size: int,
        *,
        rank: int,
        world_size: int,
        device: int,
        group: Any = None,
        alignment: int = 256,
        runtime: _IpcRuntime | None = None,
        synchronize: Callable[[], None] | None = None,
    ) -> None:
        if size <= 0 or alignment <= 0:
            raise ValueError("size and alignment must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if world_size > ipc_max_world_size():
            raise ValueError(f"IPC world_size exceeds metadata capacity (maximum {ipc_max_world_size()})")
        self._bind_distributed_backend(self.backend_name)
        self.size, self.rank, self.world_size, self.device = int(size), int(rank), int(world_size), int(device)
        self.group, self.alignment = group, int(alignment)
        self._runtime = runtime or _FfiIpcRuntime()
        self._synchronize = synchronize or self._synchronize_device
        self._base = 0
        self._peer_bases: list[int] = []
        self._offset = 0
        self._generation = 0
        self._closed = False
        self._arena_device = self.device
        self._allocations: dict[int, Allocation] = {}
        self._storages: set[object] = set()
        self._storage_lock = RLock()

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def metadata(self) -> tuple[int, ...]:
        """Native ABI header/status slots followed by process-local peer VAs."""
        self._require_open()
        layout = ipc_metadata_layout()
        payload = [0] * layout["peer_base"]
        payload[layout["version"]] = layout["abi_version"]
        payload[layout["backend"]] = self.backend_id
        payload[layout["rank"]] = self.rank
        payload[layout["world_size"]] = self.world_size
        payload[layout["payload_size"]] = layout["private_header_size"] + self.world_size
        payload[layout["arena_size"]] = self.size
        return (*payload, *self._peer_bases)

    @property
    def base_ptr(self) -> int:
        self._require_open()
        return self._base

    def initialize(self, handles: list[bytes] | None = None, device_ids: list[int] | None = None) -> None:
        if self._closed:
            raise RuntimeError("IPC allocator cannot be initialized after close or failed initialization")
        if self._base:
            return
        # Explicit handle/device lists are useful for local tests and callers
        # that perform their own exchange.  Otherwise every local stage reports
        # success or failure before ranks advance to the next collective.
        collective = self.world_size > 1 and (handles is None or device_ids is None)
        try:
            # HIP IPC allocation and handle import are device-context sensitive.
            # PyTorch uses the CUDA namespace for both CUDA and HIP builds.
            import torch

            self._run_initialization_stage("device selection", lambda: torch.cuda.set_device(self.device), collective)

            def allocate_local() -> None:
                self._base = self._runtime.malloc(self.size)

            self._run_initialization_stage("allocation", allocate_local, collective)

            if handles is None:
                local_handle = self._run_initialization_stage(
                    "handle creation", lambda: self._runtime.create_handle(self._base), collective
                )
                handles = self._run_initialization_stage("handle exchange", lambda: self._exchange(local_handle), collective)
            if device_ids is None:
                device_ids = self._run_initialization_stage("device exchange", lambda: self._exchange(self.device), collective)

            def map_peers() -> None:
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

            self._run_initialization_stage("peer mapping", map_peers, collective)
            self._generation += 1
        except Exception as initialization_error:
            self._closed = True
            cleanup_error = None
            try:
                self._run_initialization_stage("failed initialization cleanup", self._release_local_resources, collective)
            except Exception as exc:
                cleanup_error = exc
            finally:
                self._generation += 1
            if cleanup_error is not None:
                raise RuntimeError(f"{initialization_error}; cleanup failed: {cleanup_error}") from initialization_error
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
        allocation = Allocation(self._base + offset, offset, nbytes, self._generation, self, self.device)
        self._allocations[offset] = allocation
        return allocation

    def _validate_allocation(self, allocation: Allocation, nbytes: int) -> None:
        self._require_open()
        if allocation.owner is not self:
            raise RuntimeError("allocation owner does not match the allocator")
        if allocation.generation != self.generation:
            raise RuntimeError("allocation generation is stale")
        if allocation.device != self.device or self.device != self._arena_device:
            raise RuntimeError("allocation device does not match the arena device")
        if allocation.offset < 0 or allocation.nbytes <= 0 or allocation.offset + allocation.nbytes > self.size:
            raise ValueError("allocation offset/size is outside the arena")
        if not self.contains(allocation.ptr, allocation.nbytes):
            raise ValueError("allocation pointer is outside the arena")
        if allocation.ptr != self._base + allocation.offset:
            raise ValueError("allocation pointer does not match its arena offset")
        if allocation.nbytes < nbytes:
            raise ValueError("allocation is smaller than requested tensor shape")
        if self._allocations.get(allocation.offset) is not allocation:
            raise RuntimeError("allocation was not issued by this allocator")

    def _retain_storage(self, allocation: Allocation, nbytes: int) -> object:
        with self._storage_lock:
            self._validate_allocation(allocation, nbytes)
            token = object()
            self._storages.add(token)
            return token

    def _release_storage(self, token: object) -> None:
        with self._storage_lock:
            self._storages.discard(token)

    def contains(self, ptr: int, nbytes: int) -> bool:
        return self._base <= ptr and nbytes >= 0 and ptr + nbytes <= self._base + self.size

    def close(self, *, collective: bool = True) -> None:
        """Release an IPC arena after all ranks stopped using its mappings."""
        with self._storage_lock:
            self._close(collective=collective)

    def _close(self, *, collective: bool) -> None:
        if self._closed:
            return
        coordinate = collective and self._base != 0 and self.world_size > 1 and self.group is not None

        def check_views() -> None:
            if self._storages:
                raise RuntimeError(f"{len(self._storages)} active tensor view(s)")

        # Reuse the existing status exchange: every rank learns the counts and
        # failing ranks before anyone transitions to CLOSED or unmaps memory.
        # Rejection leaves generation, mappings and initialization unchanged.
        self._run_initialization_stage("close active views", check_views, coordinate)
        self._closed = True
        failures = []

        if collective and self._base:
            try:
                self._run_initialization_stage("close synchronization", self._synchronize, coordinate)
            except Exception as exc:
                failures.append(str(exc))
            if not failures:
                try:
                    self._run_initialization_stage("close barrier", self._barrier, coordinate)
                except Exception as exc:
                    failures.append(str(exc))

        try:
            self._run_initialization_stage("close resource cleanup", self._release_local_resources, coordinate)
        except Exception as exc:
            failures.append(str(exc))
        finally:
            self._generation += 1
        if failures:
            raise RuntimeError("IPC collective close failed: " + "; ".join(failures))

    def _release_local_resources(self) -> None:
        self._allocations.clear()
        cleanup_errors = []
        for peer, ptr in enumerate(self._peer_bases):
            if peer != self.rank and ptr:
                try:
                    self._runtime.close_handle(ptr)
                except Exception as exc:
                    cleanup_errors.append(f"peer rank {peer} unmap: {type(exc).__name__}: {exc}")
        self._peer_bases = []
        if self._base:
            try:
                self._runtime.free(self._base)
            except Exception as exc:
                cleanup_errors.append(f"local free: {type(exc).__name__}: {exc}")
            finally:
                self._base = 0
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))

    def _run_initialization_stage(self, stage: str, operation: Callable[[], Any], collective: bool) -> Any:
        value, error = None, None
        try:
            value = operation()
        except Exception as exc:
            error = exc
        if not collective:
            if error is not None:
                raise error
            return value

        status = (self.rank, stage, None if error is None else f"{type(error).__name__}: {error}")
        statuses = self._exchange(status)
        if len(statuses) != self.world_size:
            raise RuntimeError(f"IPC {stage} status exchange returned an unexpected world size")
        failures = []
        for item in statuses:
            if not isinstance(item, (tuple, list)) or len(item) != 3 or item[1] != stage:
                raise RuntimeError(f"IPC {stage} status exchange returned an invalid status")
            if item[2] is not None:
                failures.append(f"rank {item[0]}: {item[2]}")
        if failures:
            raise RuntimeError(f"IPC {stage} failed across ranks: {'; '.join(failures)}") from error
        return value

    def _synchronize_device(self) -> None:
        import torch

        torch.cuda.synchronize(self.device)

    def _barrier(self) -> None:
        if self.world_size == 1 or self.group is None:
            return
        if hasattr(self.group, "barrier"):
            self.group.barrier()
            return
        import torch.distributed as dist

        dist.barrier(group=self.group)

    def _exchange(self, value: Any) -> list[Any]:
        if self.world_size == 1:
            return [value]
        if self.group is None:
            raise RuntimeError("world_size > 1 requires an initialized process group")
        if hasattr(self.group, "all_gather_object"):
            return list(self.group.all_gather_object(value))
        import torch.distributed as dist

        values = [None] * self.world_size
        dist.all_gather_object(values, value, group=self.group)
        return values

    def _require_open(self) -> None:
        if self._closed or not self._base:
            raise RuntimeError("IPC allocator is not initialized or has been closed")


# Runtime implementations own their registration; the factory has no IPC branch.
IPC_BACKEND = register_distributed_backend(
    "ipc", 1, IpcAllocator,
)
