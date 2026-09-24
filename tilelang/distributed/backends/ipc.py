"""HIP IPC arena and peer mapping lifecycle for HCU distributed kernels."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Protocol

from ..allocator import Allocation, DistributedAllocator


class _IpcRuntime(Protocol):
    def malloc(self, nbytes: int) -> int: ...
    def free(self, ptr: int) -> None: ...
    def create_handle(self, ptr: int) -> bytes: ...
    def open_handle(self, handle: bytes) -> int: ...
    def close_handle(self, ptr: int) -> None: ...
    def can_access_peer(self, device: int, peer_device: int) -> bool: ...


@contextmanager
def tvm_hcu_stream(device: int, stream: int, *, get_stream=None, set_stream=None):
    """Temporarily bind TVM's ROCm device stream to a PyTorch HIP stream."""
    if get_stream is None or set_stream is None:
        from tilelang import tvm

        get_stream = tvm.ffi.get_global_func("tl.hcu.ipc.get_tvm_stream")
        set_stream = tvm.ffi.get_global_func("tl.hcu.ipc.set_tvm_stream")
    previous = int(get_stream(device))
    set_stream(device, int(stream))
    try:
        yield
    finally:
        set_stream(device, previous)


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
                 group: Any = None, alignment: int = 256, runtime: _IpcRuntime | None = None,
                 synchronize: Callable[[], None] | None = None) -> None:
        if size <= 0 or alignment <= 0:
            raise ValueError("size and alignment must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.size, self.rank, self.world_size, self.device = int(size), int(rank), int(world_size), int(device)
        self.group, self.alignment = group, int(alignment)
        self._runtime = runtime or _FfiIpcRuntime()
        self._synchronize = synchronize or self._synchronize_device
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
        # Explicit handle/device lists are useful for local tests and callers
        # that perform their own exchange.  Otherwise every local stage reports
        # success or failure before ranks advance to the next collective.
        collective = self.world_size > 1 and (handles is None or device_ids is None)
        try:
            # HIP IPC allocation and handle import are device-context sensitive.
            # PyTorch uses the CUDA namespace for both CUDA and HIP builds.
            import torch

            self._run_initialization_stage(
                "device selection", lambda: torch.cuda.set_device(self.device), collective
            )

            def allocate_local() -> None:
                self._base = self._runtime.malloc(self.size)

            self._run_initialization_stage("allocation", allocate_local, collective)

            if handles is None:
                local_handle = self._run_initialization_stage(
                    "handle creation", lambda: self._runtime.create_handle(self._base), collective
                )
                handles = self._run_initialization_stage(
                    "handle exchange", lambda: self._exchange(local_handle), collective
                )
            if device_ids is None:
                device_ids = self._run_initialization_stage(
                    "device exchange", lambda: self._exchange(self.device), collective
                )

            def map_peers() -> None:
                if len(handles) != self.world_size or len(device_ids) != self.world_size:
                    raise RuntimeError("IPC collective exchange returned an unexpected world size")
                self._peer_bases = [0] * self.world_size
                for peer, (handle, peer_device) in enumerate(zip(handles, device_ids)):
                    if peer == self.rank:
                        self._peer_bases[peer] = self._base
                    else:
                        if not self._runtime.can_access_peer(self.device, int(peer_device)):
                            raise RuntimeError(
                                f"HCU device {self.device} cannot access peer rank {peer} device {peer_device}"
                            )
                        self._peer_bases[peer] = self._runtime.open_handle(handle)

            self._run_initialization_stage("peer mapping", map_peers, collective)
            self._generation += 1
        except Exception:
            self.close(collective=False)
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

    def close(self, *, collective: bool = True) -> None:
        """Release an IPC arena after all ranks stopped using its mappings."""
        if self._closed:
            return
        self._closed = True
        try:
            if collective and self._base:
                self._synchronize()
                self._barrier()
        finally:
            for peer, ptr in enumerate(self._peer_bases):
                if peer != self.rank and ptr:
                    try: self._runtime.close_handle(ptr)
                    except Exception: pass
            self._peer_bases = []
            if self._base:
                try: self._runtime.free(self._base)
                finally: self._base = 0
            self._generation += 1

    def _run_initialization_stage(
        self, stage: str, operation: Callable[[], Any], collective: bool
    ) -> Any:
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


class IpcMetadataInitializer:
    """Initialize an IPC module's device metadata once per allocator generation."""

    def __init__(self) -> None:
        self._initialized: dict[tuple[int, int], int] = {}
        # Keep both the PyTorch owner and its TVM DLPack view alive until this
        # module/device pair is initialized again for a newer generation.
        self._metadata_payloads: dict[tuple[int, int], tuple[Any, Any]] = {}

    def initialize(self, module: Any, allocator: IpcAllocator, *, metadata_tensor: Any = None) -> bool:
        """Launch the module helper when its cached allocator generation is stale.

        Returns ``True`` when a helper launch occurred and ``False`` when the
        module/device pair was already initialized for this generation.
        """
        key = (id(module), allocator.device)
        if self._initialized.get(key) == allocator.generation:
            return False
        helper = module.get_function("__tilelang_init_ipc_metadata", query_imports=False)
        if helper is None:
            raise RuntimeError("module does not contain the IPC metadata helper")
        payload = allocator.metadata
        if len(payload) != allocator.world_size + 2:
            raise RuntimeError("IPC metadata count must equal world_size + 2")
        if not 0 <= payload[0] < payload[1] or payload[1] != allocator.world_size:
            raise RuntimeError("IPC metadata rank/world_size is inconsistent with the allocator")
        if len(payload) > 1024:
            raise RuntimeError("IPC metadata exceeds the device metadata capacity")
        if metadata_tensor is None:
            import torch

            metadata_tensor = torch.tensor(payload, dtype=torch.uint64, device=f"cuda:{allocator.device}")
        import torch

        if not isinstance(metadata_tensor, torch.Tensor):
            helper_arg = metadata_tensor
        else:
            import ctypes

            # ROCm Module FunctionInfo declares an opaque handle.  A ctypes
            # void pointer maps to HANDLE_TO_HANDLE; passing a Tensor/NDArray
            # would instead pass a host-side descriptor address.
            helper_arg = ctypes.c_void_p(metadata_tensor.data_ptr())
        helper(helper_arg, len(payload))
        self._metadata_payloads[key] = (metadata_tensor, helper_arg)
        self._initialized[key] = allocator.generation
        return True
