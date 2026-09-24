"""Run with torchrun or the MPI command in test_ipc_runtime.py."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from tilelang.distributed.backends.ipc import IpcAllocator


class _FailHandleCreationRuntime:
    def __init__(self, runtime, rank: int) -> None:
        self._runtime = runtime
        self._rank = rank

    def malloc(self, nbytes: int) -> int:
        return self._runtime.malloc(nbytes)

    def free(self, ptr: int) -> None:
        self._runtime.free(ptr)

    def create_handle(self, ptr: int) -> bytes:
        if self._rank == 1:
            raise RuntimeError("injected handle creation failure")
        return self._runtime.create_handle(ptr)

    def open_handle(self, handle: bytes) -> int:
        return self._runtime.open_handle(handle)

    def close_handle(self, ptr: int) -> None:
        self._runtime.close_handle(ptr)

    def can_access_peer(self, device: int, peer_device: int) -> bool:
        return self._runtime.can_access_peer(device, peer_device)


def main() -> None:
    rank = int(os.environ.get("RANK") or os.environ["OMPI_COMM_WORLD_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE") or os.environ["OMPI_COMM_WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK") or os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError(f"IPC smoke requires exactly 2 ranks, got {world_size}")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(f"IPC smoke requires {world_size} HCU devices")

    torch.cuda.set_device(local_rank)
    print(f"ipc-smoke rank={rank} stage=set-device device={local_rank}", flush=True)
    # torchrun exports RANK/WORLD_SIZE, whereas Open MPI exposes its rank
    # variables under OMPI_COMM_WORLD_*. Pass normalized values rather than
    # depending on env:// to find torchrun-specific variable names.
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"ipc-smoke rank={rank} stage=process-group", flush=True)
    allocator = None
    try:
        allocator = IpcAllocator(
            4096, rank=rank, world_size=world_size, device=local_rank, group=dist.group.WORLD
        )
        print(f"ipc-smoke rank={rank} stage=allocator-created", flush=True)
        allocator.initialize()
        print(f"ipc-smoke rank={rank} stage=allocator-initialized", flush=True)
        metadata = allocator.metadata
        if len(metadata) != 2 + world_size:
            raise RuntimeError(f"rank {rank}: invalid metadata length {len(metadata)}")
        if metadata[2 + rank] != allocator.base_ptr:
            raise RuntimeError(f"rank {rank}: local peer base does not match arena base")
        print(json.dumps({"rank": rank, "device": local_rank, "metadata": metadata}), flush=True)
        dist.barrier()
        allocator.close()
        allocator = None
        dist.barrier()

        # Inject a local failure on rank 1 before handle exchange.  The status
        # collective must make rank 0 fail at the same stage and both ranks
        # must release the allocation made in the preceding stage.
        failing = IpcAllocator(
            4096, rank=rank, world_size=world_size, device=local_rank, group=dist.group.WORLD
        )
        failing._runtime = _FailHandleCreationRuntime(failing._runtime, rank)
        error = None
        try:
            failing.initialize()
        except RuntimeError as exc:
            error = str(exc)
        if error is None or "IPC handle creation failed across ranks" not in error:
            raise RuntimeError(f"rank {rank}: failure was not propagated: {error!r}")
        if failing._base != 0 or failing._peer_bases:
            raise RuntimeError(f"rank {rank}: failed initialization leaked IPC resources")
        errors = [None] * world_size
        dist.all_gather_object(errors, error)
        if not all("rank 1: RuntimeError: injected handle creation failure" in item for item in errors):
            raise RuntimeError(f"ranks observed inconsistent propagated errors: {errors}")
        print(f"ipc-smoke rank={rank} stage=failure-propagated", flush=True)
        dist.barrier()

        closing = IpcAllocator(
            4096, rank=rank, world_size=world_size, device=local_rank, group=dist.group.WORLD
        )
        closing.initialize()
        if rank == 1:
            def fail_synchronize():
                raise RuntimeError("injected close synchronization failure")

            closing._synchronize = fail_synchronize
        close_error = None
        try:
            closing.close()
        except RuntimeError as exc:
            close_error = str(exc)
        if close_error is None or "IPC close synchronization failed across ranks" not in close_error:
            raise RuntimeError(f"rank {rank}: close failure was not propagated: {close_error!r}")
        if closing._base != 0 or closing._peer_bases:
            raise RuntimeError(f"rank {rank}: failed collective close leaked IPC resources")
        close_errors = [None] * world_size
        dist.all_gather_object(close_errors, close_error)
        if not all(
            "rank 1: RuntimeError: injected close synchronization failure" in item
            for item in close_errors
        ):
            raise RuntimeError(f"ranks observed inconsistent close errors: {close_errors}")
        print(f"ipc-smoke rank={rank} stage=close-failure-propagated", flush=True)
        dist.barrier()
    finally:
        if allocator is not None:
            allocator.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
