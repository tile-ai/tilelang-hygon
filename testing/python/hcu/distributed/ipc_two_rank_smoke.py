"""Run with: torchrun --standalone --nproc_per_node=2 ipc_two_rank_smoke.py."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from tilelang.distributed.backends.ipc import IpcAllocator


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError(f"IPC smoke requires exactly 2 ranks, got {world_size}")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(f"IPC smoke requires {world_size} HCU devices")

    torch.cuda.set_device(local_rank)
    print(f"ipc-smoke rank={rank} stage=set-device device={local_rank}", flush=True)
    dist.init_process_group("gloo")
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
    finally:
        if allocator is not None:
            allocator.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
