"""Two-rank HCU IPC device remote-get smoke, launched by MPI."""


import os

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
from tilelang.distributed.backends.ipc import IpcAllocator


N = 128


def remote_get_kernel():
    @T.prim_func
    def main(out: T.Tensor((N,), "float32"), src: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=64):
            rank = T.get_rank()
            T.get_block(
                T.address_of(src[0]),
                T.address_of(out[0]),
                N,
                rank ^ 1,
            )

    return main


def main() -> None:
    rank = int(os.environ.get("RANK") or os.environ["OMPI_COMM_WORLD_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE") or os.environ["OMPI_COMM_WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK") or os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError(f"IPC remote-get smoke requires exactly 2 ranks, got {world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    allocator = IpcAllocator(4096, rank=rank, world_size=world_size, device=local_rank, group=dist.group.WORLD)
    try:
        allocator.initialize()
        arch = torch.cuda.get_device_properties(local_rank).gcnArchName.split(":", 1)[0]
        src = tilelang.tensor((N,), torch.float32, allocator=allocator)
        src.fill_(float(rank + 1))
        torch.cuda.synchronize(local_rank)
        if not torch.all(src == float(rank + 1)):
            raise RuntimeError(f"rank {rank}: IPC tensor view write failed: {src.cpu().tolist()}")
        print(f"ipc-remote-get rank={rank} source={src[0].item()}", flush=True)
        dist.barrier()

        kernel = tilelang.compile(
            remote_get_kernel(),
            out_idx=[0],
            target={"kind": "hcu", "mcpu": arch, "dist_backend": "ipc"},
        )
        kernel.initialize(allocator)
        torch.cuda.synchronize(local_rank)
        out = kernel(src)
        torch.cuda.synchronize(local_rank)
        expected = float((rank ^ 1) + 1)
        if not torch.all(out == expected):
            raise RuntimeError(f"rank {rank}: remote get result {out.cpu().tolist()} != {expected}")
        print(f"ipc-remote-get rank={rank} value={out[0].item()}", flush=True)
        dist.barrier()
    finally:
        allocator.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
