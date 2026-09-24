"""Multi-rank HCU IPC all-peer pull AllReduce smoke, launched by MPI."""

import gc
import os

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
from tilelang.distributed.backends.ipc import IpcAllocator


TILE = 128
NUM_TILES = 4
N = TILE * NUM_TILES
ITERATIONS = 3


def allreduce_kernel():
    @T.prim_func
    def main(out: T.Tensor((N,), "float32"), src: T.Tensor((N,), "float32")):
        with T.Kernel(NUM_TILES, threads=64) as tile:
            peer = T.alloc_shared((TILE,), "float32")
            for i in T.Parallel(TILE):
                out[tile * TILE + i] = 0.0
            T.sync_threads()
            for src_pe in T.serial(T.get_num_ranks()):
                T.get_block(
                    T.address_of(src[tile * TILE]),
                    T.address_of(peer[0]),
                    TILE,
                    src_pe,
                )
                T.sync_threads()
                for i in T.Parallel(TILE):
                    out[tile * TILE + i] += peer[i]
                T.sync_threads()

    return main


def allreduce_bf16_kernel():
    """All-peer BF16 pull with FP32 accumulation."""

    @T.prim_func
    def main(out: T.Tensor((N,), "float32"), src: T.Tensor((N,), "bfloat16")):
        with T.Kernel(NUM_TILES, threads=64) as tile:
            peer = T.alloc_shared((TILE,), "bfloat16")
            for i in T.Parallel(TILE):
                out[tile * TILE + i] = 0.0
            T.sync_threads()
            for src_pe in T.serial(T.get_num_ranks()):
                T.get_block(
                    T.address_of(src[tile * TILE]),
                    T.address_of(peer[0]),
                    TILE,
                    src_pe,
                )
                T.sync_threads()
                for i in T.Parallel(TILE):
                    out[tile * TILE + i] += T.Cast("float32", peer[i])
                T.sync_threads()

    return main


def _rank_env(name: str) -> int:
    mpi_name = {
        "RANK": "OMPI_COMM_WORLD_RANK",
        "WORLD_SIZE": "OMPI_COMM_WORLD_SIZE",
        "LOCAL_RANK": "OMPI_COMM_WORLD_LOCAL_RANK",
    }[name]
    return int(os.environ.get(name) or os.environ[mpi_name])


def main() -> None:
    rank, world_size, local_rank = (_rank_env("RANK"), _rank_env("WORLD_SIZE"), _rank_env("LOCAL_RANK"))
    if world_size < 2:
        raise RuntimeError(f"all-peer IPC smoke requires at least two ranks, got {world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    allocator = IpcAllocator(4096, rank=rank, world_size=world_size, device=local_rank, group=dist.group.WORLD)
    try:
        allocator.initialize()
        arch = torch.cuda.get_device_properties(local_rank).gcnArchName.split(":", 1)[0]
        src = tilelang.tensor((N,), torch.float32, allocator=allocator)
        allreduce = tilelang.compile(
            allreduce_kernel(),
            out_idx=[0],
            target={"kind": "hcu", "mcpu": arch, "dist_backend": "ipc"},
        )
        # One metadata initialization must remain valid across repeated launches.
        allreduce.initialize(allocator)
        out = None
        for iteration in range(ITERATIONS):
            src.fill_(float(rank + 1 + iteration))
            torch.cuda.synchronize(local_rank)
            dist.barrier()
            out = allreduce(src)
            torch.cuda.synchronize(local_rank)
            expected = float(world_size * (world_size + 1) // 2 + iteration * world_size)
            if not torch.all(out == expected):
                raise RuntimeError(
                    f"rank {rank}: iteration {iteration} result {out.cpu().tolist()} != {expected}"
                )
        print(
            f"ipc-allpeer-allreduce rank={rank} world_size={world_size} "
            f"tiles={NUM_TILES} iterations={ITERATIONS} value={out[0].item()}",
            flush=True,
        )
        dist.barrier()

        src_bf16 = tilelang.tensor((N,), torch.bfloat16, allocator=allocator)
        src_bf16.fill_(float(rank + 1))
        torch.cuda.synchronize(local_rank)
        bf16_allreduce = tilelang.compile(
            allreduce_bf16_kernel(),
            out_idx=[0],
            target={"kind": "hcu", "mcpu": arch, "dist_backend": "ipc"},
        )
        bf16_allreduce.initialize(allocator)
        bf16_out = bf16_allreduce(src_bf16)
        torch.cuda.synchronize(local_rank)
        bf16_expected = float(world_size * (world_size + 1) // 2)
        if not torch.all(bf16_out == bf16_expected):
            raise RuntimeError(
                f"rank {rank}: BF16 all-peer result {bf16_out.cpu().tolist()} != {bf16_expected}"
            )
        print(f"ipc-allpeer-bf16 rank={rank} world_size={world_size} value={bf16_out[0].item()}", flush=True)
        dist.barrier()

        del bf16_out, bf16_allreduce, src_bf16, out, allreduce, src
        gc.collect()
        torch.cuda.synchronize(local_rank)
        dist.barrier()
    finally:
        allocator.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
