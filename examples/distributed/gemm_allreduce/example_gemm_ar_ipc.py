"""Two-kernel HCU IPC GEMM + AllReduce example.

Launch with two same-node P2P-capable HCU ranks, for example:

  HSA_USE_SVM=0 HSA_FORCE_FINE_GRAIN_PCIE=1 \
  mpirun --allow-run-as-root -np 2 python3 example_gemm_ar_ipc.py
"""

import gc
import os

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
from tilelang.distributed.backends.ipc import IpcAllocator


M = N = K = 64
IPC_ARENA_BYTES = 32768


def gemm_kernel():
    """Rank-local FP32 GEMM writing its partial-C to an IPC arena view."""

    @T.prim_func
    def main(
        a: T.Tensor((M, K), "float32"),
        b: T.Tensor((K, N), "float32"),
        partial_c: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, 1, threads=128):
            a_shared = T.alloc_shared((M, 32), "float32")
            b_shared = T.alloc_shared((32, N), "float32")
            c_local = T.alloc_fragment((M, N), "float32")
            T.clear(c_local)
            for k in T.Pipelined(K // 32, num_stages=0):
                T.copy(a[0, k * 32], a_shared)
                T.copy(b[k * 32, 0], b_shared)
                T.gemm(a_shared, b_shared, c_local)
            T.copy(c_local, partial_c)

    return main


def allreduce_kernel():
    """Pull the peer partial-C to LDS and add it to this rank's partial-C."""

    @T.prim_func
    def main(
        output: T.Tensor((M, N), "float32"),
        partial_c: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=128):
            peer_partial = T.alloc_shared((M, N), "float32")
            T.get_block(
                T.address_of(partial_c[0, 0]),
                T.address_of(peer_partial[0, 0]),
                M * N,
                T.get_rank() ^ 1,
            )
            T.sync_threads()
            for i, j in T.Parallel(M, N):
                output[i, j] = partial_c[i, j] + peer_partial[i, j]

    return main


def _rank_env(name: str) -> int:
    mpi_name = {
        "RANK": "OMPI_COMM_WORLD_RANK",
        "WORLD_SIZE": "OMPI_COMM_WORLD_SIZE",
        "LOCAL_RANK": "OMPI_COMM_WORLD_LOCAL_RANK",
    }[name]
    return int(os.environ.get(name) or os.environ[mpi_name])


def main() -> None:
    rank = _rank_env("RANK")
    world_size = _rank_env("WORLD_SIZE")
    local_rank = _rank_env("LOCAL_RANK")
    if world_size != 2:
        raise RuntimeError(f"this IPC example requires exactly two ranks, got {world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    allocator = IpcAllocator(
        IPC_ARENA_BYTES,
        rank=rank,
        world_size=world_size,
        device=local_rank,
        group=dist.group.WORLD,
    )
    try:
        allocator.initialize()
        arch = torch.cuda.get_device_properties(local_rank).gcnArchName.split(":", 1)[0]

        # Kernel 1 is an ordinary HCU GEMM; out_idx=[] lets it write the
        # allocator-backed partial-C supplied by the caller.
        a = torch.full((M, K), float(rank + 1), device=f"cuda:{local_rank}")
        b = torch.full((K, N), 2.0, device=f"cuda:{local_rank}")
        partial_c = tilelang.tensor((M, N), torch.float32, allocator=allocator)
        gemm = tilelang.compile(gemm_kernel(), out_idx=[], target={"kind": "hcu", "mcpu": arch})
        gemm(a, b, partial_c)
        torch.cuda.synchronize(local_rank)
        expected_partial = torch.full_like(partial_c, float((rank + 1) * 2 * K))
        if not torch.allclose(partial_c, expected_partial, atol=1e-3, rtol=1e-3):
            raise RuntimeError(f"rank {rank}: local GEMM partial-C is incorrect")
        dist.barrier()

        # Kernel 2 is separately compiled with IPC lowering and needs one
        # metadata initialization for this allocator generation.
        allreduce = tilelang.compile(
            allreduce_kernel(),
            out_idx=[0],
            target={"kind": "hcu", "mcpu": arch, "dist_backend": "ipc"},
        )
        allreduce.initialize(allocator)
        output = allreduce(partial_c)
        torch.cuda.synchronize(local_rank)
        expected = torch.full_like(output, float(6 * K))
        if not torch.allclose(output, expected, atol=1e-3, rtol=1e-3):
            raise RuntimeError(f"rank {rank}: GEMM+AllReduce output is incorrect")
        print(f"ipc-gemm-allreduce rank={rank} value={output[0, 0].item()}", flush=True)
        dist.barrier()

        # tensor() views are non-owning: release them before closing the arena.
        del output, allreduce, expected, expected_partial, partial_c, gemm, a, b
        gc.collect()
        torch.cuda.synchronize(local_rank)
        dist.barrier()
    finally:
        allocator.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
