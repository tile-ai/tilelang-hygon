"""Two-rank HCU IPC device remote-get smoke, launched by MPI."""


import os

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
from tilelang.distributed.backends.ipc import IpcAllocator


N = 128
GEMM_M = 64
GEMM_N = 64
GEMM_K = 64


def allreduce_kernel():
    @T.prim_func
    def main(out: T.Tensor((N,), "float32"), src: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=64):
            peer = T.alloc_shared((N,), "float32")
            T.get_block(
                T.address_of(src[0]),
                T.address_of(peer[0]),
                N,
                T.get_rank() ^ 1,
            )
            T.sync_threads()
            for i in T.Parallel(N):
                out[i] = src[i] + peer[i]

    return main


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


def gemm_kernel():
    """Local GEMM writing directly into an IPC-arena partial-C buffer."""

    @T.prim_func
    def main(
        a: T.Tensor((GEMM_M, GEMM_K), "float32"),
        b: T.Tensor((GEMM_K, GEMM_N), "float32"),
        c: T.Tensor((GEMM_M, GEMM_N), "float32"),
    ):
        with T.Kernel(1, 1, threads=128):
            a_shared = T.alloc_shared((GEMM_M, 32), "float32")
            b_shared = T.alloc_shared((32, GEMM_N), "float32")
            c_local = T.alloc_fragment((GEMM_M, GEMM_N), "float32")
            T.clear(c_local)
            for k in T.Pipelined(2, num_stages=0):
                T.copy(a[0, k * 32], a_shared)
                T.copy(b[k * 32, 0], b_shared)
                T.gemm(a_shared, b_shared, c_local)
            T.copy(c_local, c)

    return main


def gemm_allreduce_kernel():
    """IPC pull AllReduce of two rank-local GEMM partial-C buffers."""

    @T.prim_func
    def main(
        out: T.Tensor((GEMM_M, GEMM_N), "float32"),
        src: T.Tensor((GEMM_M, GEMM_N), "float32"),
    ):
        with T.Kernel(1, threads=128):
            peer = T.alloc_shared((GEMM_M, GEMM_N), "float32")
            T.get_block(
                T.address_of(src[0, 0]),
                T.address_of(peer[0, 0]),
                GEMM_M * GEMM_N,
                T.get_rank() ^ 1,
            )
            T.sync_threads()
            for i, j in T.Parallel(GEMM_M, GEMM_N):
                out[i, j] = src[i, j] + peer[i, j]

    return main


def main() -> None:
    rank = int(os.environ.get("RANK") or os.environ["OMPI_COMM_WORLD_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE") or os.environ["OMPI_COMM_WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK") or os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError(f"IPC remote-get smoke requires exactly 2 ranks, got {world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    allocator = IpcAllocator(32768, rank=rank, world_size=world_size, device=local_rank, group=dist.group.WORLD)
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
        allreduce = tilelang.compile(
            allreduce_kernel(),
            out_idx=[0],
            target={"kind": "hcu", "mcpu": arch, "dist_backend": "ipc"},
        )
        allreduce.initialize(allocator)
        reduced = allreduce(src)
        torch.cuda.synchronize(local_rank)
        if not torch.all(reduced == 3.0):
            raise RuntimeError(f"rank {rank}: allreduce result {reduced.cpu().tolist()} != 3.0")
        print(f"ipc-allreduce rank={rank} value={reduced[0].item()}", flush=True)
        dist.barrier()

        # The GEMM output is supplied as an input (out_idx=[]), so the first
        # kernel writes into the same IPC arena consumed by the second kernel.
        a = torch.full((GEMM_M, GEMM_K), float(rank + 1), device=f"cuda:{local_rank}")
        b = torch.full((GEMM_K, GEMM_N), 2.0, device=f"cuda:{local_rank}")
        partial = tilelang.tensor((GEMM_M, GEMM_N), torch.float32, allocator=allocator)
        gemm = tilelang.compile(
            gemm_kernel(), out_idx=[], target={"kind": "hcu", "mcpu": arch}
        )
        gemm(a, b, partial)
        torch.cuda.synchronize(local_rank)
        expected_partial = torch.full_like(partial, float((rank + 1) * 2 * GEMM_K))
        if not torch.allclose(partial, expected_partial, atol=1e-3, rtol=1e-3):
            raise RuntimeError(f"rank {rank}: GEMM partial-C is incorrect")
        dist.barrier()

        gemm_allreduce = tilelang.compile(
            gemm_allreduce_kernel(),
            out_idx=[0],
            target={"kind": "hcu", "mcpu": arch, "dist_backend": "ipc"},
        )
        gemm_allreduce.initialize(allocator)
        result = gemm_allreduce(partial)
        torch.cuda.synchronize(local_rank)
        expected = torch.full_like(result, float(6 * GEMM_K))
        if not torch.allclose(result, expected, atol=1e-3, rtol=1e-3):
            raise RuntimeError(f"rank {rank}: GEMM+AllReduce result is incorrect")
        print(f"ipc-gemm-allreduce rank={rank} value={result[0, 0].item()}", flush=True)
        dist.barrier()

        # tilelang.tensor creates non-owning views. Drop every view before
        # closing the backing IPC arena; otherwise HCU cleanup can dereference
        # an already-unmapped HIP allocation during process teardown.
        del result, expected, gemm_allreduce, partial, expected_partial, gemm, a, b
        del reduced, allreduce, out, kernel, src
        import gc
        gc.collect()
        torch.cuda.synchronize(local_rank)
        dist.barrier()
    finally:
        allocator.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
