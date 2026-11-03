"""Persistent GEMM + reduce-scatter with system-scope IPC ready flags.

Two TileLang kernels: persistent GEMM/push and waiting local reduce, followed
by a stream-ordered RCCL barrier protecting inbox reuse. Payload and ready flags
use hipDeviceMallocFinegrained. Two ranks use row order; four ranks rotate tile
groups by source rank. No host flag writes after setup. Signal waits use standard blocking semantics.
"""

import tilelang.language as T


def gemm_push_waves(
    M,
    N,
    K,
    rank,
    world_size,
    block_M=256,
    block_N=256,
    block_K=16,
    dtype="float16",
    accum_dtype="float32",
    steady_wait=8,
    shared_0_wait=6,
    persistent_blocks=80,
):
    if (block_M, block_N, block_K, dtype, accum_dtype) != (256, 256, 16, "float16", "float32"):
        raise ValueError("push_waves retains the original 256x256x16 FP16/FP32 GEMM")
    if world_size < 1 or M % (world_size * block_M) or N % block_N or K < 64 or K % 64:
        raise ValueError("push_waves requires full M/N tiles, rank-aligned row shards and K divisible by 64")
    if not isinstance(persistent_blocks, int) or persistent_blocks < 1:
        raise ValueError("persistent_blocks must be a positive integer")
    m_tiles, n_tiles = M // block_M, N // block_N
    total_tiles = m_tiles * n_tiles
    # gfx936 on this machine reports 80 CUs. This is a CTA count, not a CU
    # affinity: each CTA processes grid-strided output tiles below.
    grid_size = min(persistent_blocks, total_tiles)
    rows_per_rank = M // world_size
    transpose_B = False
    k_groups = K // (4 * block_K)

    def tile_coordinates(index):
        # Rotate traversal only: tile ownership and the inbox source plane stay fixed.
        if world_size == 4 and rank != 0:
            index = (index + rank * (total_tiles // world_size)) % total_tiles
        return index % n_tiles, index // n_tiles

    @T.macro
    def async_copy_a(A, A_shared, by, k_tile):
        T.async_copy(
            A[
                by * block_M : (by + 1) * block_M,
                k_tile * block_K : (k_tile + 1) * block_K,
            ],
            A_shared,
        )

    @T.macro
    def async_copy_b(B, B_shared, bx, k_tile):
        T.async_copy(
            B[
                k_tile * block_K : (k_tile + 1) * block_K,
                bx * block_N : (bx + 1) * block_N,
            ],
            B_shared,
        )

    @T.macro
    def store_output(acc, s0, s1, s2, s3, part):
        # Eight compute waves contribute 16 rows from each 64-row band.
        # Each band reuses exactly one 8 KiB input-stage allocation.
        for i, j in T.Parallel(block_M, block_N):
            if i % 64 // 16 == part:
                ptr = T.if_then_else(
                    i // 64 == 0,
                    T.address_of(s0[0, 0]),
                    T.if_then_else(
                        i // 64 == 1, T.address_of(s1[0, 0]), T.if_then_else(i // 64 == 2, T.address_of(s2[0, 0]), T.address_of(s3[0, 0]))
                    ),
                )
                shared = T.make_tensor_from_addr(ptr, (16, block_N), dtype, storage_scope="shared.dyn")
                shared[i % 16, j] = T.cast(acc[i, j], dtype)
        T.s_waitcnt(0, "lgkmcnt")

    @T.macro
    def push_output(payload, peers, by, bx, part):
        remote = T.make_tensor_from_addr(peers[by * block_M // rows_per_rank], (world_size, rows_per_rank, N // 2), "uint32")
        # Each register payload contains four 16-row strips, one per 64-row band.
        # Restore their global rows; each copy keeps the N dimension contiguous.
        for band in T.unroll(4):
            T.copy(
                payload[band * 16 : (band + 1) * 16, :],
                remote[
                    rank,
                    by * block_M % rows_per_rank + band * 64 + part * 16 : by * block_M % rows_per_rank + band * 64 + part * 16 + 16,
                    bx * (block_N // 2) : (bx + 1) * (block_N // 2),
                ],
            )
        T.s_waitcnt(0, "lgkmcnt")

    @T.macro
    def load_payload(s0, s1, s2, s3, payload):
        # Bind the loop layout to the push-wave range (threads 512..767).
        # An unbound buffer annotation would use the wrong thread origin.
        for i, j in T.Parallel(64, 128, loop_layout=T.Fragment((64, 128), lambda i, j: ((i % 8) * 32 + j // 4, (i // 8) * 4 + j % 4))):
            ptr = T.if_then_else(
                i // 16 == 0,
                T.address_of(s0[0, 0]),
                T.if_then_else(
                    i // 16 == 1, T.address_of(s1[0, 0]), T.if_then_else(i // 16 == 2, T.address_of(s2[0, 0]), T.address_of(s3[0, 0]))
                ),
            )
            shared = T.make_tensor_from_addr(ptr, (16, block_N), dtype, storage_scope="shared.dyn")
            payload[i, j] = T.cast(T.reinterpret(shared[i % 16, j * 2], "uint16"), "uint32") | (
                T.cast(T.reinterpret(shared[i % 16, j * 2 + 1], "uint16"), "uint32") << 16
            )
        T.s_waitcnt(0, "lgkmcnt")

    @T.macro
    def receive_output(p0, p1, p2, p3, a0, a1, a2, a3, b0, b1, b2, b3):
        T.sync_warp()
        load_payload(a0, a1, a2, a3, p0)
        T.sync_warp()
        load_payload(b0, b1, b2, b3, p1)
        T.sync_warp()
        load_payload(a0, a1, a2, a3, p2)
        T.sync_warp()
        load_payload(b0, b1, b2, b3, p3)
        T.sync_warp()

    @T.macro
    def wait_first_compute():
        for _ in T.serial(6):
            T.sync_warp()
        for _kg in T.serial(k_groups - 1):
            for _ in T.serial(7):
                T.sync_warp()
        for _ in T.serial(3):
            T.sync_warp()

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        peers: T.Tensor((world_size,), "uint64"),
        control: T.Tensor((grid_size + 1, 32), "uint64"),
    ):
        with T.Kernel(grid_size, threads=768) as block_id:
            tx = T.get_thread_binding()
            warp_idx = T.get_warp_idx()
            turns = T.ceildiv(total_tiles - block_id, grid_size)
            # Preserve the original eight independent LDS stages. Reuse each
            # stage through an 8 KiB linear output view only after input reads.
            A_shared_0 = T.alloc_shared((block_M, block_K), dtype)
            A_shared_1 = T.alloc_shared((block_M, block_K), dtype)
            A_shared_2 = T.alloc_shared((block_M, block_K), dtype)
            A_shared_3 = T.alloc_shared((block_M, block_K), dtype)
            B_shared_0 = T.alloc_shared((block_K, block_N), dtype)
            B_shared_1 = T.alloc_shared((block_K, block_N), dtype)
            B_shared_2 = T.alloc_shared((block_K, block_N), dtype)
            B_shared_3 = T.alloc_shared((block_K, block_N), dtype)
            out_a0 = T.make_tensor_from_addr(T.address_of(A_shared_0[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            out_a1 = T.make_tensor_from_addr(T.address_of(A_shared_1[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            out_a2 = T.make_tensor_from_addr(T.address_of(A_shared_2[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            out_a3 = T.make_tensor_from_addr(T.address_of(A_shared_3[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            out_b0 = T.make_tensor_from_addr(T.address_of(B_shared_0[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            out_b1 = T.make_tensor_from_addr(T.address_of(B_shared_1[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            out_b2 = T.make_tensor_from_addr(T.address_of(B_shared_2[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            out_b3 = T.make_tensor_from_addr(T.address_of(B_shared_3[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            if tx < 512:
                A_local_0 = T.alloc_fragment((block_M, block_K), dtype)
                B_local_0 = T.alloc_fragment((block_K, block_N), dtype)
                A_local_1 = T.alloc_fragment((block_M, block_K), dtype)
                B_local_1 = T.alloc_fragment((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                # Persistent tile loop: block_id, block_id + grid_size, ...
                for turn in T.serial(turns):
                    bx, by = tile_coordinates(block_id + turn * grid_size)
                    T.clear(C_local)

                    T.sync_warp()
                    async_copy_a(A, A_shared_0, by, 0)
                    async_copy_b(B, B_shared_0, bx, 0)

                    T.sync_warp()
                    async_copy_a(A, A_shared_1, by, 1)
                    async_copy_b(B, B_shared_1, bx, 1)

                    T.sync_warp()
                    async_copy_a(A, A_shared_2, by, 2)
                    async_copy_b(B, B_shared_2, bx, 2)

                    T.sync_warp()
                    async_copy_a(A, A_shared_3, by, 3)
                    async_copy_b(B, B_shared_3, bx, 3)

                    T.ptx_wait_group(4)
                    T.sync_warp()

                    T.copy(A_shared_0, A_local_0)
                    T.copy(B_shared_0, B_local_0)
                    T.s_waitcnt(0, "lgkmcnt")
                    T.sync_warp()
                    T.sched_barrier()

                    if warp_idx < 4:
                        for kg in range(k_groups - 1):
                            k0 = kg * 4

                            # Phase 0
                            async_copy_a(A, A_shared_0, by, k0 + 4)
                            async_copy_b(B, B_shared_0, bx, k0 + 4)
                            T.copy(A_shared_1, A_local_1)
                            T.copy(B_shared_1, B_local_1)
                            T.ptx_wait_group(4)
                            T.sync_warp()

                            # Phase 1
                            T.s_waitcnt(steady_wait, "lgkmcnt")
                            T.sched_barrier()
                            T.gemm(A_local_0, B_local_0, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                            T.sched_barrier()
                            T.sync_warp()

                            # Phase 2
                            k1_if = kg * 4 + 1
                            async_copy_a(A, A_shared_1, by, k1_if + 4)
                            async_copy_b(B, B_shared_1, bx, k1_if + 4)
                            T.copy(A_shared_2, A_local_0)
                            T.copy(B_shared_2, B_local_0)
                            T.ptx_wait_group(4)
                            T.sync_warp()

                            # Phase 3
                            T.s_waitcnt(steady_wait, "lgkmcnt")
                            T.sched_barrier()
                            T.gemm(A_local_1, B_local_1, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                            T.sched_barrier()
                            T.sync_warp()

                            # Phase 4
                            k2_if = kg * 4 + 2
                            async_copy_a(A, A_shared_2, by, k2_if + 4)
                            async_copy_b(B, B_shared_2, bx, k2_if + 4)
                            T.copy(A_shared_3, A_local_1)
                            T.copy(B_shared_3, B_local_1)
                            T.ptx_wait_group(4)
                            T.sync_warp()

                            # Phase 5
                            T.s_waitcnt(steady_wait, "lgkmcnt")
                            T.sched_barrier()
                            T.gemm(A_local_0, B_local_0, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                            T.sched_barrier()
                            T.sync_warp()

                            # Phase 6
                            k3_if = kg * 4 + 3
                            async_copy_a(A, A_shared_3, by, k3_if + 4)
                            async_copy_b(B, B_shared_3, bx, k3_if + 4)
                            T.copy(A_shared_0, A_local_0)
                            T.copy(B_shared_0, B_local_0)
                            T.ptx_wait_group(4)
                            T.sync_warp()

                            T.s_waitcnt(shared_0_wait, "lgkmcnt")
                            T.sched_barrier()
                            T.gemm(A_local_1, B_local_1, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                            T.sched_barrier()
                    else:
                        for kg in range(k_groups - 1):
                            k0 = kg * 4

                            # Phase 0
                            T.copy(A_shared_1, A_local_1)
                            T.copy(B_shared_1, B_local_1)
                            T.call_extern("tl::promote_prio", dtype="void")
                            async_copy_a(A, A_shared_0, by, k0 + 4)
                            async_copy_b(B, B_shared_0, bx, k0 + 4)
                            T.call_extern("tl::restore_prio", dtype="void")
                            T.s_waitcnt(steady_wait, "lgkmcnt")
                            T.sched_barrier()
                            T.gemm(A_local_0, B_local_0, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                            T.sched_barrier()
                            T.ptx_wait_group(4)
                            T.sync_warp()

                            # Phase 1
                            k1_else = kg * 4 + 1
                            T.call_extern("tl::promote_prio", dtype="void")
                            async_copy_a(A, A_shared_1, by, k1_else + 4)
                            async_copy_b(B, B_shared_1, bx, k1_else + 4)
                            T.call_extern("tl::restore_prio", dtype="void")
                            T.copy(A_shared_2, A_local_0)
                            T.copy(B_shared_2, B_local_0)
                            T.ptx_wait_group(4)
                            T.sync_warp()

                            # Phase 2
                            T.s_waitcnt(steady_wait, "lgkmcnt")
                            T.sched_barrier()
                            T.gemm(A_local_1, B_local_1, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                            T.sched_barrier()
                            T.sync_warp()

                            # Phase 3
                            k2_else = kg * 4 + 2
                            T.call_extern("tl::promote_prio", dtype="void")
                            async_copy_a(A, A_shared_2, by, k2_else + 4)
                            async_copy_b(B, B_shared_2, bx, k2_else + 4)
                            T.call_extern("tl::restore_prio", dtype="void")
                            T.copy(A_shared_3, A_local_1)
                            T.copy(B_shared_3, B_local_1)
                            T.ptx_wait_group(4)
                            T.sync_warp()

                            # Phase 4
                            T.s_waitcnt(steady_wait, "lgkmcnt")
                            T.sched_barrier()
                            T.gemm(A_local_0, B_local_0, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                            T.sched_barrier()
                            T.sync_warp()

                            # Phase 5
                            k3_else = kg * 4 + 3
                            T.call_extern("tl::promote_prio", dtype="void")
                            async_copy_a(A, A_shared_3, by, k3_else + 4)
                            async_copy_b(B, B_shared_3, bx, k3_else + 4)
                            T.call_extern("tl::restore_prio", dtype="void")
                            T.copy(A_shared_0, A_local_0)
                            T.copy(B_shared_0, B_local_0)
                            T.ptx_wait_group(4)
                            T.sync_warp()

                            # Phase 6
                            T.s_waitcnt(shared_0_wait, "lgkmcnt")
                            T.sched_barrier()
                            T.gemm(A_local_1, B_local_1, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                            T.sched_barrier()
                            T.sync_warp()

                    T.copy(A_shared_1, A_local_1)
                    T.copy(B_shared_1, B_local_1)
                    T.s_waitcnt(6, "lgkmcnt")
                    T.ptx_wait_group(4)
                    T.sync_warp()
                    T.sched_barrier()
                    T.gemm(A_local_0, B_local_0, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                    T.sched_barrier()

                    T.ptx_wait_group(2)
                    T.sync_warp()
                    T.copy(A_shared_2, A_local_0)
                    T.copy(B_shared_2, B_local_0)
                    T.s_waitcnt(6, "lgkmcnt")
                    T.sched_barrier()
                    T.gemm(A_local_1, B_local_1, C_local, transpose_B=transpose_B, annotations={"trans_c": True})

                    T.ptx_wait_group(0)
                    T.sync_warp()
                    T.copy(A_shared_3, A_local_1)
                    T.copy(B_shared_3, B_local_1)
                    T.s_waitcnt(0, "lgkmcnt")
                    T.sched_barrier()
                    T.gemm(A_local_0, B_local_0, C_local, transpose_B=transpose_B, annotations={"trans_c": True})

                    T.gemm(A_local_1, B_local_1, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                    # Only completed outputs cross roles. Input LDS is dead here.
                    T.sync_warp()
                    store_output(C_local, out_a0, out_a1, out_a2, out_a3, 0)
                    T.sync_warp()
                    store_output(C_local, out_b0, out_b1, out_b2, out_b3, 1)
                    T.sync_warp()
                    store_output(C_local, out_a0, out_a1, out_a2, out_a3, 2)
                    T.sync_warp()
                    store_output(C_local, out_b0, out_b1, out_b2, out_b3, 3)
                    T.sync_warp()
                    T.sync_warp()  # All four payload fragments are in push registers.
            else:
                # Four packed fragments retain a complete FP16 output tile:
                # 128 KiB / 256 push threads = 128 uint32 registers per lane.
                p0 = T.alloc_fragment((block_M // 4, block_N // 2), "uint32")
                p1 = T.alloc_fragment((block_M // 4, block_N // 2), "uint32")
                p2 = T.alloc_fragment((block_M // 4, block_N // 2), "uint32")
                p3 = T.alloc_fragment((block_M // 4, block_N // 2), "uint32")
                # Peel the first tile: there is no previous output to send.
                wait_first_compute()
                T.sync_warp()
                receive_output(p0, p1, p2, p3, out_a0, out_a1, out_a2, out_a3, out_b0, out_b1, out_b2, out_b3)
                for turn in T.serial(1, turns):
                    prev_bx, prev_by = tile_coordinates(block_id + (turn - 1) * grid_size)
                    for _ in T.serial(6):
                        T.sync_warp()
                    # First batch overlaps the next tile's first K iteration.
                    push_output(p0, peers, prev_by, prev_bx, 0)
                    for kg in T.serial(k_groups - 1):
                        T.sync_warp()
                        if kg == (k_groups - 1) // 4:
                            push_output(p1, peers, prev_by, prev_bx, 1)
                        T.sync_warp()
                        if kg == (k_groups - 1) // 2:
                            push_output(p2, peers, prev_by, prev_bx, 2)
                        T.sync_warp()
                        if kg == 3 * (k_groups - 1) // 4:
                            push_output(p3, peers, prev_by, prev_bx, 3)
                        for _ in T.serial(4):
                            T.sync_warp()
                    T.sync_warp()
                    if k_groups == 1:
                        push_output(p1, peers, prev_by, prev_bx, 1)
                    T.sync_warp()
                    if k_groups == 1:
                        push_output(p2, peers, prev_by, prev_bx, 2)
                    T.sync_warp()
                    if k_groups == 1:
                        push_output(p3, peers, prev_by, prev_bx, 3)
                    T.sync_warp()
                    receive_output(p0, p1, p2, p3, out_a0, out_a1, out_a2, out_a3, out_b0, out_b1, out_b2, out_b3)
                # Drain the final completed tile; no later GEMM can hide this batch.
                last_bx, last_by = tile_coordinates(block_id + (turns - 1) * grid_size)
                push_output(p0, peers, last_by, last_bx, 0)
                push_output(p1, peers, last_by, last_bx, 1)
                push_output(p2, peers, last_by, last_bx, 2)
                push_output(p3, peers, last_by, last_bx, 3)

            # Every writer fences its own peer stores before the CTA leader
            # publishes completion. No grid barrier and no completion atomics.
            T.call_extern("void", "__threadfence_system")
            T.sync_warp()
            if tx == 0:
                epoch = T.alloc_var("uint64", init=control[block_id + 1, 0] + T.uint64(1))
                T.device_assert(epoch != 0, no_stack_info=True)
                T.signal_op(control[block_id + 1, 0], epoch, "set", rank)

    return gemm


def waiting_reduce(n, world, rank, grid):
    rows = n // world
    total_tiles = (n // 256) ** 2

    @T.prim_func
    def main(
        inbox: T.Tensor((world, rows, n), "float16"),
        output: T.Tensor((rows, n), "float16"),
        peers: T.Tensor((world,), "uint64"),
        consumed: T.Tensor((rows // 256, n // 256), "uint64"),
    ):
        with T.Kernel(n // 256, rows // 256, threads=512) as (bx, by):
            tile = (rank * (rows // 256) + by) * (n // 256) + bx
            if T.get_thread_binding() == 0:
                epoch = T.alloc_var("uint64", init=consumed[by, bx] + T.uint64(1))
                T.device_assert(epoch != 0, no_stack_info=True)
                for peer in T.serial(world):
                    scheduled = tile
                    if world == 4:
                        scheduled = (tile + total_tiles - peer * (total_tiles // world)) % total_tiles
                    producer = scheduled % grid
                    flags = T.make_tensor_from_addr(peers[peer], (grid + 1, 32), "uint64")
                    T.signal_wait_until(flags[producer + 1, 0], "ge", epoch)
                consumed[by, bx] = epoch
            T.sync_threads()
            # Every reader wave acquires before reading the peer-written payload.
            T.call_extern("void", "__builtin_amdgcn_fence", 2, "")
            acc = T.alloc_fragment((256, 256), "float32")
            T.clear(acc)
            for peer in T.serial(world):
                for i, j in T.Parallel(256, 256):
                    acc[i, j] += inbox[peer, by * 256 + i, bx * 256 + j]
            T.copy(acc, output[by * 256 : (by + 1) * 256, bx * 256 : (bx + 1) * 256])

    return main


def run():
    import argparse
    import ctypes
    import gc
    import os
    import statistics
    import time
    from datetime import timedelta
    from pathlib import Path

    import torch
    import torch.distributed as dist
    import tilelang as tl
    from tilelang.distributed.backends.ipc import IpcAllocator, _FfiIpcRuntime, ipc_metadata_layout
    from examples.distributed.gemm_reduce_scatter.utils import rank_env

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--dump-source", type=Path)
    args = parser.parse_args()
    rank, world, device = (rank_env(key) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    n = args.size
    if world not in (2, 4) or n < 256 * world or n % (256 * world):
        raise ValueError("use 2 or 4 ranks and size divisible by 256*world")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    os.environ["TILELANG_CACHE_DIR"] = str(Path(os.environ.get("TILELANG_CACHE_DIR", "/tmp/tl-kernel-barrier")) / f"rank-{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=180))
    rccl = dist.new_group(backend="nccl", timeout=timedelta(seconds=180))
    hip = ctypes.CDLL("/opt/dtk/hip/lib/libamdhip64.so")

    def hip_check(code, operation):
        if code:
            raise RuntimeError(f"{operation}: HIP error {code}")

    grid = min(80, (n // 256) ** 2)
    runtime = _FfiIpcRuntime()
    hip.hipExtMallocWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]

    def fine_malloc(size):
        pointer = ctypes.c_void_p()
        hip_check(hip.hipExtMallocWithFlags(ctypes.byref(pointer), size, 1), "hipDeviceMallocFinegrained")
        return pointer.value

    runtime.malloc = fine_malloc
    arena = IpcAllocator(
        n * n * 2 + (grid + 1) * 256 + 4096,
        rank=rank,
        world_size=world,
        device=device,
        group=dist.group.WORLD,
        runtime=runtime,
    )
    arena.initialize()

    inbox = tl.tensor((world, n // world, n), torch.float16, allocator=arena)
    control = tl.tensor((grid + 1, 32), torch.uint64, allocator=arena)
    inbox.fill_(float("nan"))
    # Initialization only. Slot zero is reserved; each producer advances its own ready counter.
    control.zero_()
    consumed = torch.zeros((n // world // 256, n // 256), device="cuda", dtype=torch.uint64)

    def peer_addresses(tensor):
        bases = arena.metadata[ipc_metadata_layout()["peer_base"] :]
        offset = tensor.data_ptr() - arena.base_ptr
        return torch.tensor([base + offset for base in bases], dtype=torch.uint64, device="cuda")

    peers = peer_addresses(inbox)
    flag_peers = peer_addresses(control)
    target = {"kind": "hcu", "mcpu": torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0], "dist_backend": "ipc"}
    gemm = tl.compile(
        gemm_push_waves(n, n, n, rank, world),
        out_idx=[],
        target=target,
        pass_configs={tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True},
    )
    reduce = tl.compile(waiting_reduce(n, world, rank, grid), out_idx=[], target=target)
    gemm.initialize(arena)
    reduce.initialize(arena)
    if args.dump_source:
        args.dump_source.mkdir(parents=True, exist_ok=True)
        for name, kernel in [("gemm_push", gemm), ("reduce_wait", reduce)]:
            (args.dump_source / f"{name}.rank-{rank}.hip").write_text(kernel.get_kernel_source())

    a = torch.empty((n, n), device="cuda", dtype=torch.float16)
    b = torch.empty_like(a)
    output = torch.empty((n // world, n), device="cuda", dtype=torch.float16)

    def enqueue():
        gemm(a, b, peers, control)
        reduce(inbox, output, flag_peers, consumed)
        # Protect previous readers before the next call can overwrite any inbox.
        work = dist.barrier(group=rccl, device_ids=[device], async_op=True)
        work.block_current_stream()

    def random_inputs():
        a.normal_().mul_(n**-0.5)
        b.normal_()

    def reference():
        full = (a.float() @ b.float()).cpu()
        dist.all_reduce(full)
        return full[rank * (n // world) : (rank + 1) * (n // world)].to(device="cuda", dtype=torch.float16)

    def verify(expected, label):
        torch.cuda.current_stream().synchronize()
        torch.testing.assert_close(
            output,
            expected,
            atol=1e-2,
            rtol=1e-2,
            msg=lambda msg: f"{label} rank={rank}: {msg}",
        )
        print(
            f"{label} rank={rank}/{world} sample={output[0, 0].item():.6g} max_abs={(output - expected).abs().max().item():.6g} passed",
            flush=True,
        )

    def queued_check(fn, label):
        # Retain every output to detect intermittent reuse failures.
        saved = []
        for iteration in range(20):
            random_inputs()
            local_reference = a.float() @ b.float()
            if iteration == 7 and rank == world - 1:
                time.sleep(0.05)
            fn()
            saved.append((output.clone(), local_reference))
        torch.cuda.current_stream().synchronize()
        for iteration, (actual, full) in enumerate(saved):
            total = full.cpu()
            dist.all_reduce(total)
            expected = total[rank * (n // world) : (rank + 1) * (n // world)].to(device="cuda", dtype=torch.float16)
            torch.testing.assert_close(
                actual,
                expected,
                atol=1e-2,
                rtol=1e-2,
                msg=lambda msg, iteration=iteration: f"{label} rank={rank} iteration={iteration}: {msg}",
            )
        print(f"{label} rank={rank}/{world} calls=20 checked=20 passed", flush=True)

    def capture(fn):
        for _ in range(3):
            fn()
        torch.cuda.current_stream().synchronize()
        dist.barrier()
        before = (control.cpu().clone(), consumed.cpu().clone())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.current_stream().synchronize()
        assert torch.equal(control.cpu(), before[0]), "capture advanced producer counters"
        assert torch.equal(consumed.cpu(), before[1]), "capture advanced consumer counters"
        return graph

    torch.manual_seed(1234 + rank)
    for trial in range(3):
        random_inputs()
        expected = reference()
        if trial == 1 and rank == world - 1:
            time.sleep(0.05)
        enqueue()
        verify(expected, f"random-check trial={trial}")
    queued_check(enqueue, "queued-random-check")

    graph = capture(enqueue)
    for trial in range(3):
        random_inputs()
        expected = reference()
        if trial == 1 and rank == world - 1:
            time.sleep(0.05)
        graph.replay()
        verify(expected, f"graph-check trial={trial}")
    queued_check(graph.replay, "queued-graph-check")
    print(f"correctness-complete rank={rank}/{world} size={n}", flush=True)

    def device_ms(fn):
        # Like the Triton tutorial: warm up, enqueue timed calls, then wait once.
        # The full path includes the RCCL tail; no CPU barrier between calls.
        torch.cuda.current_stream().synchronize()
        dist.barrier()
        for _ in range(5):
            fn()
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.repeats)]
        for start, end in events:
            start.record()
            fn()
            end.record()
        torch.cuda.current_stream().synchronize()
        return statistics.median(start.elapsed_time(end) for start, end in events)

    if args.benchmark:
        random_inputs()
        expected = reference()
        results = {"gpu_event_ms": device_ms(enqueue), "graph_gpu_event_ms": device_ms(graph.replay)}
        verify(expected, "post-timing-check")
        reports = [None] * world
        dist.all_gather_object(reports, results)
        if rank == 0:
            worst = {key: max(row[key] for row in reports) for key in results}
            print(f"timings world_size={world} size={n} by_rank={reports}", flush=True)
            print(f"summary world_size={world} size={n} max_rank={worst}", flush=True)

    torch.cuda.current_stream().synchronize()
    dist.barrier()
    graph.reset()
    inbox = control = peers = flag_peers = consumed = None
    gc.collect()
    arena.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
