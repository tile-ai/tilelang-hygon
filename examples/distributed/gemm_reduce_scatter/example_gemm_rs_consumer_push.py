"""GEMM + reduce-scatter with direct stores from the eight compute waves."""

import tilelang.language as T


def gemm_consumer_push(
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
    swizzle_panel_size=0,
    swizzle_order="row",
):
    """Keep the original 4+4 async-copy GEMM and store its FP16 partials to peers."""
    if dtype != "float16":
        raise ValueError("consumer_push requires FP16 input/output")
    if world_size < 1 or M % (world_size * block_M) or N % block_N or 4 * block_K > K or K % (4 * block_K):
        raise ValueError("consumer_push requires rank-aligned full M/N tiles and K divisible by 4*block_K")
    rows_per_rank = M // world_size
    transpose_B = False
    k_tiles = (K + block_K - 1) // block_K
    k_groups = (k_tiles + 3) // 4

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

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        peer_inboxes: T.Tensor((world_size,), "uint64"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (bx, by):
            T.use_swizzle(
                panel_size=swizzle_panel_size,
                order=swizzle_order,
                enable=swizzle_panel_size > 0,
            )
            warp_idx = T.get_warp_idx()
            A_shared_0 = T.alloc_shared((block_M, block_K), dtype)
            A_shared_1 = T.alloc_shared((block_M, block_K), dtype)
            A_shared_2 = T.alloc_shared((block_M, block_K), dtype)
            A_shared_3 = T.alloc_shared((block_M, block_K), dtype)
            B_shared_0 = T.alloc_shared((block_K, block_N), dtype)
            B_shared_1 = T.alloc_shared((block_K, block_N), dtype)
            B_shared_2 = T.alloc_shared((block_K, block_N), dtype)
            B_shared_3 = T.alloc_shared((block_K, block_N), dtype)
            A_local_0 = T.alloc_fragment((block_M, block_K), dtype)
            B_local_0 = T.alloc_fragment((block_K, block_N), dtype)
            A_local_1 = T.alloc_fragment((block_M, block_K), dtype)
            B_local_1 = T.alloc_fragment((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
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
            # Each sender owns a separate slot at the destination rank.
            remote_output = T.make_tensor_from_addr(
                peer_inboxes[(by * block_M) // rows_per_rank],
                (world_size, rows_per_rank, N),
                dtype,
            )
            T.copy(
                C_local,
                remote_output[
                    rank, (by * block_M) % rows_per_rank : (by * block_M) % rows_per_rank + block_M, bx * block_N : (bx + 1) * block_N
                ],
            )

    return gemm


if __name__ == "__main__":
    from examples.distributed.gemm_reduce_scatter._benchmark import run_benchmark

    run_benchmark(gemm_consumer_push, "consumer_push")
