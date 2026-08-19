"""Vanilla GEMM using asynchronous Global-to-LDS copies."""

import tilelang as tl
import tilelang.language as T


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def gemm_async_copy_vanilla(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype="float16",
    accum_dtype="float32",
    transpose_B=False,
):
    """Four-stage GEMM using async copies and compiler-derived LDS layouts."""
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
        if transpose_B:
            T.async_copy(
                B[
                    bx * block_N : (bx + 1) * block_N,
                    k_tile * block_K : (k_tile + 1) * block_K,
                ],
                B_shared,
            )
        else:
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
        B: T.Tensor((N, K) if transpose_B else (K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (bx, by):
            warp_idx = T.get_warp_idx()
            A_shared_0 = T.alloc_shared((block_M, block_K), dtype)
            A_shared_1 = T.alloc_shared((block_M, block_K), dtype)
            A_shared_2 = T.alloc_shared((block_M, block_K), dtype)
            A_shared_3 = T.alloc_shared((block_M, block_K), dtype)
            B_shared_0 = T.alloc_shared((block_N, block_K) if transpose_B else (block_K, block_N), dtype)
            B_shared_1 = T.alloc_shared((block_N, block_K) if transpose_B else (block_K, block_N), dtype)
            B_shared_2 = T.alloc_shared((block_N, block_K) if transpose_B else (block_K, block_N), dtype)
            B_shared_3 = T.alloc_shared((block_N, block_K) if transpose_B else (block_K, block_N), dtype)
            A_local_0 = T.alloc_fragment((block_M, block_K), dtype)
            B_local_0 = T.alloc_fragment((block_N, block_K) if transpose_B else (block_K, block_N), dtype)
            A_local_1 = T.alloc_fragment((block_M, block_K), dtype)
            B_local_1 = T.alloc_fragment((block_N, block_K) if transpose_B else (block_K, block_N), dtype)
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

            for kg in range(k_groups - 1):
                k0 = kg * 4
                if warp_idx < 4:
                    T.copy(A_shared_1, A_local_1)
                    T.copy(B_shared_1, B_local_1)
                    async_copy_a(A, A_shared_0, by, k0 + 4)
                    async_copy_b(B, B_shared_0, bx, k0 + 4)

                    T.s_waitcnt(6, "lgkmcnt")
                    T.ptx_wait_group(4)
                    T.sync_warp()
                    T.sched_barrier()
                    T.gemm(A_local_0, B_local_0, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                    T.sched_barrier()

                    k1_if = kg * 4 + 1
                    T.copy(A_shared_2, A_local_0)
                    T.copy(B_shared_2, B_local_0)
                    async_copy_a(A, A_shared_1, by, k1_if + 4)
                    async_copy_b(B, B_shared_1, bx, k1_if + 4)

                    T.s_waitcnt(6, "lgkmcnt")
                    T.ptx_wait_group(4)
                    T.sync_warp()
                    T.sched_barrier()
                    T.gemm(A_local_1, B_local_1, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                    T.sched_barrier()

                    k2_if = kg * 4 + 2
                    T.copy(A_shared_3, A_local_1)
                    T.copy(B_shared_3, B_local_1)
                    async_copy_a(A, A_shared_2, by, k2_if + 4)
                    async_copy_b(B, B_shared_2, bx, k2_if + 4)

                    T.s_waitcnt(6, "lgkmcnt")
                    T.ptx_wait_group(4)
                    T.sync_warp()
                    T.sched_barrier()
                    T.gemm(A_local_0, B_local_0, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                    T.sched_barrier()

                    k3_if = kg * 4 + 3
                    T.copy(A_shared_0, A_local_0)
                    T.copy(B_shared_0, B_local_0)
                    async_copy_a(A, A_shared_3, by, k3_if + 4)
                    async_copy_b(B, B_shared_3, bx, k3_if + 4)

                    T.s_waitcnt(6, "lgkmcnt")
                    T.ptx_wait_group(4)
                    T.sync_warp()
                    T.sched_barrier()
                    T.gemm(A_local_1, B_local_1, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                    T.sched_barrier()
                else:
                    T.copy(A_shared_1, A_local_1)
                    T.copy(B_shared_1, B_local_1)
                    async_copy_a(A, A_shared_0, by, k0 + 4)
                    async_copy_b(B, B_shared_0, bx, k0 + 4)

                    T.s_waitcnt(6, "lgkmcnt")
                    T.sched_barrier()
                    T.gemm(A_local_0, B_local_0, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                    T.sched_barrier()
                    T.ptx_wait_group(4)
                    T.sync_warp()

                    k1_else = kg * 4 + 1
                    T.copy(A_shared_2, A_local_0)
                    T.copy(B_shared_2, B_local_0)
                    async_copy_a(A, A_shared_1, by, k1_else + 4)
                    async_copy_b(B, B_shared_1, bx, k1_else + 4)

                    T.s_waitcnt(6, "lgkmcnt")
                    T.sched_barrier()
                    T.gemm(A_local_1, B_local_1, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                    T.sched_barrier()
                    T.ptx_wait_group(4)
                    T.sync_warp()

                    k2_else = kg * 4 + 2
                    T.copy(A_shared_3, A_local_1)
                    T.copy(B_shared_3, B_local_1)
                    async_copy_a(A, A_shared_2, by, k2_else + 4)
                    async_copy_b(B, B_shared_2, bx, k2_else + 4)

                    T.s_waitcnt(6, "lgkmcnt")
                    T.sched_barrier()
                    T.gemm(A_local_0, B_local_0, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                    T.sched_barrier()
                    T.ptx_wait_group(4)
                    T.sync_warp()

                    k3_else = kg * 4 + 3
                    T.copy(A_shared_0, A_local_0)
                    T.copy(B_shared_0, B_local_0)
                    async_copy_a(A, A_shared_3, by, k3_else + 4)
                    async_copy_b(B, B_shared_3, bx, k3_else + 4)

                    T.s_waitcnt(6, "lgkmcnt")
                    T.sched_barrier()
                    T.gemm(A_local_1, B_local_1, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                    T.sched_barrier()
                    T.ptx_wait_group(4)
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
            T.copy(C_local, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return gemm
