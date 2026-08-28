"""Double-buffered GEMM using asynchronous Global-to-LDS copies."""

import tilelang as tl
import tilelang.language as T


BLOCK_K = 32
K_PACK = 2


def _gemm_async_copy_pingpong(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K=BLOCK_K,
    dtype="float16",
    accum_dtype="float32",
):
    """GEMM with two LDS stages and a loop-carried LDS-to-register prefetch."""
    transpose_B = True
    assert block_K == BLOCK_K, "async_copy_gemm_pp requires block_K == 32"
    assert block_N % 2 == 0, "block_N must be divisible by two"

    sub_block_N = block_N // 2
    k_tiles = (K + block_K - 1) // block_K
    async_copy_stage_requests = 4

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
                bx * block_N : (bx + 1) * block_N,
                k_tile * block_K : (k_tile + 1) * block_K,
            ],
            B_shared,
        )

    @T.macro
    def copy_b_half(B_shared, B_local, n_half):
        T.copy(
            B_shared[n_half * sub_block_N : (n_half + 1) * sub_block_N, :],
            B_local,
        )

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (bx, by):
            
            T.use_swizzle(
                panel_size=4,
                order="col",
                enable=True,
            )
            
            warp_idx = T.get_warp_idx()
            A_shared_0 = T.alloc_shared((block_M, block_K), dtype)
            A_shared_1 = T.alloc_shared((block_M, block_K), dtype)
            B_shared_0 = T.alloc_shared((block_N, block_K), dtype)
            B_shared_1 = T.alloc_shared((block_N, block_K), dtype)

            A_local_0 = T.alloc_fragment((block_M, block_K), dtype)
            A_local_1 = T.alloc_fragment((block_M, block_K), dtype)
            B_local_n0_0 = T.alloc_fragment((sub_block_N, block_K), dtype)
            B_local_n1_0 = T.alloc_fragment((sub_block_N, block_K), dtype)
            B_local_n0_1 = T.alloc_fragment((sub_block_N, block_K), dtype)
            B_local_n1_1 = T.alloc_fragment((sub_block_N, block_K), dtype)
            C_local_n0 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)
            C_local_n1 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)
            T.clear(C_local_n0)
            T.clear(C_local_n1)

            # Prime both LDS stages, then prepare the complete ping fragments.
            T.sync_warp()
            async_copy_a(A, A_shared_0, by, 0)
            async_copy_b(B, B_shared_0, bx, 0)

            async_copy_a(A, A_shared_1, by, 1)
            async_copy_b(B, B_shared_1, bx, 1)
            
            # A/B shared_0 ready
            T.ptx_wait_group(async_copy_stage_requests)
            T.sync_warp()
            
            T.copy(A_shared_0, A_local_0)
            copy_b_half(B_shared_0, B_local_n0_0, 0)
            copy_b_half(B_shared_0, B_local_n1_0, 1)

            T.s_waitcnt(0, "lgkmcnt")
            T.sync_warp()

            if warp_idx < 4:
                for pair in T.Serial(k_tiles // 2 - 1):
                    base = pair * 2 + 2

                    # Phase 0: prepare pong A while the back warps compute ping.
                    async_copy_a(A, A_shared_0, by, base)
                    T.ptx_wait_group(4)
                    T.sync_warp()

                    # Phase 1: compute ping n0 while the back warps prepare pong A.
                    T.s_waitcnt(4, "lgkmcnt")
                    T.sched_barrier()
                    T.gemm(
                        A_local_0,
                        B_local_n0_0,
                        C_local_n0,
                        transpose_B=transpose_B,
                        k_pack=K_PACK,
                        annotations={"trans_c": True},
                    )
                    T.sched_barrier()
                    T.sync_warp()

                    # Phase 2: prepare pong B while the back warps compute ping n1.
                    async_copy_b(B, B_shared_0, bx, base)
                    T.copy(A_shared_1, A_local_1)
                    T.ptx_wait_group(4)
                    T.sync_warp()

                    # Phase 3 also starts the next pong half-cycle.
                    T.s_waitcnt(4, "lgkmcnt")
                    T.sched_barrier()
                    T.gemm(
                        A_local_0,
                        B_local_n1_0,
                        C_local_n1,
                        transpose_B=transpose_B,
                        k_pack=K_PACK,
                        annotations={"trans_c": True},
                    )
                    T.sched_barrier()
                    T.sync_warp()
                    
                    # Phase 4
                    copy_b_half(B_shared_1, B_local_n0_1, 0)
                    copy_b_half(B_shared_1, B_local_n1_1, 1)
                    async_copy_a(A, A_shared_1, by, base + 1)
                    T.ptx_wait_group(4)
                    T.sync_warp()

                    # Phase 5
                    T.s_waitcnt(4, "lgkmcnt")
                    T.sched_barrier()
                    T.gemm(
                        A_local_1,
                        B_local_n0_1,
                        C_local_n0,
                        transpose_B=transpose_B,
                        k_pack=K_PACK,
                        annotations={"trans_c": True},
                    )
                    T.sched_barrier()
                    T.sync_warp()

                    # Phase 6
                    T.copy(A_shared_0, A_local_0)
                    async_copy_b(B, B_shared_1, bx, base + 1)
                    T.ptx_wait_group(4)
                    T.sync_warp()

                    # Phase 7
                    T.s_waitcnt(4, "lgkmcnt")
                    T.sched_barrier()
                    T.gemm(
                        A_local_1,
                        B_local_n1_1,
                        C_local_n1,
                        transpose_B=transpose_B,
                        k_pack=K_PACK,
                        annotations={"trans_c": True},
                    )
                    T.sched_barrier()
                    T.sync_warp()
                    
                    copy_b_half(B_shared_0, B_local_n0_0, 0)
                    copy_b_half(B_shared_0, B_local_n1_0, 1)
            else:
                for pair in T.Serial(k_tiles // 2 - 1):
                    base = pair * 2 + 2

                    # Phase 0: compute ping while the front warps prepare pong A.
                    T.s_waitcnt(4, "lgkmcnt")
                    T.sched_barrier()
                    T.gemm(
                        A_local_0,
                        B_local_n0_0,
                        C_local_n0,
                        transpose_B=transpose_B,
                        k_pack=K_PACK,
                        annotations={"trans_c": True},
                    )
                    T.sched_barrier()
                    T.ptx_wait_group(2)
                    T.sync_warp() # A_shared_1 ready

                    # Phase 1: prepare pong A while the front warps compute ping.
                    T.call_extern("tl::promote_prio", dtype="void")
                    async_copy_a(A, A_shared_0, by, base)
                    T.call_extern("tl::restore_prio", dtype="void")
                    T.copy(A_shared_1, A_local_1)
                    T.sync_warp()

                    # Phase 2: compute ping n1 while the front warps prepare pong B.
                    T.s_waitcnt(4, "lgkmcnt")
                    T.sched_barrier()
                    T.gemm(
                        A_local_0,
                        B_local_n1_0,
                        C_local_n1,
                        transpose_B=transpose_B,
                        k_pack=K_PACK,
                        annotations={"trans_c": True},
                    )
                    T.sched_barrier()
                    T.ptx_wait_group(2)
                    T.sync_warp() # B_shared_1 ready

                    # Phase 3 also starts the next pong half-cycle.
                    T.call_extern("tl::promote_prio", dtype="void")
                    async_copy_b(B, B_shared_0, bx, base)
                    T.call_extern("tl::restore_prio", dtype="void")
                    copy_b_half(B_shared_1, B_local_n0_1, 0)
                    copy_b_half(B_shared_1, B_local_n1_1, 1)
                    T.sync_warp()
                    
                    # Phase 4
                    T.s_waitcnt(4, "lgkmcnt")
                    T.sched_barrier()
                    T.gemm(
                        A_local_1,
                        B_local_n0_1,
                        C_local_n0,
                        transpose_B=transpose_B,
                        k_pack=K_PACK,
                        annotations={"trans_c": True},
                    )
                    T.sched_barrier()
                    T.ptx_wait_group(2)
                    T.sync_warp() # A_shared_0 ready

                    # Phase 5
                    T.call_extern("tl::promote_prio", dtype="void")
                    async_copy_a(A, A_shared_1, by, base + 1)
                    T.call_extern("tl::restore_prio", dtype="void")
                    T.copy(A_shared_0, A_local_0)
                    T.sync_warp()

                    # Phase 6
                    T.s_waitcnt(4, "lgkmcnt")
                    T.sched_barrier()
                    T.gemm(
                        A_local_1,
                        B_local_n1_1,
                        C_local_n1,
                        transpose_B=transpose_B,
                        k_pack=K_PACK,
                        annotations={"trans_c": True},
                    )
                    T.sched_barrier()
                    T.ptx_wait_group(2)
                    T.sync_warp() # B_shared_0 ready

                    # Phase 7
                    T.call_extern("tl::promote_prio", dtype="void")
                    async_copy_b(B, B_shared_1, bx, base + 1)
                    T.call_extern("tl::restore_prio", dtype="void")
                    copy_b_half(B_shared_0, B_local_n0_0, 0)
                    copy_b_half(B_shared_0, B_local_n1_0, 1)
                    T.sync_warp()

            # The loop leaves the final pong tile in LDS and the final ping tile in local.
            T.ptx_wait_group(0)
            T.s_waitcnt(0, "lgkmcnt")
            T.sync_warp()
            T.copy(A_shared_1, A_local_1)
            copy_b_half(B_shared_1, B_local_n0_1, 0)
            copy_b_half(B_shared_1, B_local_n1_1, 1)
            T.s_waitcnt(0, "lgkmcnt")
            T.sched_barrier()
            T.gemm(
                A_local_0,
                B_local_n0_0,
                C_local_n0,
                transpose_B=transpose_B,
                k_pack=K_PACK,
                annotations={"trans_c": True},
            )
            T.sched_barrier()
            T.gemm(
                A_local_0,
                B_local_n1_0,
                C_local_n1,
                transpose_B=transpose_B,
                k_pack=K_PACK,
                annotations={"trans_c": True},
            )
            T.sched_barrier()
            T.gemm(
                A_local_1,
                B_local_n0_1,
                C_local_n0,
                transpose_B=transpose_B,
                k_pack=K_PACK,
                annotations={"trans_c": True},
            )
            T.sched_barrier()
            T.gemm(
                A_local_1,
                B_local_n1_1,
                C_local_n1,
                transpose_B=transpose_B,
                k_pack=K_PACK,
                annotations={"trans_c": True},
            )
            T.sched_barrier()

            T.copy(
                C_local_n0,
                C[
                    by * block_M : (by + 1) * block_M,
                    bx * block_N : bx * block_N + sub_block_N,
                ],
            )
            T.copy(
                C_local_n1,
                C[
                    by * block_M : (by + 1) * block_M,
                    bx * block_N + sub_block_N : (bx + 1) * block_N,
                ],
            )

    return gemm


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def gemm_async_copy_k_major(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K=BLOCK_K,
    dtype="float16",
    accum_dtype="float32",
):
    return _gemm_async_copy_pingpong(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )
