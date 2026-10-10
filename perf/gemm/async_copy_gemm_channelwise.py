# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Channelwise-scaled GEMM using asynchronous Global-to-LDS copies.

Computes ``C[m, n] = scale_a[m] * scale_b[n] * sum_k A[m, k] * B[n, k]``
where ``A`` is (M, K), ``B`` is (N, K) (N-major, ``transpose_B=True``),
``scale_a`` is a (M,) float32 per-token scale and ``scale_b`` is a (N,)
float32 per-channel scale. The scales are applied as an epilogue multiply
on the float32 accumulator fragment before the final store.

The kernel is a hand-written four-stage async pipeline (manual
``T.async_copy`` + ``T.ptx_wait_group`` + ``T.s_waitcnt`` +
``T.sched_barrier`` + ``tl::promote_prio``/``tl::restore_prio``) that
overlaps global-to-LDS copies with MMAC. It requires
``K >= 4 * block_K`` (the prologue unconditionally issues four async
copy groups) and is validated for ``block_N = 128`` on HCU (gfx938).
"""

import tilelang as tl
import tilelang.language as T


def _gemm_async_copy_channelwise(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype="float8_e4m3fn",
    accum_dtype="float32",
    steady_wait=6,
    shared_0_wait=6,
    swizzle_panel_size=0,
    swizzle_order="row",
):
    """Four-stage channelwise GEMM using async copies and compiler-derived LDS layouts."""
    transpose_B = True
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
                bx * block_N : (bx + 1) * block_N,
                k_tile * block_K : (k_tile + 1) * block_K,
            ],
            B_shared,
        )

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        scale_a: T.Tensor((M,), "float32"),
        scale_b: T.Tensor((N,), "float32"),
        C: T.Tensor((M, N), accum_dtype),
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
            B_shared_0 = T.alloc_shared((block_N, block_K), dtype)
            B_shared_1 = T.alloc_shared((block_N, block_K), dtype)
            B_shared_2 = T.alloc_shared((block_N, block_K), dtype)
            B_shared_3 = T.alloc_shared((block_N, block_K), dtype)
            A_local_0 = T.alloc_fragment((block_M, block_K), dtype)
            B_local_0 = T.alloc_fragment((block_N, block_K), dtype)
            A_local_1 = T.alloc_fragment((block_M, block_K), dtype)
            B_local_1 = T.alloc_fragment((block_N, block_K), dtype)
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
            # Channelwise epilogue: per-token (M) and per-channel (N) fp32 scales.
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = C_local[i, j] * scale_a[by * block_M + i] * scale_b[bx * block_N + j]
            T.copy(C_local, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return gemm


@tl.jit(
    out_idx=[-1],
)
def gemm_async_copy_channelwise(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype="float8_e4m3fn",
    accum_dtype="float32",
    swizzle_panel_size=0,
    swizzle_order="row",
    steady_wait=8,
    shared_0_wait=6,
):
    """Channelwise-scaled GEMM (A (M,K) x B (N,K) with fp32 per-token/per-channel scales).

    Validated configuration on HCU (gfx938): block 256x128x32, M=N=K=4096,
    ``dtype="float8_e4m3fn"``. Requires ``K >= 4 * block_K``.
    """
    return _gemm_async_copy_channelwise(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        dtype=dtype,
        accum_dtype=accum_dtype,
        steady_wait=steady_wait,
        shared_0_wait=shared_0_wait,
        swizzle_panel_size=swizzle_panel_size,
        swizzle_order=swizzle_order,
    )
