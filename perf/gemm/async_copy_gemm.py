# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Vanilla GEMM using asynchronous Global-to-LDS copies."""

import tilelang as tl
import tilelang.language as T

tl.disable_cache()
def _gemm_async_copy_vanilla(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype="float16",
    accum_dtype="float32",
    steady_wait=6,
    shared_0_wait=6,
    swizzle_panel_size=0,
    swizzle_order="row",
):
    """Four-stage GEMM using async copies and compiler-derived LDS layouts."""
    transpose_B = False
    k_tiles = (K + block_K - 1) // block_K
    k_groups = (k_tiles + 3) // 4

    @T.macro
    def async_copy_a(A, A_shared, by, k_tile):
        T.copy(
            A[
                by * block_M : (by + 1) * block_M,
                k_tile * block_K : (k_tile + 1) * block_K,
            ],
            A_shared,
            enable_async=True,
        )

    @T.macro
    def async_copy_b(B, B_shared, bx, k_tile):
        T.copy(
            B[
                k_tile * block_K : (k_tile + 1) * block_K,
                bx * block_N : (bx + 1) * block_N,
            ],
            B_shared,
            enable_async=True,
        )

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
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
            T.copy(C_local, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])

    return gemm


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def gemm_async_copy_n_major(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype="float16",
    accum_dtype="float32",
    swizzle_panel_size=0,
    swizzle_order="row",
):
    return _gemm_async_copy_vanilla(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        dtype=dtype,
        accum_dtype=accum_dtype,
        steady_wait=8,
        shared_0_wait=6,
        swizzle_panel_size=swizzle_panel_size,
        swizzle_order=swizzle_order,
    )


def main():
    """Correctness check and latency benchmark for ``gemm_async_copy_n_major``."""
    M, N, K = 10240, 10240, 10240
    block_M, block_N, block_K = 256, 256, 16

    kernel = gemm_async_copy_n_major(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        dtype="float16",
        accum_dtype="float32",
    )

    import torch

    a = torch.randn(M, K, device="cuda").half()
    b = torch.randn(K, N, device="cuda").half()

    c = kernel(a, b)

    # B is N-major [K, N], so the reference is A @ B (no transpose).
    ref_c = a @ b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # Get CUDA Source
    print("CUDA Source:")
    print(kernel.get_kernel_source())

    # benchmark (default "event" backend works on both CUDA and ROCm)
    profiler = kernel.get_profiler()
    latency = profiler.do_bench()
    tflops = 2 * M * N * K / latency * 1e-9
    print(f"tilelang Latency: {latency}ms")
    print(f"tilelang TFlops: {tflops:.4f}")


def run_regression_perf():
    """Compile a small shape and return the benchmark latency in milliseconds."""
    M, N, K = 1024, 1024, 1024
    block_M, block_N, block_K = 128, 128, 32

    kernel = gemm_async_copy_n_major(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        dtype="float16",
        accum_dtype="float32",
    )
    profiler = kernel.get_profiler()
    return profiler.do_bench()


if __name__ == "__main__":
    main()