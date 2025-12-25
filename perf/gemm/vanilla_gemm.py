"""
Vanilla GEMM implementation.
"""

import tilelang as tl
import tilelang.language as T
from perf.gemm.utils import get_configs, _run_autotuner


@tl.jit(out_idx=[-1])
def gemm_vanilla(M, N, K, block_M, block_N, block_K,
           num_stages, thread_num, enable_rasteration=True, group_size=8,
           dtype="float16", accum_dtype="float"):
    """
    Vanilla GEMM kernel with optional group swizzling optimization.

    Args:
        group_size: Size of tile groups for swizzling (1 disables group swizzling).
                   When > 1, uses the same group swizzling pattern as gemm_persistent.
    """
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    total_tiles = m_blocks * n_blocks

    @T.prim_func
    def _gemm_vanilla(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        if group_size > 1:
            # Use group swizzling optimization
            with T.Kernel(total_tiles, threads=thread_num) as (tile_id,):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.clear(C_local)

                # Group swizzling: same pattern as persistent_gemm
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size
                if bx * block_M < M and by * block_N < N:
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(A[bx * block_M, k * block_K], A_shared, coalesced_width=8)
                        T.copy(B[by * block_N, k * block_K], B_shared, coalesced_width=8)
                        T.gemm(A_shared, B_shared, C_local, k_pack=2, transpose_B=True)
                    T.copy(C_local, C[bx * block_M, by * block_N])
        else:
            # Standard 2D grid with simple swizzle
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared, coalesced_width=8)
                    T.copy(B[bx * block_N, k * block_K], B_shared, coalesced_width=8)
                    T.gemm(A_shared, B_shared, C_local, k_pack=2, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

    return _gemm_vanilla


def get_best_vanilla_config(M, N, K, with_roller=False):
    """Autotune vanilla GEMM kernel and return the best configuration."""
    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        thread_num=None,
        enable_rasteration=None,
        group_size=None,
    ):
        dtype = "float16"
        accum_dtype = "float"

        m_blocks = T.ceildiv(M, block_M)
        n_blocks = T.ceildiv(N, block_N)
        total_tiles = m_blocks * n_blocks

        @T.prim_func
        def main(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((N, K), dtype),
                C: T.Tensor((M, N), dtype),
        ):
            if group_size > 1:
                # Use group swizzling optimization
                with T.Kernel(total_tiles, threads=thread_num) as (tile_id,):
                    A_shared = T.alloc_shared((block_M, block_K), dtype)
                    B_shared = T.alloc_shared((block_N, block_K), dtype)
                    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    C_shared = T.alloc_shared((block_M, block_N), dtype)

                    # Group swizzling
                    bx = (tile_id // group_size) % m_blocks
                    by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                    if bx * block_M < M and by * block_N < N:
                        T.clear(C_local)
                        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                            T.copy(A[bx * block_M, k * block_K], A_shared)
                            T.copy(B[by * block_N, k * block_K], B_shared)
                            T.gemm(A_shared, B_shared, C_local)
                        T.copy(C_local, C_shared)
                        T.copy(C_shared, C[bx * block_M, by * block_N])
            else:
                # Standard 2D grid with simple swizzle
                with T.Kernel(
                        T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                    A_shared = T.alloc_shared((block_M, block_K), dtype)
                    B_shared = T.alloc_shared((block_N, block_K), dtype)
                    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    C_shared = T.alloc_shared((block_M, block_N), dtype)
                    T.use_swizzle(panel_size=10, enable=enable_rasteration)
                    T.clear(C_local)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(A[by * block_M, k * block_K], A_shared)
                        T.copy(B[bx * block_N, k * block_K], B_shared)
                        T.gemm(A_shared, B_shared, C_local)
                    T.copy(C_local, C_shared)
                    T.copy(C_shared, C[by * block_M, bx * block_N])

        return main

    return _run_autotuner(kernel, get_configs(M, N, K, with_roller))
