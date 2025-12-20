"""
Split-K GEMM implementation.
"""

import tilelang as tl
import tilelang.language as T


@tl.jit(out_idx=[-1])
def gemm_splitk(M, N, K, block_M, block_N, block_K,
           num_stages, thread_num, enable_rasteration=True, split_k=2,
           dtype="float16", accum_dtype="float"):

    @T.prim_func
    def _gemm_splitk(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=thread_num) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)
            #for k in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=0):
            for k in T.Pipelined(T.ceildiv(K, split_k * block_K), num_stages=num_stages):
                #T.copy(A[by * block_M, bz * splitK + k * block_K], A_shared, coalesced_width=8)
                #T.copy(B[bx * block_N, bz * splitK + k * block_K], B_shared, coalesced_width=8)
                T.copy(A[by * block_M, bz * block_K + k * split_k * block_K], A_shared, coalesced_width=8)
                T.copy(B[bx * block_N, bz * block_K + k * split_k * block_K], B_shared, coalesced_width=8)
                T.gemm(A_shared, B_shared, C_local, k_pack=2, transpose_B=True)
            #T.copy(C_local, C_shared, coalesced_width=8)
            for i, j in T.Parallel(block_M, block_N):
                T.atomic_add(C[by * block_M + i, bx * block_N + j], C_local[i, j])

    return _gemm_splitk
