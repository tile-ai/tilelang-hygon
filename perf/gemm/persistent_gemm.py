"""
Persistent GEMM implementation.
"""

import torch
import tilelang as tl
import tilelang.language as T
from tilelang.intrinsics import get_swizzle_layout
from tilelang.layout.swizzle import make_linear_layout
from tvm import DataType
from perf.gemm.utils import _generate_configs_from_product, _run_autotuner


def make_block_swizzle_layout(buffer, block_m, block_n, swizzle_bytes=128):
    """Create a layout that applies the swizzle inside each block tile.

    The transform splits a global index (i, j) into tile and local indices,
    applies make_mmac_swizzle_layout to the local indices (within block_m x block_n),
    and maps back to global coordinates.
    If the block tile isn't swizzleable for the given dtype, return identity.
    """
    dtype = buffer.dtype
    shape = buffer.shape
    # Check if swizzle is possible for the block width
    can_swizzle = (block_n * DataType(dtype).bits) % 512 == 0
    if not can_swizzle:
        return T.Layout(shape, lambda *args: args)

    def transform(i, j):
        tile_i = i // block_m
        tile_j = j // block_n
        local_i = i % block_m
        local_j = j % block_n
        new_local_i, new_local_j = get_swizzle_layout(local_i, local_j, block_n, dtype, swizzle_bytes)
        return tile_i * block_m + new_local_i, tile_j * block_n + new_local_j
    return T.Layout(shape, transform)

def get_persistent_configs(M, N, K):
    """
    Generate a list of kernel tuning configuration dictionaries for persistent GEMM.

    Similar to get_configs but for persistent kernels. Adds wgs_per_cu parameter.
    """
    param_dict = {
        "block_M": [64, 128, 256],
        "block_N": [64, 128, 256],
        "block_K": [32, 64],
        "num_stages": [0, 2, 3],  # Reduced from [0, 1, 2, 3]
        "thread_num": [128, 256],
        "wgs_per_cu": [1, 2, 4],  # Grid size multiplier
    }
    return _generate_configs_from_product(param_dict)


def get_best_persistent_config(M, N, K):
    """
    Autotune persistent GEMM kernel and return the best configuration.
    """
    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        thread_num=None,
        wgs_per_cu=None,
    ):
        dtype = "float16"  # Match the default in matmul_persistent
        accum_dtype = "float"

        #cu_num = torch.cuda.get_device_properties("cuda").multi_processor_count

        grid_size = wgs_per_cu * 80
        m_blocks = T.ceildiv(M, block_M)
        n_blocks = T.ceildiv(N, block_N)
        waves = T.ceildiv(m_blocks * n_blocks, grid_size)
        group_size = 8

        @T.prim_func
        def main(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((N, K), dtype),
                C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(grid_size, threads=thread_num) as (block_id):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                for w in T.serial(waves):
                    tile_id = grid_size * w + block_id
                    bx = (tile_id // group_size) % m_blocks
                    by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                    if bx * block_M < M and by * block_N < N:
                        T.clear(C_local)
                        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                            T.copy(A[bx * block_M, k * block_K], A_shared, coalesced_width=8)
                            T.copy(B[by * block_N, k * block_K], B_shared, coalesced_width=8)
                            T.gemm(A_shared, B_shared, C_local, k_pack=2, transpose_B=True)

                        T.copy(C_local, C[bx * block_M, by * block_N])

        return main

    return _run_autotuner(kernel, get_persistent_configs(M, N, K))


# FIXME: Boudary check is not considered, so non-divisible block_N and group_size may cause
#        correctness issue.
# Note: Use pass_configs={"tl.disable_safe_memory_legalize": True} to disable safe memory legalize
#       during using vectorized with swizzled layout.
@tl.jit(out_idx=[-1])
def gemm_persistent(M, N, K, block_M, block_N, block_K,
                    num_stages, thread_num, group_size=8, wgs_per_cu=2,
                    dtype="float16", accum_dtype="float"):
    cu_num = torch.cuda.get_device_properties("cuda").multi_processor_count
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    grid_size = T.min(m_blocks * n_blocks, wgs_per_cu * cu_num)
    waves = T.ceildiv(m_blocks * n_blocks, grid_size)

    @T.prim_func
    def _gemm_persistent(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_size, threads=thread_num) as (block_id):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            for w in T.serial(waves):
                tile_id = grid_size * w + block_id
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                if bx * block_M < M and by * block_N < N:
                    T.clear(C_local)
                    #T.annotate_layout({
                    #    A: make_block_swizzle_layout(A, block_M, block_K),
                    #    A_shared: make_linear_layout(A_shared),
                    #    B_shared: make_linear_layout(B_shared),
                    #})
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        #for i in T.Parallel(block_M):
                        #    for j in T.Parallel(block_K):
                        #        # Apply swizzle layout to local block indices
                        #        si, sj = get_swizzle_layout(i, j, block_K, dtype, 128)
                        #        # Global indices in A (swizzled)
                        #        gi = bx * block_M + si
                        #        gk = k * block_K + sj
                        #        # Load from swizzled global positions
                        #        A_shared[i, j] = A[gi, gk]
                        #for i in T.Parallel(block_N):
                        #    for j in T.Parallel(block_K):
                        #        # Apply swizzle layout to local block indices
                        #        si, sj = get_swizzle_layout(i, j, block_K, dtype, 128)
                        #        # Global indices in A (swizzled)
                        #        gi = by * block_N + si
                        #        gk = k * block_K + sj
                        #        # Load from swizzled global positions
                        #        B_shared[i, j] = B[gi, gk]
                        T.copy(A[bx * block_M, k * block_K], A_shared, coalesced_width=8)
                        T.copy(B[by * block_N, k * block_K], B_shared, coalesced_width=8)
                        T.gemm(A_shared, B_shared, C_local, k_pack=2, transpose_B=True)

                    T.copy(C_local, C[bx * block_M, by * block_N])

    return _gemm_persistent
