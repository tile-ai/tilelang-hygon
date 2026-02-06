"""
Persistent GEMM implementation.
"""

import torch
import tilelang as tl
import tilelang.language as T
from tilelang.intrinsics import get_swizzle_layout
from tilelang.layout.swizzle import make_linear_layout, make_hcu_swizzled_layout
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
        "group_size": [1, 2, 4],  # Enable group swizzling optimization
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
        group_size=None,
    ):
        dtype = "float16"  # Match the default in matmul_persistent
        accum_dtype = "float"

        #cu_num = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid_size = wgs_per_cu * 80
        # grid_size = T.min(m_blocks * n_blocks, wgs_per_cu * cu_num)
        m_blocks = T.ceildiv(M, block_M)
        n_blocks = T.ceildiv(N, block_N)
        waves = T.ceildiv(m_blocks * n_blocks, grid_size)
        # group_size = 8

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

    return _run_autotuner(
        kernel,
        get_persistent_configs(M, N, K),
        pass_configs={"tl.enable_aggressive_shared_memory_merge": True},
    )

def get_best_persistent_config_v1(M, N, K):
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
        group_size=None,
    ):
        dtype = "float16"  # Match the default in matmul_persistent
        accum_dtype = "float"

        # grid_size = wgs_per_cu * 80
        cu_num = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid_size = T.min(m_blocks * n_blocks, wgs_per_cu * cu_num)
        m_blocks = T.ceildiv(M, block_M)
        n_blocks = T.ceildiv(N, block_N)
        waves = T.ceildiv(m_blocks * n_blocks, grid_size)

        split_n = 2
        sub_block_N = block_N // split_n
        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(grid_size, threads=thread_num) as (block_id):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                # B_shared = T.alloc_shared((block_N, block_K), dtype)
                B_shared_0 = T.alloc_shared((sub_block_N, block_K), dtype)

                C_local_0 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)
                C_local_1 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)
                C_shared_0 = T.alloc_shared((block_M, sub_block_N), dtype)
                T.annotate_layout({
                    C_shared_0: tl.layout.make_hcu_swizzled_layout(C_shared_0, major_pack=2),
                })

                for w in T.serial(waves):
                    tile_id = grid_size * w + block_id
                    bx = (tile_id // group_size) % m_blocks
                    by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                    if bx * block_M < M and by * block_N < N:
                        T.clear(C_local_0)
                        T.clear(C_local_1)
                        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                            T.copy(A[bx * block_M, k * block_K], A_shared, coalesced_width=8)
                            # T.copy(B[by * block_N, k * block_K], B_shared, coalesced_width=8)
                            # T.gemm(A_shared, B_shared, C_local, k_pack=2, transpose_B=True)
                            T.copy(B[by * block_N, k * block_K], B_shared_0, coalesced_width=8)
                            T.gemm(A_shared, B_shared_0, C_local_0, k_pack=2, transpose_B=True)
                            T.copy(B[by * block_N + sub_block_N, k * block_K], B_shared_0, coalesced_width=8)
                            T.gemm(A_shared, B_shared_0, C_local_1, k_pack=2, transpose_B=True)

                        # T.copy(C_local_0, C[bx * block_M, by * block_N])
                        # T.copy(C_local_1, C[bx * block_M, by * block_N + sub_block_N])
                        T.copy(C_local_0, C_shared_0)
                        T.copy(C_shared_0, C[bx * block_M, by * block_N])
                        T.copy(C_local_1, C_shared_0)
                        T.copy(C_shared_0, C[bx * block_M, by * block_N + sub_block_N])

        return main

    return _run_autotuner(
        kernel,
        get_persistent_configs(M, N, K),
        pass_configs={"tl.enable_aggressive_shared_memory_merge": True},
    )


# Impl:
#   1. annotate C Layout to leverage buffer_store_dwordx4
#   2. split block_n // 2 to limit LDS
@tl.jit(out_idx=[-1], pass_configs={"tl.enable_aggressive_shared_memory_merge": True})
def gemm_persistent_v1(M, N, K, block_M, block_N, block_K,
                    num_stages, thread_num, group_size=8, wgs_per_cu=2,
                    dtype="float16", accum_dtype="float"):
    cu_num = torch.cuda.get_device_properties("cuda").multi_processor_count
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    grid_size = T.min(m_blocks * n_blocks, wgs_per_cu * cu_num)
    waves = T.ceildiv(m_blocks * n_blocks, grid_size)

    split_n = 2
    sub_block_N = block_N // split_n

    @T.prim_func
    def _gemm_persistent(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_size, threads=thread_num) as (block_id):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared_0 = T.alloc_shared((sub_block_N, block_K), dtype)

            C_local_0 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)
            C_local_1 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)

            C_shared_0 = T.alloc_shared((block_M, sub_block_N), dtype)
            T.annotate_layout({
                C_shared_0: tl.layout.make_hcu_swizzled_layout(C_shared_0, major_pack=2),
            })

            for w in T.serial(waves):
                tile_id = grid_size * w + block_id
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                if bx * block_M < M and by * block_N < N:
                    T.clear(C_local_0)
                    T.clear(C_local_1)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(A[bx * block_M, k * block_K], A_shared, coalesced_width=8)
                        T.copy(B[by * block_N, k * block_K], B_shared_0, coalesced_width=8)
                        T.gemm(A_shared, B_shared_0, C_local_0, k_pack=2, transpose_B=True)
                        T.copy(B[by * block_N + sub_block_N, k * block_K], B_shared_0, coalesced_width=8)
                        T.gemm(A_shared, B_shared_0, C_local_1, k_pack=2, transpose_B=True)

                    T.copy(C_local_0, C_shared_0)
                    T.copy(C_shared_0, C[bx * block_M, by * block_N])
                    T.copy(C_local_1, C_shared_0)
                    T.copy(C_shared_0, C[bx * block_M, by * block_N + sub_block_N])

    return _gemm_persistent


# Impl:
#   preload A/B to register swizzled before T.gemm
@tl.jit(out_idx=[-1], pass_configs={"tl.enable_aggressive_shared_memory_merge": True})
def gemm_persistent_v2(M, N, K, block_M, block_N, block_K,
                    num_stages, thread_num, group_size=8, wgs_per_cu=2,
                    dtype="float16", accum_dtype="float"):
    cu_num = torch.cuda.get_device_properties("cuda").multi_processor_count
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    grid_size = T.min(m_blocks * n_blocks, wgs_per_cu * cu_num)
    waves = T.ceildiv(m_blocks * n_blocks, grid_size)

    split_n = 2
    sub_block_N = block_N // split_n

    @T.prim_func
    def _gemm_persistent(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_size, threads=thread_num) as (block_id):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared_0 = T.alloc_shared((sub_block_N, block_K), dtype)
            
            A_local_0 = T.alloc_fragment((block_M, block_K), dtype)
            A_local_0_ = T.alloc_fragment((block_M, block_K), dtype)
            
            B_local_0 = T.alloc_fragment((sub_block_N, block_K), dtype)
            B_local_1 = T.alloc_fragment((sub_block_N, block_K), dtype)
            
            B_local_0_ = T.alloc_fragment((sub_block_N, block_K), dtype)
            B_local_1_ = T.alloc_fragment((sub_block_N, block_K), dtype)

            C_local_0 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)
            C_local_1 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)

            C_shared_0 = T.alloc_shared((block_M, sub_block_N), dtype)
            T.annotate_layout({
                C_shared_0: tl.layout.make_hcu_swizzled_layout(C_shared_0, major_pack=2),
                B_shared_0: tl.layout.make_hcu_swizzled_layout(B_shared_0, major_pack=2),
                A_shared: tl.layout.make_hcu_swizzled_layout(A_shared, major_pack=2),
            })

            for w in T.serial(waves):
                tile_id = grid_size * w + block_id
                
                # swizzle along N
                bx = (tile_id % group_size) + (tile_id // group_size) // n_blocks * group_size
                by = (tile_id // group_size) % n_blocks

                # if bx * block_M < M and by * block_N < N:
                if tile_id < m_blocks * n_blocks:
                    T.clear(C_local_0)
                    T.clear(C_local_1)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        # T.copy(A[bx * block_M, k * block_K], A_shared, coalesced_width=8)
                        T.copy(A[bx * block_M, k * block_K], A_local_0, coalesced_width=8)
                        # A Block swizzle
                        T.copy(A_local_0, A_shared)

                        # preload B Block N_0
                        T.copy(B[by * block_N, k * block_K], B_local_0, coalesced_width=8)
                        # preload B Block N_1
                        T.copy(B[by * block_N + sub_block_N, k * block_K], B_local_1, coalesced_width=8)
                        
                        # B Block N_0 swizzle
                        T.copy(B_local_0, B_shared_0)
                        T.copy(B_shared_0, B_local_0_)
                        
                        # B Block N_1 swizzle
                        T.copy(B_local_1, B_shared_0)
                        T.copy(B_shared_0, B_local_1_)
                        
                        # A local
                        T.copy(A_shared, A_local_0_)
                        
                        T.gemm(A_local_0_, B_local_0_, C_local_0, k_pack=2, transpose_B=True)
                        T.gemm(A_local_0_, B_local_1_, C_local_1, k_pack=2, transpose_B=True)

                    T.copy(C_local_0, C_shared_0)
                    T.copy(C_shared_0, C[bx * block_M, by * block_N])
                    T.copy(C_local_1, C_shared_0)
                    T.copy(C_shared_0, C[bx * block_M, by * block_N + sub_block_N])

    return _gemm_persistent


# Impl:
#   preload A/B to register swizzled before T.gemm
#   use T.persistent instead of swizzle manually
@tl.jit(out_idx=[-1], pass_configs={"tl.enable_aggressive_shared_memory_merge": True})
def gemm_persistent_v3(M, N, K, block_M, block_N, block_K,
                    num_stages, thread_num, group_size=8, wgs_per_cu=2,
                    dtype="float16", accum_dtype="float"):
    cu_num = torch.cuda.get_device_properties("cuda").multi_processor_count
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    grid_size = T.min(m_blocks * n_blocks, wgs_per_cu * cu_num)
    # waves = T.ceildiv(m_blocks * n_blocks, grid_size)

    split_n = 2
    sub_block_N = block_N // split_n

    @T.prim_func
    def _gemm_persistent(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_size, threads=thread_num) as (block_id):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared_0 = T.alloc_shared((sub_block_N, block_K), dtype)
            
            A_local_0 = T.alloc_fragment((block_M, block_K), dtype)
            A_local_0_ = T.alloc_fragment((block_M, block_K), dtype)
            
            B_local_0 = T.alloc_fragment((sub_block_N, block_K), dtype)
            B_local_1 = T.alloc_fragment((sub_block_N, block_K), dtype)
            
            B_local_0_ = T.alloc_fragment((sub_block_N, block_K), dtype)
            B_local_1_ = T.alloc_fragment((sub_block_N, block_K), dtype)

            C_local_0 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)
            C_local_1 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)

            C_shared_0 = T.alloc_shared((block_M, sub_block_N), dtype)
            T.annotate_layout({
                C_shared_0: tl.layout.make_hcu_swizzled_layout(C_shared_0, major_pack=2),
                B_shared_0: tl.layout.make_hcu_swizzled_layout(B_shared_0, major_pack=2),
                A_shared: tl.layout.make_hcu_swizzled_layout(A_shared, major_pack=2),
            })
            
            # bx: N, by: M
            for bx, by in T.Persistent(
                [T.ceildiv(N, block_N), T.ceildiv(M, block_M)],
                wgs_per_cu * cu_num,
                block_id,
                group_size=1
            ):
                if by * block_M < M and bx * block_N < N:
                    T.clear(C_local_0)
                    T.clear(C_local_1)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(A[by * block_M, k * block_K], A_local_0, coalesced_width=8)
                        # A Block swizzle
                        T.copy(A_local_0, A_shared)

                        # preload B Block N_0
                        T.copy(B[bx * block_N, k * block_K], B_local_0, coalesced_width=8)
                        # preload B Block N_1
                        T.copy(B[bx * block_N + sub_block_N, k * block_K], B_local_1, coalesced_width=8)
                        
                        # B Block N_0 swizzle
                        T.copy(B_local_0, B_shared_0)
                        T.copy(B_shared_0, B_local_0_)
                        
                        # B Block N_1 swizzle
                        T.copy(B_local_1, B_shared_0)
                        T.copy(B_shared_0, B_local_1_)
                        
                        # A local
                        T.copy(A_shared, A_local_0_)
                        
                        T.gemm(A_local_0_, B_local_0_, C_local_0, k_pack=2, transpose_B=True)
                        T.gemm(A_local_0_, B_local_1_, C_local_1, k_pack=2, transpose_B=True)

                    T.copy(C_local_0, C_shared_0)
                    T.copy(C_shared_0, C[by * block_M, bx * block_N])
                    T.copy(C_local_1, C_shared_0)
                    T.copy(C_shared_0, C[by * block_M, bx * block_N + sub_block_N])

    return _gemm_persistent


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
