"""
Persistent GEMM implementation.
"""

import torch
import tilelang as tl
import tilelang.language as T
from tilelang.intrinsics import get_swizzle_layout
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

        # cu_num = torch.cuda.get_device_properties("cuda").multi_processor_count
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
        m_blocks = T.ceildiv(M, block_M)
        n_blocks = T.ceildiv(N, block_N)
        grid_size = T.min(m_blocks * n_blocks, wgs_per_cu * cu_num)
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
                T.annotate_layout(
                    {
                        C_shared_0: tl.layout.make_hcu_swizzled_layout(C_shared_0, major_pack=2),
                    }
                )

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
def gemm_persistent_v1(
    M, N, K, block_M, block_N, block_K, num_stages, thread_num, group_size=8, wgs_per_cu=2, dtype="float16", accum_dtype="float"
):
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
            T.annotate_layout(
                {
                    C_shared_0: tl.layout.make_hcu_swizzled_layout(C_shared_0, major_pack=2),
                }
            )

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
def gemm_persistent_v2(
    M, N, K, block_M, block_N, block_K, num_stages, thread_num, group_size=8, wgs_per_cu=2, dtype="float16", accum_dtype="float"
):
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
            T.annotate_layout(
                {
                    C_shared_0: tl.layout.make_hcu_swizzled_layout(C_shared_0, major_pack=2),
                    B_shared_0: tl.layout.make_hcu_swizzled_layout(B_shared_0, major_pack=2),
                    A_shared: tl.layout.make_hcu_swizzled_layout(A_shared, major_pack=2),
                }
            )

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
@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def gemm_persistent_v3(
    M, N, K, block_M, block_N, block_K, num_stages, thread_num, group_size=8, wgs_per_cu=2, dtype="float16", accum_dtype="float"
):
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
            T.annotate_layout(
                {
                    C_shared_0: tl.layout.make_hcu_swizzled_layout(C_shared_0, major_pack=2),
                    B_shared_0: tl.layout.make_hcu_swizzled_layout(B_shared_0, major_pack=2),
                    A_shared: tl.layout.make_hcu_swizzled_layout(A_shared, major_pack=2),
                }
            )

            # bx: N, by: M
            for bx, by in T.Persistent(
                [T.ceildiv(N, block_N), T.ceildiv(M, block_M)], wgs_per_cu * cu_num, block_id, group_size=group_size
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

                        # # A local
                        T.copy(A_shared, A_local_0_)

                        T.gemm(A_local_0_, B_local_0_, C_local_0, k_pack=2, transpose_B=True)
                        T.gemm(A_local_0_, B_local_1_, C_local_1, k_pack=2, transpose_B=True)

                    T.copy(C_local_0, C_shared_0)
                    T.copy(C_shared_0, C[by * block_M, bx * block_N])
                    T.copy(C_local_1, C_shared_0)
                    T.copy(C_shared_0, C[by * block_M, bx * block_N + sub_block_N])

    return _gemm_persistent


# Impl:
#   preload A/B to register swizzled before T.gemm
#   use T.persistent instead of swizzle manually
@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def gemm_persistent_v4(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    num_stages,
    thread_num,
    group_size=1,
    wgs_per_cu=1,
    dtype="float16",
    accum_dtype="float",
    transpose_B=True,
):
    """Persistent GEMM v4 supporting K-major [N, K] and N-major [K, N] B layouts."""
    cu_num = torch.cuda.get_device_properties("cuda").multi_processor_count
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    grid_size = T.min(m_blocks * n_blocks, wgs_per_cu * cu_num)
    # waves = T.ceildiv(m_blocks * n_blocks, grid_size)

    split_n = 2
    sub_block_N = block_N // split_n
    k_loop_ = T.ceildiv(K, block_K)
    k_loop = T.ceildiv(k_loop_, 2)
    remain = 2 - (k_loop_ % 2)
    store_vecsize = 8 if N % 8 == 0 else 4 if N % 4 == 0 else 2 if N % 2 == 0 else 1

    @T.macro
    def matrix_load_b(B, n_offset, k_offset, B_shared):
        if transpose_B:
            T.matrix_load(B[n_offset, k_offset], B_shared)
        else:
            T.matrix_load(B[k_offset, n_offset], B_shared)

    @T.prim_func
    def _gemm_persistent(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K) if transpose_B else (K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_size, threads=thread_num) as (block_id):
            warp_idx = T.get_warp_idx()
            A_shared = T.alloc_shared((2, block_M, block_K), dtype)
            B_shared_0 = T.alloc_shared((2, sub_block_N, block_K) if transpose_B else (2, block_K, sub_block_N), dtype)
            B_shared_1 = T.alloc_shared((2, sub_block_N, block_K) if transpose_B else (2, block_K, sub_block_N), dtype)

            A_local_0 = T.alloc_fragment((block_M, block_K), dtype)
            B_local_0 = T.alloc_fragment((sub_block_N, block_K) if transpose_B else (block_K, sub_block_N), dtype)
            B_local_1 = T.alloc_fragment((sub_block_N, block_K) if transpose_B else (block_K, sub_block_N), dtype)

            A_local_0_ = T.alloc_fragment((block_M, block_K), dtype)
            B_local_0_ = T.alloc_fragment((sub_block_N, block_K) if transpose_B else (block_K, sub_block_N), dtype)
            B_local_1_ = T.alloc_fragment((sub_block_N, block_K) if transpose_B else (block_K, sub_block_N), dtype)

            C_local_0 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)
            C_local_1 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)

            local_out0 = T.alloc_fragment((block_M, sub_block_N), dtype)
            local_out1 = T.alloc_fragment((block_M, sub_block_N), dtype)

            C_shared_0 = T.alloc_shared((block_M, sub_block_N), dtype)
            T.annotate_layout(
                {
                    C_shared_0: tl.layout.make_hcu_swizzled_layout(C_shared_0, major_pack=2),
                }
            )

            # bx: N, by: M
            for bx, by in T.Persistent(
                [T.ceildiv(N, block_N), T.ceildiv(M, block_M)], wgs_per_cu * cu_num, block_id, group_size=group_size
            ):
                if by * block_M < M and bx * block_N < N:
                    T.clear(C_local_0)
                    T.clear(C_local_1)

                    if warp_idx < 4:
                        T.matrix_load(A[by * block_M, 0], A_shared[0, :, :])
                        matrix_load_b(B, bx * block_N, 0, B_shared_0[0, :, :])
                        matrix_load_b(B, bx * block_N + sub_block_N, 0, B_shared_1[0, :, :])
                        T.sched_barrier()
                        T.s_waitcnt(0)
                        T.sync_warp()
                        T.matrix_load(A[by * block_M, block_K], A_shared[1, :, :])
                        matrix_load_b(B, bx * block_N, block_K, B_shared_0[1, :, :])
                        matrix_load_b(B, bx * block_N + sub_block_N, block_K, B_shared_1[1, :, :])
                        T.sched_barrier()
                        T.ds_read_format(A_shared[0, :, :], A_local_0)
                        T.ds_read_format(B_shared_0[0, :, :], B_local_0)
                        T.ds_read_format(B_shared_1[0, :, :], B_local_1)

                        for k in T.Serial(k_loop - 1):
                            base = 2 * (k + 1)
                            T.s_waitcnt(0, flag="lgkmcnt")
                            T.sync_warp()
                            T.sched_barrier()
                            T.matrix_load(A[by * block_M, base * block_K], A_shared[0, :, :])
                            matrix_load_b(B, bx * block_N, base * block_K, B_shared_0[0, :, :])
                            matrix_load_b(B, bx * block_N + sub_block_N, base * block_K, B_shared_1[0, :, :])

                            T.s_waitcnt(3)
                            T.sync_warp()
                            T.ds_read_format(A_shared[1, :, :], A_local_0_)
                            T.ds_read_format(B_shared_0[1, :, :], B_local_0_)
                            T.sched_barrier()
                            T.gemm(A_local_0, B_local_0, C_local_0, transpose_B=transpose_B)
                            T.gemm(A_local_0, B_local_1, C_local_1, transpose_B=transpose_B)
                            T.sched_barrier()
                            T.ds_read_format(B_shared_1[1, :, :], B_local_1_)

                            T.sched_barrier()
                            T.s_waitcnt(0, flag="lgkmcnt")
                            T.sync_warp()
                            T.sched_barrier()
                            T.matrix_load(A[by * block_M, (base + 1) * block_K], A_shared[1, :, :])
                            matrix_load_b(B, bx * block_N, (base + 1) * block_K, B_shared_0[1, :, :])
                            matrix_load_b(B, bx * block_N + sub_block_N, (base + 1) * block_K, B_shared_1[1, :, :])

                            T.sched_barrier()
                            T.s_waitcnt(3)
                            T.sync_warp()
                            T.ds_read_format(A_shared[0, :, :], A_local_0)
                            T.ds_read_format(B_shared_0[0, :, :], B_local_0)
                            T.sched_barrier()
                            T.gemm(A_local_0_, B_local_0_, C_local_0, transpose_B=transpose_B)
                            T.gemm(A_local_0_, B_local_1_, C_local_1, transpose_B=transpose_B)
                            T.sched_barrier()
                            T.ds_read_format(B_shared_1[0, :, :], B_local_1)
                    else:
                        T.matrix_load(A[by * block_M, 0], A_shared[0, :, :])
                        matrix_load_b(B, bx * block_N, 0, B_shared_0[0, :, :])
                        matrix_load_b(B, bx * block_N + sub_block_N, 0, B_shared_1[0, :, :])
                        T.sched_barrier()
                        T.s_waitcnt(0)
                        T.sync_warp()
                        T.ds_read_format(A_shared[0, :, :], A_local_0)
                        T.ds_read_format(B_shared_0[0, :, :], B_local_0)
                        T.ds_read_format(B_shared_1[0, :, :], B_local_1)
                        T.sched_barrier()
                        T.matrix_load(A[by * block_M, block_K], A_shared[1, :, :])
                        matrix_load_b(B, bx * block_N, block_K, B_shared_0[1, :, :])
                        matrix_load_b(B, bx * block_N + sub_block_N, block_K, B_shared_1[1, :, :])

                        for k in T.Serial(k_loop - 1):
                            base = 2 * (k + 1)
                            T.s_waitcnt(0, flag="lgkmcnt")
                            T.sync_warp()

                            T.sched_barrier()
                            T.gemm(A_local_0, B_local_0, C_local_0, transpose_B=transpose_B)
                            T.sched_barrier()
                            T.s_waitcnt(0)
                            T.sync_warp()
                            T.matrix_load(A[by * block_M, base * block_K], A_shared[0, :, :])
                            matrix_load_b(B, bx * block_N, base * block_K, B_shared_0[0, :, :])
                            matrix_load_b(B, bx * block_N + sub_block_N, base * block_K, B_shared_1[0, :, :])
                            T.s_waitcnt(15)  # avoid clang auto insert vmcnt(0)
                            T.ds_read_format(A_shared[1, :, :], A_local_0_)
                            T.ds_read_format(B_shared_0[1, :, :], B_local_0_)
                            T.ds_read_format(B_shared_1[1, :, :], B_local_1_)
                            T.sched_barrier()
                            T.gemm(A_local_0, B_local_1, C_local_1, transpose_B=transpose_B)

                            T.sched_barrier()
                            T.s_waitcnt(0, flag="lgkmcnt")
                            T.sync_warp()

                            T.sched_barrier()
                            T.gemm(A_local_0_, B_local_0_, C_local_0, transpose_B=transpose_B)
                            T.sched_barrier()
                            T.s_waitcnt(0)
                            T.sync_warp()
                            T.matrix_load(A[by * block_M, (base + 1) * block_K], A_shared[1, :, :])
                            matrix_load_b(B, bx * block_N, (base + 1) * block_K, B_shared_0[1, :, :])
                            matrix_load_b(B, bx * block_N + sub_block_N, (base + 1) * block_K, B_shared_1[1, :, :])
                            T.s_waitcnt(15)  # avoid clang auto insert vmcnt(0)
                            T.ds_read_format(A_shared[0, :, :], A_local_0)
                            T.ds_read_format(B_shared_0[0, :, :], B_local_0)
                            T.ds_read_format(B_shared_1[0, :, :], B_local_1)
                            T.sched_barrier()
                            T.gemm(A_local_0_, B_local_1_, C_local_1, transpose_B=transpose_B)

                    if remain > 1:
                        T.sched_barrier()
                        T.s_waitcnt(4, flag="lgkmcnt")
                        T.sched_barrier()
                        T.gemm(A_local_0, B_local_0, C_local_0, transpose_B=transpose_B)
                        T.sched_barrier()
                        T.s_waitcnt(0)
                        T.sync_warp()
                        T.ds_read_format(A_shared[1, :, :], A_local_0_)
                        T.ds_read_format(B_shared_0[1, :, :], B_local_0_)
                        T.ds_read_format(B_shared_1[1, :, :], B_local_1_)
                        T.sched_barrier()
                        T.s_waitcnt(12, flag="lgkmcnt")
                        T.sched_barrier()
                        T.gemm(A_local_0, B_local_1, C_local_1, transpose_B=transpose_B)
                        T.sched_barrier()

                        T.s_waitcnt(4, flag="lgkmcnt")
                        T.sched_barrier()
                        T.gemm(A_local_0_, B_local_0_, C_local_0, transpose_B=transpose_B)
                        T.sync_warp()
                        T.copy(C_local_0, C_shared_0)
                        T.sync_threads()
                        T.copy(C_shared_0, local_out0, coalesced_width=store_vecsize)
                        T.copy(local_out0, C[by * block_M, bx * block_N])
                        T.s_waitcnt(0, flag="lgkmcnt")
                        T.sched_barrier()
                        T.gemm(A_local_0_, B_local_1_, C_local_1, transpose_B=transpose_B)
                        T.sync_threads()
                        T.copy(C_local_1, C_shared_0)
                        T.sync_threads()
                        T.copy(C_shared_0, local_out1, coalesced_width=store_vecsize)
                        T.copy(local_out1, C[by * block_M, bx * block_N + sub_block_N])
                    else:
                        T.sched_barrier()
                        T.s_waitcnt(4, flag="lgkmcnt")
                        T.sched_barrier()
                        T.gemm(A_local_0, B_local_0, C_local_0, transpose_B=transpose_B)
                        T.sync_warp()
                        T.copy(C_local_0, C_shared_0)
                        T.sync_threads()
                        T.copy(C_shared_0, local_out0, coalesced_width=store_vecsize)
                        T.copy(local_out0, C[by * block_M, bx * block_N])
                        T.sched_barrier()
                        T.s_waitcnt(0, flag="lgkmcnt")
                        T.sched_barrier()
                        T.gemm(A_local_0, B_local_1, C_local_1, transpose_B=transpose_B)
                        T.sync_threads()
                        T.copy(C_local_1, C_shared_0)
                        T.sync_threads()
                        T.copy(C_shared_0, local_out1, coalesced_width=store_vecsize)
                        T.copy(local_out1, C[by * block_M, bx * block_N + sub_block_N])

    return _gemm_persistent


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def gemm_persistent_v4_split_m(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    num_stages,
    thread_num,
    group_size=1,
    wgs_per_cu=1,
    dtype="float16",
    accum_dtype="float",
    transpose_B=True,
):
    """Persistent GEMM v4 supporting K-major [N, K] and N-major [K, N] B layouts."""
    cu_num = torch.cuda.get_device_properties("cuda").multi_processor_count
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    grid_size = T.min(m_blocks * n_blocks, wgs_per_cu * cu_num)
    # waves = T.ceildiv(m_blocks * n_blocks, grid_size)

    split_m = 2
    sub_block_M = block_M // split_m
    split_n = 1
    sub_block_N = block_N
    k_loop_ = T.ceildiv(K, block_K)
    k_loop = T.ceildiv(k_loop_, 2)
    remain = 2 - (k_loop_ % 2)
    store_vecsize = 8 if N % 8 == 0 else 4 if N % 4 == 0 else 2 if N % 2 == 0 else 1

    @T.macro
    def matrix_load_b(B, n_offset, k_offset, B_shared):
        if transpose_B:
            T.matrix_load(B[n_offset, k_offset], B_shared)
        else:
            T.matrix_load(B[k_offset, n_offset], B_shared)

    @T.prim_func
    def _gemm_persistent(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K) if transpose_B else (K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_size, threads=thread_num) as (block_id):
            warp_idx = T.get_warp_idx()
            A_shared_0 = T.alloc_shared((2, sub_block_M, block_K), dtype)
            A_shared_1 = T.alloc_shared((2, sub_block_M, block_K), dtype)
            B_shared_0 = T.alloc_shared((2, sub_block_N, block_K) if transpose_B else (2, block_K, sub_block_N), dtype)

            A_local_0 = T.alloc_fragment((sub_block_M, block_K), dtype)
            A_local_1 = T.alloc_fragment((sub_block_M, block_K), dtype)
            B_local_0 = T.alloc_fragment((sub_block_N, block_K) if transpose_B else (block_K, sub_block_N), dtype)

            A_local_0_ = T.alloc_fragment((sub_block_M, block_K), dtype)
            A_local_1_ = T.alloc_fragment((sub_block_M, block_K), dtype)
            B_local_0_ = T.alloc_fragment((sub_block_N, block_K) if transpose_B else (block_K, sub_block_N), dtype)

            C_local_0 = T.alloc_fragment((sub_block_M, sub_block_N), accum_dtype)
            C_local_1 = T.alloc_fragment((sub_block_M, sub_block_N), accum_dtype)

            local_out0 = T.alloc_fragment((sub_block_M, sub_block_N), dtype)
            local_out1 = T.alloc_fragment((sub_block_M, sub_block_N), dtype)

            C_shared_0 = T.alloc_shared((sub_block_M, sub_block_N), dtype)
            T.annotate_layout(
                {
                    C_shared_0: tl.layout.make_hcu_swizzled_layout(C_shared_0, major_pack=2),
                }
            )

            # bx: N, by: M
            for bx, by in T.Persistent(
                [T.ceildiv(N, block_N), T.ceildiv(M, block_M)], wgs_per_cu * cu_num, block_id, group_size=group_size
            ):
                if by * block_M < M and bx * block_N < N:
                    T.clear(C_local_0)
                    T.clear(C_local_1)

                    if warp_idx < 4:
                        T.matrix_load(A[by * block_M, 0], A_shared_0[0, :, :])
                        T.matrix_load(A[by * block_M + sub_block_M, 0], A_shared_1[0, :, :])
                        matrix_load_b(B, bx * block_N, 0, B_shared_0[0, :, :])
                        T.sched_barrier()
                        T.s_waitcnt(0)
                        T.sync_warp()
                        T.matrix_load(A[by * block_M, block_K], A_shared_0[1, :, :])
                        T.matrix_load(A[by * block_M + sub_block_M, block_K], A_shared_1[1, :, :])
                        matrix_load_b(B, bx * block_N, block_K, B_shared_0[1, :, :])
                        T.sched_barrier()
                        T.ds_read_format(A_shared_0[0, :, :], A_local_0)
                        T.ds_read_format(A_shared_1[0, :, :], A_local_1)
                        T.ds_read_format(B_shared_0[0, :, :], B_local_0)

                        for k in T.Serial(k_loop - 1):
                            base = 2 * (k + 1)
                            T.s_waitcnt(0, flag="lgkmcnt")
                            T.sync_warp()
                            T.sched_barrier()
                            T.matrix_load(A[by * block_M, base * block_K], A_shared_0[0, :, :])
                            T.matrix_load(A[by * block_M + sub_block_M, base * block_K], A_shared_1[0, :, :])
                            matrix_load_b(B, bx * block_N, base * block_K, B_shared_0[0, :, :])

                            T.s_waitcnt(3)
                            T.sync_warp()
                            T.ds_read_format(A_shared_0[1, :, :], A_local_0_)
                            T.ds_read_format(B_shared_0[1, :, :], B_local_0_)
                            T.sched_barrier()
                            T.gemm(A_local_0, B_local_0, C_local_0, transpose_B=transpose_B)
                            T.gemm(A_local_1, B_local_0, C_local_1, transpose_B=transpose_B)
                            T.sched_barrier()
                            T.ds_read_format(A_shared_1[1, :, :], A_local_1_)

                            T.sched_barrier()
                            T.s_waitcnt(0, flag="lgkmcnt")
                            T.sync_warp()
                            T.sched_barrier()
                            T.matrix_load(A[by * block_M, (base + 1) * block_K], A_shared_0[1, :, :])
                            T.matrix_load(A[by * block_M + sub_block_M, (base + 1) * block_K], A_shared_1[1, :, :])
                            matrix_load_b(B, bx * block_N, (base + 1) * block_K, B_shared_0[1, :, :])

                            T.sched_barrier()
                            T.s_waitcnt(3)
                            T.sync_warp()
                            T.ds_read_format(A_shared_0[0, :, :], A_local_0)
                            T.ds_read_format(B_shared_0[0, :, :], B_local_0)
                            T.sched_barrier()
                            T.gemm(A_local_0_, B_local_0_, C_local_0, transpose_B=transpose_B)
                            T.gemm(A_local_1_, B_local_0_, C_local_1, transpose_B=transpose_B)
                            T.sched_barrier()
                            T.ds_read_format(A_shared_1[0, :, :], A_local_1)
                    else:
                        T.matrix_load(A[by * block_M, 0], A_shared_0[0, :, :])
                        T.matrix_load(A[by * block_M + sub_block_M, 0], A_shared_1[0, :, :])
                        matrix_load_b(B, bx * block_N, 0, B_shared_0[0, :, :])
                        T.sched_barrier()
                        T.s_waitcnt(0)
                        T.sync_warp()
                        T.ds_read_format(A_shared_0[0, :, :], A_local_0)
                        T.ds_read_format(A_shared_1[0, :, :], A_local_1)
                        T.ds_read_format(B_shared_0[0, :, :], B_local_0)
                        T.sched_barrier()
                        T.matrix_load(A[by * block_M, block_K], A_shared_0[1, :, :])
                        T.matrix_load(A[by * block_M + sub_block_M, block_K], A_shared_1[1, :, :])
                        matrix_load_b(B, bx * block_N, block_K, B_shared_0[1, :, :])

                        for k in T.Serial(k_loop - 1):
                            base = 2 * (k + 1)
                            T.s_waitcnt(0, flag="lgkmcnt")
                            T.sync_warp()

                            T.sched_barrier()
                            T.gemm(A_local_0, B_local_0, C_local_0, transpose_B=transpose_B)
                            T.sched_barrier()
                            T.s_waitcnt(0)
                            T.sync_warp()
                            T.matrix_load(A[by * block_M, base * block_K], A_shared_0[0, :, :])
                            T.matrix_load(A[by * block_M + sub_block_M, base * block_K], A_shared_1[0, :, :])
                            matrix_load_b(B, bx * block_N, base * block_K, B_shared_0[0, :, :])
                            T.s_waitcnt(15)  # avoid clang auto insert vmcnt(0)
                            T.ds_read_format(A_shared_0[1, :, :], A_local_0_)
                            T.ds_read_format(A_shared_1[1, :, :], A_local_1_)
                            T.ds_read_format(B_shared_0[1, :, :], B_local_0_)
                            T.sched_barrier()
                            T.gemm(A_local_1, B_local_0, C_local_1, transpose_B=transpose_B)

                            T.sched_barrier()
                            T.s_waitcnt(0, flag="lgkmcnt")
                            T.sync_warp()

                            T.sched_barrier()
                            T.gemm(A_local_0_, B_local_0_, C_local_0, transpose_B=transpose_B)
                            T.sched_barrier()
                            T.s_waitcnt(0)
                            T.sync_warp()
                            T.matrix_load(A[by * block_M, (base + 1) * block_K], A_shared_0[1, :, :])
                            T.matrix_load(A[by * block_M + sub_block_M, (base + 1) * block_K], A_shared_1[1, :, :])
                            matrix_load_b(B, bx * block_N, (base + 1) * block_K, B_shared_0[1, :, :])
                            T.s_waitcnt(15)  # avoid clang auto insert vmcnt(0)
                            T.ds_read_format(A_shared_0[0, :, :], A_local_0)
                            T.ds_read_format(A_shared_1[0, :, :], A_local_1)
                            T.ds_read_format(B_shared_0[0, :, :], B_local_0)
                            T.sched_barrier()
                            T.gemm(A_local_1_, B_local_0_, C_local_1, transpose_B=transpose_B)

                    if remain > 1:
                        T.sched_barrier()
                        T.s_waitcnt(4, flag="lgkmcnt")
                        T.sched_barrier()
                        T.gemm(A_local_0, B_local_0, C_local_0, transpose_B=transpose_B)
                        T.sched_barrier()
                        T.s_waitcnt(0)
                        T.sync_warp()
                        T.ds_read_format(A_shared_0[1, :, :], A_local_0_)
                        T.ds_read_format(A_shared_1[1, :, :], A_local_1_)
                        T.ds_read_format(B_shared_0[1, :, :], B_local_0_)
                        T.sched_barrier()
                        T.s_waitcnt(12, flag="lgkmcnt")
                        T.sched_barrier()
                        T.gemm(A_local_1, B_local_0, C_local_1, transpose_B=transpose_B)
                        T.sched_barrier()

                        T.s_waitcnt(4, flag="lgkmcnt")
                        T.sched_barrier()
                        T.gemm(A_local_0_, B_local_0_, C_local_0, transpose_B=transpose_B)
                        T.sync_warp()
                        T.copy(C_local_0, C_shared_0)
                        T.sync_threads()
                        T.copy(C_shared_0, local_out0, coalesced_width=store_vecsize)
                        T.copy(local_out0, C[by * block_M, bx * block_N])
                        T.s_waitcnt(0, flag="lgkmcnt")
                        T.sched_barrier()
                        T.gemm(A_local_1_, B_local_0_, C_local_1, transpose_B=transpose_B)
                        T.sync_threads()
                        T.copy(C_local_1, C_shared_0)
                        T.sync_threads()
                        T.copy(C_shared_0, local_out1, coalesced_width=store_vecsize)
                        T.copy(local_out1, C[by * block_M + sub_block_M, bx * block_N])
                    else:
                        T.sched_barrier()
                        T.s_waitcnt(4, flag="lgkmcnt")
                        T.sched_barrier()
                        T.gemm(A_local_0, B_local_0, C_local_0, transpose_B=transpose_B)
                        T.sync_warp()
                        T.copy(C_local_0, C_shared_0)
                        T.sync_threads()
                        T.copy(C_shared_0, local_out0, coalesced_width=store_vecsize)
                        T.copy(local_out0, C[by * block_M, bx * block_N])
                        T.sched_barrier()
                        T.s_waitcnt(0, flag="lgkmcnt")
                        T.sched_barrier()
                        T.gemm(A_local_1, B_local_0, C_local_1, transpose_B=transpose_B)
                        T.sync_threads()
                        T.copy(C_local_1, C_shared_0)
                        T.sync_threads()
                        T.copy(C_shared_0, local_out1, coalesced_width=store_vecsize)
                        T.copy(local_out1, C[by * block_M + sub_block_M, bx * block_N])

    return _gemm_persistent


# for small size like(1024 * 1024 * 1024)
@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def gemm_persistent_v5(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    num_stages,
    thread_num,
    group_size=8,
    wgs_per_cu=2,
    dtype="float16",
    accum_dtype="float",
    use_mls=False,
):
    cu_num = torch.cuda.get_device_properties("cuda").multi_processor_count
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    grid_size = T.min(m_blocks * n_blocks, wgs_per_cu * cu_num)
    # waves = T.ceildiv(m_blocks * n_blocks, grid_size)
    k_loop = T.ceildiv(K, block_K)

    @T.prim_func
    def _gemm_persistent(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_size, threads=thread_num) as (block_id):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            A_local = T.alloc_fragment((block_M, block_K), dtype)
            B_local = T.alloc_fragment((block_N, block_K), dtype)

            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            C_shared = T.alloc_shared((block_M, block_N), dtype)
            if use_mls:
                # MLS: matrix_load / ds_read_format / gemm infer A/B shared layout.
                T.annotate_layout(
                    {
                        C_shared: tl.layout.make_hcu_swizzled_layout(C_shared, major_pack=2),
                    }
                )
            else:
                T.annotate_layout(
                    {
                        C_shared: tl.layout.make_hcu_swizzled_layout(C_shared, major_pack=2),
                        B_shared: tl.layout.make_hcu_swizzled_layout(B_shared, major_pack=2),
                        A_shared: tl.layout.make_hcu_swizzled_layout(A_shared, major_pack=2),
                    }
                )

            # bx: N, by: M
            for bx, by in T.Persistent(
                [T.ceildiv(N, block_N), T.ceildiv(M, block_M)], wgs_per_cu * cu_num, block_id, group_size=group_size
            ):
                if by * block_M < M and bx * block_N < N:
                    T.clear(C_local)

                    if use_mls:
                        T.matrix_load(A[by * block_M, 0], A_shared)
                        T.matrix_load(B[bx * block_N, 0], B_shared)
                        for k in T.Serial(k_loop - 1):
                            T.s_waitcnt(0)
                            T.sync_warp()
                            T.ds_read_format(A_shared, A_local)
                            T.ds_read_format(B_shared, B_local)
                            T.sync_threads()
                            T.matrix_load(A[by * block_M, (k + 1) * block_K], A_shared)
                            T.matrix_load(B[bx * block_N, (k + 1) * block_K], B_shared)
                            T.gemm(A_local, B_local, C_local, transpose_B=True)

                        T.s_waitcnt(0)
                        T.sync_warp()
                        T.ds_read_format(A_shared, A_local)
                        T.ds_read_format(B_shared, B_local)
                        T.gemm(A_local, B_local, C_local, transpose_B=True)
                        # T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                    else:
                        A_local_pre = T.alloc_fragment((block_M, block_K), dtype)
                        B_local_pre = T.alloc_fragment((block_N, block_K), dtype)
                        T.copy(A[by * block_M, 0], A_local_pre)
                        T.copy(B[bx * block_N, 0], B_local_pre)
                        for k in T.Serial(k_loop - 1):
                            T.copy(A_local_pre, A_shared)
                            T.sync_threads()
                            T.copy(A[by * block_M, (k + 1) * block_K], A_local_pre)
                            T.copy(A_shared, A_local)
                            T.copy(B_local_pre, B_shared)
                            T.sync_threads()
                            T.copy(B[bx * block_N, (k + 1) * block_K], B_local_pre)
                            T.copy(B_shared, B_local)
                            T.gemm(A_local, B_local, C_local, k_pack=2, transpose_B=True)

                        T.copy(A_local_pre, A_shared)
                        T.copy(B_local_pre, B_shared)
                        T.sync_threads()
                        T.copy(A_shared, A_local)
                        T.copy(B_shared, B_local)
                        T.gemm(A_local, B_local, C_local, k_pack=2, transpose_B=True)

                    T.copy(C_local, C_shared)
                    T.sync_threads()
                    T.copy(C_shared, C[by * block_M, bx * block_N])

    return _gemm_persistent


# FIXME: Boundary check is not considered, so non-divisible block_N and group_size may cause
#        correctness issue.
# Note: Use pass_configs={"tl.disable_safe_memory_legalize": True} to disable safe memory legalize
#       during using vectorized with swizzled layout.
@tl.jit(out_idx=[-1])
def gemm_persistent(
    M, N, K, block_M, block_N, block_K, num_stages, thread_num, group_size=8, wgs_per_cu=2, dtype="float16", accum_dtype="float"
):
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
                    # T.annotate_layout({
                    #    A: make_block_swizzle_layout(A, block_M, block_K),
                    #    A_shared: make_linear_layout(A_shared),
                    #    B_shared: make_linear_layout(B_shared),
                    # })
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        # for i in T.Parallel(block_M):
                        #    for j in T.Parallel(block_K):
                        #        # Apply swizzle layout to local block indices
                        #        si, sj = get_swizzle_layout(i, j, block_K, dtype, 128)
                        #        # Global indices in A (swizzled)
                        #        gi = bx * block_M + si
                        #        gk = k * block_K + sj
                        #        # Load from swizzled global positions
                        #        A_shared[i, j] = A[gi, gk]
                        # for i in T.Parallel(block_N):
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
