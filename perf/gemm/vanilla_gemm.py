"""
Vanilla GEMM implementation.
"""

import tilelang as tl
import tilelang.language as T
from perf.gemm.utils import get_configs, _run_autotuner


@tl.jit(out_idx=[-1])
def gemm_vanilla(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    num_stages,
    thread_num,
    enable_rasteration=True,
    group_size=8,
    wgs_per_cu=1,
    dtype="float16",
    accum_dtype="float",
    use_mls=False,
):
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
                        if use_mls:
                            T.matrix_load(A[bx * block_M, k * block_K], A_shared)
                            T.matrix_load(B[by * block_N, k * block_K], B_shared)
                            T.gemm(A_shared, B_shared, C_local, k_pack=1, transpose_B=True)
                        else:
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
                    if use_mls:
                        T.matrix_load(A[by * block_M, k * block_K], A_shared)
                        T.matrix_load(B[bx * block_N, k * block_K], B_shared)
                        T.gemm(A_shared, B_shared, C_local, k_pack=1, transpose_B=True)
                    else:
                        T.copy(A[by * block_M, k * block_K], A_shared, coalesced_width=8)
                        T.copy(B[bx * block_N, k * block_K], B_shared, coalesced_width=8)
                        T.gemm(A_shared, B_shared, C_local, k_pack=2, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

    return _gemm_vanilla


@tl.jit(
    out_idx=[-1],
    pass_configs={
        "tl.disable_thread_storage_sync": False,
    },
)
def gemm_vanilla_v1(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    num_stages,
    thread_num,
    enable_rasteration=True,
    group_size=8,
    wgs_per_cu=1,
    dtype="float16",
    accum_dtype="float",
    use_mls=False,
):
    """
    Vanilla GEMM kernel with optional group swizzling optimization.

    Args:
        group_size: Size of tile groups for swizzling (1 disables group swizzling).
                   When > 1, uses the same group swizzling pattern as gemm_persistent.
    """
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    total_tiles = m_blocks * n_blocks  # noqa: F841

    split_n = 2
    sub_block_N = block_N // split_n

    @T.prim_func
    def _gemm_vanilla(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Use group swizzling optimization
        # with T.Kernel(total_tiles, threads=thread_num) as (tile_id):
        # with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=thread_num) as (bx, by):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            # bx = (tile_id % group_size) + (tile_id // group_size) // n_blocks * group_size
            # by = (tile_id // group_size) % n_blocks

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared_0 = T.alloc_shared((sub_block_N, block_K), dtype)
            # B_shared_1 = T.alloc_shared((sub_block_N, block_K), dtype)

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

            # T.use_swizzle(panel_size=8, order="row", enable=True)
            T.clear(C_local_0)
            T.clear(C_local_1)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if not use_mls:
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
                else:
                    # A -> A_shared
                    T.matrix_load(A[by * block_M, k * block_K], A_shared)

                    # preload B Block N_0
                    T.matrix_load(B[bx * block_N, k * block_K], B_shared_0)

                    # A_shared -> A_local_0_
                    T.ptx_wait_group(1)
                    T.sync_threads()
                    T.ds_read_format(A_shared, A_local_0_)

                    # B_shared_0 -> B_local_0_
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    T.ds_read_format(B_shared_0, B_local_0_)

                    # preload B Block N_1
                    T.matrix_load(B[bx * block_N + sub_block_N, k * block_K], B_shared_0)

                    # B_shared_0 -> B_local_1_
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    T.ds_read_format(B_shared_0, B_local_1_)

                    T.gemm(A_local_0_, B_local_0_, C_local_0, k_pack=1, transpose_B=True)
                    T.gemm(A_local_0_, B_local_1_, C_local_1, k_pack=1, transpose_B=True)

            T.copy(C_local_0, C_shared_0)
            T.copy(C_shared_0, C[by * block_M, bx * block_N])
            T.copy(C_local_1, C_shared_0)
            T.copy(C_shared_0, C[by * block_M, bx * block_N + sub_block_N])

    return _gemm_vanilla


@tl.jit(
    out_idx=[-1],
    pass_configs={
        tl.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def gemm_vanilla_v2(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    num_stages,
    thread_num,
    enable_rasteration=False,
    group_size=1,
    wgs_per_cu=1,
    dtype="float16",
    accum_dtype="float",
    use_mls=False,
    transpose_B=True,
):
    """
    Vanilla GEMM kernel with optional group swizzling optimization.

    Args:
        group_size: Size of tile groups for swizzling (1 disables group swizzling).
                   When > 1, uses the same group swizzling pattern as gemm_persistent.
        transpose_B: Whether B uses K-major [N, K] layout. Set to False for N-major [K, N].
    """
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    total_tiles = m_blocks * n_blocks  # noqa: F841

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
    def _gemm_vanilla(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K) if transpose_B else (K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            T.use_swizzle(
                panel_size=group_size,
                order="row",
                enable=group_size > 1 or enable_rasteration,
            )
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

            local_out0 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)
            local_out1 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)

            C_shared_0 = T.alloc_shared((block_M, sub_block_N), dtype)
            T.annotate_layout(
                {
                    C_shared_0: tl.layout.make_hcu_swizzled_layout(C_shared_0, major_pack=2),
                }
            )

            T.clear(C_local_0)
            T.clear(C_local_1)

            if warp_idx < 4:
                T.matrix_load(A[by * block_M, 0], A_shared[0, :, :])
                matrix_load_b(B, bx * block_N, 0, B_shared_0[0, :, :])
                matrix_load_b(B, bx * block_N + sub_block_N, 0, B_shared_1[0, :, :])
                T.sched_barrier()
                T.ptx_wait_group(0)
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
                    # T.sched_barrier()
                    T.s_waitcnt(0, flag="lgkmcnt")
                    T.sync_warp()
                    T.sched_barrier()
                    T.matrix_load(A[by * block_M, base * block_K], A_shared[0, :, :])
                    matrix_load_b(B, bx * block_N, base * block_K, B_shared_0[0, :, :])
                    matrix_load_b(B, bx * block_N + sub_block_N, base * block_K, B_shared_1[0, :, :])

                    # Keep the three newer shared_0 MatrixLoad groups in flight.
                    T.ptx_wait_group(3)
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
                    # Keep the three newer shared_1 MatrixLoad groups in flight.
                    T.ptx_wait_group(3)
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
                T.ptx_wait_group(0)
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
                    # T.sched_barrier()
                    T.s_waitcnt(0, flag="lgkmcnt")
                    T.sync_warp()

                    T.sched_barrier()
                    T.gemm(A_local_0, B_local_0, C_local_0, transpose_B=transpose_B)
                    T.sched_barrier()
                    T.ptx_wait_group(0)
                    T.sync_warp()
                    T.matrix_load(A[by * block_M, base * block_K], A_shared[0, :, :])
                    matrix_load_b(B, bx * block_N, base * block_K, B_shared_0[0, :, :])
                    matrix_load_b(B, bx * block_N + sub_block_N, base * block_K, B_shared_1[0, :, :])
                    T.s_waitcnt(15)  # just to avoid clang auto insert vmcnt 0
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
                    T.ptx_wait_group(0)
                    T.sync_warp()
                    T.matrix_load(A[by * block_M, (base + 1) * block_K], A_shared[1, :, :])
                    matrix_load_b(B, bx * block_N, (base + 1) * block_K, B_shared_0[1, :, :])
                    matrix_load_b(B, bx * block_N + sub_block_N, (base + 1) * block_K, B_shared_1[1, :, :])
                    T.s_waitcnt(15)  # just to avoid clang auto insert vmcnt 0
                    T.ds_read_format(A_shared[0, :, :], A_local_0)
                    T.ds_read_format(B_shared_0[0, :, :], B_local_0)
                    T.ds_read_format(B_shared_1[0, :, :], B_local_1)
                    T.sched_barrier()
                    T.gemm(A_local_0_, B_local_1_, C_local_1, transpose_B=transpose_B)

            if remain > 1:
                T.gemm(A_local_0, B_local_0, C_local_0, transpose_B=transpose_B)
                T.ptx_wait_group(0)
                T.sync_warp()
                T.ds_read_format(A_shared[1, :, :], A_local_0_)
                T.ds_read_format(B_shared_0[1, :, :], B_local_0_)
                T.ds_read_format(B_shared_1[1, :, :], B_local_1_)
                T.gemm(A_local_0, B_local_1, C_local_1, transpose_B=transpose_B)
                T.sched_barrier()
                T.gemm(A_local_0_, B_local_0_, C_local_0, transpose_B=transpose_B)
                T.sync_warp()
                T.copy(C_local_0, C_shared_0)
                T.sync_threads()
                T.copy(C_shared_0, local_out0, coalesced_width=store_vecsize)
                T.copy(local_out0, C[by * block_M, bx * block_N])
                T.gemm(A_local_0_, B_local_1_, C_local_1, transpose_B=transpose_B)
                T.sync_threads()
                T.copy(C_local_1, C_shared_0)
                T.sync_threads()
                T.copy(C_shared_0, local_out1, coalesced_width=store_vecsize)
                T.copy(local_out1, C[by * block_M, bx * block_N + sub_block_N])
            else:
                T.gemm(A_local_0, B_local_0, C_local_0, transpose_B=transpose_B)
                T.sync_warp()
                T.copy(C_local_0, C_shared_0)
                T.sync_threads()
                T.copy(C_shared_0, local_out0, coalesced_width=store_vecsize)
                T.copy(local_out0, C[by * block_M, bx * block_N])
                T.gemm(A_local_0, B_local_1, C_local_1, transpose_B=transpose_B)
                T.sync_threads()
                T.copy(C_local_1, C_shared_0)
                T.sync_threads()
                T.copy(C_shared_0, local_out1, coalesced_width=store_vecsize)
                T.copy(local_out1, C[by * block_M, bx * block_N + sub_block_N])

            # T.sync_warp()
            # T.copy(C_local_0, C_shared_0)
            # T.sync_threads()
            # T.copy(C_shared_0, C[by * block_M, bx * block_N])
            # T.sync_threads()
            # T.copy(C_local_1, C_shared_0)
            # T.sync_threads()
            # T.copy(C_shared_0, C[by * block_M, bx * block_N + sub_block_N])

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
                with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
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
