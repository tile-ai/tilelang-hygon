"""
Stream-K GEMM implementation.
"""

import torch
import tilelang as tl
import tilelang.language as T


@tl.jit(out_idx=[-1])
def gemm_streamk(M, N, K, block_M, block_N, block_K, num_stages, thread_num,
           dtype="float16", accum_dtype="float"):
    """
    Stream-K GEMM implementation that linearizes the iteration space
    to achieve perfect load balancing across CUs.
    """
    cu_num = torch.cuda.get_device_properties("cuda").multi_processor_count
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    total_tiles = m_blocks * n_blocks
    iters_per_tile = T.ceildiv(K, block_K)

    # Calculate Stream-K parameters
    streamk_programs = cu_num
    streamk_tiles = total_tiles % streamk_programs
    if (total_tiles - streamk_tiles > streamk_programs):
        streamk_tiles += streamk_programs

    blocking_tiles = total_tiles - streamk_tiles
    streamk_iters = streamk_tiles * iters_per_tile
    streamk_full_tiles = streamk_iters // streamk_programs
    streamk_partial_tiles = streamk_iters % streamk_programs
    sm_partition_factor = T.max(blocking_tiles // streamk_programs, 1)

    @T.macro
    def compute_first_wave(
        pid: T.int32,
        A_buf: T.Tensor,
        A_buf_shared: T.SharedBuffer,
        B_buf: T.Tensor,
        B_buf_shared: T.SharedBuffer,
        C: T.Tensor,
        C_local: T.LocalBuffer,
    ):
        start_iter = T.alloc_fragment((1,), "int32", "local")
        end_iter = T.alloc_fragment((1,), "int32", "local")

        start_iter[0] = pid * streamk_full_tiles + T.min(pid, streamk_partial_tiles)
        last_iter = (pid + 1) * streamk_full_tiles + T.min(pid + 1, streamk_partial_tiles)

        while start_iter[0] < last_iter:
            end_iter[0] = T.min(
                start_iter[0] + (iters_per_tile - (start_iter[0] % iters_per_tile)),
                last_iter,
            )

            tile_id = start_iter[0] // iters_per_tile
            remain_iters = start_iter[0] % iters_per_tile
            pid_m = tile_id // n_blocks
            pid_n = tile_id % n_blocks

            T.clear(C_local)
            for k in T.Pipelined(end_iter[0] - start_iter[0], num_stages=num_stages):
                k_offset = k + (start_iter[0] % iters_per_tile)
                T.copy(
                    A_buf[pid_m * block_M, k_offset * block_K],
                    A_buf_shared,
                    coalesced_width=8
                )
                T.copy(
                    B_buf[pid_n * block_N, k_offset * block_K],
                    B_buf_shared,
                    coalesced_width=8
                )
                T.gemm(A_buf_shared, B_buf_shared, C_local, k_pack=2, transpose_B=True)

            # If this is a complete tile (started at iteration 0 and ends at tile boundary)
            if remain_iters == 0 and (end_iter[0] % iters_per_tile == 0):
                T.copy(C_local, C[pid_m * block_M, pid_n * block_N])
            else:
                # Partial tile: use atomic add
                for i, j in T.Parallel(block_M, block_N):
                    T.atomic_add(C[pid_m * block_M + i, pid_n * block_N + j], C_local[i, j])

            start_iter[0] = end_iter[0]

    @T.macro
    def compute_full_tiles(
        pid: T.int32,
        A_buf: T.Tensor,
        A_shared: T.SharedBuffer,
        B_buf: T.Tensor,
        B_shared: T.SharedBuffer,
        C: T.Tensor,
        C_local: T.LocalBuffer,
    ):
        for p in T.serial(sm_partition_factor):
            tile_id = pid + streamk_tiles + p * streamk_programs
            if tile_id < total_tiles:
                pid_m = tile_id // n_blocks
                pid_n = tile_id % n_blocks
                T.clear(C_local)

                for k in T.Pipelined(iters_per_tile, num_stages=num_stages):
                    T.copy(A_buf[pid_m * block_M, k * block_K], A_shared, coalesced_width=8)
                    T.copy(B_buf[pid_n * block_N, k * block_K], B_shared, coalesced_width=8)
                    T.gemm(A_shared, B_shared, C_local, k_pack=2, transpose_B=True)
                T.copy(C_local, C[pid_m * block_M, pid_n * block_N])

    @T.prim_func
    def _gemm_streamk(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            #C: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(streamk_programs, threads=thread_num) as pid:
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            A_shared_full = T.alloc_shared((block_M, block_K), dtype)
            B_shared_full = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            compute_first_wave(pid, A, A_shared, B, B_shared, C, C_local)

            if sm_partition_factor > 0:
                compute_full_tiles(pid, A, A_shared_full, B, B_shared_full, C, C_local)

    return _gemm_streamk
