"""
Split-K GEMM implementation.
"""

import tilelang as tl
import tilelang.language as T

# Note: It is a little tricky to integrate the atomic_barrier splitk implementation with the other
# implementations into the same profiler because they have different output tensors.
# As a reference, the performance of atomic_barrier splitk is better than the atomic_direct splitk,
# which brings about 1.21x speedup: 73TFlops --> 88TFlops, while both of them under-perform compared
# to the persistent kernel. So we leave it here as a reference for future optimization but adopt the
# atomic_direct version by default for the benchmark.
_USE_ATOMIC_BARRIER = False


def gemm_splitk(
    M, N, K, block_M, block_N, block_K, num_stages, thread_num, enable_rasteration=True, split_k=2, dtype="float16", accum_dtype="float"
):
    N_blocks = T.ceildiv(N, block_N)
    M_blocks = T.ceildiv(M, block_M)
    # The final barrier value when all split_k blocks are done
    # Each block sets bit bz, so final value is (1 << split_k) - 1
    barrier_final = (1 << split_k) - 1

    @T.prim_func
    def _gemm_splitk_atomic_barrier(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C_partial: T.Tensor((M_blocks, N_blocks, block_M, block_N, split_k), accum_dtype),
        C: T.Tensor((M, N), accum_dtype),
        barriers: T.Tensor((M_blocks, N_blocks, split_k), "uint32"),
    ):
        with T.Kernel(N_blocks, M_blocks, split_k + 1, threads=thread_num) as (bx, by, bz):
            if bz != split_k:
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                # T.use_swizzle(panel_size=10, enable=enable_rasteration)

                # Step 1: Each split-k block computes its partial sum
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, split_k * block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, bz * block_K + k * split_k * block_K], A_shared, coalesced_width=8)
                    T.copy(B[bx * block_N, bz * block_K + k * split_k * block_K], B_shared, coalesced_width=8)
                    T.gemm(A_shared, B_shared, C_local, k_pack=2, transpose_B=True)

                # Step 2: Write partial sum to global memory (C_partial)
                T.copy(C_local, C_partial[by, bx, :, :, bz])

                # Step 3: Signal completion via atomic barrier (use release to ensure visibility)
                # The release memory order ensures that the write to C_partial is visible
                # before the barrier update is seen by other threads
                T.atomic_store(barriers[by, bx, bz], 1, memory_order="release")

            # Block (by, bx, split_k) is the final block that accumulates all partial sums
            else:
                C_accum_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                processed_mask = T.alloc_local((1,), "uint32")

                T.clear(C_accum_local)
                T.fill(processed_mask, 0)
                # Reusable buffer for loading partial sums (better memory coalescing than element-wise access)
                C_partial_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                # Busy-wait until all split_k blocks for this (by, bx) tile finish
                # Use acquire memory ordering to ensure we see all updates from other blocks
                while processed_mask[0] != barrier_final:
                    # T.print(current_barrier, "current_barrier")
                    # Check each split_k block
                    for k in T.serial(split_k):
                        current_barrier = T.atomic_load(barriers[by, bx, k], memory_order="acquire")
                        bit_mask = 1 << k
                        # If this block is done and not yet processed
                        if (current_barrier != 0) and (processed_mask[0] & bit_mask) == 0:
                            # Copy the partial sum tile to local buffer (coalesced memory access)
                            T.copy(C_partial[by, bx, :, :, k], C_partial_local)
                            # Accumulate this partial sum
                            for i, j in T.Parallel(block_M, block_N):
                                C_accum_local[i, j] = C_accum_local[i, j] + C_partial_local[i, j]
                            # Mark this block as processed
                            processed_mask[0] = processed_mask[0] | bit_mask

                # Write final result to output
                T.copy(C_accum_local, C[by * block_M, bx * block_N])

    @T.prim_func
    def _gemm_splitk_atomic_direct(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(N_blocks, M_blocks, split_k, threads=thread_num) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, split_k * block_K), num_stages=num_stages):
                T.copy(A[by * block_M, bz * block_K + k * split_k * block_K], A_shared, coalesced_width=8)
                T.copy(B[bx * block_N, bz * block_K + k * split_k * block_K], B_shared, coalesced_width=8)
                T.gemm(A_shared, B_shared, C_local, k_pack=2, transpose_B=True)

            # Direct atomic accumulation
            for i, j in T.Parallel(block_M, block_N):
                T.atomic_add(C[by * block_M + i, bx * block_N + j], C_local[i, j])

    # Compile with appropriate out_idx for each variant
    if _USE_ATOMIC_BARRIER:
        # For atomic_barrier: params are A, B, C_partial, C, barriers
        # Outputs are C_partial (index 2) and C (index 3)
        return tl.compile(_gemm_splitk_atomic_barrier, out_idx=[2, 3])
    else:
        # For atomic_direct: params are A, B, C
        # Output is C (index 2)
        return tl.compile(_gemm_splitk_atomic_direct, out_idx=[2])
