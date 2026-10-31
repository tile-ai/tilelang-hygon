"""GEMM + reduce-scatter with eight compute waves and four dedicated push waves.

Persistent CTAs retain the previous completed FP16 tile in push registers.
Its peer stores are issued between barriers of the next tile's GEMM.
Eight independent 8 KiB LDS stages are reused for the inter-tile handoff.
The four payload stores are spread across the next tile's K loop.
Random/reuse/graph correctness has been checked on two, four and eight ranks.
"""

import tilelang.language as T


def gemm_push_waves(
    M,
    N,
    K,
    rank,
    world_size,
    block_M=256,
    block_N=256,
    block_K=16,
    dtype="float16",
    accum_dtype="float32",
    steady_wait=8,
    shared_0_wait=6,
    swizzle_panel_size=0,
    swizzle_order="row",
    persistent_blocks=80,
):
    if (block_M, block_N, block_K, dtype, accum_dtype) != (256, 256, 16, "float16", "float32"):
        raise ValueError("push_waves retains the original 256x256x16 FP16/FP32 GEMM")
    if world_size < 1 or M % (world_size * block_M) or N % block_N or K < 64 or K % 64:
        raise ValueError("push_waves requires full M/N tiles, rank-aligned row shards and K divisible by 64")
    if not isinstance(persistent_blocks, int) or persistent_blocks < 1:
        raise ValueError("persistent_blocks must be a positive integer")
    if swizzle_order not in ("row", "col"):
        raise ValueError("swizzle_order must be row or col")
    m_tiles, n_tiles = M // block_M, N // block_N
    total_tiles = m_tiles * n_tiles
    # gfx936 on this machine reports 80 CUs. This is a CTA count, not a CU
    # affinity: each CTA processes grid-strided output tiles below.
    grid_size = min(persistent_blocks, total_tiles)
    rows_per_rank = M // world_size
    transpose_B = False
    k_groups = K // (4 * block_K)

    def tile_coordinates(index):
        if not swizzle_panel_size:
            return index % n_tiles, index // n_tiles
        if swizzle_order == "row":
            panel = index // (swizzle_panel_size * n_tiles)
            offset = index % (swizzle_panel_size * n_tiles)
            width = T.min(swizzle_panel_size, m_tiles - panel * swizzle_panel_size)
            column = offset // width
            return T.if_then_else(panel % 2 == 0, column, n_tiles - 1 - column), panel * swizzle_panel_size + offset % width
        panel = index // (swizzle_panel_size * m_tiles)
        offset = index % (swizzle_panel_size * m_tiles)
        width = T.min(swizzle_panel_size, n_tiles - panel * swizzle_panel_size)
        row = offset // width
        return panel * swizzle_panel_size + offset % width, T.if_then_else(panel % 2 == 0, row, m_tiles - 1 - row)

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
                k_tile * block_K : (k_tile + 1) * block_K,
                bx * block_N : (bx + 1) * block_N,
            ],
            B_shared,
        )

    @T.macro
    def store_output(acc, s0, s1, s2, s3, part):
        # Eight compute waves contribute 16 rows from each 64-row band.
        # Each band reuses exactly one 8 KiB input-stage allocation.
        for i, j in T.Parallel(block_M, block_N):
            if i % 64 // 16 == part:
                ptr = T.if_then_else(
                    i // 64 == 0,
                    T.address_of(s0[0, 0]),
                    T.if_then_else(
                        i // 64 == 1, T.address_of(s1[0, 0]), T.if_then_else(i // 64 == 2, T.address_of(s2[0, 0]), T.address_of(s3[0, 0]))
                    ),
                )
                shared = T.make_tensor_from_addr(ptr, (16, block_N), dtype, storage_scope="shared.dyn")
                shared[i % 16, j] = T.cast(acc[i, j], dtype)
        T.s_waitcnt(0, "lgkmcnt")

    @T.macro
    def push_output(payload, peers, by, bx, part):
        remote = T.make_tensor_from_addr(peers[by * block_M // rows_per_rank], (world_size, rows_per_rank, N // 2), "uint32")
        # Each register payload contains four 16-row strips, one per 64-row band.
        # Restore their global rows; each copy keeps the N dimension contiguous.
        for band in T.unroll(4):
            T.copy(
                payload[band * 16 : (band + 1) * 16, :],
                remote[
                    rank,
                    by * block_M % rows_per_rank + band * 64 + part * 16 : by * block_M % rows_per_rank + band * 64 + part * 16 + 16,
                    bx * (block_N // 2) : (bx + 1) * (block_N // 2),
                ],
            )
        T.s_waitcnt(0, "lgkmcnt")

    @T.macro
    def load_payload(s0, s1, s2, s3, payload):
        # Bind the loop layout to the push-wave range (threads 512..767).
        # An unbound buffer annotation would use the wrong thread origin.
        for i, j in T.Parallel(64, 128, loop_layout=T.Fragment((64, 128), lambda i, j: ((i % 8) * 32 + j // 4, (i // 8) * 4 + j % 4))):
            ptr = T.if_then_else(
                i // 16 == 0,
                T.address_of(s0[0, 0]),
                T.if_then_else(
                    i // 16 == 1, T.address_of(s1[0, 0]), T.if_then_else(i // 16 == 2, T.address_of(s2[0, 0]), T.address_of(s3[0, 0]))
                ),
            )
            shared = T.make_tensor_from_addr(ptr, (16, block_N), dtype, storage_scope="shared.dyn")
            payload[i, j] = T.cast(T.reinterpret(shared[i % 16, j * 2], "uint16"), "uint32") | (
                T.cast(T.reinterpret(shared[i % 16, j * 2 + 1], "uint16"), "uint32") << 16
            )
        T.s_waitcnt(0, "lgkmcnt")

    @T.macro
    def receive_output(p0, p1, p2, p3, a0, a1, a2, a3, b0, b1, b2, b3):
        T.sync_warp()
        load_payload(a0, a1, a2, a3, p0)
        T.sync_warp()
        load_payload(b0, b1, b2, b3, p1)
        T.sync_warp()
        load_payload(a0, a1, a2, a3, p2)
        T.sync_warp()
        load_payload(b0, b1, b2, b3, p3)
        T.sync_warp()

    @T.macro
    def wait_first_compute():
        for _ in T.serial(6):
            T.sync_warp()
        for _kg in T.serial(k_groups - 1):
            for _ in T.serial(7):
                T.sync_warp()
        for _ in T.serial(3):
            T.sync_warp()

    @T.prim_func
    def gemm(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), peers: T.Tensor((world_size,), "uint64")):
        with T.Kernel(grid_size, threads=768) as block_id:
            tx = T.get_thread_binding()
            warp_idx = T.get_warp_idx()
            turns = T.ceildiv(total_tiles - block_id, grid_size)
            # Preserve the original eight independent LDS stages. Reuse each
            # stage through an 8 KiB linear output view only after input reads.
            A_shared_0 = T.alloc_shared((block_M, block_K), dtype)
            A_shared_1 = T.alloc_shared((block_M, block_K), dtype)
            A_shared_2 = T.alloc_shared((block_M, block_K), dtype)
            A_shared_3 = T.alloc_shared((block_M, block_K), dtype)
            B_shared_0 = T.alloc_shared((block_K, block_N), dtype)
            B_shared_1 = T.alloc_shared((block_K, block_N), dtype)
            B_shared_2 = T.alloc_shared((block_K, block_N), dtype)
            B_shared_3 = T.alloc_shared((block_K, block_N), dtype)
            out_a0 = T.make_tensor_from_addr(T.address_of(A_shared_0[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            out_a1 = T.make_tensor_from_addr(T.address_of(A_shared_1[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            out_a2 = T.make_tensor_from_addr(T.address_of(A_shared_2[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            out_a3 = T.make_tensor_from_addr(T.address_of(A_shared_3[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            out_b0 = T.make_tensor_from_addr(T.address_of(B_shared_0[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            out_b1 = T.make_tensor_from_addr(T.address_of(B_shared_1[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            out_b2 = T.make_tensor_from_addr(T.address_of(B_shared_2[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            out_b3 = T.make_tensor_from_addr(T.address_of(B_shared_3[0, 0]), (16, block_N), dtype, storage_scope="shared.dyn")
            if tx < 512:
                A_local_0 = T.alloc_fragment((block_M, block_K), dtype)
                B_local_0 = T.alloc_fragment((block_K, block_N), dtype)
                A_local_1 = T.alloc_fragment((block_M, block_K), dtype)
                B_local_1 = T.alloc_fragment((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                # Persistent tile loop: block_id, block_id + grid_size, ...
                for turn in T.serial(turns):
                    bx, by = tile_coordinates(block_id + turn * grid_size)
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
                    # Only completed outputs cross roles. Input LDS is dead here.
                    T.sync_warp()
                    store_output(C_local, out_a0, out_a1, out_a2, out_a3, 0)
                    T.sync_warp()
                    store_output(C_local, out_b0, out_b1, out_b2, out_b3, 1)
                    T.sync_warp()
                    store_output(C_local, out_a0, out_a1, out_a2, out_a3, 2)
                    T.sync_warp()
                    store_output(C_local, out_b0, out_b1, out_b2, out_b3, 3)
                    T.sync_warp()
                    T.sync_warp()  # All four payload fragments are in push registers.
            else:
                # Four packed fragments retain a complete FP16 output tile:
                # 128 KiB / 256 push threads = 128 uint32 registers per lane.
                p0 = T.alloc_fragment((block_M // 4, block_N // 2), "uint32")
                p1 = T.alloc_fragment((block_M // 4, block_N // 2), "uint32")
                p2 = T.alloc_fragment((block_M // 4, block_N // 2), "uint32")
                p3 = T.alloc_fragment((block_M // 4, block_N // 2), "uint32")
                # Peel the first tile: there is no previous output to send.
                wait_first_compute()
                T.sync_warp()
                receive_output(p0, p1, p2, p3, out_a0, out_a1, out_a2, out_a3, out_b0, out_b1, out_b2, out_b3)
                for turn in T.serial(1, turns):
                    prev_bx, prev_by = tile_coordinates(block_id + (turn - 1) * grid_size)
                    for _ in T.serial(6):
                        T.sync_warp()
                    # First batch overlaps the next tile's first K iteration.
                    push_output(p0, peers, prev_by, prev_bx, 0)
                    for kg in T.serial(k_groups - 1):
                        T.sync_warp()
                        if kg == (k_groups - 1) // 4:
                            push_output(p1, peers, prev_by, prev_bx, 1)
                        T.sync_warp()
                        if kg == (k_groups - 1) // 2:
                            push_output(p2, peers, prev_by, prev_bx, 2)
                        T.sync_warp()
                        if kg == 3 * (k_groups - 1) // 4:
                            push_output(p3, peers, prev_by, prev_bx, 3)
                        for _ in T.serial(4):
                            T.sync_warp()
                    T.sync_warp()
                    if k_groups == 1:
                        push_output(p1, peers, prev_by, prev_bx, 1)
                    T.sync_warp()
                    if k_groups == 1:
                        push_output(p2, peers, prev_by, prev_bx, 2)
                    T.sync_warp()
                    if k_groups == 1:
                        push_output(p3, peers, prev_by, prev_bx, 3)
                    T.sync_warp()
                    receive_output(p0, p1, p2, p3, out_a0, out_a1, out_a2, out_a3, out_b0, out_b1, out_b2, out_b3)
                # Drain the final completed tile; no later GEMM can hide this batch.
                last_bx, last_by = tile_coordinates(block_id + (turns - 1) * grid_size)
                push_output(p0, peers, last_by, last_bx, 0)
                push_output(p1, peers, last_by, last_bx, 1)
                push_output(p2, peers, last_by, last_bx, 2)
                push_output(p3, peers, last_by, last_bx, 3)

    return gemm


if __name__ == "__main__":
    from examples.distributed.gemm_reduce_scatter._benchmark import run_benchmark

    run_benchmark(gemm_push_waves, "push_waves")
