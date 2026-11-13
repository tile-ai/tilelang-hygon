"""Persistent Pull AllGather + GEMM: local-first computation and reclaimed CTAs.

Each rank provides FP16 A_local[M/world, K] and B[K, N], and receives C[M, N].
M is the global gathered row count. The input shards remain immutable while
any rank is reading them. Communication uses 16x256 register batches within
256x256 panels, ordinary payload memory, and fine-grained ready signals.

Run with torchrun --nproc_per_node=2, 4 or 8. Default execution checks eager
and Graph correctness, including changed inputs and repeated invocations.
--benchmark adds TileLang/Torch eager and Graph timing; --dump-source saves HIP.
Uses the repository's ordinary tilelang.compile / initialize / call path.
"""
import tilelang.language as T


def persistent_allgather_gemm_pull(M, N, K, rank, world, compute_blocks=64):
    """Build the V2 independent scheduler with fixed 16x256 Pull batches."""
    if world not in (2, 4, 8) or not 0 <= rank < world:
        raise ValueError("world must be 2, 4, or 8 and rank must be valid")
    if M <= 0 or M % (256 * world) or N <= 0 or N % 256 or K < 256 or K % 256:
        raise ValueError("M must be divisible by 256*world; N and K by 256")
    if not isinstance(compute_blocks, int) or not 2 <= compute_blocks <= 78 or compute_blocks % 2:
        raise ValueError("compute_blocks must be even and in 2..78")
    BM = BN = block_M = block_N = 256
    block_K = 16
    chunk_rows, chunk_cols = 16, 256
    dtype, accum_dtype = "float16", "float32"
    transpose_B = False
    k_groups = K // 64
    total_tiles = (M // 256) * (N // 256)
    panels = K // 256
    local_rows = M // world // 256
    # Own rows stay in local_A and are never pulled, so they are not enumerated.
    tasks = (M // 256 - local_rows) * panels
    comm_blocks = 80 - compute_blocks
    # Tickets visit (row 0, col 0), (row 1, col 0), (row 0, col 1), ...
    # Each CTA claims one ticket independently; there are no fixed partners.
    @T.macro
    def claim_tile(queue, ticket_slot, ticket):
        if T.get_thread_binding() == 0:
            q = T.atomic_add(queue[0], 1, return_prev=True)
            ticket_slot[0] = T.if_then_else(
                q < total_tiles,
                (2 * ((q // 2) // (N // BN)) + q % 2) * (N // BN)
                + (q // 2) % (N // BN),
                -1,
            )
        T.sync_threads()
        ticket[0] = ticket_slot[0]
        # All waves must read the ticket before GEMM reuses this LDS region.
        T.sync_threads()

    @T.macro
    def async_copy_a(A, A_shared, by, k_tile):
        T.async_copy(A[by*block_M:(by+1)*block_M,
                       k_tile*block_K:(k_tile+1)*block_K], A_shared)

    @T.macro
    def async_copy_b(B, B_shared, bx, k_tile):
        T.async_copy(B[k_tile*block_K:(k_tile+1)*block_K,
                       bx*block_N:(bx+1)*block_N], B_shared)

    @T.macro
    def gemm_core(A_input, B, C, input_by, by, bx, A_shared_0, A_shared_1, A_shared_2, A_shared_3, B_shared_0, B_shared_1, B_shared_2, B_shared_3, A_local_0, A_local_1, B_local_0, B_local_1, C_local, warp_idx):
        T.clear(C_local)

        T.sync_warp()
        async_copy_a(A_input, A_shared_0, input_by, 0)
        async_copy_b(B, B_shared_0, bx, 0)

        T.sync_warp()
        async_copy_a(A_input, A_shared_1, input_by, 1)
        async_copy_b(B, B_shared_1, bx, 1)

        T.sync_warp()
        async_copy_a(A_input, A_shared_2, input_by, 2)
        async_copy_b(B, B_shared_2, bx, 2)

        T.sync_warp()
        async_copy_a(A_input, A_shared_3, input_by, 3)
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
                T.call_extern("tl::promote_prio", dtype="void")
                async_copy_a(A_input, A_shared_0, input_by, kg * 4 + 4)
                async_copy_b(B, B_shared_0, bx, kg * 4 + 4)
                T.call_extern("tl::restore_prio", dtype="void")
                T.copy(A_shared_1, A_local_1)
                T.copy(B_shared_1, B_local_1)
                T.ptx_wait_group(4)
                T.sync_warp()
                T.s_waitcnt(8, "lgkmcnt")
                T.sched_barrier()
                T.gemm(A_local_0, B_local_0, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                T.sched_barrier()
                T.call_extern("tl::promote_prio", dtype="void")
                async_copy_a(A_input, A_shared_1, input_by, kg * 4 + 5)
                async_copy_b(B, B_shared_1, bx, kg * 4 + 5)
                T.call_extern("tl::restore_prio", dtype="void")
                T.copy(A_shared_2, A_local_0)
                T.copy(B_shared_2, B_local_0)
                T.ptx_wait_group(4)
                T.sync_warp()
                T.s_waitcnt(8, "lgkmcnt")
                T.sched_barrier()
                T.gemm(A_local_1, B_local_1, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                T.sched_barrier()
                T.call_extern("tl::promote_prio", dtype="void")
                async_copy_a(A_input, A_shared_2, input_by, kg * 4 + 6)
                async_copy_b(B, B_shared_2, bx, kg * 4 + 6)
                T.call_extern("tl::restore_prio", dtype="void")
                T.copy(A_shared_3, A_local_1)
                T.copy(B_shared_3, B_local_1)
                T.ptx_wait_group(4)
                T.sync_warp()
                T.s_waitcnt(8, "lgkmcnt")
                T.sched_barrier()
                T.gemm(A_local_0, B_local_0, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                T.sched_barrier()
                T.call_extern("tl::promote_prio", dtype="void")
                async_copy_a(A_input, A_shared_3, input_by, kg * 4 + 7)
                async_copy_b(B, B_shared_3, bx, kg * 4 + 7)
                T.call_extern("tl::restore_prio", dtype="void")
                T.copy(A_shared_0, A_local_0)
                T.copy(B_shared_0, B_local_0)
                T.ptx_wait_group(4)
                T.sync_warp()
                T.s_waitcnt(6, "lgkmcnt")
                T.sched_barrier()
                T.gemm(A_local_1, B_local_1, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                T.sched_barrier()
        else:
            for kg in range(k_groups - 1):
                T.call_extern("tl::promote_prio", dtype="void")
                async_copy_a(A_input, A_shared_0, input_by, kg * 4 + 4)
                async_copy_b(B, B_shared_0, bx, kg * 4 + 4)
                T.call_extern("tl::restore_prio", dtype="void")
                T.copy(A_shared_1, A_local_1)
                T.copy(B_shared_1, B_local_1)
                T.ptx_wait_group(4)
                T.s_waitcnt(8, "lgkmcnt")
                T.sched_barrier()
                T.gemm(A_local_0, B_local_0, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                T.sched_barrier()
                T.sync_warp()
                T.call_extern("tl::promote_prio", dtype="void")
                async_copy_a(A_input, A_shared_1, input_by, kg * 4 + 5)
                async_copy_b(B, B_shared_1, bx, kg * 4 + 5)
                T.call_extern("tl::restore_prio", dtype="void")
                T.copy(A_shared_2, A_local_0)
                T.copy(B_shared_2, B_local_0)
                T.ptx_wait_group(4)
                T.s_waitcnt(8, "lgkmcnt")
                T.sched_barrier()
                T.gemm(A_local_1, B_local_1, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                T.sched_barrier()
                T.sync_warp()
                T.call_extern("tl::promote_prio", dtype="void")
                async_copy_a(A_input, A_shared_2, input_by, kg * 4 + 6)
                async_copy_b(B, B_shared_2, bx, kg * 4 + 6)
                T.call_extern("tl::restore_prio", dtype="void")
                T.copy(A_shared_3, A_local_1)
                T.copy(B_shared_3, B_local_1)
                T.ptx_wait_group(4)
                T.s_waitcnt(8, "lgkmcnt")
                T.sched_barrier()
                T.gemm(A_local_0, B_local_0, C_local, transpose_B=transpose_B, annotations={"trans_c": True})
                T.sched_barrier()
                T.sync_warp()
                T.call_extern("tl::promote_prio", dtype="void")
                async_copy_a(A_input, A_shared_3, input_by, kg * 4 + 7)
                async_copy_b(B, B_shared_3, bx, kg * 4 + 7)
                T.call_extern("tl::restore_prio", dtype="void")
                T.copy(A_shared_0, A_local_0)
                T.copy(B_shared_0, B_local_0)
                T.ptx_wait_group(4)
                T.s_waitcnt(6, "lgkmcnt")
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
        T.copy(C_local,C[by*BM:(by+1)*BM,bx*BN:(bx+1)*BN])
        T.s_waitcnt(0,"lgkmcnt")
        T.sync_threads()

    @T.macro
    def compute_tile(tile, local_A, A, B, C, ready, epoch_value, block_id, A_shared_0, A_shared_1, A_shared_2, A_shared_3, B_shared_0, B_shared_1, B_shared_2, B_shared_3, A_local_0, A_local_1, B_local_0, B_local_1, C_local, warp_idx):
        # Logical tasks visit this rank's input first; output retains rank order.
        physical = (tile + rank * (total_tiles // world)) % total_tiles
        (bx, by) = (physical % (N // BN), physical // (N // BN))
        is_local = tile < total_tiles // world
        if not is_local:
            # Eight wave leaders share the K-panel readiness checks.
            if T.get_thread_binding() % 64 == 0:
                for p in T.serial(T.ceildiv(panels, 8)):
                    panel = p * 8 + T.get_thread_binding() // 64
                    if panel < panels:
                        T.signal_wait_until(ready[by, panel, 0], 'ge', epoch_value)
            T.sync_threads()
            T.call_extern('void', '__builtin_amdgcn_fence', 2, '')
        if is_local:
            gemm_core(local_A, B, C, by % local_rows, by, bx, A_shared_0, A_shared_1, A_shared_2, A_shared_3, B_shared_0, B_shared_1, B_shared_2, B_shared_3, A_local_0, A_local_1, B_local_0, B_local_1, C_local, warp_idx)
        else:
            gemm_core(A, B, C, by, by, bx, A_shared_0, A_shared_1, A_shared_2, A_shared_3, B_shared_0, B_shared_1, B_shared_2, B_shared_3, A_local_0, A_local_1, B_local_0, B_local_1, C_local, warp_idx)

    @T.macro
    def communicate(local_A, A, peers, ready, epoch_value, block_id):
        payload = T.alloc_fragment((chunk_rows, chunk_cols), dtype)
        for turn in T.serial(T.ceildiv(tasks, comm_blocks)):
            task = block_id - compute_blocks + turn * comm_blocks
            if task < tasks:
                row = task // panels
                panel = task % panels
                # Own rows are owned by this rank and stay in local_A, so the
                # task list starts after them and wraps around them: no pulled
                # panel ever has peer == rank.
                global_row = (row + (rank + 1) * local_rows) % (M // 256)
                peer = global_row // local_rows
                remote = T.make_tensor_from_addr(peers[peer], (M // world, K), dtype)
                # Keep each read/wait/write group small, but publish only one
                # ready flag after the whole 256x256 panel has been written.
                for sub_row in T.serial(256 // chunk_rows):
                    for sub_col in T.serial(256 // chunk_cols):
                        src_row = global_row % local_rows * 256 + sub_row * chunk_rows
                        dst_row = global_row * 256 + sub_row * chunk_rows
                        col = panel * 256 + sub_col * chunk_cols
                        T.copy(remote[src_row:src_row + chunk_rows, col:col + chunk_cols], payload)
                        T.copy(payload, A[dst_row:dst_row + chunk_rows, col:col + chunk_cols])
                T.call_extern('void', '__threadfence')
                T.sync_threads()
                if T.get_thread_binding() == 0:
                    T.signal_op(ready[global_row, panel, 0], epoch_value, 'set', rank)
                T.sync_threads()

    @T.prim_func
    def fused(local_A: T.Tensor((M // world, K), dtype),
              B: T.Tensor((K, N), dtype), A: T.Tensor((M, K), dtype),
              C: T.Tensor((M, N), dtype),
              ready: T.Tensor((M // 256, panels, 32), 'uint64'),
              peers: T.Tensor((world,), 'uint64'),
              epoch: T.Tensor((4,), 'int64'),
              queue: T.Tensor((1 + 40 * 3,), 'int32')):
        with T.Kernel(80, threads=512) as block_id:
            epoch_value = epoch[0]
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
            if block_id >= compute_blocks:
                communicate(local_A, A, peers, ready, epoch_value, block_id)
            # Every CTA independently claims a tile; communication CTAs join
            # immediately after completing their own communication workload.
            # Alias existing GEMM storage: no extra LDS beyond the 64 KiB.
            ticket_slot = T.make_tensor_from_addr(
                T.address_of(A_shared_0[0, 0]), (1,), "int32",
                storage_scope="shared.dyn",
            )
            ticket = T.alloc_local((1,), 'int32')
            claim_tile(queue, ticket_slot, ticket)
            while ticket[0] >= 0:
                compute_tile(ticket[0], local_A, A, B, C, ready, epoch_value, block_id, A_shared_0, A_shared_1, A_shared_2, A_shared_3, B_shared_0, B_shared_1, B_shared_2, B_shared_3, A_local_0, A_local_1, B_local_0, B_local_1, C_local, warp_idx)
                claim_tile(queue, ticket_slot, ticket)
    return fused


def run():
    import argparse
    import ctypes as C
    import gc
    import os
    import statistics
    import time
    from datetime import timedelta
    from pathlib import Path

    import torch
    import torch.distributed as dist
    import tilelang as tl
    from tilelang.env import env
    from tilelang.distributed.backends.ipc import IpcAllocator, _FfiIpcRuntime, ipc_metadata_layout

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--M', dest='m', type=int, default=4096, help='global gathered row count')
    parser.add_argument('--N', dest='n', type=int, default=4096)
    parser.add_argument('--K', dest='k', type=int, default=4096)
    parser.add_argument('--compute-blocks', type=int, choices=(64, 72), default=64)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--graph-batch', type=int, default=5)
    parser.add_argument('--dump-source', type=Path)
    args = parser.parse_args()
    rank, world, device = (int(os.environ[key]) for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    m, n, k = args.m, args.n, args.k
    if world not in (2, 4, 8):
        raise ValueError('use 2, 4, or 8 ranks on one node')
    if m <= 0 or m % (256 * world) or n <= 0 or n % 256 or k < 256 or k % 256:
        raise ValueError('M must be divisible by 256*world; N and K must be positive multiples of 256')
    if args.repeats <= 0 or args.graph_batch <= 0:
        raise ValueError('repeats and graph-batch must be positive')
    os.environ['TILELANG_CACHE_DIR'] = str(Path(os.environ.get('TILELANG_CACHE_DIR', '/tmp/tl-block-pull-ag')) / f'rank-{rank}')
    env.disable_cache()
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if props.gcnArchName.split(":")[0] != "gfx936" or props.multi_processor_count != 80:
        raise RuntimeError("this example requires a full 80-CU gfx936 device")
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('gloo', rank=rank, world_size=world, timeout=timedelta(seconds=1800))
    rccl = dist.new_group(backend='nccl', timeout=timedelta(seconds=1800))
    hip = C.CDLL('/opt/dtk/hip/lib/libamdhip64.so')
    hip.hipExtMallocWithFlags.argtypes = [C.POINTER(C.c_void_p), C.c_size_t, C.c_uint]
    hip.hipExtMallocWithFlags.restype = C.c_int
    arenas, graphs = [], []
    kernel = None
    local_A = gathered_A = ready = peers = None

    def allocate(size, flags):
        def malloc(count):
            pointer = C.c_void_p()
            status = hip.hipExtMallocWithFlags(C.byref(pointer), count, flags)
            if status:
                raise RuntimeError(f'hipExtMallocWithFlags: HIP error {status}')
            return pointer.value
        runtime = _FfiIpcRuntime()
        runtime.malloc = malloc
        arena = IpcAllocator(size + 131072, rank=rank, world_size=world,
                             device=device, group=dist.group.WORLD, runtime=runtime)
        arenas.append(arena)
        arena.initialize()
        return arena

    def synchronize():
        torch.cuda.synchronize()
        dist.barrier()

    try:
        # Only local communication CTAs write gathered_A. Remote GPUs only
        # read immutable local_A, so neither payload allocation needs fine grain.
        signals = allocate((m // 256) * (k // 256) * 32 * 8, 1)
        payload = allocate(m * k * 2, 0)
        sources = allocate((m // world) * k * 2, 0)
        local_A = tl.tensor((m // world, k), torch.float16, allocator=sources)
        gathered_A = tl.tensor((m, k), torch.float16, allocator=payload)
        ready = tl.tensor((m // 256, k // 256, 32), torch.uint64, allocator=signals)
        ready.zero_()
        offset = local_A.data_ptr() - sources.base_ptr
        bases = sources.metadata[ipc_metadata_layout()['peer_base']:]
        peers = torch.tensor([base + offset for base in bases], device='cuda', dtype=torch.uint64)
        B = torch.empty((k, n), device='cuda', dtype=torch.float16)
        output = torch.empty((m, n), device='cuda', dtype=torch.float16)
        epoch = torch.zeros(4, device='cuda', dtype=torch.int64)
        # Keep the tested buffer shape; only queue[0] is the task counter.
        # The remaining entries stay zero and are checked for unexpected writes.
        queue = torch.zeros(121, device='cuda', dtype=torch.int32)
        torch_A = torch.empty_like(gathered_A)
        torch_output = torch.empty_like(output)
        kernel = tl.compile(
            persistent_allgather_gemm_pull(m, n, k, rank, world, args.compute_blocks),
            out_idx=[],
            target={"kind": "hcu", "mcpu": props.gcnArchName.split(":")[0], "dist_backend": "ipc"},
            pass_configs={tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True},
        )
        kernel.initialize(signals)
        if args.dump_source:
            args.dump_source.mkdir(parents=True, exist_ok=True)
            path = args.dump_source / f'allgather_gemm.rank-{rank}.hip'
            path.write_text(kernel.get_kernel_source())
            print(f'source rank={rank}: {path}', flush=True)

        def enqueue():
            # Finish all prior readers before reusing gathered_A / queue.
            # Input mutation needs its own preceding all-rank rendezvous.
            work = dist.barrier(group=rccl, device_ids=[device], async_op=True)
            work.block_current_stream()
            epoch[:1].add_(1)
            queue.zero_()
            kernel(local_A, B, gathered_A, output, ready, peers, epoch, queue)

        def random_inputs():
            # A rank finishing its own GEMM does not imply all peers finished
            # pulling its source. This barrier MUST precede source mutation.
            synchronize()
            local_A.normal_().mul_(k ** -0.5)
            B.normal_()
            synchronize()

        def reference():
            dist.all_gather_into_tensor(torch_A, local_A, group=rccl)
            return (torch_A.float() @ B.float()).half()

        def verify(expected, label, actual=None):
            actual = output if actual is None else actual
            synchronize()
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2,
                                       msg=lambda msg: f'{label} rank={rank}: {msg}')
            if actual is output:
                # This rank's own rows are never pulled into gathered_A.
                owned = torch.arange(m, device=gathered_A.device) // (m // world) == rank
                assert torch.equal(gathered_A[~owned], torch_A[~owned]), 'AllGather output mismatch'
                # Own rows are computed from local_A, so they are never signalled.
                local_row = owned.view(m // 256, 256)[:, 0]
                ready_values = ready.to(torch.int64)
                assert bool((ready_values[~local_row, :, 0] == epoch[0]).all()), 'missing panel readiness'
                assert bool((ready_values[local_row] == 0).all()), 'local rows signalled'
                assert bool((ready_values[:, :, 1:] == 0).all()), 'signal padding overwritten'
                tiles = (m // 256) * (n // 256)
                assert queue[0].item() == tiles + 80, 'independent queue task count'
                assert bool((queue[1:] == 0).all()), 'unexpected queue writes'
            print(f'{label} rank={rank}/{world} max_abs={(actual - expected).abs().max().item():.6g} passed', flush=True)

        def capture(fn, count=1):
            for _ in range(3):
                fn()
            synchronize()
            before = epoch.cpu().clone()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            graphs.append(graph)
            with torch.cuda.graph(graph):
                for _ in range(count):
                    fn()
            graph.instantiate()
            synchronize()
            assert torch.equal(epoch.cpu(), before), 'capture advanced the epoch'
            return graph

        def repeated_check(fn, expected, label):
            # Retain each result, not only the last one. Include rank skew.
            saved = []
            for iteration in range(8):
                if iteration == 3 and rank == world - 1:
                    time.sleep(0.05)
                fn()
                saved.append(output.clone())
            synchronize()
            for iteration, actual in enumerate(saved):
                torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2,
                    msg=lambda msg, i=iteration: f'{label} rank={rank} call={i}: {msg}')
            verify(expected, label)

        torch.manual_seed(1234 + rank)
        synchronize()
        for trial in range(3):
            random_inputs()
            expected = reference()
            gathered_A.fill_(float('nan'))
            output.fill_(float('nan'))
            enqueue()
            verify(expected, f'random-check trial={trial}')
        repeated_check(enqueue, expected, 'repeated-eager-check')
        graph = capture(enqueue)
        for trial in range(3):
            random_inputs()
            expected = reference()
            gathered_A.fill_(float('nan'))
            output.fill_(float('nan'))
            graph.replay()
            verify(expected, f'graph-check trial={trial}')
        repeated_check(graph.replay, expected, 'repeated-graph-check')
        print(f'correctness-complete rank={rank}/{world} M_global={m} N={n} K={k} launch=tilelang', flush=True)

        def device_ms(fn, batch=1):
            synchronize()
            for _ in range(5):
                fn()
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(args.repeats)]
            for start, end in events:
                start.record()
                fn()
                end.record()
            torch.cuda.synchronize()
            return statistics.median(start.elapsed_time(end) / batch for start, end in events)

        if args.benchmark:
            def torch_path():
                dist.all_gather_into_tensor(torch_A, local_A, group=rccl)
                torch.mm(torch_A, B, out=torch_output)

            random_inputs()
            expected = reference()
            batched = capture(enqueue, args.graph_batch)
            torch_graph = capture(torch_path, args.graph_batch)
            results = {'gpu_event_ms': device_ms(enqueue),
                       'graph_gpu_event_ms': device_ms(batched.replay, args.graph_batch),
                       'torch_gpu_event_ms': device_ms(torch_path),
                       'torch_graph_gpu_event_ms': device_ms(torch_graph.replay, args.graph_batch)}
            verify(expected, 'post-timing-check')
            verify(expected, 'torch-post-timing-check', torch_output)
            reports = [None] * world
            dist.all_gather_object(reports, results)
            if rank == 0:
                worst = {key: max(row[key] for row in reports) for key in results}
                print(f'timings world_size={world} M_global={m} N={n} K={k} by_rank={reports}', flush=True)
                print(f'summary max_rank={worst}', flush=True)
                for mode, ours, baseline in [('eager', 'gpu_event_ms', 'torch_gpu_event_ms'),
                                             ('graph', 'graph_gpu_event_ms', 'torch_graph_gpu_event_ms')]:
                    print(f'mode={mode} tilelang_ms={worst[ours]:.6f} torch_ms={worst[baseline]:.6f} '
                          f'speedup={worst[baseline] / worst[ours]:.3f}x', flush=True)
        synchronize()
    finally:
        for graph in graphs:
            graph.reset()
        kernel = None
        local_A = gathered_A = ready = peers = None
        gc.collect()
        for arena in reversed(arenas):
            arena.close()
        dist.destroy_process_group(rccl)
        dist.destroy_process_group()


if __name__ == '__main__':
    run()
