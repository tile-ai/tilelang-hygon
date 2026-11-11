"""Persistent owner-Push ReduceScatter with device-owned producer/consumer counters.
Initialize ready, arrived, done, comm_completed and consumed once to zero.
Enqueue fused, completion wait, then a stream-ordered RCCL barrier before reuse.
Concurrent invocations sharing these buffers are unsupported.

Run with torchrun --nproc_per_node=2, 4 or 8. --M/--N/--K set M, N and local K
(each defaults to 4096). --compute-blocks selects 64/16 or 72/8.
Default execution checks eager and graph correctness. Add --benchmark for
TileLang/Torch timings, or --dump-source to save generated HIP.

The kernel launches 80 blocks, divided into compute and communication blocks.
Compute blocks perform GEMM and write partial output tiles to local global memory.
Communication blocks consume ready tiles and push their contributions to the
corresponding owner ranks. For tiles owned by the local rank, they combine
incoming contributions with the local contribution and write the reduced results
to the local output shard. Data transfers and reductions operate on chunks
within each tile.
"""
import tilelang.language as T


def persistent_gemm_owner_push_rs(
    M, N, K, rank, compute_blocks=64,
    chunk_rows=16, chunk_cols=256, tile_parts=2, world=8,
    BM=256, BN=256, BK=16, BLOCKS=80,
):
    if (BM, BN, BK) != (256, 256, 16):
        raise ValueError("the hand-written pipeline requires 256x256x16 GEMM tiles")
    if not isinstance(BLOCKS, int) or BLOCKS < 2:
        raise ValueError("BLOCKS must be an integer greater than one")
    if world not in (2,4,8):
        raise ValueError("world must be 2,4,or 8")
    if min(M,N)<=0 or M%BM or N%BN or K<64 or K%64:
        raise ValueError("full256x256tiles and local K divisible by64 required")
    total_tiles=(M//BM)*(N//BN)
    if M%(BM*world) or total_tiles%world or not 0<=rank<world or not 1<=compute_blocks<BLOCKS:
        raise ValueError("valid rank, tiles divisible by world, and 1 <= compute_blocks < BLOCKS required")
    if tile_parts not in (1,2,4,8) or chunk_rows not in (16,32,64,128,256) or chunk_cols not in (32,64,128,256):
        raise ValueError("unsupported stripe count or chunk geometry")
    stripe_rows=BM//tile_parts
    if chunk_rows>stripe_rows:
        raise ValueError("chunk must fit within a tile stripe")
    owned=total_tiles//world;comm_blocks=BLOCKS-compute_blocks
    tasks=owned*tile_parts;chunks=stripe_rows//chunk_rows*(BN//chunk_cols)
    block_M,block_N,block_K=BM,BN,BK
    dtype,accum_dtype="float16","float32"
    transpose_B=False
    steady_wait,shared_0_wait=8,6
    k_groups=K//(4*BK)

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

    @T.prim_func
    def fused(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype),
             partial: T.Tensor((total_tiles, BM, BN), dtype),
             ready: T.Tensor((total_tiles, 32), "uint64"),
             done: T.Tensor((owned, tile_parts, 32), "uint64"),
             receive: T.Tensor((world, owned, BM, BN), dtype),
             arrived: T.Tensor((owned, world, tile_parts, 32), "uint64"),
             peers: T.Tensor((world, 4), "uint64"), comm_completed: T.Tensor((owned, tile_parts), "uint64")):
        with T.Kernel(BLOCKS, threads=512) as block_id:
            if block_id < compute_blocks:
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
                for turn in T.serial(T.ceildiv(total_tiles, compute_blocks)):
                    tile = block_id + turn * compute_blocks
                    if tile < total_tiles:
                        physical_tile = (tile % world) * owned + tile // world
                        bx, by = physical_tile % (N // BN), physical_tile // (N // BN)
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
                                T.call_extern("tl::promote_prio", dtype="void")
                                async_copy_a(A, A_shared_0, by, kg * 4 + 4)
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
                                async_copy_a(A, A_shared_1, by, kg * 4 + 5)
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
                                async_copy_a(A, A_shared_2, by, kg * 4 + 6)
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
                                async_copy_a(A, A_shared_3, by, kg * 4 + 7)
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
                                async_copy_a(A, A_shared_0, by, kg * 4 + 4)
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
                                async_copy_a(A, A_shared_1, by, kg * 4 + 5)
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
                                async_copy_a(A, A_shared_2, by, kg * 4 + 6)
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
                                async_copy_a(A, A_shared_3, by, kg * 4 + 7)
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
                        T.copy(C_local, partial[tile, :, :])
                        # All waves finish consuming registers/LDS before the next tile.
                        T.s_waitcnt(0, "lgkmcnt")
                        T.sync_threads()
                        # Every producing wave publishes its payload before the
                        # CTA leader release-signals this rank's tile readiness.
                        T.call_extern("void", "__threadfence_system")
                        T.sync_threads()
                        if T.get_thread_binding() == 0:
                            # One producer owns this tile's counter across calls.
                            produced = T.alloc_var("uint64", init=ready[tile, 0] + T.uint64(1))
                            T.device_assert(produced != 0, no_stack_info=True)
                            T.signal_op(ready[tile, 0], produced, "set", rank)
                        T.sync_threads()
            else:
                send_values=T.alloc_fragment((world-1,chunk_rows,chunk_cols),dtype)
                values=T.alloc_fragment((world,chunk_rows,chunk_cols),dtype)
                acc=T.alloc_fragment((chunk_rows,chunk_cols),accum_dtype)
                result=T.alloc_fragment((chunk_rows,chunk_cols),dtype)
                for turn in T.serial(T.ceildiv(tasks,comm_blocks)):
                    task=block_id-compute_blocks+turn*comm_blocks
                    if task<tasks:
                        q=task//tile_parts;part=task%tile_parts
                        # Consumer-owned generation: never derive it from ready/arrived.
                        epoch_value = T.alloc_var("uint64", init=comm_completed[q, part] + T.uint64(1))
                        if T.get_thread_binding() == 0:
                            T.device_assert(epoch_value != 0, no_stack_info=True)
                        # One wave leader per LOCAL GEMM tile,including our own.
                        if T.get_thread_binding()%64==0 and T.get_thread_binding()//64<world:
                            local_owner=T.get_thread_binding()//64
                            T.signal_wait_until(ready[q*world+local_owner,0],"ge",epoch_value)
                        T.sync_threads()
                        T.call_extern("void","__builtin_amdgcn_fence",2,"")
                        for chunk in T.serial(chunks):
                            cy=part*stripe_rows+chunk//(BN//chunk_cols)*chunk_rows
                            cx=chunk%(BN//chunk_cols)*chunk_cols
                            for hop in T.unroll(world-1):
                                owner=(rank+hop+1)%world
                                T.copy(partial[q*world+owner,cy:cy+chunk_rows,cx:cx+chunk_cols],send_values[hop,:,:],coalesced_width=8)
                            T.sched_barrier()
                            for hop in T.unroll(world-1):
                                owner=(rank+hop+1)%world
                                remote=T.make_tensor_from_addr(peers[owner,3],(world,owned,BM,BN),dtype)
                                T.copy(send_values[hop,:,:],remote[rank,q,cy:cy+chunk_rows,cx:cx+chunk_cols],coalesced_width=8)
                        T.call_extern("void","__threadfence_system")
                        T.sync_threads()
                        if T.get_thread_binding()%64==0 and T.get_thread_binding()//64<world-1:
                            peer=(rank+T.get_thread_binding()//64+1)%world
                            T.signal_op(arrived[q,rank,part,0],epoch_value,"set",peer)
                        T.sync_threads()
                        if T.get_thread_binding()%64==0 and T.get_thread_binding()//64<world-1:
                            peer=(rank+T.get_thread_binding()//64+1)%world
                            T.signal_wait_until(arrived[q,peer,part,0],"ge",epoch_value)
                        T.sync_threads()
                        T.call_extern("void","__builtin_amdgcn_fence",2,"")
                        tile=q*world+rank;by=q//(N//BN);bx=q%(N//BN)
                        for chunk in T.serial(chunks):
                            cy=part*stripe_rows+chunk//(BN//chunk_cols)*chunk_rows
                            cx=chunk%(BN//chunk_cols)*chunk_cols
                            for peer in T.unroll(world):
                                if peer==rank:
                                    T.copy(partial[tile,cy:cy+chunk_rows,cx:cx+chunk_cols],values[peer,:,:],coalesced_width=8)
                                else:
                                    T.copy(receive[peer,q,cy:cy+chunk_rows,cx:cx+chunk_cols],values[peer,:,:],coalesced_width=8)
                            T.sched_barrier()
                            T.clear(acc)
                            for peer in T.unroll(world):
                                for i,j in T.Parallel(chunk_rows,chunk_cols):
                                    acc[i,j]+=T.cast(values[peer,i,j],"float32")
                            for i,j in T.Parallel(chunk_rows,chunk_cols):result[i,j]=T.cast(acc[i,j],dtype)
                            output=T.make_tensor_from_addr(peers[rank,1],(M//world,N),dtype)
                            T.copy(result,output[by*BM+cy:by*BM+cy+chunk_rows,bx*BN+cx:bx*BN+cx+chunk_cols],coalesced_width=8)
                        T.call_extern("void","__threadfence_system")
                        T.sync_threads()
                        if T.get_thread_binding()==0:
                            T.signal_op(done[q,part,0],epoch_value,"set",rank)
                        T.sync_threads()
                        # All waves have finished using this generation before advancing.
                        if T.get_thread_binding() == 0:
                            comm_completed[q, part] = epoch_value
    return fused


def wait_for_owner_push_rs(M,N,tile_parts=2,world=8,BM=256,BN=256,BLOCKS=80):
    tiles=(M//BM)*(N//BN)//world
    @T.prim_func
    def wait(done:T.Tensor((tiles,tile_parts,32),"uint64"),consumed:T.Tensor((tiles,tile_parts),"uint64")):
        with T.Kernel(BLOCKS,threads=64) as block:
            for turn in T.serial(T.ceildiv(tiles*tile_parts,BLOCKS)):
                task=block+turn*BLOCKS
                if task<tiles*tile_parts:
                    if T.get_thread_binding()==0:
                        tile=task//tile_parts;part=task%tile_parts
                        expected = T.alloc_var("uint64", init=consumed[tile, part] + T.uint64(1))
                        T.device_assert(expected != 0, no_stack_info=True)
                        T.signal_wait_until(done[tile,part,0],"ge",expected)
                        consumed[tile, part] = expected
            T.sync_threads()
            T.call_extern("void","__builtin_amdgcn_fence",2,"")
    return wait


def run():
    import argparse
    import ctypes
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
    from examples.distributed.gemm_reduce_scatter.utils import rank_env

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", dest="m", type=int, default=4096)
    parser.add_argument("--N", dest="n", type=int, default=4096)
    parser.add_argument("--K", "--k", dest="k", type=int, default=4096, help="local K on each rank")
    parser.add_argument("--compute-blocks", type=int, choices=(64, 72), default=64)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--dump-source", type=Path)
    args = parser.parse_args()
    rank, world, device = (rank_env(key) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    m, n, k = args.m, args.n, args.k
    BM, BN, BK, BLOCKS = 256, 256, 16, 80
    if world not in (2, 4, 8):
        raise ValueError("use 2, 4 or 8 ranks")
    if min(m, n) <= 0 or m % BM or n % BN or k < 64 or k % 64:
        raise ValueError("M/N must be positive multiples of 256 and local K a positive multiple of 64")
    tiles = (m // BM) * (n // BN)
    if m % (BM * world):
        raise ValueError("M must be divisible by 256 * world size")
    owned, parts = tiles // world, 2
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    os.environ["TILELANG_CACHE_DIR"] = str(Path(os.environ.get("TILELANG_CACHE_DIR", "/tmp/tl-block-push-rs")) / f"rank-{rank}")
    # Each live IPC module is bound to this allocator's metadata.
    env.disable_cache()
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=1800))
    rccl = dist.new_group(backend="nccl", timeout=timedelta(seconds=1800))
    hip = ctypes.CDLL("/opt/dtk/hip/lib/libamdhip64.so")

    def hip_check(code, operation):
        if code:
            raise RuntimeError(f"{operation}: HIP error {code}")

    runtime = _FfiIpcRuntime()
    hip.hipExtMallocWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    hip.hipExtMallocWithFlags.restype = ctypes.c_int

    def fine_malloc(size):
        pointer = ctypes.c_void_p()
        hip_check(hip.hipExtMallocWithFlags(ctypes.byref(pointer), size, 1), "hipDeviceMallocFinegrained")
        return pointer.value

    runtime.malloc = fine_malloc
    arena = IpcAllocator(
        m * n * 4 + tiles * (512 * 2 + 512) + 131072,
        rank=rank,
        world_size=world,
        device=device,
        group=dist.group.WORLD,
        runtime=runtime,
    )
    graph = torch_graph = None
    partial = output = receive = ready = done = arrived = peers = None
    comm_completed = consumed = counters = None
    try:
        arena.initialize()

        partial = torch.empty((tiles, BM, BN), device="cuda", dtype=torch.float16)
        output = tl.tensor((m // world, n), torch.float16, allocator=arena)
        receive = tl.tensor((world, owned, BM, BN), torch.float16, allocator=arena)
        ready = tl.tensor((tiles, 32), torch.uint64, allocator=arena)
        done = tl.tensor((owned, parts, 32), torch.uint64, allocator=arena)
        arrived = tl.tensor((owned, world, parts, 32), torch.uint64, allocator=arena)
        # Initialize once; producers and consumers advance their own counters.
        ready.zero_()
        done.zero_()
        arrived.zero_()
        comm_completed = torch.zeros((owned, parts), device="cuda", dtype=torch.uint64)
        consumed = torch.zeros((owned, parts), device="cuda", dtype=torch.uint64)
        counters = (ready, arrived, done, comm_completed, consumed)

        def peer_addresses(tensor):
            bases = arena.metadata[ipc_metadata_layout()["peer_base"] :]
            offset = tensor.data_ptr() - arena.base_ptr
            return torch.tensor([base + offset for base in bases], dtype=torch.uint64, device="cuda")

        # Column zero is reserved: partial lives in ordinary local GPU memory.
        peers = torch.stack([
            torch.zeros(world, dtype=torch.uint64, device="cuda"),
            peer_addresses(output), peer_addresses(ready), peer_addresses(receive),
        ], dim=1)
        target = {"kind": "hcu", "mcpu": torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0], "dist_backend": "ipc"}
        kernel = tl.compile(
            persistent_gemm_owner_push_rs(
                m, n, k, rank, compute_blocks=args.compute_blocks, world=world,
                BM=BM, BN=BN, BK=BK, BLOCKS=BLOCKS),
            out_idx=[],
            target=target,
            pass_configs={tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True},
        )
        wait = tl.compile(
            wait_for_owner_push_rs(m, n, parts, world, BM=BM, BN=BN, BLOCKS=BLOCKS),
            out_idx=[], target=target,
        )
        kernel.initialize(arena)
        wait.initialize(arena)
        if args.dump_source:
            args.dump_source.mkdir(parents=True, exist_ok=True)
            for name, compiled in [("gemm_reducescatter", kernel), ("completion_wait", wait)]:
                path = args.dump_source / f"{name}.rank-{rank}.hip"
                path.write_text(compiled.get_kernel_source())
                print(f"source rank={rank}: {path}", flush=True)

        a = torch.empty((m, k), device="cuda", dtype=torch.float16)
        b = torch.empty((k, n), device="cuda", dtype=torch.float16)

        def enqueue():
            kernel(a, b, partial, ready, done, receive, arrived, peers, comm_completed)
            wait(done, consumed)
            # Protect previous readers before the next call can overwrite any inbox.
            work = dist.barrier(group=rccl, device_ids=[device], async_op=True)
            work.block_current_stream()

        def random_inputs():
            a.normal_().mul_(k**-0.5)
            b.normal_()

        def reference():
            full = (a.float() @ b.float()).cpu()
            dist.all_reduce(full)
            return full[rank * (m // world) : (rank + 1) * (m // world)].to(device="cuda", dtype=torch.float16)

        def verify(expected, label, actual=None):
            actual = output if actual is None else actual
            torch.cuda.current_stream().synchronize()
            torch.testing.assert_close(
                actual,
                expected,
                atol=1e-2,
                rtol=1e-2,
                msg=lambda msg: f"{label} rank={rank}: {msg}",
            )
            print(
                f"{label} rank={rank}/{world} sample={actual[0, 0].item():.6g} max_abs={(actual - expected).abs().max().item():.6g} passed",
                flush=True,
            )

        def queued_check(fn, label):
            # Retain every output to detect intermittent reuse failures.
            saved = []
            for iteration in range(20):
                random_inputs()
                local_reference = a.float() @ b.float()
                if iteration == 7 and rank == world - 1:
                    time.sleep(0.05)
                fn()
                saved.append((output.clone(), local_reference))
            torch.cuda.current_stream().synchronize()
            for iteration, (actual, full) in enumerate(saved):
                total = full.cpu()
                dist.all_reduce(total)
                expected = total[rank * (m // world) : (rank + 1) * (m // world)].to(device="cuda", dtype=torch.float16)
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=1e-2,
                    rtol=1e-2,
                    msg=lambda msg, iteration=iteration: f"{label} rank={rank} iteration={iteration}: {msg}",
                )
            print(f"{label} rank={rank}/{world} calls=20 checked=20 passed", flush=True)

        def capture(fn):
            for _ in range(3):
                fn()
            torch.cuda.current_stream().synchronize()
            dist.barrier()
            before = [counter.cpu().clone() for counter in counters]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.current_stream().synchronize()
            for counter, previous in zip(counters, before):
                assert torch.equal(counter.cpu(), previous), "capture advanced invocation counters"
            return graph

        # Every rank must finish zeroing IPC signals before the first publication.
        # This also warms up RCCL before graph capture.
        dist.barrier(group=rccl, device_ids=[device])
        torch.cuda.current_stream().synchronize()
        torch.manual_seed(1234 + rank)
        for trial in range(3):
            random_inputs()
            expected = reference()
            if trial == 1 and rank == world - 1:
                time.sleep(0.05)
            enqueue()
            verify(expected, f"random-check trial={trial}")
        queued_check(enqueue, "queued-random-check")

        graph = capture(enqueue)
        for trial in range(3):
            random_inputs()
            expected = reference()
            if trial == 1 and rank == world - 1:
                time.sleep(0.05)
            graph.replay()
            verify(expected, f"graph-check trial={trial}")
        queued_check(graph.replay, "queued-graph-check")
        print(f"correctness-complete rank={rank}/{world} M={m} N={n} K_local={k}", flush=True)

        def device_ms(fn):
            # Like the Triton tutorial: warm up, enqueue timed calls, then wait once.
            # The full path includes the RCCL tail; no CPU barrier between calls.
            torch.cuda.current_stream().synchronize()
            dist.barrier()
            for _ in range(5):
                fn()
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.repeats)]
            for start, end in events:
                start.record()
                fn()
                end.record()
            torch.cuda.current_stream().synchronize()
            return statistics.median(start.elapsed_time(end) for start, end in events)

        if args.benchmark:
            random_inputs()
            expected = reference()
            results = {"gpu_event_ms": device_ms(enqueue), "graph_gpu_event_ms": device_ms(graph.replay)}
            verify(expected, "post-timing-check")

            # Preallocated Torch baseline; the timed path stays entirely on the GPU.
            torch_partial = torch.empty((m, n), device="cuda", dtype=torch.float16)
            torch_output = torch.empty_like(output)

            def torch_path():
                torch.mm(a, b, out=torch_partial)
                dist.reduce_scatter_tensor(torch_output, torch_partial, group=rccl)

            torch_path()
            verify(expected, "torch-check", torch_output)
            torch_graph = capture(torch_path)
            torch_graph.replay()
            verify(expected, "torch-graph-check", torch_output)
            results["torch_gpu_event_ms"] = device_ms(torch_path)
            results["torch_graph_gpu_event_ms"] = device_ms(torch_graph.replay)
            verify(expected, "torch-post-timing-check", torch_output)

            reports = [None] * world
            dist.all_gather_object(reports, results)
            if rank == 0:
                worst = {key: max(row[key] for row in reports) for key in results}
                print(f"timings world_size={world} M={m} N={n} K_local={k} by_rank={reports}", flush=True)
                print(f"summary world_size={world} M={m} N={n} K_local={k} max_rank={worst}", flush=True)
                for mode, ours_key, torch_key in (
                    ("eager", "gpu_event_ms", "torch_gpu_event_ms"),
                    ("graph", "graph_gpu_event_ms", "torch_graph_gpu_event_ms"),
                ):
                    ours_ms, torch_ms = worst[ours_key], worst[torch_key]
                    print(
                        f"world_size={world} M={m} N={n} K_local={k} mode={mode} "
                        f"tilelang_ms={ours_ms:.6f} torch_ms={torch_ms:.6f} speedup={torch_ms / ours_ms:.3f}x",
                        flush=True,
                    )

        torch.cuda.current_stream().synchronize()
        dist.barrier()
    finally:
        if torch_graph is not None:
            torch_graph.reset()
        if graph is not None:
            graph.reset()
        partial = output = receive = ready = done = arrived = peers = None
        comm_completed = consumed = counters = None
        gc.collect()
        arena.close()
        dist.destroy_process_group(rccl)
        dist.destroy_process_group()


if __name__ == "__main__":
    run()
