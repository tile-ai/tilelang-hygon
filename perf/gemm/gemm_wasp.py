"""
HCU gfx946 WASP GEMM POC: warp-specialized producer/consumer handoff via ABarrier.

Kernels (gfx946 + WDRA only): MLS 4p4c4c one-shot and persistent variants.
"""

import argparse
import sys

import torch
import tilelang as tl
import tilelang.language as T
from tilelang.contrib.rocm import get_rocm_arch

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64
WARP_SIZE = 64

FREE_PING = 0
READY_PING = 1
FREE_PONG = 2
READY_PONG = 3
EBAR_ID = 0
CONSUMER_PING_EBAR_ID = 1
CONSUMER_PONG_EBAR_ID = 2

# 4+244+244 is four-VGPR aligned and its sum is divisible by three branches.
MLS_PRODUCER_MAX_NREG_3BR = 4
MLS_CONSUMER_MAX_NREG_3BR = 244

GFX946_ARCH = "gfx946"

WASP_PASS_CONFIGS = {
    # tl.PassConfigKey.TL_ENABLE_DUMP_IR: True,
    # tl.PassConfigKey.TL_DUMP_IR_DIR: "./dump_ir_gemm_wasp",
    tl.PassConfigKey.TL_ENABLE_HCU_WDRA: True,
}


def require_gfx946():
    arch = get_rocm_arch()
    if arch != GFX946_ARCH:
        raise RuntimeError(f"gemm_wasp requires HCU target {GFX946_ARCH!r}, got {arch!r}.")


@tl.jit(
    out_idx=[-1],
    pass_configs=WASP_PASS_CONFIGS,
)
def gemm_wasp_mls_4p4c4c(
    M,
    N,
    K,
    block_M: int = 256,
    block_N: int = 256,
    block_K: int = 64,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """4p4c4c with MLS: producer matrix_load, consumer ds_read_format (WDRA 3-branch)."""
    num_producer_waves = 4
    num_consumer_waves_per_group = 4
    num_consumer_groups = 2
    num_consumer_waves = num_consumer_waves_per_group * num_consumer_groups
    threads = (num_producer_waves + num_consumer_waves) * WARP_SIZE
    producer_tx = num_producer_waves * WARP_SIZE
    consumer0_tx = (num_producer_waves + num_consumer_waves_per_group) * WARP_SIZE
    k_tiles = T.ceildiv(K, block_K)
    full_k_pairs = k_tiles // 2
    half_N = block_N // 2
    mls_annotations = {"no_implicit_async_commit_wait": T.int32(1)}

    @T.prim_func
    def _gemm_wasp_mls_4p4c4c(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            tx = T.get_thread_binding()

            A_shared_ping = T.alloc_shared((block_M, block_K), dtype)
            A_shared_pong = T.alloc_shared((block_M, block_K), dtype)
            B_shared_ping = T.alloc_shared((block_N, block_K), dtype)
            B_shared_pong = T.alloc_shared((block_N, block_K), dtype)

            A_local_ping = T.alloc_fragment((block_M, block_K), dtype)
            A_local_pong = T.alloc_fragment((block_M, block_K), dtype)
            B_local_ping = T.alloc_fragment((half_N, block_K), dtype)
            B_local_pong = T.alloc_fragment((half_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, half_N), accum_dtype)
            A_local1_ping = T.alloc_fragment((block_M, block_K), dtype)
            A_local1_pong = T.alloc_fragment((block_M, block_K), dtype)
            B_local1_ping = T.alloc_fragment((half_N, block_K), dtype)
            B_local1_pong = T.alloc_fragment((half_N, block_K), dtype)
            C_local1 = T.alloc_fragment((block_M, half_N), accum_dtype)

            T.abarrier_init(FREE_PING, num_consumer_waves)
            T.abarrier_init(READY_PING, num_producer_waves)
            T.abarrier_init(FREE_PONG, num_consumer_waves)
            T.abarrier_init(READY_PONG, num_producer_waves)
            T.ebarrier_sync_cnt(EBAR_ID, num_producer_waves + num_consumer_waves)

            if tx < producer_tx:
                T.set_max_nreg(MLS_PRODUCER_MAX_NREG_3BR, 0)
                for k_pair in T.Serial(full_k_pairs):
                    k_ping = k_pair * 2
                    phase = k_pair & 1
                    T.abarrier_try_wait(FREE_PING, phase)
                    # Bind following async matrix_load(s) to READY_PING for arrive tracking.
                    T.abarrier_seq(READY_PING)
                    T.matrix_load(
                        A[by * block_M, k_ping * block_K],
                        A_shared_ping,
                        boundary=(None, False),
                        annotations=mls_annotations,
                    )
                    T.matrix_load(
                        B[bx * block_N, k_ping * block_K],
                        B_shared_ping,
                        boundary=(None, False),
                        annotations=mls_annotations,
                    )
                    T.abarrier_arrive(READY_PING)

                    k_pong = k_ping + 1
                    T.sched_barrier(0)
                    T.abarrier_try_wait(FREE_PONG, phase)
                    T.abarrier_seq(READY_PONG)
                    T.matrix_load(
                        A[by * block_M, k_pong * block_K],
                        A_shared_pong,
                        boundary=(None, False),
                        annotations=mls_annotations,
                    )
                    T.matrix_load(
                        B[bx * block_N, k_pong * block_K],
                        B_shared_pong,
                        boundary=(None, False),
                        annotations=mls_annotations,
                    )
                    T.abarrier_arrive(READY_PONG)

                # Only an odd number of K tiles leaves a single ping tail.
                if k_tiles % 2 != 0:
                    tail_k_ping = full_k_pairs * 2
                    tail_phase = full_k_pairs & 1
                    T.abarrier_try_wait(FREE_PING, tail_phase)
                    T.abarrier_seq(READY_PING)
                    T.matrix_load(
                        A[by * block_M, tail_k_ping * block_K],
                        A_shared_ping,
                        annotations=mls_annotations,
                    )
                    T.matrix_load(
                        B[bx * block_N, tail_k_ping * block_K],
                        B_shared_ping,
                        annotations=mls_annotations,
                    )
                    T.abarrier_arrive(READY_PING)
            elif tx < consumer0_tx:
                T.set_max_nreg(MLS_CONSUMER_MAX_NREG_3BR, 0)
                T.abarrier_arrive(FREE_PING)
                T.abarrier_arrive(FREE_PONG)
                T.clear(C_local)
                for k_pair in T.Serial(full_k_pairs):
                    phase = k_pair & 1
                    T.abarrier_try_wait(READY_PING, phase)
                    T.ds_read_format(A_shared_ping, A_local_ping)
                    T.ds_read_format(B_shared_ping[0:half_N, :], B_local_ping)
                    T.abarrier_arrive(FREE_PING)
                    T.ebarrier_sync_cnt(CONSUMER_PING_EBAR_ID, num_consumer_waves)
                    T.call_extern("tl::promote_prio", dtype="void")
                    T.gemm(
                        A_local_ping,
                        B_local_ping,
                        C_local,
                        transpose_B=True,
                        k_pack=1,
                        annotations={"trans_c": True},
                    )
                    T.call_extern("tl::restore_prio", dtype="void")
                    T.abarrier_try_wait(READY_PONG, phase)
                    T.ds_read_format(A_shared_pong, A_local_pong)
                    T.ds_read_format(B_shared_pong[0:half_N, :], B_local_pong)
                    T.abarrier_arrive(FREE_PONG)
                    T.ebarrier_sync_cnt(CONSUMER_PONG_EBAR_ID, num_consumer_waves)
                    T.call_extern("tl::promote_prio", dtype="void")
                    T.gemm(
                        A_local_pong,
                        B_local_pong,
                        C_local,
                        transpose_B=True,
                        k_pack=1,
                        annotations={"trans_c": True},
                    )
                    T.call_extern("tl::restore_prio", dtype="void")

                if k_tiles % 2 != 0:
                    tail_phase0 = full_k_pairs & 1
                    T.abarrier_try_wait(READY_PING, tail_phase0)
                    T.ds_read_format(A_shared_ping, A_local_ping)
                    T.ds_read_format(B_shared_ping[0:half_N, :], B_local_ping)
                    T.abarrier_arrive(FREE_PING)
                    T.ebarrier_sync_cnt(CONSUMER_PING_EBAR_ID, num_consumer_waves)
                    T.call_extern("tl::promote_prio", dtype="void")
                    T.gemm(
                        A_local_ping,
                        B_local_ping,
                        C_local,
                        transpose_B=True,
                        k_pack=1,
                        annotations={"trans_c": True},
                    )
                    T.call_extern("tl::restore_prio", dtype="void")
                T.copy(C_local, C[by * block_M, bx * block_N])
            else:
                T.set_max_nreg(MLS_CONSUMER_MAX_NREG_3BR, 0)
                T.abarrier_arrive(FREE_PING)
                T.abarrier_arrive(FREE_PONG)
                T.clear(C_local1)
                for k_pair in T.Serial(full_k_pairs):
                    phase = k_pair & 1
                    T.ebarrier_sync_cnt(CONSUMER_PING_EBAR_ID, num_consumer_waves)
                    T.abarrier_try_wait(READY_PING, phase)
                    T.ds_read_format(A_shared_ping, A_local1_ping)
                    T.ds_read_format(B_shared_ping[half_N:block_N, :], B_local1_ping)
                    T.abarrier_arrive(FREE_PING)
                    T.sched_barrier(0)
                    T.gemm(
                        A_local1_ping,
                        B_local1_ping,
                        C_local1,
                        transpose_B=True,
                        k_pack=1,
                        annotations={"trans_c": True},
                    )
                    T.sched_barrier(0)
                    T.ebarrier_sync_cnt(CONSUMER_PONG_EBAR_ID, num_consumer_waves)
                    T.abarrier_try_wait(READY_PONG, phase)
                    T.ds_read_format(A_shared_pong, A_local1_pong)
                    T.ds_read_format(B_shared_pong[half_N:block_N, :], B_local1_pong)
                    T.abarrier_arrive(FREE_PONG)
                    T.sched_barrier(0)
                    T.gemm(
                        A_local1_pong,
                        B_local1_pong,
                        C_local1,
                        transpose_B=True,
                        k_pack=1,
                        annotations={"trans_c": True},
                    )
                    T.sched_barrier(0)

                if k_tiles % 2 != 0:
                    tail_phase1 = full_k_pairs & 1
                    T.ebarrier_sync_cnt(CONSUMER_PING_EBAR_ID, num_consumer_waves)
                    T.abarrier_try_wait(READY_PING, tail_phase1)
                    T.ds_read_format(A_shared_ping, A_local1_ping)
                    T.ds_read_format(B_shared_ping[half_N:block_N, :], B_local1_ping)
                    T.abarrier_arrive(FREE_PING)
                    T.sched_barrier(0)
                    T.gemm(
                        A_local1_ping,
                        B_local1_ping,
                        C_local1,
                        transpose_B=True,
                        k_pack=1,
                        annotations={"trans_c": True},
                    )
                    T.sched_barrier(0)
                T.copy(C_local1, C[by * block_M, bx * block_N + half_N])

    return _gemm_wasp_mls_4p4c4c


@tl.jit(
    out_idx=[-1],
    pass_configs=WASP_PASS_CONFIGS,
)
def gemm_wasp_mls_4p4c4c_persistent(
    M,
    N,
    K,
    block_M: int = 256,
    block_N: int = 256,
    block_K: int = 64,
    num_persistent_blocks: int = 48,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """Persistent 4p4c4c MLS kernel with role-local output-block loops."""
    k_pair_boundary = (None, False)
    num_producer_waves = 4
    num_consumer_waves_per_group = 4
    num_consumer_waves = num_consumer_waves_per_group * 2
    threads = (num_producer_waves + num_consumer_waves) * WARP_SIZE
    producer_tx = num_producer_waves * WARP_SIZE
    consumer0_tx = (num_producer_waves + num_consumer_waves_per_group) * WARP_SIZE
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    total_blocks = m_blocks * n_blocks
    # Do not launch idle blocks.  T.Serial(ceildiv) is compiled with the max
    # trip count, so extra blocks would still issue MLS at OOB tile_id.
    grid_size = T.min(num_persistent_blocks, total_blocks)
    k_tiles = T.ceildiv(K, block_K)
    full_k_pairs = k_tiles // 2
    ping_uses_per_block = full_k_pairs + k_tiles % 2
    pong_uses_per_block = full_k_pairs
    half_N = block_N // 2
    mls_annotations = {"no_implicit_async_commit_wait": T.int32(1)}

    @T.prim_func
    def _gemm_wasp_mls_4p4c4c_persistent(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_size, threads=threads) as block_id:
            tx = T.get_thread_binding()

            A_shared_ping = T.alloc_shared((block_M, block_K), dtype)
            A_shared_pong = T.alloc_shared((block_M, block_K), dtype)
            B_shared_ping = T.alloc_shared((block_N, block_K), dtype)
            B_shared_pong = T.alloc_shared((block_N, block_K), dtype)

            A_local_ping = T.alloc_fragment((block_M, block_K), dtype)
            A_local_pong = T.alloc_fragment((block_M, block_K), dtype)
            B_local_ping = T.alloc_fragment((half_N, block_K), dtype)
            B_local_pong = T.alloc_fragment((half_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, half_N), accum_dtype)
            A_local1_ping = T.alloc_fragment((block_M, block_K), dtype)
            A_local1_pong = T.alloc_fragment((block_M, block_K), dtype)
            B_local1_ping = T.alloc_fragment((half_N, block_K), dtype)
            B_local1_pong = T.alloc_fragment((half_N, block_K), dtype)
            C_local1 = T.alloc_fragment((block_M, half_N), accum_dtype)

            T.abarrier_init(FREE_PING, num_consumer_waves)
            T.abarrier_init(READY_PING, num_producer_waves)
            T.abarrier_init(FREE_PONG, num_consumer_waves)
            T.abarrier_init(READY_PONG, num_producer_waves)
            T.ebarrier_sync_cnt(EBAR_ID, num_producer_waves + num_consumer_waves)

            if tx < producer_tx:
                T.set_max_nreg(MLS_PRODUCER_MAX_NREG_3BR, 0)
                producer_block_count = T.ceildiv(total_blocks - block_id, grid_size)
                for block_iter in T.Serial(producer_block_count):
                    tile_id = block_iter * grid_size + block_id
                    bx = tile_id % n_blocks
                    by = tile_id // n_blocks
                    # Keeping this loop rolled prevents LLVM from hoisting
                    # every K tile's MLS descriptors at once.  That greatly
                    # shortens producer SGPR live ranges in the persistent
                    # kernel and avoids SGPR-to-VGPR-lane spills.
                    for k_pair in T.Serial(
                        full_k_pairs,
                        annotations={"tl.hcu_loop_unroll_disable": True},
                    ):
                        # Compute both phases before the first synchronization point.
                        ping_phase = (block_iter * ping_uses_per_block + k_pair) & 1
                        pong_phase = (block_iter * pong_uses_per_block + k_pair) & 1
                        k_ping = k_pair * 2
                        k_pong = k_ping + 1
                        T.abarrier_try_wait(FREE_PING, ping_phase)
                        T.abarrier_seq(READY_PING)
                        T.matrix_load(
                            A[by * block_M, k_ping * block_K],
                            A_shared_ping,
                            boundary=k_pair_boundary,
                            annotations=mls_annotations,
                        )
                        T.matrix_load(
                            B[bx * block_N, k_ping * block_K],
                            B_shared_ping,
                            boundary=k_pair_boundary,
                            annotations=mls_annotations,
                        )
                        T.abarrier_arrive(READY_PING)
                        T.sched_barrier(0)
                        T.abarrier_try_wait(FREE_PONG, pong_phase)
                        T.abarrier_seq(READY_PONG)
                        T.matrix_load(
                            A[by * block_M, k_pong * block_K],
                            A_shared_pong,
                            boundary=k_pair_boundary,
                            annotations=mls_annotations,
                        )
                        T.matrix_load(
                            B[bx * block_N, k_pong * block_K],
                            B_shared_pong,
                            boundary=k_pair_boundary,
                            annotations=mls_annotations,
                        )
                        T.abarrier_arrive(READY_PONG)

                    if k_tiles % 2 != 0:
                        tail_ping_phase = (block_iter * ping_uses_per_block + full_k_pairs) & 1
                        tail_k_ping = full_k_pairs * 2
                        T.abarrier_try_wait(FREE_PING, tail_ping_phase)
                        T.abarrier_seq(READY_PING)
                        T.matrix_load(
                            A[by * block_M, tail_k_ping * block_K],
                            A_shared_ping,
                            annotations=mls_annotations,
                        )
                        T.matrix_load(
                            B[bx * block_N, tail_k_ping * block_K],
                            B_shared_ping,
                            annotations=mls_annotations,
                        )
                        T.abarrier_arrive(READY_PING)
            elif tx < consumer0_tx:
                T.set_max_nreg(MLS_CONSUMER_MAX_NREG_3BR, 0)
                T.abarrier_arrive(FREE_PING)
                T.abarrier_arrive(FREE_PONG)
                consumer0_block_count = T.ceildiv(total_blocks - block_id, grid_size)
                for block_iter in T.Serial(consumer0_block_count):
                    tile_id = block_iter * grid_size + block_id
                    bx = tile_id % n_blocks
                    by = tile_id // n_blocks
                    T.clear(C_local)
                    for k_pair in T.Serial(full_k_pairs):
                        ping_phase = (block_iter * ping_uses_per_block + k_pair) & 1
                        pong_phase = (block_iter * pong_uses_per_block + k_pair) & 1
                        T.abarrier_try_wait(READY_PING, ping_phase)
                        T.ds_read_format(A_shared_ping, A_local_ping)
                        T.ds_read_format(B_shared_ping[0:half_N, :], B_local_ping)
                        T.abarrier_arrive(FREE_PING)
                        T.ebarrier_arrive(CONSUMER_PING_EBAR_ID, num_consumer_waves)
                        T.call_extern("tl::set_prio<1>", dtype="void")
                        T.gemm(
                            A_local_ping,
                            B_local_ping,
                            C_local,
                            transpose_B=True,
                            k_pack=1,
                            annotations={"trans_c": True},
                        )
                        T.call_extern("tl::set_prio<0>", dtype="void")
                        T.abarrier_try_wait(READY_PONG, pong_phase)
                        T.ds_read_format(A_shared_pong, A_local_pong)
                        T.ds_read_format(B_shared_pong[0:half_N, :], B_local_pong)
                        T.abarrier_arrive(FREE_PONG)
                        T.ebarrier_sync_cnt(CONSUMER_PONG_EBAR_ID, num_consumer_waves)
                        T.call_extern("tl::set_prio<3>", dtype="void")
                        T.gemm(
                            A_local_pong,
                            B_local_pong,
                            C_local,
                            transpose_B=True,
                            k_pack=1,
                            annotations={"trans_c": True},
                        )
                        T.call_extern("tl::set_prio<0>", dtype="void")

                    if k_tiles % 2 != 0:
                        tail_ping_phase = (block_iter * ping_uses_per_block + full_k_pairs) & 1
                        T.abarrier_try_wait(READY_PING, tail_ping_phase)
                        T.ds_read_format(A_shared_ping, A_local_ping)
                        T.ds_read_format(B_shared_ping[0:half_N, :], B_local_ping)
                        T.abarrier_arrive(FREE_PING)
                        T.ebarrier_arrive(CONSUMER_PING_EBAR_ID, num_consumer_waves)
                        T.call_extern("tl::set_prio<1>", dtype="void")
                        T.gemm(
                            A_local_ping,
                            B_local_ping,
                            C_local,
                            transpose_B=True,
                            k_pack=1,
                            annotations={"trans_c": True},
                        )
                        T.call_extern("tl::set_prio<0>", dtype="void")
                    T.copy(C_local, C[by * block_M, bx * block_N])
            else:
                T.set_max_nreg(MLS_CONSUMER_MAX_NREG_3BR, 0)
                T.abarrier_arrive(FREE_PING)
                T.abarrier_arrive(FREE_PONG)
                consumer1_block_count = T.ceildiv(total_blocks - block_id, grid_size)
                for block_iter in T.Serial(consumer1_block_count):
                    tile_id = block_iter * grid_size + block_id
                    bx = tile_id % n_blocks
                    by = tile_id // n_blocks
                    T.clear(C_local1)
                    for k_pair in T.Serial(full_k_pairs):
                        ping_phase = (block_iter * ping_uses_per_block + k_pair) & 1
                        pong_phase = (block_iter * pong_uses_per_block + k_pair) & 1
                        T.ebarrier_sync_cnt(CONSUMER_PING_EBAR_ID, num_consumer_waves)
                        T.abarrier_try_wait(READY_PING, ping_phase)
                        T.ds_read_format(A_shared_ping, A_local1_ping)
                        T.ds_read_format(B_shared_ping[half_N:block_N, :], B_local1_ping)
                        T.abarrier_arrive(FREE_PING)
                        T.sched_barrier(0)
                        T.gemm(
                            A_local1_ping,
                            B_local1_ping,
                            C_local1,
                            transpose_B=True,
                            k_pack=1,
                            annotations={"trans_c": True},
                        )
                        T.sched_barrier(0)
                        T.ebarrier_arrive(CONSUMER_PONG_EBAR_ID, num_consumer_waves)
                        T.abarrier_try_wait(READY_PONG, pong_phase)
                        T.ds_read_format(A_shared_pong, A_local1_pong)
                        T.ds_read_format(B_shared_pong[half_N:block_N, :], B_local1_pong)
                        T.abarrier_arrive(FREE_PONG)
                        T.call_extern("tl::set_prio<2>", dtype="void")
                        T.gemm(
                            A_local1_pong,
                            B_local1_pong,
                            C_local1,
                            transpose_B=True,
                            k_pack=1,
                            annotations={"trans_c": True},
                        )
                        T.call_extern("tl::set_prio<0>", dtype="void")

                    if k_tiles % 2 != 0:
                        tail_ping_phase = (block_iter * ping_uses_per_block + full_k_pairs) & 1
                        T.ebarrier_sync_cnt(CONSUMER_PING_EBAR_ID, num_consumer_waves)
                        T.abarrier_try_wait(READY_PING, tail_ping_phase)
                        T.ds_read_format(A_shared_ping, A_local1_ping)
                        T.ds_read_format(B_shared_ping[half_N:block_N, :], B_local1_ping)
                        T.abarrier_arrive(FREE_PING)
                        T.sched_barrier(0)
                        T.gemm(
                            A_local1_ping,
                            B_local1_ping,
                            C_local1,
                            transpose_B=True,
                            k_pack=1,
                            annotations={"trans_c": True},
                        )
                        T.sched_barrier(0)
                    T.copy(C_local1, C[by * block_M, bx * block_N + half_N])

    return _gemm_wasp_mls_4p4c4c_persistent


KERNELS = {
    "mls_4p4c4c": gemm_wasp_mls_4p4c4c,
    "mls_4p4c4c_persistent": gemm_wasp_mls_4p4c4c_persistent,
}


def cpu_ref_program(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.cpu().float() @ b.cpu().float().T).to(a.dtype)


def run_check(
    variant: str,
    m: int = BLOCK_M,
    n: int = BLOCK_N,
    k: int = BLOCK_K,
    device: int = 0,
    dtype: str = "float16",
):
    require_gfx946()
    torch.cuda.set_device(device)
    kernel_fn = KERNELS[variant]
    kernel = kernel_fn(m, n, k, dtype=dtype)
    a = torch.randn(m, k, device=f"cuda:{device}", dtype=getattr(torch, dtype))
    b = torch.randn(n, k, device=f"cuda:{device}", dtype=getattr(torch, dtype))
    c = kernel(a, b)
    ref = cpu_ref_program(a, b)
    torch.testing.assert_close(c.cpu(), ref, rtol=1e-2, atol=1e-2)
    print(f"OK: gemm_wasp_{variant} M={m} N={n} K={k} on device {device}")


def main():
    parser = argparse.ArgumentParser(description="gfx946 WASP ABarrier GEMM smoke tests")
    parser.add_argument(
        "--variant",
        choices=[
            "mls_4p4c4c",
            "mls_4p4c4c_persistent",
            "all",
        ],
        default="all",
        help="kernel variant; all: run MLS 4p4c4c one-shot + persistent",
    )
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--dump-source", action="store_true")
    args = parser.parse_args()

    require_gfx946()
    torch.cuda.set_device(args.device)

    variants = ["mls_4p4c4c", "mls_4p4c4c_persistent"] if args.variant == "all" else [args.variant]
    for variant in variants:
        kernel_fn = KERNELS[variant]
        kernel = kernel_fn(args.m, args.n, args.k, dtype=args.dtype)
        if args.dump_source:
            print(f"===== gemm_wasp_{variant} =====")
            print(kernel.get_kernel_source())
            continue
        run_check(
            variant,
            args.m,
            args.n,
            args.k,
            device=args.device,
            dtype=args.dtype,
        )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
