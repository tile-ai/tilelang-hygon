"""
HCU gfx946 WASP GEMM POC: warp-specialized producer/consumer handoff via ABarrier.

Kernels (gfx946 + WDRA only):
  - gemm_wasp_4p4c:       4 producer + 4 consumer waves (8 waves / 512 threads)
  - gemm_wasp_4p4c4c:     4 producer + 4 + 4 consumer waves (12 waves / 768 threads)
  - gemm_wasp_mls_4p4c:   same 8-wave split, producer matrix_load + consumer ds_read_format
  - gemm_wasp_mls_4p4c4c: same 12-wave split, producer matrix_load + consumer ds_read_format
"""

import argparse
import sys

import torch
import tilelang as tl
import tilelang.language as T
from tilelang.contrib.rocm import get_rocm_arch

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
WARP_SIZE = 64

FREE_PING = 0
READY_PING = 1
FREE_PONG = 2
READY_PONG = 3
EBAR_ID = 0

PRODUCER_MAX_NREG = 64
CONSUMER_MAX_NREG = 192
# WDRA: sum(set_max_nreg per branch) must be divisible by branch count
# 3-branch 4p4c4c: 64+192+192=448 fails; use producer 72 -> 72+192+192=456.
PRODUCER_MAX_NREG_3BR = 72

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
def gemm_wasp_4p4c(
    M,
    N,
    K,
    block_M: int = BLOCK_M,
    block_N: int = BLOCK_N,
    block_K: int = BLOCK_K,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """4 producer waves + 4 consumer waves (WDRA 2-branch)."""
    num_producer_waves = 4
    num_consumer_waves = 4
    threads = (num_producer_waves + num_consumer_waves) * WARP_SIZE
    producer_tx = num_producer_waves * WARP_SIZE
    k_tiles = T.ceildiv(K, block_K)
    k_pairs = T.ceildiv(k_tiles, 2)

    @T.prim_func
    def _gemm_wasp_4p4c(
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
            B_local_ping = T.alloc_fragment((block_N, block_K), dtype)
            B_local_pong = T.alloc_fragment((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.annotate_layout(
                {
                    A_shared_ping: tl.layout.make_hcu_swizzled_layout(A_shared_ping, major_pack=2),
                    A_shared_pong: tl.layout.make_hcu_swizzled_layout(A_shared_pong, major_pack=2),
                    B_shared_ping: tl.layout.make_hcu_swizzled_layout(B_shared_ping, major_pack=2),
                    B_shared_pong: tl.layout.make_hcu_swizzled_layout(B_shared_pong, major_pack=2),
                }
            )

            T.abarrier_init(FREE_PING, num_consumer_waves)
            T.abarrier_init(READY_PING, num_producer_waves)
            T.abarrier_init(FREE_PONG, num_consumer_waves)
            T.abarrier_init(READY_PONG, num_producer_waves)
            T.ebarrier_sync_cnt(EBAR_ID, num_producer_waves + num_consumer_waves)

            if tx < producer_tx:
                T.set_max_nreg(PRODUCER_MAX_NREG, 0)
                for k_pair in T.Serial(k_pairs):
                    k_ping = k_pair * 2
                    phase = k_pair & 1
                    T.abarrier_try_wait(FREE_PING, phase)
                    T.copy(A[by * block_M, k_ping * block_K], A_shared_ping, coalesced_width=8)
                    T.copy(B[bx * block_N, k_ping * block_K], B_shared_ping, coalesced_width=8)
                    T.abarrier_arrive(READY_PING)

                    k_pong = k_ping + 1
                    T.sched_barrier(0)
                    T.abarrier_try_wait(FREE_PONG, phase)
                    T.copy(A[by * block_M, k_pong * block_K], A_shared_pong, coalesced_width=8)
                    T.copy(B[bx * block_N, k_pong * block_K], B_shared_pong, coalesced_width=8)
                    T.abarrier_arrive(READY_PONG)
            else:
                T.set_max_nreg(CONSUMER_MAX_NREG, 0)
                T.abarrier_arrive(FREE_PING)
                T.abarrier_arrive(FREE_PONG)
                T.clear(C_local)
                for k_pair in T.Serial(k_pairs):
                    phase = k_pair & 1
                    T.abarrier_try_wait(READY_PING, phase)
                    T.copy(A_shared_ping, A_local_ping)
                    T.copy(B_shared_ping, B_local_ping)
                    T.abarrier_arrive(FREE_PING)
                    T.gemm(A_local_ping, B_local_ping, C_local, transpose_B=True, k_pack=2)
                    T.sched_barrier(0)
                    T.abarrier_try_wait(READY_PONG, phase)
                    T.copy(A_shared_pong, A_local_pong)
                    T.copy(B_shared_pong, B_local_pong)
                    T.abarrier_arrive(FREE_PONG)
                    T.gemm(A_local_pong, B_local_pong, C_local, transpose_B=True, k_pack=2)
                T.copy(C_local, C[by * block_M, bx * block_N])

    return _gemm_wasp_4p4c


@tl.jit(
    out_idx=[-1],
    pass_configs=WASP_PASS_CONFIGS,
)
def gemm_wasp_4p4c4c(
    M,
    N,
    K,
    block_M: int = BLOCK_M,
    block_N: int = BLOCK_N,
    block_K: int = BLOCK_K,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """4 producer waves + 4 + 4 consumer waves (WDRA 3-branch, split N/2 per consumer group)."""
    num_producer_waves = 4
    num_consumer_waves_per_group = 4
    num_consumer_groups = 2
    num_consumer_waves = num_consumer_waves_per_group * num_consumer_groups
    threads = (num_producer_waves + num_consumer_waves) * WARP_SIZE
    producer_tx = num_producer_waves * WARP_SIZE
    consumer0_tx = (num_producer_waves + num_consumer_waves_per_group) * WARP_SIZE
    k_tiles = T.ceildiv(K, block_K)
    k_pairs = T.ceildiv(k_tiles, 2)
    half_N = block_N // 2

    @T.prim_func
    def _gemm_wasp_4p4c4c(
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
            # Consumer1 needs distinct fragment names: shared locals get one
            # layout/predicate per buffer; reusing names across WDRA branches
            # makes LowerTileOp guard copies with consumer0's tx range.
            A_local1_ping = T.alloc_fragment((block_M, block_K), dtype)
            A_local1_pong = T.alloc_fragment((block_M, block_K), dtype)
            B_local1_ping = T.alloc_fragment((half_N, block_K), dtype)
            B_local1_pong = T.alloc_fragment((half_N, block_K), dtype)
            C_local1 = T.alloc_fragment((block_M, half_N), accum_dtype)

            T.annotate_layout(
                {
                    A_shared_ping: tl.layout.make_hcu_swizzled_layout(A_shared_ping, major_pack=2),
                    A_shared_pong: tl.layout.make_hcu_swizzled_layout(A_shared_pong, major_pack=2),
                    B_shared_ping: tl.layout.make_hcu_swizzled_layout(B_shared_ping, major_pack=2),
                    B_shared_pong: tl.layout.make_hcu_swizzled_layout(B_shared_pong, major_pack=2),
                }
            )

            T.abarrier_init(FREE_PING, num_consumer_waves)
            T.abarrier_init(READY_PING, num_producer_waves)
            T.abarrier_init(FREE_PONG, num_consumer_waves)
            T.abarrier_init(READY_PONG, num_producer_waves)
            T.ebarrier_sync_cnt(EBAR_ID, num_producer_waves + num_consumer_waves)

            if tx < producer_tx:
                T.set_max_nreg(PRODUCER_MAX_NREG_3BR, 0)
                for k_pair in T.Serial(k_pairs):
                    k_ping = k_pair * 2
                    phase = k_pair & 1
                    T.abarrier_try_wait(FREE_PING, phase)
                    T.copy(A[by * block_M, k_ping * block_K], A_shared_ping, coalesced_width=8)
                    T.copy(B[bx * block_N, k_ping * block_K], B_shared_ping, coalesced_width=8)
                    T.abarrier_arrive(READY_PING)

                    k_pong = k_ping + 1
                    T.sched_barrier(0)
                    T.abarrier_try_wait(FREE_PONG, phase)
                    T.copy(A[by * block_M, k_pong * block_K], A_shared_pong, coalesced_width=8)
                    T.copy(B[bx * block_N, k_pong * block_K], B_shared_pong, coalesced_width=8)
                    T.abarrier_arrive(READY_PONG)
            elif tx < consumer0_tx:
                T.set_max_nreg(CONSUMER_MAX_NREG, 0)
                T.abarrier_arrive(FREE_PING)
                T.abarrier_arrive(FREE_PONG)
                T.clear(C_local)
                for k_pair in T.Serial(k_pairs):
                    phase = k_pair & 1
                    T.abarrier_try_wait(READY_PING, phase)
                    T.copy(A_shared_ping, A_local_ping)
                    T.copy(B_shared_ping[0:half_N, :], B_local_ping)
                    T.abarrier_arrive(FREE_PING)
                    T.gemm(A_local_ping, B_local_ping, C_local, transpose_B=True, k_pack=2)
                    T.sched_barrier(0)
                    T.abarrier_try_wait(READY_PONG, phase)
                    T.copy(A_shared_pong, A_local_pong)
                    T.copy(B_shared_pong[0:half_N, :], B_local_pong)
                    T.abarrier_arrive(FREE_PONG)
                    T.gemm(A_local_pong, B_local_pong, C_local, transpose_B=True, k_pack=2)
                T.copy(C_local, C[by * block_M, bx * block_N])
            else:
                T.set_max_nreg(CONSUMER_MAX_NREG, 0)
                T.abarrier_arrive(FREE_PING)
                T.abarrier_arrive(FREE_PONG)
                T.clear(C_local1)
                for k_pair in T.Serial(k_pairs):
                    phase = k_pair & 1
                    T.abarrier_try_wait(READY_PING, phase)
                    T.copy(A_shared_ping, A_local1_ping)
                    T.copy(B_shared_ping[half_N:block_N, :], B_local1_ping)
                    T.abarrier_arrive(FREE_PING)
                    T.gemm(A_local1_ping, B_local1_ping, C_local1, transpose_B=True, k_pack=2)
                    T.sched_barrier(0)
                    T.abarrier_try_wait(READY_PONG, phase)
                    T.copy(A_shared_pong, A_local1_pong)
                    T.copy(B_shared_pong[half_N:block_N, :], B_local1_pong)
                    T.abarrier_arrive(FREE_PONG)
                    T.gemm(A_local1_pong, B_local1_pong, C_local1, transpose_B=True, k_pack=2)
                T.copy(C_local1, C[by * block_M, bx * block_N + half_N])

    return _gemm_wasp_4p4c4c


@tl.jit(
    out_idx=[-1],
    pass_configs=WASP_PASS_CONFIGS,
)
def gemm_wasp_mls_4p4c4c(
    M,
    N,
    K,
    block_M: int = BLOCK_M,
    block_N: int = BLOCK_N,
    block_K: int = BLOCK_K,
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
    k_pairs = T.ceildiv(k_tiles, 2)
    half_N = block_N // 2

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
                T.set_max_nreg(PRODUCER_MAX_NREG_3BR, 0)
                for k_pair in T.Serial(k_pairs):
                    k_ping = k_pair * 2
                    phase = k_pair & 1
                    T.abarrier_try_wait(FREE_PING, phase)
                    # Bind following async matrix_load(s) to READY_PING for arrive tracking.
                    T.abarrier_seq(READY_PING)
                    T.matrix_load(A[by * block_M, k_ping * block_K], A_shared_ping)
                    T.matrix_load(B[bx * block_N, k_ping * block_K], B_shared_ping)
                    T.abarrier_arrive(READY_PING)

                    k_pong = k_ping + 1
                    T.sched_barrier(0)
                    T.abarrier_try_wait(FREE_PONG, phase)
                    T.abarrier_seq(READY_PONG)
                    T.matrix_load(A[by * block_M, k_pong * block_K], A_shared_pong)
                    T.matrix_load(B[bx * block_N, k_pong * block_K], B_shared_pong)
                    T.abarrier_arrive(READY_PONG)
            elif tx < consumer0_tx:
                T.set_max_nreg(CONSUMER_MAX_NREG, 0)
                T.abarrier_arrive(FREE_PING)
                T.abarrier_arrive(FREE_PONG)
                T.clear(C_local)
                for k_pair in T.Serial(k_pairs):
                    phase = k_pair & 1
                    T.abarrier_try_wait(READY_PING, phase)
                    T.ds_read_format(A_shared_ping, A_local_ping)
                    T.ds_read_format(B_shared_ping[0:half_N, :], B_local_ping)
                    T.abarrier_arrive(FREE_PING)
                    T.gemm(A_local_ping, B_local_ping, C_local, transpose_B=True, k_pack=1)
                    T.sched_barrier(0)
                    T.abarrier_try_wait(READY_PONG, phase)
                    T.ds_read_format(A_shared_pong, A_local_pong)
                    T.ds_read_format(B_shared_pong[0:half_N, :], B_local_pong)
                    T.abarrier_arrive(FREE_PONG)
                    T.gemm(A_local_pong, B_local_pong, C_local, transpose_B=True, k_pack=1)
                T.copy(C_local, C[by * block_M, bx * block_N])
            else:
                T.set_max_nreg(CONSUMER_MAX_NREG, 0)
                T.abarrier_arrive(FREE_PING)
                T.abarrier_arrive(FREE_PONG)
                T.clear(C_local1)
                for k_pair in T.Serial(k_pairs):
                    phase = k_pair & 1
                    T.abarrier_try_wait(READY_PING, phase)
                    T.ds_read_format(A_shared_ping, A_local1_ping)
                    T.ds_read_format(B_shared_ping[half_N:block_N, :], B_local1_ping)
                    T.abarrier_arrive(FREE_PING)
                    T.gemm(A_local1_ping, B_local1_ping, C_local1, transpose_B=True, k_pack=1)
                    T.sched_barrier(0)
                    T.abarrier_try_wait(READY_PONG, phase)
                    T.ds_read_format(A_shared_pong, A_local1_pong)
                    T.ds_read_format(B_shared_pong[half_N:block_N, :], B_local1_pong)
                    T.abarrier_arrive(FREE_PONG)
                    T.gemm(A_local1_pong, B_local1_pong, C_local1, transpose_B=True, k_pack=1)
                T.copy(C_local1, C[by * block_M, bx * block_N + half_N])

    return _gemm_wasp_mls_4p4c4c


@tl.jit(
    out_idx=[-1],
    pass_configs=WASP_PASS_CONFIGS,
)
def gemm_wasp_mls_4p4c(
    M,
    N,
    K,
    block_M: int = BLOCK_M,
    block_N: int = BLOCK_N,
    block_K: int = BLOCK_K,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """4p4c with MLS: producer matrix_load, consumer ds_read_format (WDRA 2-branch)."""
    num_producer_waves = 4
    num_consumer_waves = 4
    threads = (num_producer_waves + num_consumer_waves) * WARP_SIZE
    producer_tx = num_producer_waves * WARP_SIZE
    k_tiles = T.ceildiv(K, block_K)
    k_pairs = T.ceildiv(k_tiles, 2)

    @T.prim_func
    def _gemm_wasp_mls_4p4c(
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
            B_local_ping = T.alloc_fragment((block_N, block_K), dtype)
            B_local_pong = T.alloc_fragment((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.abarrier_init(FREE_PING, num_consumer_waves)
            T.abarrier_init(READY_PING, num_producer_waves)
            T.abarrier_init(FREE_PONG, num_consumer_waves)
            T.abarrier_init(READY_PONG, num_producer_waves)
            T.ebarrier_sync_cnt(EBAR_ID, num_producer_waves + num_consumer_waves)

            if tx < producer_tx:
                T.set_max_nreg(PRODUCER_MAX_NREG, 0)
                for k_pair in T.Serial(k_pairs):
                    k_ping = k_pair * 2
                    phase = k_pair & 1
                    T.abarrier_try_wait(FREE_PING, phase)
                    T.abarrier_seq(READY_PING)
                    T.matrix_load(A[by * block_M, k_ping * block_K], A_shared_ping)
                    T.matrix_load(B[bx * block_N, k_ping * block_K], B_shared_ping)
                    T.abarrier_arrive(READY_PING)

                    k_pong = k_ping + 1
                    T.sched_barrier(0)
                    T.abarrier_try_wait(FREE_PONG, phase)
                    T.abarrier_seq(READY_PONG)
                    T.matrix_load(A[by * block_M, k_pong * block_K], A_shared_pong)
                    T.matrix_load(B[bx * block_N, k_pong * block_K], B_shared_pong)
                    T.abarrier_arrive(READY_PONG)
            else:
                T.set_max_nreg(CONSUMER_MAX_NREG, 0)
                T.abarrier_arrive(FREE_PING)
                T.abarrier_arrive(FREE_PONG)
                T.clear(C_local)
                for k_pair in T.Serial(k_pairs):
                    phase = k_pair & 1
                    T.abarrier_try_wait(READY_PING, phase)
                    T.ds_read_format(A_shared_ping, A_local_ping)
                    T.ds_read_format(B_shared_ping, B_local_ping)
                    T.abarrier_arrive(FREE_PING)
                    T.gemm(A_local_ping, B_local_ping, C_local, transpose_B=True, k_pack=1)
                    T.sched_barrier(0)
                    T.abarrier_try_wait(READY_PONG, phase)
                    T.ds_read_format(A_shared_pong, A_local_pong)
                    T.ds_read_format(B_shared_pong, B_local_pong)
                    T.abarrier_arrive(FREE_PONG)
                    T.gemm(A_local_pong, B_local_pong, C_local, transpose_B=True, k_pack=1)
                T.copy(C_local, C[by * block_M, bx * block_N])

    return _gemm_wasp_mls_4p4c


KERNELS = {
    "4p4c": gemm_wasp_4p4c,
    "4p4c4c": gemm_wasp_4p4c4c,
    "mls_4p4c": gemm_wasp_mls_4p4c,
    "mls_4p4c4c": gemm_wasp_mls_4p4c4c,
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
            "4p4c",
            "4p4c4c",
            "mls_4p4c",
            "mls_4p4c4c",
            "all",
        ],
        default="all",
        help="kernel variant; all: run 4p4c + 4p4c4c",
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

    variants = ["4p4c", "4p4c4c"] if args.variant == "all" else [args.variant]
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
