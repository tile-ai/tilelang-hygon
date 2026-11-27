"""
HCU gfx94x WASP GEMM FP8 FP16-accumulator POC: warp-specialized producer/consumer handoff via ABarrier.

Kernel (gfx946/gfx948 + WDRA): persistent MLS 4p4c4c with native-B16
MatrixStore output.

The CLI uses dynamic M/N/K by default; --static-shape provides a baseline.
Use --variant mls_4p4c4c_persistent --matrix-store for native B16 output.
The demo interface independently infers the largest power-of-two divisor
of the actual M/N/K, capped at twice each axis's block size, and passes
those constraints to the JIT kernel. K retains ping/pong-pair alignment.
No additional alignment cache or shape bucketing is used.
Direct kernel-factory callers must honor any supplied alignment preconditions.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

ARTIFACT_DIR = Path(__file__).with_name("gemm_wasp_fp8_artifacts")
WORK_DIR = ARTIFACT_DIR
CACHE_DIR = WORK_DIR / "cache"
PERF_DIR = WORK_DIR / "dump"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PERF_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TILELANG_KERNEL_DUMP"] = "1"
os.environ["TILELANG_KERNEL_CACHE"] = str(CACHE_DIR)
# Current TileLang cache resolution uses TILELANG_CACHE_DIR internally. Keep
# the requested kernel-specific variable above and mirror it for that path.
os.environ["TILELANG_CACHE_DIR"] = str(CACHE_DIR)
os.environ["TRITON_CACHE_DIR"] = str(CACHE_DIR)

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
# MLS_PRODUCER_MAX_NREG_3BR = 16
MLS_PRODUCER_MAX_NREG_3BR = 4
MLS_CONSUMER_MAX_NREG_3BR = 244

GFX94X_ARCHES = {"gfx946", "gfx948"}

WASP_PASS_CONFIGS = {
    # tl.PassConfigKey.TL_ENABLE_DUMP_IR: True,
    # tl.PassConfigKey.TL_DUMP_IR_DIR: "./dump_ir_gemm_wasp",
    tl.PassConfigKey.TL_ENABLE_HCU_WDRA: True,
    # Prove dynamic persistent tile bounds without expression-specific rules.
    tl.PassConfigKey.TL_Z3_RLIMIT: 2_000_000,
    # PMD does not implement the WDRA trap handler used by hardware.
    tl.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: [
        "-mllvm=-turn-off-wdra-trap-handler=true",
    ],
}


def require_gfx94x():
    arch = get_rocm_arch()
    if arch not in GFX94X_ARCHES:
        raise RuntimeError(f"gemm_wasp_fp8 requires HCU target in {sorted(GFX94X_ARCHES)}, got {arch!r}.")


def _validate_alignment(alignment, axis):
    if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"{axis} alignment must be a positive power of two, got {alignment}")


def _max_power_of_two_alignment(size):
    if size <= 0:
        raise ValueError(f"dimension must be positive, got {size}")
    return size & -size


# IDs name physical slots; logical ping/pong rotate when a block has a tail.
A_FREE_PING = 0
A_READY_PING = 1
A_FREE_PONG = 2
A_READY_PONG = 3
B_FREE_PING = 4
B_READY_PING = 5
B_FREE_PONG = 6
B_READY_PONG = 7


@tl.jit(
    out_idx=[-1],
    execution_backend="cython",
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
    dtype: str = "float8_e4m3fn",
    accum_dtype: str = "float16",
    out_dtype: str = "float16",
    use_matrix_store: bool = False,
    max_m_align: int = 1,
    max_n_align: int = 1,
    max_k_align: int = 1,
    min_full_k_pairs: int = 0,
    has_tail: bool | None = None,
):
    """Persistent MLS kernel with optional native B16 MatrixStore output.

    max_m/n/k_align independently specifies each power-of-two divisor.
    A value of 1 adds no alignment assumption for that dimension.
    min_full_k_pairs is a static lower bound on ceildiv(K, block_K) // 2.
    has_tail optionally fixes whether ceildiv(K, block_K) is odd.
    These are caller preconditions, not runtime bounds checks. M/N/K may
    remain dynamic; use min_full_k_pairs=1, has_tail=False for paired K tiles.
    """
    for alignment, axis in ((max_m_align, "M"), (max_n_align, "N"), (max_k_align, "K")):
        _validate_alignment(alignment, axis)
    if use_matrix_store and out_dtype != "float16":
        raise ValueError("MatrixStore output currently requires out_dtype=float16")

    if not isinstance(min_full_k_pairs, int) or isinstance(min_full_k_pairs, bool) or min_full_k_pairs < 0:
        raise ValueError("min_full_k_pairs must be a nonnegative static integer")
    if has_tail is not None and not isinstance(has_tail, bool):
        raise ValueError("has_tail must be a static bool or None")
    if isinstance(K, int):
        actual_tiles = (K + block_K - 1) // block_K
        if actual_tiles // 2 < min_full_k_pairs:
            raise ValueError("K violates min_full_k_pairs")
        if has_tail is not None and bool(actual_tiles % 2) != has_tail:
            raise ValueError("K violates has_tail")

    k_pair_boundary = (None, False)
    num_producer_waves = 4
    num_consumer_waves_per_group = 4
    num_consumer_waves = num_consumer_waves_per_group * 2
    threads = (num_producer_waves + num_consumer_waves) * WARP_SIZE
    producer_tx = num_producer_waves * WARP_SIZE
    consumer0_tx = (num_producer_waves + num_consumer_waves_per_group) * WARP_SIZE
    m_blocks = M // block_M if max_m_align % block_M == 0 else T.ceildiv(M, block_M)
    n_blocks = N // block_N if max_n_align % block_N == 0 else T.ceildiv(N, block_N)
    total_blocks = m_blocks * n_blocks
    # Do not launch idle blocks.  T.Serial(ceildiv) is compiled with the max
    # trip count, so extra blocks would still issue MLS at OOB tile_id.
    grid_size = T.min(num_persistent_blocks, total_blocks)
    k_tiles = K // block_K if max_k_align % block_K == 0 else T.ceildiv(K, block_K)
    full_k_pairs = k_tiles // 2
    tail_tiles = k_tiles % 2 if has_tail is None else int(has_tail)
    has_first_pair = True if min_full_k_pairs >= 1 else full_k_pairs > 0
    remaining_pair_end = full_k_pairs if min_full_k_pairs >= 1 else T.max(1, full_k_pairs)
    tail_clear_accum = False if min_full_k_pairs >= 1 else full_k_pairs == 0
    if block_M % 2 != 0:
        raise ValueError("M-partitioned consumers require an even block_M")
    half_M = block_M // 2

    mls_annotations = {"no_implicit_async_commit_wait": T.int32(1)}
    b_ds_read_alt = 2
    mmac_annotations = {
        "tl.hcu_mmac_lit": 1,
        "trans_c": 0,
        "tl.hcu_b_ds_read_alt": 2,
    }

    @T.prim_func
    def _gemm_wasp_mls_4p4c4c_persistent(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        if min_full_k_pairs > 0:
            T.assume(full_k_pairs >= min_full_k_pairs)
        if has_tail is not None:
            T.assume(k_tiles % 2 == tail_tiles)
        if max_m_align > 1:
            T.assume(M > 0)
            T.assume(M % max_m_align == 0)
        if max_n_align > 1:
            T.assume(N > 0)
            T.assume(N % max_n_align == 0)
        if max_k_align > 1:
            T.assume(K > 0)
            T.assume(K % max_k_align == 0)
        with T.Kernel(grid_size, threads=threads) as block_id:
            tx = T.get_thread_binding()

            A_shared = T.alloc_shared((2, block_M, block_K), dtype)
            B_shared = T.alloc_shared((2, block_N, block_K), dtype)
            A_local_ping = T.alloc_fragment((half_M, block_K), dtype)
            A_local_pong = T.alloc_fragment((half_M, block_K), dtype)
            B_local_ping = T.alloc_fragment((block_N, block_K), dtype)
            B_local_pong = T.alloc_fragment((block_N, block_K), dtype)
            # Keep the MMAC accumulator in FP32, then explicitly stage the
            # result in an FP16 fragment before MatrixStore.
            C_local = T.alloc_fragment((half_M, block_N), accum_dtype)
            C_store_local = T.alloc_fragment((half_M, block_N), out_dtype)
            A_local1_ping = T.alloc_fragment((half_M, block_K), dtype)
            A_local1_pong = T.alloc_fragment((half_M, block_K), dtype)
            B_local1_ping = T.alloc_fragment((block_N, block_K), dtype)
            B_local1_pong = T.alloc_fragment((block_N, block_K), dtype)
            C_local1 = T.alloc_fragment((half_M, block_N), accum_dtype)
            C_store_local1 = T.alloc_fragment((half_M, block_N), out_dtype)
            C_packed = T.alloc_local((8,), out_dtype)

            T.abarrier_init(A_FREE_PING, num_consumer_waves)
            T.abarrier_init(A_READY_PING, num_producer_waves)
            T.abarrier_init(A_FREE_PONG, num_consumer_waves)
            T.abarrier_init(A_READY_PONG, num_producer_waves)
            T.abarrier_init(B_FREE_PING, num_consumer_waves)
            T.abarrier_init(B_READY_PING, num_producer_waves)
            T.abarrier_init(B_FREE_PONG, num_consumer_waves)
            T.abarrier_init(B_READY_PONG, num_producer_waves)
            T.ebarrier_sync_cnt(EBAR_ID, num_producer_waves + num_consumer_waves)

            if tx < producer_tx:
                T.set_max_nreg(MLS_PRODUCER_MAX_NREG_3BR, 0)
                producer_block_count = T.ceildiv(total_blocks - block_id, grid_size)
                for block_iter in T.Serial(producer_block_count):
                    ping_slot = (block_iter * tail_tiles) & 1
                    pong_slot = ping_slot ^ 1
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
                        ping_phase = ((block_iter * (2 * full_k_pairs + tail_tiles)) // 2 + k_pair) & 1
                        pong_phase = ((block_iter * (2 * full_k_pairs + tail_tiles) + 1) // 2 + k_pair) & 1
                        k_ping = k_pair * 2
                        k_pong = k_ping + 1
                        T.abarrier_try_wait(A_FREE_PING + ping_slot * 2, ping_phase)
                        T.abarrier_seq(A_READY_PING + ping_slot * 2)
                        T.matrix_load(
                            A[by * block_M, k_ping * block_K],
                            A_shared[ping_slot, :, :],
                            boundary=k_pair_boundary,
                            annotations=mls_annotations,
                        )
                        T.abarrier_arrive(A_READY_PING + ping_slot * 2)
                        T.abarrier_try_wait(B_FREE_PING + ping_slot * 2, ping_phase)
                        T.abarrier_seq(B_READY_PING + ping_slot * 2)
                        T.matrix_load(
                            B[bx * block_N, k_ping * block_K],
                            B_shared[ping_slot, :, :],
                            boundary=k_pair_boundary,
                            annotations=mls_annotations,
                        )
                        T.abarrier_arrive(B_READY_PING + ping_slot * 2)
                        T.sched_barrier(0)
                        T.abarrier_try_wait(A_FREE_PING + pong_slot * 2, pong_phase)
                        T.abarrier_seq(A_READY_PING + pong_slot * 2)
                        # Keep default boundary inference for pong: the last
                        # tile may be partial even when ceildiv(K, block_K)
                        # is even. Only ping is guaranteed full in a pair.
                        T.matrix_load(
                            A[by * block_M, k_pong * block_K],
                            A_shared[pong_slot, :, :],
                            annotations=mls_annotations,
                        )
                        T.abarrier_arrive(A_READY_PING + pong_slot * 2)
                        T.abarrier_try_wait(B_FREE_PING + pong_slot * 2, pong_phase)
                        T.abarrier_seq(B_READY_PING + pong_slot * 2)
                        T.matrix_load(
                            B[bx * block_N, k_pong * block_K],
                            B_shared[pong_slot, :, :],
                            annotations=mls_annotations,
                        )
                        T.abarrier_arrive(B_READY_PING + pong_slot * 2)

                    if tail_tiles != 0:
                        tail_ping_phase = ((block_iter * (2 * full_k_pairs + tail_tiles)) // 2 + full_k_pairs) & 1
                        tail_k_ping = full_k_pairs * 2
                        T.abarrier_try_wait(A_FREE_PING + ping_slot * 2, tail_ping_phase)
                        T.abarrier_seq(A_READY_PING + ping_slot * 2)
                        T.matrix_load(
                            A[by * block_M, tail_k_ping * block_K],
                            A_shared[ping_slot, :, :],
                            annotations=mls_annotations,
                        )
                        T.abarrier_arrive(A_READY_PING + ping_slot * 2)
                        T.abarrier_try_wait(B_FREE_PING + ping_slot * 2, tail_ping_phase)
                        T.abarrier_seq(B_READY_PING + ping_slot * 2)
                        T.matrix_load(
                            B[bx * block_N, tail_k_ping * block_K],
                            B_shared[ping_slot, :, :],
                            annotations=mls_annotations,
                        )
                        T.abarrier_arrive(B_READY_PING + ping_slot * 2)
            elif tx < consumer0_tx:
                T.set_max_nreg(MLS_CONSUMER_MAX_NREG_3BR, 0)
                T.abarrier_arrive(A_FREE_PING)
                T.abarrier_arrive(A_FREE_PONG)
                T.abarrier_arrive(B_FREE_PING)
                T.abarrier_arrive(B_FREE_PONG)
                consumer0_block_count = T.ceildiv(total_blocks - block_id, grid_size)
                for block_iter in T.Serial(consumer0_block_count):
                    ping_slot = (block_iter * tail_tiles) & 1
                    pong_slot = ping_slot ^ 1
                    tile_id = block_iter * grid_size + block_id
                    bx = tile_id % n_blocks
                    by = tile_id // n_blocks
                    # Peel the first ping/pong pair: only its ping starts from zero.
                    if has_first_pair:
                        # Keep the first A/B reads together before releasing either slot.
                        ping_phase = ((block_iter * (2 * full_k_pairs + tail_tiles)) // 2 + 0) & 1
                        pong_phase = ((block_iter * (2 * full_k_pairs + tail_tiles) + 1) // 2 + 0) & 1
                        T.abarrier_try_wait((A_READY_PING + ping_slot * 2), ping_phase)
                        T.ds_read_format(A_shared[ping_slot, 0:half_M, :], A_local_ping)
                        T.abarrier_try_wait(B_READY_PING + ping_slot * 2, ping_phase)
                        T.ds_read_format(B_shared[ping_slot, :, :], B_local_ping, alt=b_ds_read_alt)
                        T.abarrier_arrive(A_FREE_PING + ping_slot * 2)
                        T.abarrier_arrive(B_FREE_PING + ping_slot * 2)
                        T.ebarrier_arrive(CONSUMER_PING_EBAR_ID, num_consumer_waves)
                        T.call_extern("tl::set_prio<3>", dtype="void")
                        T.gemm(
                            A_local_ping,
                            B_local_ping,
                            C_local,
                            clear_accum=True,
                            transpose_B=True,
                            k_pack=1,
                            annotations=mmac_annotations,
                        )
                        T.call_extern("tl::set_prio<0>", dtype="void")
                        T.abarrier_try_wait((A_READY_PING + pong_slot * 2), pong_phase)
                        T.ds_read_format(A_shared[pong_slot, 0:half_M, :], A_local_pong)
                        T.abarrier_try_wait(B_READY_PING + pong_slot * 2, pong_phase)
                        T.ds_read_format(B_shared[pong_slot, :, :], B_local_pong, alt=b_ds_read_alt)
                        T.abarrier_arrive(A_FREE_PING + pong_slot * 2)
                        T.abarrier_arrive(B_FREE_PING + pong_slot * 2)
                        T.call_extern("tl::set_prio<2>", dtype="void")
                        T.gemm(
                            A_local_pong,
                            B_local_pong,
                            C_local,
                            transpose_B=True,
                            k_pack=1,
                            annotations=mmac_annotations,
                        )
                        T.call_extern("tl::set_prio<0>", dtype="void")

                    for k_pair in T.Serial(1, remaining_pair_end):
                        ping_phase = ((block_iter * (2 * full_k_pairs + tail_tiles)) // 2 + k_pair) & 1
                        pong_phase = ((block_iter * (2 * full_k_pairs + tail_tiles) + 1) // 2 + k_pair) & 1
                        T.abarrier_try_wait((A_READY_PING + ping_slot * 2), ping_phase)
                        T.ds_read_format(A_shared[ping_slot, 0:half_M, :], A_local_ping)
                        T.abarrier_arrive(A_FREE_PING + ping_slot * 2)
                        T.abarrier_try_wait(B_READY_PING + ping_slot * 2, ping_phase)
                        T.ds_read_format(B_shared[ping_slot, :, :], B_local_ping, alt=b_ds_read_alt)
                        T.abarrier_arrive(B_FREE_PING + ping_slot * 2)
                        T.ebarrier_arrive(CONSUMER_PING_EBAR_ID, num_consumer_waves)
                        T.call_extern("tl::set_prio<1>", dtype="void")
                        T.gemm(
                            A_local_ping,
                            B_local_ping,
                            C_local,
                            transpose_B=True,
                            k_pack=1,
                            annotations=mmac_annotations,
                        )
                        T.call_extern("tl::set_prio<0>", dtype="void")
                        T.abarrier_try_wait((A_READY_PING + pong_slot * 2), pong_phase)
                        T.ds_read_format(A_shared[pong_slot, 0:half_M, :], A_local_pong)
                        T.abarrier_arrive(A_FREE_PING + pong_slot * 2)
                        T.abarrier_try_wait(B_READY_PING + pong_slot * 2, pong_phase)
                        T.ds_read_format(B_shared[pong_slot, :, :], B_local_pong, alt=b_ds_read_alt)
                        T.abarrier_arrive(B_FREE_PING + pong_slot * 2)
                        T.ebarrier_sync_cnt(CONSUMER_PONG_EBAR_ID, num_consumer_waves)
                        T.call_extern("tl::set_prio<3>", dtype="void")
                        T.gemm(
                            A_local_pong,
                            B_local_pong,
                            C_local,
                            transpose_B=True,
                            k_pack=1,
                            annotations=mmac_annotations,
                        )
                        T.call_extern("tl::set_prio<0>", dtype="void")

                    if tail_tiles != 0:
                        tail_ping_phase = ((block_iter * (2 * full_k_pairs + tail_tiles)) // 2 + full_k_pairs) & 1
                        T.abarrier_try_wait((A_READY_PING + ping_slot * 2), tail_ping_phase)
                        T.ds_read_format(A_shared[ping_slot, 0:half_M, :], A_local_ping)
                        T.abarrier_arrive(A_FREE_PING + ping_slot * 2)
                        T.abarrier_try_wait(B_READY_PING + ping_slot * 2, tail_ping_phase)
                        T.ds_read_format(B_shared[ping_slot, :, :], B_local_ping, alt=b_ds_read_alt)
                        T.abarrier_arrive(B_FREE_PING + ping_slot * 2)
                        # Rendezvous before reusing PING at the next persistent block.
                        # Its first ping uses the other LDS slot and can be ready early.
                        T.ebarrier_sync_cnt(CONSUMER_PING_EBAR_ID, num_consumer_waves)
                        T.call_extern("tl::set_prio<1>", dtype="void")
                        T.gemm(
                            A_local_ping,
                            B_local_ping,
                            C_local,
                            clear_accum=tail_clear_accum,
                            transpose_B=True,
                            k_pack=1,
                            annotations=mmac_annotations,
                        )
                        T.call_extern("tl::set_prio<0>", dtype="void")
                    if use_matrix_store:
                        # Prioritize output conversion/store over the next tile's clear.
                        T.call_extern("tl::set_prio<1>", dtype="void")
                        T.copy(C_local, C_store_local)
                        T.matrix_store(
                            C_store_local,
                            C[by * block_M, bx * block_N],
                            trans=False,
                        )
                        T.call_extern("tl::set_prio<0>", dtype="void")
                    else:
                        for i, j in T.Parallel(half_M, block_N // 8):
                            for v in T.unroll(8):
                                C_packed[v] = C_local[i, j * 8 + v]
                            for v in T.vectorized(8):
                                if by * block_M + i < M and bx * block_N + j * 8 + v < N:
                                    C[by * block_M + i, bx * block_N + j * 8 + v] = C_packed[v]
            else:
                T.set_max_nreg(MLS_CONSUMER_MAX_NREG_3BR, 0)
                T.abarrier_arrive(A_FREE_PING)
                T.abarrier_arrive(A_FREE_PONG)
                T.abarrier_arrive(B_FREE_PING)
                T.abarrier_arrive(B_FREE_PONG)
                consumer1_block_count = T.ceildiv(total_blocks - block_id, grid_size)
                for block_iter in T.Serial(consumer1_block_count):
                    ping_slot = (block_iter * tail_tiles) & 1
                    pong_slot = ping_slot ^ 1
                    tile_id = block_iter * grid_size + block_id
                    bx = tile_id % n_blocks
                    by = tile_id // n_blocks
                    # Peel the first ping/pong pair: only its ping starts from zero.
                    if has_first_pair:
                        # Keep the first A/B reads together before releasing either slot.
                        ping_phase = ((block_iter * (2 * full_k_pairs + tail_tiles)) // 2 + 0) & 1
                        pong_phase = ((block_iter * (2 * full_k_pairs + tail_tiles) + 1) // 2 + 0) & 1
                        T.ebarrier_sync_cnt(CONSUMER_PING_EBAR_ID, num_consumer_waves)
                        T.abarrier_try_wait((A_READY_PING + ping_slot * 2), ping_phase)
                        T.ds_read_format(A_shared[ping_slot, half_M:block_M, :], A_local1_ping)
                        T.abarrier_try_wait(B_READY_PING + ping_slot * 2, ping_phase)
                        T.ds_read_format(B_shared[ping_slot, :, :], B_local1_ping, alt=b_ds_read_alt)
                        T.abarrier_arrive(A_FREE_PING + ping_slot * 2)
                        T.abarrier_arrive(B_FREE_PING + ping_slot * 2)
                        T.sched_barrier(0)
                        T.call_extern("tl::set_prio<3>", dtype="void")
                        T.gemm(
                            A_local1_ping,
                            B_local1_ping,
                            C_local1,
                            clear_accum=True,
                            transpose_B=True,
                            k_pack=1,
                            annotations=mmac_annotations,
                        )
                        T.call_extern("tl::set_prio<0>", dtype="void")
                        T.sched_barrier(0)
                        T.abarrier_try_wait((A_READY_PING + pong_slot * 2), pong_phase)
                        T.ds_read_format(A_shared[pong_slot, half_M:block_M, :], A_local1_pong)
                        T.abarrier_try_wait(B_READY_PING + pong_slot * 2, pong_phase)
                        T.ds_read_format(B_shared[pong_slot, :, :], B_local1_pong, alt=b_ds_read_alt)
                        T.abarrier_arrive(A_FREE_PING + pong_slot * 2)
                        T.abarrier_arrive(B_FREE_PING + pong_slot * 2)
                        T.call_extern("tl::set_prio<2>", dtype="void")
                        T.gemm(
                            A_local1_pong,
                            B_local1_pong,
                            C_local1,
                            transpose_B=True,
                            k_pack=1,
                            annotations=mmac_annotations,
                        )
                        T.call_extern("tl::set_prio<0>", dtype="void")

                    for k_pair in T.Serial(1, remaining_pair_end):
                        ping_phase = ((block_iter * (2 * full_k_pairs + tail_tiles)) // 2 + k_pair) & 1
                        pong_phase = ((block_iter * (2 * full_k_pairs + tail_tiles) + 1) // 2 + k_pair) & 1
                        T.ebarrier_sync_cnt(CONSUMER_PING_EBAR_ID, num_consumer_waves)
                        T.abarrier_try_wait((A_READY_PING + ping_slot * 2), ping_phase)
                        T.ds_read_format(A_shared[ping_slot, half_M:block_M, :], A_local1_ping)
                        T.abarrier_arrive(A_FREE_PING + ping_slot * 2)
                        T.abarrier_try_wait(B_READY_PING + ping_slot * 2, ping_phase)
                        T.ds_read_format(B_shared[ping_slot, :, :], B_local1_ping, alt=b_ds_read_alt)
                        T.abarrier_arrive(B_FREE_PING + ping_slot * 2)
                        T.sched_barrier(0)
                        T.gemm(
                            A_local1_ping,
                            B_local1_ping,
                            C_local1,
                            transpose_B=True,
                            k_pack=1,
                            annotations=mmac_annotations,
                        )
                        T.sched_barrier(0)
                        T.ebarrier_arrive(CONSUMER_PONG_EBAR_ID, num_consumer_waves)
                        T.abarrier_try_wait((A_READY_PING + pong_slot * 2), pong_phase)
                        T.ds_read_format(A_shared[pong_slot, half_M:block_M, :], A_local1_pong)
                        T.abarrier_arrive(A_FREE_PING + pong_slot * 2)
                        T.abarrier_try_wait(B_READY_PING + pong_slot * 2, pong_phase)
                        T.ds_read_format(B_shared[pong_slot, :, :], B_local1_pong, alt=b_ds_read_alt)
                        T.abarrier_arrive(B_FREE_PING + pong_slot * 2)
                        T.call_extern("tl::set_prio<2>", dtype="void")
                        T.gemm(
                            A_local1_pong,
                            B_local1_pong,
                            C_local1,
                            transpose_B=True,
                            k_pack=1,
                            annotations=mmac_annotations,
                        )
                        T.call_extern("tl::set_prio<0>", dtype="void")

                    if tail_tiles != 0:
                        tail_ping_phase = ((block_iter * (2 * full_k_pairs + tail_tiles)) // 2 + full_k_pairs) & 1
                        T.ebarrier_sync_cnt(CONSUMER_PING_EBAR_ID, num_consumer_waves)
                        T.abarrier_try_wait((A_READY_PING + ping_slot * 2), tail_ping_phase)
                        T.ds_read_format(A_shared[ping_slot, half_M:block_M, :], A_local1_ping)
                        T.abarrier_arrive(A_FREE_PING + ping_slot * 2)
                        T.abarrier_try_wait(B_READY_PING + ping_slot * 2, tail_ping_phase)
                        T.ds_read_format(B_shared[ping_slot, :, :], B_local1_ping, alt=b_ds_read_alt)
                        T.abarrier_arrive(B_FREE_PING + ping_slot * 2)
                        T.sched_barrier(0)
                        T.gemm(
                            A_local1_ping,
                            B_local1_ping,
                            C_local1,
                            clear_accum=tail_clear_accum,
                            transpose_B=True,
                            k_pack=1,
                            annotations=mmac_annotations,
                        )
                        T.sched_barrier(0)
                    if use_matrix_store:
                        # Use the same output priority for both consumer groups.
                        T.call_extern("tl::set_prio<1>", dtype="void")
                        T.copy(C_local1, C_store_local1)
                        T.matrix_store(
                            C_store_local1,
                            C[by * block_M + half_M, bx * block_N],
                            trans=False,
                        )
                        T.call_extern("tl::set_prio<0>", dtype="void")
                    else:
                        for i, j in T.Parallel(half_M, block_N // 8):
                            for v in T.unroll(8):
                                C_packed[v] = C_local1[i, j * 8 + v]
                            for v in T.vectorized(8):
                                if by * block_M + half_M + i < M and bx * block_N + j * 8 + v < N:
                                    C[by * block_M + half_M + i, bx * block_N + j * 8 + v] = C_packed[v]

    return _gemm_wasp_mls_4p4c4c_persistent


KERNELS = {"mls_4p4c4c_persistent": gemm_wasp_mls_4p4c4c_persistent}


def cpu_ref_program(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    return (a.cpu().float() @ b.cpu().float().T).to(out_dtype)


def fp16_acc_ref(a: torch.Tensor, b: torch.Tensor, k_chunk: int = 32) -> torch.Tensor:
    """Reference the FP16 accumulator by rounding after every MMAC K chunk."""
    a = a.cpu().float()
    b = b.cpu().float()
    result = torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float16)
    for k0 in range(0, a.shape[1], k_chunk):
        partial = (a[:, k0 : k0 + k_chunk] @ b[:, k0 : k0 + k_chunk].T).to(torch.float16)
        result = (result + partial).to(torch.float16)
    return result


def _collect_perf_artifacts():
    """Copy PMD perf and trace outputs into the script-local artifact directory."""
    search_roots = (Path.cwd() / "m5out", Path("/tmp/m5out"))
    patterns = ("*.perf", "ttrace*.out")
    copied = set()
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for source in root.rglob(pattern):
                if source.is_file() and source not in copied:
                    shutil.copy2(source, PERF_DIR / source.name)
                    copied.add(source)


def run_check(
    variant: str,
    m: int = BLOCK_M,
    n: int = BLOCK_N,
    k: int = BLOCK_K,
    device: int = 0,
    dtype: str = "float8_e4m3fn",
    dynamic: bool = True,
    matrix_store: bool = False,
    dump_source: bool = False,
    num_persistent_blocks: int = 48,
):
    require_gfx94x()
    torch.cuda.set_device(device)
    kernel_fn = KERNELS[variant]
    if num_persistent_blocks <= 0:
        raise ValueError("num_persistent_blocks must be positive")
    if min(m, n, k) <= 0:
        raise ValueError("M/N/K must be positive")
    max_m_align = min(_max_power_of_two_alignment(m), 2 * BLOCK_M)
    max_n_align = min(_max_power_of_two_alignment(n), 2 * BLOCK_N)
    max_k_align = min(_max_power_of_two_alignment(k), 2 * BLOCK_K)
    if matrix_store and variant != "mls_4p4c4c_persistent":
        raise ValueError("--matrix-store requires the persistent variant")
    shape = (T.dynamic("M"), T.dynamic("N"), T.dynamic("K")) if dynamic else (m, n, k)
    options = {
        "dtype": dtype,
        "out_dtype": "float16",
        "max_m_align": max_m_align,
        "max_n_align": max_n_align,
        "max_k_align": max_k_align,
        "num_persistent_blocks": num_persistent_blocks,
    }
    if variant == "mls_4p4c4c_persistent":
        options["use_matrix_store"] = matrix_store
    kernel = kernel_fn(*shape, **options)
    if dump_source:
        print(kernel.get_kernel_source())
        return
    generator = torch.Generator().manual_seed(321)
    a = torch.randn(m, k, generator=generator, dtype=torch.float32).to(getattr(torch, dtype))
    b = torch.randn(n, k, generator=generator, dtype=torch.float32).to(getattr(torch, dtype))
    expected = fp16_acc_ref(a, b)
    a, b = a.to(device=f"cuda:{device}"), b.to(device=f"cuda:{device}")
    c = kernel(a, b)
    torch.testing.assert_close(c.cpu(), expected, rtol=1e-2, atol=1e-1)
    _collect_perf_artifacts()
    print(
        f"OK: gemm_wasp_{variant} M={m} N={n} K={k} dynamic={dynamic} matrix_store={matrix_store} alignment={(max_m_align, max_n_align, max_k_align)} on device {device}"
    )


def main():
    parser = argparse.ArgumentParser(description="gfx94x WASP ABarrier GEMM FP8 FP16-accumulator smoke tests")
    parser.add_argument(
        "--variant",
        choices=["mls_4p4c4c_persistent"],
        default="mls_4p4c4c_persistent",
        help="persistent FP8-input kernel",
    )
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float8_e4m3fn", choices=["float8_e4m3fn", "float8_e5m2"])
    parser.add_argument("--dump-source", action="store_true")
    parser.add_argument("--static-shape", action="store_true", help="use static dimensions instead of the default dynamic M/N/K")
    parser.add_argument("--matrix-store", action="store_true", help="use native B16 MatrixStore (persistent variant only)")
    parser.add_argument("--num-persistent-blocks", type=int, default=48)
    args = parser.parse_args()
    require_gfx94x()
    torch.cuda.set_device(args.device)

    run_check(
        args.variant,
        args.m,
        args.n,
        args.k,
        device=args.device,
        dtype=args.dtype,
        dynamic=not args.static_shape,
        matrix_store=args.matrix_store,
        dump_source=args.dump_source,
        num_persistent_blocks=args.num_persistent_blocks,
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
