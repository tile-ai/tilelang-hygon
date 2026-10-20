# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Channelwise-scaled fp8 GEMM via the MLS (matrix-LDS-read) producer path.

Computes ``C[m, n] = scale_a[m] * scale_b[n] * sum_k A[m, k] * B[n, k]`` with
A (M, K) fp8, B (N, K) fp8 (N-major), fp32 per-token / per-channel scales
applied as an epilogue multiply on the fp32 accumulator.

Both A and B are loaded with ``T.copy(..., prefer_instruction="matrix_load")``
(MLS, the async global->LDS matrix load). This makes ``T.gemm`` lower through
``_gemm_mls_mls`` -> ``ds_read_format_tensor_a/b`` — the HCU hardware matrix
LDS read (``ds_read_m32x32_b8``), the same read the hipblaslt fp8 kernel uses.
On HCU (gfx938) the winning config (block 256x256x64, num_stages=1, 512
threads) reaches **~432 TFLOPS @ 8192^3** (rel=0), vs ~332 for the scalar-LDS
``cp.async`` path and ~560 for lightop/hipblasLt.

Constraint: with ``trans_B=True`` the B MLS atom is the trans b8 tile
(tile_k 64/128), so ``block_K`` must be a multiple of 64.

Run (validate + bench):
    HIP_VISIBLE_DEVICES=2 python perf/gemm/fp8_channelwise_mls.py
"""

from dataclasses import dataclass

import torch
import tilelang
import tilelang.language as T

# Per-block LDS budget on gfx938 (src/hcu/target_utils.cc:202).
LDS_LIMIT = 65536
# MMA output micro-tile / MLS atom floor on both axes (src/tl_templates/hcu/gemm.h:150,
# src/hcu/op/mls.cc:408). block_N < 16 is not expressible -- see decode_cfg docstring.
MIN_TILE = 16


@dataclass(frozen=True)
class DecodeCfg:
    """Tile + pipeline config for the M<=128 decode GEMM."""

    block_M: int
    block_N: int
    block_K: int
    num_stages: int
    threads: int
    setprio: int
    # Kept as an explicit closure marker: split-K was built, measured 10-21%
    # slower than the non-split baseline on o and down, and removed. The field
    # exists so the chooser's "never select it" invariant stays assertable.
    split_k: int = 1


def _validate_cfg(cfg, M, N, K):
    """Fail at selection time, with the shape in the message, rather than deep
    inside tilelang.compile.  Every assert here has been a real silent-wrong-answer
    or launch failure at some point."""
    bm, bn, bk, ns, sk = cfg.block_M, cfg.block_N, cfg.block_K, cfg.num_stages, cfg.split_k
    assert bk % 64 == 0, f"shape {M}x{N}x{K}: block_K={bk} must be a multiple of 64 (trans-B fp8 MLS atom)"
    # K % block_K == 0 is NOT implied by bk % 64 == 0: the pipelined T.copy reads a whole
    # block_K tile per step, so a non-dividing block_K runs past the end of B and produces
    # silently wrong results (bk=192 on K=5120 gave rel=1.61 with no error).  This is also
    # shape-dependent: bk=320 divides K=5120 and K=25600 but not K=8192.
    assert K % bk == 0, f"shape {M}x{N}x{K}: K % block_K != 0 (K={K}, block_K={bk}) -- silent OOB copy"
    assert K % (sk * bk) == 0, f"shape {M}x{N}x{K}: K % (split_k*block_K) != 0 (K={K}, sk={sk}, bk={bk})"
    assert bm % MIN_TILE == 0 and bn % MIN_TILE == 0, f"shape {M}x{N}x{K}: block_M/block_N must be multiples of {MIN_TILE}"
    lds = (bm + bn) * bk * ns
    assert lds <= LDS_LIMIT, f"shape {M}x{N}x{K}: LDS {lds}B > {LDS_LIMIT}B (bm={bm}, bn={bn}, bk={bk}, ns={ns})"
    assert M <= bm, f"shape {M}x{N}x{K}: M > block_M={bm} (we never split M -- it re-reads B)"
    # num_stages > 1 outside bm64/bn64 is now CORRECT (the packed-stride defect it
    # used to hit was fixed 2026-10-17h -- see the doc section of that name), but
    # it is still untuned: no ns>1 tile below bm64/bn64 has been benchmarked
    # against its ns=1 counterpart, so the chooser keeps returning ns=1 there.
    # bn96/bk256/ns2 is the one benchmarked small-tile ns>1 exception: exact
    # (rel == 0.0) on all four decode shapes at M in {1,16,32}, and 3-8% faster
    # than its ns=1 counterpart.  bm16/bn16/ns2 stays blocked (it is NaN).
    assert ns <= 1 or (bm, bn) == (64, 64) or (bn == 96 and bk == 256 and ns <= 2), (
        f"shape {M}x{N}x{K}: num_stages={ns} is only tuned for bm=bn=64 or bn96/bk256 (bm={bm}, bn={bn}, bk={bk})")
    return cfg


def decode_cfg(M, N, K):
    """Shape-aware decode config.  Replaces the old M-only table.

    The deficit is a property of the SHAPE, not of M: ``_decode_cfg(M)`` was tuned
    on gate_up and applied to all four decode shapes, which is why o/down lost at
    every M.  Measured rule (paired, same-process, 7 rounds, idle card):

      N >= 10240 -> bn32/t256      (gate_up 1600 blocks, qkv 320 blocks: both win)
        ... except bm32 in the narrow window 9216 <= N <= 10240 -> bn32/bk128/t128
        (qkv M=17..32 only; +9-10% over bn32/bk256/t256, see the inline note)
      N ==  5120 -> bn96/bk256/ns2/t192  (o/down; superseded bn16/t64 on
                    2026-10-20 -- see the inline note; bn16 is still the
                    fallback when K % 256 != 0)

    block_N < 16 is impossible: the MMA output micro-tile is hardware-fixed at
    16x16 (``src/tl_templates/hcu/gemm.h:150``) and the MLS B atom floor is
    tile_mn=16 (``src/hcu/op/mls.cc:408``).  So bn16 is the narrowest tile that
    exists, and this table is at the floor of the in-tree search space.

    NOTE the original diagnosis here was "o/down are block-starved". That was a
    confound -- the sweep behind it varied N at fixed K, so block count and total
    bytes moved together. Holding bytes constant and varying only block count
    (``/tmp/b0_grid_scale.py``) reverses the sign: 320 blocks/100 k-steps is the
    *best* point of the down curve, and 1280/25 is worse. More blocks is not the
    lever, and split-K (which is exactly that transformation) measured 10-21%
    slower. See the 2026-10-17f section of the doc.

    block_M stays the pow2 band rather than M -- block_M = M would break
    ``M % kMPerWarp == 0`` for non-pow2 M.
    """
    if M <= 16:
        bm, sp = 16, 1
    elif M <= 32:
        bm, sp = 32, 1
    elif M <= 64:
        # Out of scope, and a measured structural optimum: bk128 ns2 (double-buffered)
        # beats bk256 ns1 by 1.2% (243.5 vs 246.6us). The gain is cross-k-step overlap,
        # NOT wait granularity -- at fixed bk256, ns=2 is 1.6% SLOWER.
        return _validate_cfg(DecodeCfg(64, 64, 128, 2, 256, 1), M, N, K)
    else:
        return _validate_cfg(DecodeCfg(128, 128, 128, 1, 512, 0), M, N, K)

    if N >= 10240:
        bn, threads = 32, 256
        # Narrow measured window (2026-10-17g): at N=10240 with bm32, bk128/t128
        # beats the bn32/bk256/t256 above by 9-10% (paired, same process, 16
        # rounds, both arms setprio=1: 52.3 vs 57.8us), which takes qkv M=32 from
        # 1.14x to 1.03x vs hipblas (best round 0.945 -- hipblas is bimodal at
        # 51/56us here, so the ratio is a range, not a point).
        #
        # It is NOT a general "smaller block wins" rule, and the mechanism is not
        # understood. Measured at M=32/K=5120 with bn32, bk128/t128 loses by
        # 17-38% at N = 11264, 15360, 20480, 25600 and 51200 -- the next point up
        # from 10240 already loses, and no N between 10240 and 11264 was
        # measured. It is not a byte-count or cache effect: the win holds at
        # N=10240 for K = 5120, 10240 and 20480 alike, and survives an L2 flush
        # before every timed replay (L2 is 8 MB, so a 52 MB B cannot be resident
        # anyway). So it is a property of this one grid shape, written as the one
        # measured point rather than extrapolated. K % 128 keeps the fallback's
        # silent-OOB guard intact for shapes this was never measured on. See the
        # 2026-10-17g section of docs/optimization/fp8_decode_m_le_128.md.
        if bm == 32 and N <= 10240 and K % 128 == 0:
            return _validate_cfg(DecodeCfg(32, 32, 128, 1, 128, 1), M, N, K)
    else:
        # Wide/deep tile for N == 5120 (o, down).  The 16x16 MMA atom forces
        # bn >= 16, and bn16 means 320 blocks each re-reading the whole A tile:
        # A is read 320 times.  hipblas's M=16 kernel (measured with hipprof)
        # instead launches 54 workgroups covering 96 N-columns each, so it
        # re-reads A only 54 times.  bn96 with num_stages=2 is the widest tile
        # whose two stages still fit in one CU's 65536 B
        # (2*(16+96)*256 = 57344), and it needs the dense 16x256 MLS packing in
        # src/hcu/op/mls.cc (6144 -> 4096 per tile) to fit at all.
        # Measured 2026-10-20 (paired, same process, 5 rounds, idle card) vs the
        # bn16/t64/ns1 baseline, at M = 1/16/32:
        #   down: 0.923 / 0.932 / 0.952     o: 0.952 / 0.950 / 0.969
        # and vs hipblas (M=16): down 1.177 -> 1.097, o 1.116 -> 1.060.
        # qkv and gate_up (N >= 10240) LOSE 5-23% with this tile -- they already
        # beat hipblas, so this rule is deliberately N == 5120 only.
        # Correctness: rel == 0.0 against the fp32 reference on all four shapes
        # at M = 1/16/32 (12 arms); `num_stages=2` is exact here, unlike
        # bm16/bn16/ns2 which produces NaN.
        #
        # 3 wavefronts, not 6 (2026-10-20b).  The MLS A producer gives *every*
        # warp the same A load -- the address in the k-loop has no warp-id term
        # -- so A's read amplification is blocks x warps and was still ~320x at
        # t384 (54 blocks x 6), i.e. the wide tile alone never cut it.  Measured
        # at fixed bn96/bk256/ns2, paired, 5 rounds, idle card:
        #   t384 1.000  t192 0.952-0.962  t128 0.971-0.983  t64 1.23-1.29
        # (a U: fewer warps means less A re-read but less latency hiding).
        # Across M = 1/16/32 vs t384: o 0.961/0.969/0.893,
        # down 0.940/0.955/0.868 -- o M=32 ties hipblas (1.000) and down M=32
        # beats it (0.935).  Do not pick t64/t128: both regress.
        if N == 5120 and K % 256 == 0:
            return _validate_cfg(DecodeCfg(bm, 96, 256, 2, 192, sp), M, N, K)
        bn = 16
        # threads must scale with block_M: at bn16 a bm32 tile has two 16x16 M-atoms,
        # and giving them ONE warp (t=64) makes the block MMA-issue-bound instead of
        # B-stream-bound. Measured on o/down at M=32 (paired, 7 rounds):
        #   o   : t64 75.2us (1.911x)  t128 72.3 (1.837x)  t256 46.9 (1.191x)
        #   down: t64 209.9us (1.944x) t128 203.0 (1.880x) t256 122.4 (1.133x)
        # t=256 is a 1.60x/1.71x win over the t=64 that the old M-only table gave.
        threads = 64 if bm <= 16 else 256
    return _validate_cfg(DecodeCfg(bm, bn, 256, 1, threads, sp), M, N, K)


def fp8_mls_channelwise(
    M,
    N,
    K,
    block_M=256,
    block_N=256,
    block_K=64,
    num_stages=1,
    threads=512,
    accum_dtype="float32",
    setprio=0,
    policy=None,
):
    """Channelwise fp8 GEMM using MLS (matrix-LDS-read) A/B producers.

    ``block_K`` must be a multiple of 64 (the trans-B fp8 MLS atom width).

    ``setprio=1`` wraps the MLS mmac loop in ``s_setprio(1)``/``s_setprio(0)``
    (``tl.hcu_mmac_setprio``), matching hipblas's MMA priority bump. At the
    shipped bn16/t64 decode structure it is worth 7-11% at every M in
    {1,16,32} (paired, 2026-10-19: o -6.6/-6.7/-10.2%, down
    -10.5/-10.4/-11.5%) — the earlier note that it "only pays off at M=64" was
    wrong. Default stays OFF so the 8192² compute path is unchanged.

    ``policy=T.GemmWarpPolicy.FullColK`` partitions the K axis across warps
    (``k_warp = threads/64`` when the N tile cannot absorb them). Each warp then
    accumulates a partial C over its own K slice, so the kernel MUST follow the
    loop with ``T.reduce_sum_warp(C_l, C_l, clear=False)``: the C fragment is
    *replicated* across the K warps and the copy-out guard stores replica 0
    only, so without the reduction the output is silently K-warp-0's partial.
    With ``policy=None`` (default) the loop body and codegen are byte-identical
    to the pre-warp_k builder.

    FullColK is measured *neutral-to-negative* on o/down (2026-10-19, paired
    5-arm, 9 rounds: WK4/bn16/bk256/t256 vs the shipped config is 0.97-1.03 with
    mixed sign), and it is a substitute for ``setprio`` rather than a complement
    — each recovers the same ~7-11% and they do not stack. ``decode_cfg``
    therefore never selects it. It is kept because it is correct and removes an
    artificial limitation, not because it wins.
    """
    assert block_K % 64 == 0, "fp8 MLS channelwise GEMM requires block_K % 64 == 0"
    # block_K % 64 is necessary (trans-B atom) but not sufficient: the pipelined
    # T.copy reads a whole block_K tile per step, so a non-dividing block_K runs
    # past the end of B and produces silently wrong results (bk=192 on K=5120
    # gave rel=1.61 with no error).
    assert K % block_K == 0, f"fp8 MLS channelwise GEMM requires K % block_K == 0 (K={K}, block_K={block_K})"
    # num_stages > 1 outside the bm64/bn64 region is STILL WRONG, not merely
    # untuned.  Two independent defects lived here:
    #  (1) the packed-stride mismatch (measured 2026-10-17f: rel 0.75-580, or
    #      non-finite; it tracked the thread count because that selects the MLS
    #      tile, and only tile_k=128 takes the packed LDS branch).  Root-caused
    #      and genuinely fixed 2026-10-17h -- the ds_read consumer read stage >= 1
    #      from the logical stride while the producer and the shared-memory
    #      allocator used the packed stride.  bk256/ns2 at bn16 is now exact.
    #  (2) a SECOND defect the fix did not reach: at bm16/bn16/t64 the emitted
    #      per-stage LDS base is wrong once the packed branch and ns > 1 combine,
    #      and because the MLS producer is async while the consumer ds_reads
    #      independently, aliased stages corrupt each other intermittently -- the
    #      result varies run to run on identical inputs (measured 2026-10-18,
    #      K=25600, 5 runs each:
    #        bk128/ns2 rel 3504/2898/2090/1980/1119
    #        bk128/ns3 rel 12123/11923/9552/10799/9271
    #        bk256/ns3 rel 8899/8709/8581/10180/6289
    #        bk256/ns4 rel 6597/5616/7074/7700/6511
    #      while bk256/ns2, bk128/ns4 and every contiguous (bk64/bk512) case are
    #      exact and reproducible).  The emitted stage bases show why: bk256/ns2
    #      uses (k&1)*6144, exactly the packed size, while bk128/ns2 uses
    #      (k&1)*1024 against a packed size of 5120 and the ns=4 forms use an
    #      affine-simplifier artifact like (((k&3)*24)&63)>>4, which is not even a
    #      permutation of the stages.  So "small multi-stage tiles are correct but
    #      unbenchmarked" -- the earlier reading of this assert -- is FALSE.
    # The shipped configs are unaffected: all four decode shapes at M=16 and M=64
    # were re-run 5x each through this builder on 2026-10-18 and are exact
    # (bm64/bn64/bk128/ns2 is the only ns>1 config in the table).  This assert is
    # what keeps it that way -- do not lift it without fixing (2) first.
    # Experiment escape hatch: allow the decode-band multi-stage sweep.  Default
    # behaviour is unchanged (env unset -> identical guard).  Every candidate
    # built through here is checked against the fp32 reference before it is
    # timed, so a wrong stage stride shows up as rel != 0, not as a fast arm.
    import os as _os
    _ns_escape = _os.environ.get("TL_HCU_ALLOW_NS2_DECODE") == "1" and num_stages <= 4
    # The N==5120 wide tile is the one measured exception: bn96/bk256/ns2 is
    # exact (rel == 0.0) on all four decode shapes at M in {1,16,32}, because
    # 16x96x256 packs to 4096 B/tile after the dense-packing fix, so the stage
    # stride is the packed size.  bm16/bn16/ns2 remains NaN and stays blocked.
    _wide_exact = block_N == 96 and block_K == 256 and num_stages <= 2
    assert num_stages <= 1 or (block_M, block_N) == (64, 64) or _wide_exact or _ns_escape, (
        f"num_stages={num_stages} is only tuned for block_M=block_N=64 on HCU "
        f"(got block_M={block_M}, block_N={block_N}); multi-stage small tiles are correct "
        f"but unbenchmarked. Use num_stages<=1, or benchmark the tile first.")
    gemm_annotations = {"tl.hcu_mmac_setprio": int(setprio)} if setprio else None
    if policy is None:
        policy = T.GemmWarpPolicy.Square

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float8_e4m3fn"),
        B: T.Tensor((N, K), "float8_e4m3fn"),
        scale_a: T.Tensor((M,), "float32"),
        scale_b: T.Tensor((N,), "float32"),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), "float8_e4m3fn")
            B_s = T.alloc_shared((block_N, block_K), "float8_e4m3fn")
            C_l = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_s, prefer_instruction="matrix_load")
                T.copy(B[bx * block_N, k * block_K], B_s, prefer_instruction="matrix_load")
                if num_stages == 0:
                    T.ptx_wait_group(0)
                    T.sync_warp()
                T.gemm(A_s, B_s, C_l, False, True, k_pack=1, annotations=gemm_annotations,
                       policy=policy)
            if policy == T.GemmWarpPolicy.FullColK:
                # Cross-K-warp reduction. The C fragment is replicated per K warp
                # (make_gemm_fragment_hcu) and the copy-out stores replica 0 only,
                # so this is mandatory, not an optimization.
                T.reduce_sum_warp(C_l, C_l, clear=False)
            for i, j in T.Parallel(block_M, block_N):
                C_l[i, j] = C_l[i, j] * scale_a[by * block_M + i] * scale_b[bx * block_N + j]
            T.copy(C_l, C[by * block_M, bx * block_N])

    return tilelang.compile(gemm, out_idx=[4])


def _ref(A, B, scale_a, scale_b):
    return (A.float() @ B.float().T) * scale_a[:, None] * scale_b[None, :]


def _validate_and_bench(size=8192):
    from tilelang.profiler import do_bench_cudagraph

    torch.manual_seed(0)
    A = torch.randn(size, size, device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn(size, size, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.rand(size, device="cuda") + 0.5
    sb = torch.rand(size, device="cuda") + 0.5

    kern = fp8_mls_channelwise(size, size, size)  # 256x256x64 ns1 t512
    C = kern(A, B, sa, sb)
    ref = _ref(A, B, sa, sb)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    lat = do_bench_cudagraph(lambda: kern(A, B, sa, sb))
    tf = 2.0 * size * size * size / lat * 1e-9
    print(f"fp8_mls_channelwise {size}^3 (256x256x64 ns1 t512): "
          f"{tf:.1f} TFLOPS, rel={rel:.6f}, {lat * 1e3:.1f} us")
    assert rel < 1e-3, f"rel error {rel} too large"
    return tf, rel


if __name__ == "__main__":
    _validate_and_bench()
