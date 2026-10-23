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
    HIP_VISIBLE_DEVICES=2 python tilelang/hcu/op/gemm/gemm_fp8_channelwise.py
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


# bk64 tiles admitted for num_stages=2 (2026-10-21).  Halving block_K halves the
# per-block LDS (and the per-step bytes), which doubles the resident blocks/CU
# from 1 to 2 -- the prefetch depth this path pays for in LDS rather than VGPRs.
# It only pays when the grid still fills those slots, i.e. more blocks than CUs
# (see decode_cfg).  Measured paired same-process, graph replay, 7 rounds, idle
# card, every arm bit-exact (rel == 0.0), ratio = ours/hipblas:
#   96x96x64/ns2/t384/u8   qkv32 M= 96  0.9633   (shipped bn96/bk128 1.0467)
#   96x128x64/ns2/t512/u8  qkv27 M= 96  0.9392   (shipped bn128/bk128 ns1 1.0272,
#                                                shipped bm96/bn96 1.3392)
#   128x128x64/ns2/t512/u8 qkv32 M=128  0.9807   (shipped bn96/bk128 1.1268)
#   128x128x64/ns2/t512/u8 qkv27 M=128  1.1021   (shipped bn96/bk128 1.5782)
# Negative controls, same protocol: z27 (N=6144, 64 blocks) 1.20-1.28x with bk64
# against 0.89-1.05x with bk128, o32 (N=5120) 1.32 vs 1.04, gu32 (N=51200) a tie.
_NS2_BK64_TILES = {(96, 96, 64), (96, 128, 64), (128, 128, 64)}

# Shapes where a vendor-derived (hipblaslt tuner) tile was measured to WIN, and
# only those.  These are explicit (N, K) lists, not ranges: all three tiles below
# lose on part of the shape space, and the loss is not monotonic in N, so a
# threshold rule would silently regress every shape it was never measured on.
# Anything not listed keeps the tile it had.
#   M <= 16   bn48  (16x48x256/ns2/t64, tuner MT32x48x256) -- wins on five of the
#             six decode shapes.  in_proj_qkv (10240x5120) is the exclusion: its
#             existing bn32/bk256/ns1 tile is the strongest cell of the whole
#             decode table (0.82x hipblas) and bn48 gives ~7% of that back.
#   bm64+bk128 bn192 (64x192x128/ns2/t512, tuner MT128x192x64) -- wins on
#             N = 34816 and 10240 only, and loses 1.4-1.6x on N = 14336
#             (75 bn192-blocks) and N = 6144 (32 blocks).  Hypothesis, NOT yet a
#             rule: 75 blocks is one full wave of 71 CUs plus 4 stragglers, so
#             ~half the runtime is a nearly-empty second wave; 32 blocks
#             under-fill one wave outright.  182 blocks (N=34816, 2.56 waves) and
#             54 (N=10240, 0.76 wave) are both fine.  The same arithmetic does
#             NOT explain the M<=16 losses (o/down run 1.5 waves there and still
#             win), so nothing here is extrapolated.
#   bm96+bk64 bn192 (96x192x64/ns2/t384) wins on gate_up27 alone, and only at
#             setprio=0 (see _BN192_BM96_NK).  in_proj_qkv is NOT in that list:
#             there the arm measures 69.60us against the shipped 69.92us, i.e.
#             inside noise, and it costs 2.5% at setprio=1.
_BN48_M16_NK = frozenset({(5120, 6144), (5120, 17408), (6144, 5120),
                          (14336, 5120), (34816, 5120)})
_BN192_NK = frozenset({(10240, 5120), (34816, 5120)})
# Deliberately narrower than _BN192_NK: at bm96 the tile only pays on gate_up27.
_BN192_BM96_NK = frozenset({(34816, 5120)})

# Tiles whose num_stages=2 form was validated (rel < 1e-3 against a chunked fp32
# reference on every decode shape each one is selected for, and bit-identical to
# the tile it replaces: relmax 0.0e+00, no NaN).  Kept separate from
# _NS2_BK64_TILES, which is the bk64 "2 blocks/CU" family and carries a
# different mechanism.
_NS2_VENDOR_TILES = {(16, 48, 256), (64, 192, 128), (96, 192, 64)}


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
    # bn96/bk128/ns2 at bm>=64 is the M in (64,128] band tile, added 2026-10-20:
    # exact (rel < 1e-3) at bm 64/96/128 on o and down, and 1.35x faster than its
    # ns=1 counterpart at M=128 (that is defect (2) in decode_cfg's comment).
    # bm96/bn128/bk128/ns2 is the gate_up M in (64,96] tile (2026-10-21): exact
    # (rel = 0) at M = 65/72/80/88/96 with both t512 and t256 -- the stage stride
    # depends on (bm,bn,bk) and the A copy extent on M, so both were swept.
    _wide_exact = (bn == 96 and bk == 256 and ns <= 2) or (bn == 96 and bk == 128 and bm >= 64 and ns <= 2) \
        or (bm == 128 and bn == 128 and bk == 128 and ns <= 2) \
        or (bm == 96 and bn == 128 and bk == 128 and ns <= 2) \
        or ((bm, bn, bk) in _NS2_BK64_TILES and ns <= 2) \
        or ((bm, bn, bk) in _NS2_VENDOR_TILES and ns <= 2)
    assert ns <= 1 or (bm, bn) == (64, 64) or _wide_exact, (
        f"shape {M}x{N}x{K}: num_stages={ns} is only tuned for bm=bn=64 or the bn96 wide tile "
        f"(bm={bm}, bn={bn}, bk={bk})")
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
                    M in (96,128] -> bm128/bn96/bk128/ns2/t512 (2026-10-21,
                    unrolled k-loop; it was bn128/t256 under T.Pipelined)
      M in (64,128], other N (gate_up, qkv) -- decided by BLOCK COUNT first and
                    MMA padding second (2026-10-21):
                      ceil(N/128) >= 256 -> bm128/bn128/bk128/ns2/t512
                      else M <= 96       -> bm96/bn96/bk128/ns2/t384
                      else               -> bm128/bn128/bk128/ns1/t512

    Three arms added 2026-10-23 come from the hipblaslt *tuner*'s own decode
    kernels rather than from the in-tree search, and all three are gated on an
    explicit (N, K) list instead of a rule -- see ``_BN48_M16_NK`` /
    ``_BN192_NK`` / ``_BN192_BM96_NK``:

      M <= 16, listed shapes   -> bm16/bn48/bk256/ns2/t64   (tuner MT32x48x256)
      M in (32,64], listed     -> bm64/bn192/bk128/ns2/t512 (tuner MT128x192x64)
      M in (64,96], listed     -> bm96/bn192/bk64/ns2/t384  (same tuner kernel,
                                  setprio=0 -- the one arm that does not want it)

    Worth 1-10% over the previous tile on every cell it is applied to (the
    2026-10-23 confirmation A/B, setprio-correct baseline), and all three are
    bit-exact with the tile they replace (relmax 0.0e+00) as well as within
    3.1e-03 of a chunked fp32 reference.  The lists exist because bn192 in
    particular is 1.4-1.6x SLOWER on the shapes left out of it; see the comment
    on the frozensets for the wave-quantisation hypothesis behind that and for
    why it is deliberately not written as a formula.  Two cells were dropped
    after the corrected baseline: in_proj_qkv M=96 (inside noise) and every
    qkv27/in_proj_z bn192 cell (1.4-1.6x slower).

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
        # bn48 (2026-10-23, from the hipblaslt tuner's MT32x48x256): the widest
        # bn that still fits two bk256 stages at bm16 (2*(16+48)*256 = 32768 B,
        # half the cap), i.e. 1.5-2x fewer A re-reads than bn32 and 2x fewer than
        # the bn96 tile -- without the per-warp A replication that makes t64 the
        # right thread count here (48/16 = 3 N-atoms, one warp per atom).
        # Paired same-process, graph replay, 7 rounds x 20 iters, idle card,
        # interleaved round-robin (all arms of a cell in one round), hipblas
        # re-measured as a duplicate control in every cell (spread <= 0.6%, so
        # no cell is dirty); ratio = ours/hipblas, shipped -> bn48, with the
        # shipped arm built at ITS OWN setprio=1 (a baseline at the builder
        # default of 0 understates the shipped tile by 7-11% and overstates this
        # change -- the first pass made exactly that mistake):
        #          M=1                M=8                M=16
        #   out_proj   0.9841->0.9153  0.9840->0.9202  1.0000->0.9251  (-7 to -8%)
        #   down27     1.0026->0.9304  1.0155->0.9304  1.0104->0.9377  (-7 to -9%)
        #   gate_up27  0.8812->0.8541  0.8794->0.8489  0.8910->0.8590  (-3 to -4%)
        #   qkv27      0.9650->0.9275  0.9627->0.9229  0.9698->0.9370  (-3 to -4%)
        #   in_proj_z  0.9669->0.9448  0.9560->0.9451  0.9777->0.9553  (-1 to -2%)
        #   in_proj_qkv 0.8881->0.9495 0.8161->0.8829  0.8283->0.8855  (+8%)
        # The gain is flat across that whole band, which is why this is gated on
        # shape and not on M.
        # in_proj_qkv is the one exclusion and the reason this is a shape list:
        # its shipped bn32/bk256/ns1 tile is the strongest cell in the entire
        # decode table (0.82x hipblas) and bn48 gives that back.  All arms
        # bit-exact against a chunked fp32 reference (rel max 3.1e-03, identical
        # to the shipped arm's own rel -- the two tiles agree to 0.0e+00).
        if (N, K) in _BN48_M16_NK:
            # setprio=1, like every other tile at this band: measured at both,
            # p1 wins gate_up27 (117.12 vs 123.04us), qkv27 (59.36 vs 62.08),
            # out_proj (27.84 vs 28.48) and down27 (57.92 vs 58.24) and loses
            # in_proj_z by 1% (33.44 vs 33.12), which is inside the noise this
            # card shows between rounds.  p1 also removes the special case.
            return _validate_cfg(DecodeCfg(16, 48, 256, 2, 64, 1), M, N, K)
    elif M <= 32:
        bm, sp = 32, 1
    elif M <= 64:
        # bn192 (2026-10-23, tuner MT128x192x64) where the grid can carry it --
        # see _BN192_NK for which shapes, and for why this is a list and not a
        # rule.  (64+192)*128*2 = 65536 B is exactly the LDS cap, so this is as
        # wide as bn gets at bm64 with ns2.
        #   gate_up27  M=48 0.9714 -> 0.9345   M=64 0.9934 -> 0.9548
        #   in_proj_qkv M=48 0.9664 -> 0.8753   M=64 1.0204 -> 0.9262  (-9-10%)
        # while N=14336 (0.89 -> 1.43) and N=6144 (0.96 -> 1.47) lose outright.
        # setprio=1 is the better of the two on all four of those cells.
        if (N, K) in _BN192_NK:
            return _validate_cfg(DecodeCfg(64, 192, 128, 2, 512, 1), M, N, K)
        # Out of scope, and a measured structural optimum: bk128 ns2 (double-buffered)
        # beats bk256 ns1 by 1.2% (243.5 vs 246.6us). The gain is cross-k-step overlap,
        # NOT wait granularity -- at fixed bk256, ns=2 is 1.6% SLOWER.
        return _validate_cfg(DecodeCfg(64, 64, 128, 2, 256, 1), M, N, K)
    elif N == 5120 and K % 128 == 0:
        # M in (64, 128] -- the band that the chooser used to serve with ONE tile
        # (bm128/bn128/bk128/ns1/t512, 40 blocks), which measured 1.83-2.29x
        # slower than hipblas.  Three things were wrong, all visible in hipblas's
        # own decode SASS (MT96x128x256: its N macro-tile is *96*, i.e. its grid
        # is ceildiv(5120,96) = 54 blocks, and one kernel serves M=96 and M=128):
        #   1. bn128 -> 40 blocks on 64 CUs: 24 CUs idle.  bn96 -> 54 blocks
        #      (+35% occupancy) for +17% activation bytes, and it wins even
        #      though it moves MORE bytes.
        #   2. ns=1 -> no cross-k-step overlap at all: bn96/ns1 1.85x vs
        #      bn96/ns2 1.37x (M=128 down, paired, same process).
        #   3. M=96 inherited bm128 and paid 32/128 padded MMA rows; bm96 is a
        #      legal MLS tile (6 atoms) and is 1.71x faster at the same block
        #      count.  The padded-M boundary is measured: M=65/80 -> bm96 wins
        #      (1.115/1.129), M=112 -> bm128 wins (1.337), so the switch is at 96.
        # Measured 2026-10-20, paired same-process, graph replay, 7 rounds,
        # idle card, min-of-rounds, ratio = ours/hipblas (M=128 / M=96):
        #   down 1.297 / 1.121   o 1.262 / 1.137
        # vs the old single tile's 2.287 / 1.991 (down) and 1.825 / 1.865 (o).
        # setprio=1 is +3.5% at M=128, +2.5% at M=96 (1.342 -> 1.297, 1.150 -> 1.121).
        # Correctness: rel < 1e-3 against the fp32 reference on o and down at
        # M = 65/80/96/112/128 (bn96/bk128/ns2 is exact at bm64/96/128).
        # M > 96 must move to bm=128: bm96 would need two block rows
        # (ceildiv(112,96)=2) and M=112 measures 109.1us at o vs 67.8 here.
        if M <= 96:
            return _validate_cfg(DecodeCfg(96, 96, 128, 2, 384, 1), M, N, K)
        # M == 97..128 takes bm128/bn96/bk128/ns2/t512/u8.  The 2026-10-20i
        # choice here was bn128/t256 on the byte law (A = (N/bn)*M*K + B = N*K:
        # 308MB at bn96 vs 262MB at bn128), which is the wrong objective -- see
        # the block-count argument below.  Closed at this tile: bk256/ns1
        # (369.8), bk64/ns3-4 (~470), FullColK/warp_k=4 (242.1), t256 at bn128
        # (223.8/79.7).  rel = 6.1e-8 (down) / 0 (o).
        # bn96 -- NOT bn128 -- once the k-loop is unrolled (2026-10-21, paired
        # same-process, 7 rounds, idle card, M in {112,128}, every round):
        #   down 183.2/182.6 vs 208.6/214.7us   o 67.8/67.8 vs 70.9/71.0us
        # i.e. 1.06/1.02x hipblas, against 1.21/1.20x.  This reverses the
        # 2026-10-20i bn128 choice, and the reason is NOT the byte law: with
        # bm=128 the per-step traffic is (N/bn)*bm*bk for A plus N*bk for B, so
        #   bn96  -> 54 blocks, 1.54MB/step, floor 0.89us/step, measured 1.02
        #   bn128 -> 40 blocks, 1.31MB/step, floor 0.76us/step, measured 1.11
        # bn96 moves 17% MORE bytes and still wins, because 54 of 64 CUs are
        # fed instead of 40: the exposed part of a step is the cp_async wait,
        # and it hides when more blocks are issuing.  bn64 (80 blocks, 1.26MB
        # of A alone) loses at 104us, so 54 blocks is a genuine optimum.
        # t512/U8 measured against t384 (illegal MLS tile at bm128), t1024
        # (88.5us), bm112 (90.6us) and U6/U16 (68.3/71.4us).
        return _validate_cfg(DecodeCfg(128, 96, 128, 2, 512, 1), M, N, K)
    else:
        # M in (64, 128] on the wide-N shapes that are neither o/down (handled
        # above) nor small-M: gate_up (N=51200, 400 blocks) and qkv (N=10240, 80
        # blocks).  Same tile either way; the difference is the pipeline depth.
        #
        # ns2 + the unrolled k-loop pay ONLY when there are enough blocks to
        # keep feeding the CUs (2026-10-21, paired same-process, graph replay,
        # 7 rounds, idle card, setprio=0 on both arms so ns is the only variable):
        #   gate_up M=128  311.8 -> 300.5us  (1.002x -> 0.966x hipblas)
        #   gate_up M= 96  305.4 -> 296.0us  (1.020x -> 0.989x)
        #   qkv     M=128   85.1 ->  90.9us  (1.114x -> 1.136x)   <- loses
        #   qkv     M= 96   87.4 ->  90.1us  (1.238x -> 1.277x)   <- loses
        # qkv at bn128 is 80 blocks on 64 CUs (1.25 waves): almost every CU runs
        # exactly one block, so a second pipeline stage has nothing to overlap
        # with and only costs LDS/registers.  gate_up's 400 blocks (6.25 waves)
        # always have a successor.  256 blocks = 4 waves is the measured split;
        # no point between 80 and 400 exists for these shapes.
        # NOT a wider-bn lever: bn160 is an illegal MLS tile ("No valid MLS tile
        # config for trans: block_mn=160") and bn256 forces bk64 or ns1, measured
        # 1.59x / 1.48x slower.
        #
        # qkv (80 blocks) instead buys its blocks by shrinking bm to 96, which
        # also removes MMA padding -- and that half of the effect is the larger
        # one (2026-10-21, paired, 7 rounds, setprio=0, hipblas calibrated to
        # within 1.35x of its quiet-card time before each run):
        #   M= 80  bm128/bn128/ns1 84.8 -> bm96/bn96/ns2/U8 75.5  (1.202 -> 1.070)
        #   M= 96  bm128/bn128/ns1 87.5 -> bm96/bn96/ns2/U8 73.6  (1.246 -> 1.048)
        # At M=96 bm128 rows are 25% padding; bn128 also gives only 80 blocks
        # (1.25 waves) against bn96's 107.  U8 beats U6 by 1.1% here and is
        # identical on o/down (58.56/58.56, 155.8/155.5), so the table uses 8.
        # Closed at this band: bn128/ns2 (1.261 at M=80), bm128/bn96 (1.248 at
        # M=96 -- bn is not the lever, bm is), setprio=1 (-2.2%).
        #
        # M in (96, 128] cannot use bm96 (two 96-row tiles = 192 rows for <=128,
        # 50% padding) and every wave-packing variant loses: bm64/bn128 1.467,
        # bm64/bn96 1.705, bn80/128 blocks 1.214, bn96 1.172, bn128/ns1 1.178.
        # (These are the bm96/bm128 *bk128* arms. qkv's remaining hole in this
        # band was the bk, not bm or bn -- see the bk64 hoist below, which takes
        # it to 1.02 at M=112 and 0.97 at M=128.)
        # Closed here 2026-10-21 (all paired, bit-exact, worse than the shipped
        # tile): bm96/bn96 ns1 76.48us/1.084 and bk256/ns1 108.00us/1.531 at
        # M=96; t256 at bm96/bn96 83.04/1.177; bk320 does not launch; at M=128
        # bn256/bk128/ns1 111.84/1.472 and ns2+U8 on bn128 91.20/1.200.
        # bn192 at bm96/bk64 (2026-10-23, tuner MT128x192x64) must be tested
        # BEFORE the block-count split below, because gate_up's 272 bn128-blocks
        # take the >=256 arm and the tile only pays there.  It does NOT pay on
        # in_proj_qkv (the other shape the bm64 arm takes): 69.60us against the
        # shipped 69.92us is inside noise, and setprio=1 costs 2.5% there, so
        # _BN192_BM96_NK is narrower than _BN192_NK.  bk64, not bk128: at bm96
        # bn192 needs (96+192)*128*2 = 73728 B > the 65536 B LDS cap, while
        # (96+192)*64*2 = 36864 B fits.
        # Paired same-process, graph replay, 7 rounds x 20 iters, idle card,
        # interleaved round-robin, duplicate hipblas control (spread <= 0.4%):
        #   gate_up27   M=96 208.16 -> 191.68us  0.9188 -> 0.8460  (-8.6%)
        #   in_proj_qkv M=96  69.92 ->  69.60us  0.9604 -> 0.9560  (noise, kept out)
        # setprio=0 -- the only tile in the table that asks for it at M<=96:
        # p1 measures 196.80us on gate_up27, 2.7% behind p0, and 71.68 vs 69.60
        # on in_proj_qkv.  Whatever setprio buys the other tiles, this one is
        # already A-re-read bound, not MMA-issue bound.
        # Bit-exact against both the chunked fp32 reference (rel 2.0e-03) and
        # the tile it replaces (relmax 0.0e+00, no NaN).  Measured at M=96 only;
        # the unmeasured 65..95 interior takes the same arm, as the bm96 split
        # below already does for its own band.
        if M <= 96 and (N, K) in _BN192_BM96_NK:
            return _validate_cfg(DecodeCfg(96, 192, 64, 2, 384, 0), M, N, K)
        if -(-N // 128) >= 256:
            # gate_up M in (64,96] moves to bm96 (2026-10-21, paired same-process,
            # graph replay, 7 rounds, idle card, min-of-rounds, U8 on every arm):
            # bm128 burns 32-48 of its 128 MMA rows as padding, and bm96 removes
            # that at ZERO byte cost -- A_issued = ceil(N/bn)*M*K has no bm in it.
            #   M=80  bm128/t512 293.1/293.3us   bm96/t256 288.0/288.2   bm96/t512 285.4/285.6
            #   M=96  bm128/t512 296.0/296.5us   bm96/t256 286.7/287.4   bm96/t512 297.6/298.4
            # ratio vs hipblas 0.955 -> 0.934 (M=80) and 0.990 -> 0.960 (M=96).
            # The thread optimum flips between the two ends (0.9% / 3.6%, each
            # reproduced twice), so the split is at 80: both ends take their
            # measured best and the unmeasured 81..95 interior takes the M=96 arm.
            # bn96 -- the qkv tile -- LOSES here (303.2/303.5, 1.016): 400+ blocks
            # are already enough, so halving bn only adds A bytes.  M > 96 cannot
            # use bm96 (two 96-row block rows = 50% padding).
            # The 20% of the gap that padding removal does NOT buy, and why bm96
            # is the end of this lever: gate_up M=64 stays on bm64/bn64 (1.059-1.070);
            # 11 arms measured there (bn 96/128, bm 96, t 128/512, bk 64/256, ns1,
            # setprio=0, U16) all lose -- see the 2026-10-21d section.
            if M <= 80:
                return _validate_cfg(DecodeCfg(96, 128, 128, 2, 512, 0), M, N, K)
            if M <= 96:
                return _validate_cfg(DecodeCfg(96, 128, 128, 2, 256, 0), M, N, K)
            return _validate_cfg(DecodeCfg(128, 128, 128, 2, 512, 0), M, N, K)
        if M <= 96:
            # bk64/ns2 (2026-10-21): halves the LDS per block, so 2 blocks/CU fit
            # and the second one hides the cp_async wait.  It needs the grid to
            # fill both slots -- more blocks than CUs -- which is the same
            # condition as "one wave at 2 blocks/CU" below 256 blocks:
            #   blocks <=  64  -> keep bk128 (z27: 64 blocks, 0.89-1.05x; bk64
            #                     1.20-1.28x, i.e. the second slot sits idle)
            #   blocks <= 128  -> bn96/bk64 (qkv32: 107 blocks, 0.9633x)
            #   else           -> bn128/bk64 (qkv27: 150 at bn96 = 2 waves, 1.44x;
            #                     112 at bn128 = 1 wave, 0.9392x)
            if K % 64 == 0 and -(-N // 96) > 64:
                if -(-N // 96) <= 128:
                    return _validate_cfg(DecodeCfg(96, 96, 64, 2, 384, 1), M, N, K)
                return _validate_cfg(DecodeCfg(96, 128, 64, 2, 512, 1), M, N, K)
            return _validate_cfg(DecodeCfg(96, 96, 128, 2, 384, 0), M, N, K)
        # What is LEFT in M in (112,128] after the bk64 hoist below: only shapes
        # whose bn128 grid is <= 64 blocks, i.e. ceil(N/128) <= 64 -> N <= 8192.
        # o/down (N=5120) never get here (they take the N==5120 branch), so in
        # practice this is z27 (N=6144, 48 blocks) alone.  For it, bn96's 72
        # blocks beat bn128's 48 (0.75 waves) once the block carries a full
        # bm=128 of MMA rows -- the o/down M>96 tile.  The switch is at 112, not
        # 96, because both tiles launch the same block count at any M<=128 -- only
        # the padding moves: at M=112, 12.5% of the tile's MMA rows are wasted, so
        # MMA per byte is low and the +17% A bytes of bn96 are not repaid
        # (2026-10-21, paired, 7 rounds, idle card, min-of-rounds, ours/hipblas):
        #   M=112  bn128/ns1 86.56/86.72  bn96/ns2 87.84   -> bn128 by 1.4%
        #   M=128  bn128/ns1 88.64/88.96  bn96/ns2 87.52/87.84 -> bn96 by 1.3%
        # (M=128 has zero padding, so the extra blocks do pay).  Both arms
        # bit-exact (rel=0).  qkv does NOT reach this branch any more: its bn128
        # grid is 80/112 blocks, so it takes the bk64 arm above and wins there
        # (M=128 0.9739, M=112 1.0203 vs the 1.221 this branch used to give it).
        # M in (96, 128] -- the whole band, not just M=128.  The bk64 arm below
        # used to be gated on `M > 112`, a boundary tuned when bk128 was the only
        # bk in the table; the arm was then only ever measured above it.  It is
        # legal over the whole band, and at M=112 the old fall-through
        # (bn128/bk128/ns1, i.e. no pipeline stage and 1 block/CU) is a 16.5%
        # hole: qkv32 M=112, paired same-process, graph replay, 7 rounds,
        # min-of-rounds, all arms bit-exact (rel == 0.0), same hipblas 65.2us:
        #   bk128/ns1 (was shipped here) 79.68us 1.221x
        #   bk64/ns2  (this arm)         66.52us 1.020x   <- 0.835x of the old arm
        #   bn96/bk128/ns2              78.67us 1.206x
        #   bn128/bk128/ns2             82.74us 1.266x
        # For reference at the two ends of the band, same process: M=96 bk64/bn128
        # 65.78 vs the bm96/bn96/bk64 arm's 61.30 (the bm96 tile wins there, which
        # is why M<=96 keeps its own branch), and M=128 bk64 67.14 vs 67.94 for the
        # identical config through the old path (same tile, so this is noise).
        # z27 (48 bn128-blocks) still falls through to bn96/bk128 below.
        if K % 64 == 0 and -(-N // 128) > 64:
            return _validate_cfg(DecodeCfg(128, 128, 64, 2, 512, 1), M, N, K)
        if M > 112:
            # Same bk64 trade as the M<=96 branch, one level up: at bm=128 a
            # bn96 tile is 107 (qkv32) / 150 (qkv27) blocks and bn128 is
            # 80 / 112.  Measured at M=128 (paired, 7 rounds, p1/u8 on the new
            # tile): qkv32 bn128 0.9807 (shipped bn96/bk128 1.1268), qkv27
            # bn128 1.1021 (shipped 1.5782, i.e. -30%).  bm128/bn96/bk64 is
            # qkv27's 2-wave trap (1.66x) and qkv32's second best (1.0192);
            # bn256/bk64 ties bn128 on qkv27 (1.1053) but fails LDS at bk128.
            # The test is on the bn128 block count -- the tile the new arm
            # actually launches -- not on bn96's: qkv32 = 80 (>64, wins),
            # qkv27 = 112 (>64, wins), while z27 (48), o (40) and down (40)
            # must keep bn96/bk128 (z27 M=128 0.8915 with it vs 1.12-1.20 on
            # every bk64 arm; o32 M=128 1.0410 vs 1.32-1.41).
            return _validate_cfg(DecodeCfg(128, 96, 128, 2, 512, 1), M, N, K)
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
        # 2026-10-17g section of docs/optimization/fp8_decode_gemm.md.
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
        # The same wide tile for N in (5120, 10240) -- measured on the only real
        # weight in that window, 27B's in_proj_z (N=6144, K=5120).  This used to
        # fall through to bn16 below and was the worst cell in the whole decode
        # table: paired vs hipblas 1.0678 (M=1), 1.2755 (M=8), 1.981 (M=24),
        # 1.816 (M=32).  bn16 = 384 blocks each re-reading the whole A tile
        # (A_issued = 384*32*5120 = 62.9 MB + 31.5 MB of B ~ 1970 GB/s, i.e. above
        # the 1725 GB/s ceiling), which is the entire loss; hipblas's bn96 grid
        # issues 10.5 MB of A.  Measured 2026-10-22, paired same-process,
        # graph replay, 7 rounds, bitexact 0.0e+00 vs the fp32 reference:
        #   M=24  bn16/t256 49.10 (1.818)  bn96/t192 23.42 (0.867)  [-52%]
        #   M=32  bn16/t256 48.76 (1.827)  bn96/t192 24.86 (0.931)  [-49%]
        # t128, not t192: at N=6144 the grid is exactly 64 blocks, so 2 warps is
        # 128 warps on 64 CUs; at N=5120 (54 blocks) t192 still wins, so this is
        # a property of the 64-block grid, carried as the measured point:
        #   M=16  t192 22.17  t128 20.12 (-9%)     M=32  t192 24.63  t128 23.78
        #   control N=5120: M=16 t192 20.19 vs t128 20.86; M=32 23.57 vs 26.76
        # (t64 loses at both: 24.32 / 32.22 -- too few warps to hide latency.)
        # k_unroll stays 1 at bm16 (u8 is +1.5%: 20.63 vs 20.12) and is 8 at bm32
        # (22.79 vs 23.61) -- see _UNROLL_TILES.
        if N < 10240 and K % 256 == 0:
            return _validate_cfg(DecodeCfg(bm, 96, 256, 2, 128, sp), M, N, K)
        bn = 16
        # threads must scale with block_M: at bn16 a bm32 tile has two 16x16 M-atoms,
        # and giving them ONE warp (t=64) makes the block MMA-issue-bound instead of
        # B-stream-bound. Measured on o/down at M=32 (paired, 7 rounds):
        #   o   : t64 75.2us (1.911x)  t128 72.3 (1.837x)  t256 46.9 (1.191x)
        #   down: t64 209.9us (1.944x) t128 203.0 (1.880x) t256 122.4 (1.133x)
        # t=256 is a 1.60x/1.71x win over the t=64 that the old M-only table gave.
        threads = 64 if bm <= 16 else 256
    return _validate_cfg(DecodeCfg(bm, bn, 256, 1, threads, sp), M, N, K)


# (block_M, block_N, block_K, num_stages, threads) -> measured ``k_unroll``.
# Only tiles that were measured on the N=5120 decode shapes are listed; anything
# else keeps the T.Pipelined loop.  See the ``k_unroll`` docstring for numbers.
_UNROLL_TILES = {
    (16, 96, 256, 2, 192): 8,      # o/down, M <= 16
    (32, 96, 256, 2, 192): 8,      # o/down, M in (16, 32]
    # 27B in_proj_z (N=6144) M in (16,32].  Same tile as above but t128; u8 wins
    # there (23.61 -> 22.79 us) while at bm16/t128 u8 LOSES (20.12 -> 20.63), so
    # only the bm32 entry is listed.
    (32, 96, 256, 2, 128): 8,
    (64, 64, 128, 2, 256): 8,      # o/down, M in (32, 64]
    (96, 96, 128, 2, 384): 8,      # o/down + qkv, M in (64, 96]
    # only o32/down32 ever reach this tile (qkv/gu bm128 are bn128).  u12 beats
    # u8 in 4/4 paired runs on both shapes: o 60.93/60.96 -> 60.63, down
    # 176.10/176.08 -> 174.80us (ratio 1.006 -> 1.000 on both).  u10 is between
    # the two, u16 is 7% worse.  Re-swept u8/u10/u12/u16 at every other shipped
    # tile: all within +-0.5% (noise), so 8 stands there.
    (128, 96, 128, 2, 512): 12,    # o/down M in (96,128]
    (128, 128, 128, 2, 512): 8,    # gate_up, M in (96, 128]
    (96, 128, 128, 2, 512): 8,     # gate_up, M in (64, 80]
    (96, 128, 128, 2, 256): 8,     # gate_up, M in (80, 96]
    (96, 96, 64, 2, 384): 8,       # qkv32, M in (64, 96]  (bn96 <= 128 blocks)
    (96, 128, 64, 2, 512): 8,      # qkv27, M in (64, 96]  (bn128, 1 wave)
    (128, 128, 64, 2, 512): 8,     # qkv,   M in (96, 128] (bk64: 2 blocks/CU)
    # 2026-10-23 vendor-derived tiles.  These three were SELECTED at k_unroll=8
    # (the confirm run built every ns2 arm with U8, so the shipped table's tile
    # is exactly the arm that was measured); the entry exists so the chooser
    # reproduces that arm rather than silently falling back to the pipelined
    # loop at u1, which was never measured on any of them.
    (16, 48, 256, 2, 64): 8,       # M <= 16, bn48 shapes
    (64, 192, 128, 2, 512): 8,     # M in (32, 64], bn192 shapes
    (96, 192, 64, 2, 384): 8,      # M in (64, 96], bn192 shapes
    # Measured and NOT taken (2026-10-21, paired): at M=128 bn128/40 blocks
    # 71.0/214.7us, bn64/80 blocks 104.0us, bn80 -- illegal MLS tile, t1024
    # 88.5us, bm112 90.6us (2 blocks/row group instead of a clean warp split);
    # U6 68.3/183.7, U8 67.8/182.6, U16 71.4/183.4.  At M=64 the pre-unroll
    # neighbourhood stays closed: bn96/t384 54.6us and bn128/t512 60.2us against
    # bn64's 49.6, bm96 58.6 (bm96 pads 50% of the MMA rows there), U4 50.2,
    # U16 50.9.
}


def _cp_async_wait(n):
    """``tl::cp_async_wait<n>`` -- wait until at most n cp_async groups are in
    flight.  There is no Python wrapper for this on HCU; each ``T.copy`` with
    ``prefer_instruction="matrix_load"`` commits exactly one group."""
    T.evaluate(T.call_extern("handle", "tl::cp_async_wait<%d>" % n))


def fp8_channelwise_gemm(
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
    k_unroll="auto",
    out_dtype=None,
    per_tensor_a=False,
):
    """Channelwise fp8 GEMM using MLS (matrix-LDS-read) A/B producers.

    ``out_dtype`` sets the dtype of the ``C`` tensor only (the accumulator and
    the scale epilogue stay ``accum_dtype``); ``None`` keeps ``C`` in
    ``accum_dtype``.  Passing ``"bfloat16"`` fuses the fp32->bf16 cast into the
    copy-out, which is what a ``torch._scaled_mm(out_dtype=bf16)`` drop-in needs:
    without it the caller pays a second pass over ``block_M x N`` fp32
    (~11 us on gate_up at M=64).

    ``block_K`` must be a multiple of 64 (the trans-B fp8 MLS atom width).

    ``k_unroll`` replaces ``T.Pipelined`` with an explicit k-loop that barriers
    by hand and is unrolled ``k_unroll`` times per iteration (``"auto"`` consults
    ``_UNROLL_TILES``).  Measured on the N=5120 decode tiles, paired same-process,
    graph replay, 7 rounds, idle card (2026-10-21):

      bm64/bn64/bk128/ns2/t256,  M=64  down 160.3 -> 127.0 us   o 57.0 -> 49.6
      bm96/bn96/bk128/ns2/t384,  M=96  down 195.8 -> 155.2 us   o 71.4 -> 58.7
      bm128/bn96/bk128/ns2/t512, M=128 down 214.7 -> 182.6 us   o 71.0 -> 67.8

    i.e. 0.79-0.88x of the pipelined loop, and against ``torch._scaled_mm``:
    down 0.87-1.08x, o 0.94-1.05x over all of M <= 128 (it used to be 1.03-1.23x).
    The M>96 tile shows the unroll also changes *which* tile wins -- there bn96
    beats bn128 by 15% once unrolled, though it moves 17% more bytes.  M=64 does
    not move (bm96 pads 50% of its MMA rows).
    Two effects, both from the unroll: (a) every shared-stage index becomes a
    compile-time constant -- a symbolic ``(k+d) % 3`` costs 13% on its own,
    because it lowers to v_mul_lo_u16/v_sub_u16/v_readfirstlane_b32 sitting on
    the matrix_load address path; (b) the scheduler can hoist the next step's
    load address above the MMA of the current step.  Per-step LDS bytes, stage
    count and barrier count are unchanged, and deeper prefetch was measured to
    *hurt* (the cp_async counter is FIFO, so covering step k's A also covers
    B(k+1) when B leads -- the DRAM stream is never more than one step ahead of
    the wait).  Requires ns=2 and the Square policy; a non-unrolled caller is
    byte-identical to the previous builder.

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

    ``per_tensor_a=True`` declares ``scale_a`` as one element instead of
    ``(block_M,)`` and multiplies by ``scale_a[0]``.  It exists for the M == 1
    decode step, where vLLM's ``scale_a`` is ``[1, 1]``: the caller would
    otherwise have to materialise a band-length buffer, and on this card *each*
    DtoD copy node costs ~5 us of GPU time regardless of size (profiled: a
    4-byte ``copy_`` is 5.0 us, i.e. as much as a quarter of the M=1 kernel).
    20 cycles of the epilogue become one broadcast read; the loop, the LDS
    accounting and the copy-out are unchanged.
    """
    c_dtype = out_dtype or accum_dtype
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
    # bm96/bn128/bk128/ns2 is the gate_up M in (64,96] tile (2026-10-21).  Checked
    # against the fp32 reference at M = 65/72/80/88/96 with t512 and t256 (10
    # kernels, every one rel = 0): the stage stride depends on (bm,bn,bk) and the
    # A copy extent on M, so both were swept before this tile was admitted.
    _wide_exact = (block_N == 96 and block_K == 256 and num_stages <= 2) or (
        block_N == 96 and block_K == 128 and block_M >= 64 and num_stages <= 2) or (
        block_M == 128 and block_N == 128 and block_K == 128 and num_stages <= 2) or (
        block_M == 96 and block_N == 128 and block_K == 128 and num_stages <= 2) or (
        (block_M, block_N, block_K) in _NS2_BK64_TILES and num_stages <= 2) or (
        (block_M, block_N, block_K) in _NS2_VENDOR_TILES and num_stages <= 2)
    # _NS2_VENDOR_TILES (2026-10-23): the three tiles taken from the vendor
    # tuner's own picks -- bn48/bk256 at bm16, bn192/bk128 at bm64 and
    # bn192/bk64 at bm96.  Each was built through this function and checked
    # against the chunked fp32 reference on every decode shape it is selected
    # for before it was written into the table (rel < 1e-3; the bm64 bn192 and
    # bm96 bn192 runs are now pinned by the suite as well).
    assert num_stages <= 1 or (block_M, block_N) == (64, 64) or _wide_exact or _ns_escape, (
        f"num_stages={num_stages} is only tuned for block_M=block_N=64 on HCU "
        f"(got block_M={block_M}, block_N={block_N}); multi-stage small tiles are correct "
        f"but unbenchmarked. Use num_stages<=1, or benchmark the tile first.")
    gemm_annotations = {"tl.hcu_mmac_setprio": int(setprio)} if setprio else None
    if policy is None:
        policy = T.GemmWarpPolicy.Square
    if k_unroll == "auto":
        k_unroll = _UNROLL_TILES.get((block_M, block_N, block_K, num_stages, threads), 1)
        if k_unroll > 1 and K // block_K < k_unroll + 4:
            # short K: the measured tiles assume the decode shapes' step counts,
            # so fall back to the pipelined loop instead of failing.
            k_unroll = 1
    unroll = int(k_unroll)
    if unroll > 1:
        assert num_stages == 2, (
            f"k_unroll={unroll} is only implemented at num_stages=2 (got {num_stages})")
        assert policy == T.GemmWarpPolicy.Square, (
            f"k_unroll={unroll} is only implemented for Square (got {policy})")
        nstep = K // block_K
        nmain = nstep - 1            # A is prefetched one step ahead
        L = (nmain // unroll) * unroll
        rem = nmain - L
        assert nstep >= unroll + 4, (
            f"k_unroll={unroll} needs more k-steps than that (nstep={nstep})")

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float8_e4m3fn"),
        B: T.Tensor((N, K), "float8_e4m3fn"),
        scale_a: T.Tensor((1 if per_tensor_a else M,), "float32"),
        scale_b: T.Tensor((N,), "float32"),
        C: T.Tensor((M, N), c_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            C_l = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_l)
            if unroll > 1:
                # Explicit double-buffered k-loop.  Structure per step:
                #   [sync][A copy][B copy][wait(2)][sync][gemm k]
                # Each T.copy commits its own cp_async group, so one step is two
                # groups and wait(2) leaves exactly the current step's pair in
                # flight.  The top sync is the WAR barrier (stage k+1 was last
                # read at step k-1), the second is the RAW one.
                A_s = T.alloc_shared((2, block_M, block_K), "float8_e4m3fn")
                B_s = T.alloc_shared((2, block_N, block_K), "float8_e4m3fn")
                T.copy(A[by * block_M:by * block_M + block_M, 0:block_K], A_s[0, :, :],
                       prefer_instruction="matrix_load")
                T.copy(B[bx * block_N:bx * block_N + block_N, 0:block_K], B_s[0, :, :],
                       prefer_instruction="matrix_load")
                for ko in T.serial(L // unroll):
                    for u in T.unroll(unroll):
                        T.sync_threads()
                        T.copy(A[by * block_M:by * block_M + block_M,
                                 (ko * unroll + u + 1) * block_K:
                                 (ko * unroll + u + 2) * block_K],
                               A_s[(ko * unroll + u + 1) % 2, :, :],
                               prefer_instruction="matrix_load")
                        T.copy(B[bx * block_N:bx * block_N + block_N,
                                 (ko * unroll + u + 1) * block_K:
                                 (ko * unroll + u + 2) * block_K],
                               B_s[(ko * unroll + u + 1) % 2, :, :],
                               prefer_instruction="matrix_load")
                        _cp_async_wait(2)
                        T.sync_threads()
                        T.gemm(A_s[(ko * unroll + u) % 2, :, :],
                               B_s[(ko * unroll + u) % 2, :, :], C_l, False, True,
                               k_pack=1, annotations=gemm_annotations, policy=policy)
                # leftover of the unrolled loop: same body, one step at a time
                for k in T.serial(rem):
                    T.sync_threads()
                    T.copy(A[by * block_M:by * block_M + block_M,
                             (k + L + 1) * block_K:(k + L + 2) * block_K],
                           A_s[(k + L + 1) % 2, :, :], prefer_instruction="matrix_load")
                    T.copy(B[bx * block_N:bx * block_N + block_N,
                             (k + L + 1) * block_K:(k + L + 2) * block_K],
                           B_s[(k + L + 1) % 2, :, :], prefer_instruction="matrix_load")
                    _cp_async_wait(2)
                    T.sync_threads()
                    T.gemm(A_s[(k + L) % 2, :, :], B_s[(k + L) % 2, :, :], C_l, False,
                           True, k_pack=1, annotations=gemm_annotations, policy=policy)
                for k in T.serial(K // block_K - nmain):
                    # drain: reload this step's own tiles (nothing reads them
                    # before the wait), so the tail needs no prefetch bookkeeping
                    T.copy(A[by * block_M:by * block_M + block_M,
                             (nmain + k) * block_K:(nmain + k + 1) * block_K],
                           A_s[(nmain + k) % 2, :, :], prefer_instruction="matrix_load")
                    T.copy(B[bx * block_N:bx * block_N + block_N,
                             (nmain + k) * block_K:(nmain + k + 1) * block_K],
                           B_s[(nmain + k) % 2, :, :], prefer_instruction="matrix_load")
                    _cp_async_wait(0)
                    T.sync_threads()
                    T.gemm(A_s[(nmain + k) % 2, :, :], B_s[(nmain + k) % 2, :, :], C_l,
                           False, True, k_pack=1, annotations=gemm_annotations,
                           policy=policy)
            else:
                A_s = T.alloc_shared((block_M, block_K), "float8_e4m3fn")
                B_s = T.alloc_shared((block_N, block_K), "float8_e4m3fn")
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
                C_l[i, j] = (C_l[i, j] * (scale_a[0] if per_tensor_a
                                          else scale_a[by * block_M + i])
                             * scale_b[bx * block_N + j])
            T.copy(C_l, C[by * block_M, bx * block_N])

    # The unrolled loop inserts its barriers explicitly, so the automatic
    # storage-sync pass must not add more (it would duplicate the RAW barrier).
    return tilelang.compile(
        gemm, out_idx=[4],
        **({"pass_configs": {"tl.disable_thread_storage_sync": True}} if unroll > 1 else {}))


def _ref(A, B, scale_a, scale_b):
    return (A.float() @ B.float().T) * scale_a[:, None] * scale_b[None, :]


def _validate_and_bench(size=8192):
    from tilelang.profiler import do_bench_cudagraph

    torch.manual_seed(0)
    A = torch.randn(size, size, device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn(size, size, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.rand(size, device="cuda") + 0.5
    sb = torch.rand(size, device="cuda") + 0.5

    kern = fp8_channelwise_gemm(size, size, size)  # 256x256x64 ns1 t512
    C = kern(A, B, sa, sb)
    ref = _ref(A, B, sa, sb)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    lat = do_bench_cudagraph(lambda: kern(A, B, sa, sb))
    tf = 2.0 * size * size * size / lat * 1e-9
    print(f"fp8_channelwise_gemm {size}^3 (256x256x64 ns1 t512): "
          f"{tf:.1f} TFLOPS, rel={rel:.6f}, {lat * 1e3:.1f} us")
    assert rel < 1e-3, f"rel error {rel} too large"
    return tf, rel


if __name__ == "__main__":
    _validate_and_bench()
