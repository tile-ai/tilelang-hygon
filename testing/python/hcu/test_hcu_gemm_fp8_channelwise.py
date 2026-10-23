# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Correctness test for the fp8 channelwise GEMM on the MLS (matrix-LDS-read)
producer path.

Kernel: ``tilelang/hcu/op/gemm/gemm_fp8_channelwise.py``. Semantics:
``C[m, n] = scale_a[m] * scale_b[n] * sum_k A[m, k] * B[n, k]`` with A (M, K)
fp8, B (N, K) fp8 (N-major), fp32 per-token / per-channel scales. Validated
configuration on HCU (gfx938): block 256x256x64, num_stages=1, 512 threads.
"""

import os
import sys

import pytest
import torch
from hcu_test_utils import current_hcu_arch_string, target_supports_fp8_mmac, target_supports_mls

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tilelang.hcu.op.gemm.gemm_fp8_channelwise import (  # noqa: E402
    DecodeCfg,
    _validate_cfg,
    decode_cfg,
    fp8_channelwise_gemm,
)

# The four qwen3-32B decode shapes, as (N, K).
_DECODE_SHAPES = [(51200, 5120), (10240, 5120), (5120, 8192), (5120, 25600)]


pytestmark = [
    pytest.mark.skipif(
        current_hcu_arch_string() != "gfx938",
        reason="fp8 MLS channelwise GEMM is validated on gfx938 only",
    ),
    pytest.mark.skipif(
        not target_supports_fp8_mmac(),
        reason="FP8 MMAC not supported on this target",
    ),
    pytest.mark.skipif(
        not target_supports_mls(),
        reason="MLS (matrix_load) not supported on this target",
    ),
]


def _chunk_rows(N, K, bytes_per_elem=4, budget=64 << 20):
    """Rows of N that fit in `budget` bytes at `bytes_per_elem` per element."""
    return max(1, budget // max(1, bytes_per_elem * K))


def _ref_program(A, B, scale_a, scale_b):
    # Chunked over N: B.float() alone is 1.05 GiB at (51200, 5120) and this card
    # gives one process ~1.5 GiB, so the unchunked form OOMs before the kernel
    # under test is even reached.  The k-reduction is not chunked and the two
    # scale multiplications are applied in the original order, so every element
    # is bit-identical to the unchunked expression.
    N, K = B.shape
    step = _chunk_rows(N, K)
    if step >= N:
        return (A.float() @ B.float().T) * scale_a[:, None] * scale_b[None, :]
    Af = A.float()
    out = torch.empty((A.shape[0], N), dtype=torch.float32, device=A.device)
    for i in range(0, N, step):
        j = min(i + step, N)
        out[:, i:j] = ((Af @ B[i:j].float().T) * scale_a[:, None]
                       * scale_b[i:j][None, :])
    return out


def _make_inputs(M, N, K, seed=0):
    torch.manual_seed(seed)
    A = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    # Same reason as _ref_program: torch.randn(N, K) is a 1.0 GiB fp32 temporary
    # at gate_up's shape.  Fill B in row chunks and cast each one in place.
    B = torch.empty((N, K), dtype=torch.float8_e4m3fn, device="cuda")
    step = _chunk_rows(N, K)
    for i in range(0, N, step):
        j = min(i + step, N)
        B[i:j] = torch.randn(j - i, K, device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.rand(M, device="cuda", dtype=torch.float32) + 0.5
    scale_b = torch.rand(N, device="cuda", dtype=torch.float32) + 0.5
    return A, B, scale_a, scale_b


def test_hcu_gemm_fp8_channelwise_validated_config():
    M, N, K = 8192, 8192, 8192
    block_M, block_N, block_K, num_stages, threads = 256, 256, 64, 1, 512
    A, B, scale_a, scale_b = _make_inputs(M, N, K)

    kernel = fp8_channelwise_gemm(
        M, N, K, block_M, block_N, block_K, num_stages=num_stages, threads=threads
    )

    # The MLS producer path must actually lower to the hardware matrix LDS read.
    src = kernel.get_kernel_source()
    assert "ds_read_format_tensor_a" in src or "ds_read_format_tensor_b" in src, (
        "fp8 MLS channelwise GEMM did not use the ds_read_format matrix LDS read"
    )

    C = kernel(A, B, scale_a, scale_b)
    ref = _ref_program(A, B, scale_a, scale_b)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 0.01, f"fp8 MLS channelwise GEMM rel error {rel:.6f} >= 0.01"
    del C, ref

    # The profiler harness is exercised on a small shape on purpose: its
    # assert_allclose materialises its own A/B/out plus a reference plus the
    # isclose temporaries, ~1.6 GiB at 8192^2, which does not fit in this box's
    # ~1.5 GiB per-process quota.  Correctness at 8192^2 is the `rel` check
    # above; this one is about the jit/profiler plumbing, which is shape-blind.
    del A, B, scale_a, scale_b
    torch.cuda.empty_cache()
    small = fp8_channelwise_gemm(2048, 2048, 2048, 128, 128, 64, num_stages=1, threads=256)
    small.get_profiler().assert_allclose(_ref_program, atol=1e-2, rtol=1e-2)


def test_hcu_gemm_fp8_channelwise_small():
    # Smaller shape, ns=0 (explicit wait path), to exercise the non-pipelined branch.
    M, N, K = 1024, 1024, 1024
    A, B, scale_a, scale_b = _make_inputs(M, N, K)
    kernel = fp8_channelwise_gemm(M, N, K, 128, 128, 64, num_stages=0, threads=256)
    C = kernel(A, B, scale_a, scale_b)
    ref = _ref_program(A, B, scale_a, scale_b)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 0.01, f"small fp8 MLS channelwise GEMM rel error {rel:.6f} >= 0.01"


def test_decode_cfg_is_shape_aware():
    """The chooser used to take only M, which is why o/down lost at every M.

    These assertions pin the measured N-rule and, more importantly, pin the
    invariants -- every one of which has at some point been a silent wrong answer
    rather than an error.
    """
    # N >= 10240 keeps bn32 (gate_up/qkv); N == 5120 now takes the FAT tile.
    # Measured 2026-10-20: bn16 at N=5120 means 320 blocks each re-reading the
    # whole A tile (320 redundant reads of A's L2 traffic), whereas hipblas's
    # M=16 kernel covers 96 N-columns per workgroup and re-reads A only 54
    # times.  bn96/bk256/ns2 is the widest tile whose two stages fit one CU
    # (2*(16+96)*256 = 57344 <= 65536), and it is worth 3-8% at M=1/16/32:
    #   wide/ship  down 0.923/0.932/0.952   o 0.952/0.950/0.969
    wide = decode_cfg(1, 51200, 5120)
    assert (wide.block_N, wide.threads) == (32, 256)
    assert (decode_cfg(1, 10240, 5120).block_N, decode_cfg(1, 10240, 5120).threads) == (32, 256)
    narrow = decode_cfg(1, 5120, 25600)
    assert (narrow.block_N, narrow.threads, narrow.num_stages) == (96, 192, 2)
    assert decode_cfg(1, 5120, 8192).block_N == 96

    # N in (5120, 10240) used to fall through to the bn=16 fallback.  Only 27B's
    # in_proj_z (6144x5120) lands there and it was the worst cell in the table:
    # 384 blocks each re-reading the whole A tile (A_issued = 384*32*5120 = 62.9
    # MB + 31.5 MB of B ~ 1970 GB/s, above the 1725 GB/s read ceiling).  It now
    # takes the same fat tile as N=5120 but at t128 -- the grid is exactly 64
    # blocks, so two warps is 128 warps on 64 CUs; the N=5120 control still
    # prefers t192 in the same process, so this is a property of the 64-block
    # grid, not a general rule.  Measured through this chooser, every arm exact
    # (rel == 0), paired vs hipblas: M=1 0.813, 8 0.823, 16 0.834, 24 0.833,
    # 32 0.879 (was 1.068 / 1.276 / 1.981 / 1.816 at bn16).  Since 2026-10-23
    # M <= 16 leaves this arm for bn48 (see
    # test_decode_cfg_vendor_tiles_are_shape_gated), so what is pinned here is
    # the band it still owns.
    for M in (17, 24, 32):
        z = decode_cfg(M, 6144, 5120)
        assert (z.block_N, z.block_K, z.num_stages, z.threads) == (96, 256, 2, 128), (M, z)

    # The fat tile's thread count is pinned by the N-atom split, so unlike the
    # old bn16 tile it does NOT scale with block_M -- bm32 keeps t192.  Three
    # wavefronts (not six) because the MLS A producer replicates A in every
    # warp: at fixed bn96/bk256/ns2, paired, t384 1.000 -> t192 0.952-0.962 ->
    # t128 0.971-0.983 -> t64 1.23-1.29.  Legal warp counts at bn96 are
    # {1,2,3,6} (96/n must stay a multiple of 16), and 3 is the measured optimum.
    for M in (1, 16, 32):
        for K in (8192, 25600):
            assert decode_cfg(M, 5120, K).threads == 192, (M, K)

    # Every returned config must be legal for its own shape.
    for N, K in _DECODE_SHAPES:
        for M in (1, 2, 4, 8, 16, 32):
            _validate_cfg(decode_cfg(M, N, K), M, N, K)

    # M=64 is out of scope and a measured structural optimum; split-K stays off.
    for N, K in _DECODE_SHAPES:
        assert decode_cfg(64, N, K).split_k == 1

    # M in (96,128] at N=5120 takes the 128x128 tile (2026-10-20i): -2.6% vs the
    # bm128/bn96/bk128/ns2/t512 it replaced (down 223.7 vs 229.9us, o 79.4 vs
    # 81.3us, paired, every round) because with bm=M the read traffic is
    # (N/bn)*M*K + N*K, i.e. 308MB at bn96 vs 262MB at bn128, and
    # (128+128)*128*2 = 65536 B is exactly the per-CU LDS cap.
    # 2026-10-21: with the unrolled k-loop the optimum moves to bn96/t512/u8 --
    # down 182.6 vs 214.7us, o 67.8 vs 71.0us (paired, 7 rounds, every round),
    # i.e. 1.02x hipblas against 1.20x.  bn96 moves 17% MORE bytes: 54 blocks
    # of 64 CUs issuing beats 40 blocks, and the exposed part of a step is the
    # cp_async wait.  bn64 (80 blocks) is 104us, so 54 really is the optimum.
    for K in (8192, 25600):
        m128 = decode_cfg(128, 5120, K)
        assert (m128.block_M, m128.block_N, m128.block_K, m128.num_stages,
                m128.threads) == (128, 96, 128, 2, 512)
        _validate_cfg(m128, 128, 5120, K)
    # M=96 keeps the 96x96 tile: bn128 needs a bm that is a legal MLS shape there
    # and loses (1.287x vs 1.135x hipblas, paired).
    assert decode_cfg(96, 5120, 25600).block_N == 96

    # M in (64,128] on the wide-N shapes (2026-10-21): block count decides the
    # pipeline depth, MMA padding decides block_M.  gate_up (N=51200) has 400
    # blocks at bn128 and takes ns2+unroll at every M>64 (0.934/0.960/0.966 vs
    # hipblas at M=80/96/128); qkv (N=10240) has only 80 blocks at bn128 (1.25
    # waves), so ns2 loses there (1.261/1.277) and instead it shrinks to
    # bm96/bn96 while M <= 96 fits in one tile row: 1.202 -> 1.070 at M=80 and
    # 1.246 -> 1.048 at M=96 (bn is NOT the lever there -- bm128/bn96 is 1.248).
    # 2026-10-21e: bk64/ns2 splits the step in half, so the per-block LDS halves
    # and a second block becomes resident (2 blocks/CU), which is what hides the
    # cp_async wait.  It needs the grid to fill both slots -- here 107 blocks at
    # bn96 -- and is worth qkv32 M=96 1.0467 -> 0.9633 vs hipblas (paired, every
    # arm bit-exact).  z27 (48 blocks) and o/down (40) stay on bk128: bk64 there
    # measures 1.20-1.41x because the second slot sits idle.
    for M in (65, 80, 96):
        q = decode_cfg(M, 10240, 5120)
        assert (q.block_M, q.block_N, q.block_K, q.num_stages, q.threads,
                q.setprio) == (96, 96, 64, 2, 384, 1), (M, q)
    # gate_up M in (64,96] takes bm96/bn128: bm128 spent 25-37% of its MMA rows on
    # padding, and bm96 removes it at zero byte cost (A_issued has no bm in it).
    # Paired 2x each: M=80 293.1/293.3 (bm128/t512) vs 285.4/285.6 (bm96/t512);
    # M=96 296.0/296.5 vs 286.7/287.4 (bm96/t256).  The thread optimum flips at
    # the band ends, so the split is at 80.  bn96 loses here (303.2, 1.016).
    # M > 96 cannot use bm96 (two 96-row block rows = 50% padding).
    for M, threads in ((65, 512), (80, 512), (96, 256)):
        g = decode_cfg(M, 51200, 5120)
        assert (g.block_M, g.block_N, g.block_K, g.num_stages, g.threads) == \
            (96, 128, 128, 2, threads), (M, g)
    # The M in (96,128] band takes bk64 across the WHOLE band (2026-10-22), not
    # just above M=112.  The old `M > 112` gate sent M=97..112 to bn128/bk128/ns1
    # -- no pipeline stage, 1 block/CU -- and was never measured there because
    # bk128 was the only bk in the table when the boundary was drawn.  At M=112
    # that is a 16.5% hole (qkv32, paired same-process, 7 rounds, min-of-rounds,
    # every arm bit-exact at rel=0.0, same hipblas 65.2us):
    #   bk128/ns1 (was shipped) 79.68us 1.221x | bk64/ns2 66.52 1.020x
    #   bn96/bk128/ns2 78.67 1.206x            | bn128/bk128/ns2 82.74 1.266x
    # Re-checked through this chooser at both a boundary and an interior point:
    # M=104 66.03us 1.014, M=112 66.67 1.020 (qkv32); M=104 74.85 0.998,
    # M=112 77.34 1.034 (qkv27); M=128 0.974/0.963.  M<=96 keeps bm96 (its own
    # branch, where bm96/bn96/bk64 is 61.30 against this tile's 65.78).
    for M in (104, 112):
        q = decode_cfg(M, 10240, 5120)
        assert (q.block_M, q.block_N, q.block_K, q.num_stages, q.threads,
                q.setprio) == (128, 128, 64, 2, 512, 1), (M, q)
    # ...while z27 (N=6144, 48 bn128-blocks) must still fall through: bk64 there
    # is 1.20-1.28x because the second LDS slot sits idle.  Below the 112
    # boundary it keeps bn128/bk128/ns1, above it the 72-block bn96 tile.
    for M in (104, 112):
        z = decode_cfg(M, 6144, 5120)
        assert (z.block_M, z.block_N, z.block_K, z.num_stages) == \
            (128, 128, 128, 1), (M, z)
    for M in (113, 128):
        z = decode_cfg(M, 6144, 5120)
        assert (z.block_M, z.block_N, z.block_K, z.num_stages) == \
            (128, 96, 128, 2), (M, z)
    # qkv M=128 with bk64 (2026-10-21e): the grid at bn128 is 80 blocks, which
    # does fill the second resident slot, so it takes bk128 -> bk64 and bn96 ->
    # bn128: 1.5782 -> 1.1021 vs hipblas for qkv27 (M=128) and 1.1268 -> 0.9807
    # for qkv32.  gate_up is wide enough (400 blocks) that bk128 already has two
    # blocks resident, so it keeps the tile -- bk64 buys nothing there.
    q = decode_cfg(128, 10240, 5120)
    assert (q.block_M, q.block_N, q.block_K, q.num_stages, q.threads, q.setprio) == \
        (128, 128, 64, 2, 512, 1), q
    for M in (112, 128):
        g = decode_cfg(M, 51200, 5120)
        assert (g.block_M, g.block_N, g.block_K, g.num_stages, g.threads) == \
            (128, 128, 128, 2, 512), (M, g)
        _validate_cfg(decode_cfg(M, 10240, 5120), M, 10240, 5120)


def test_decode_cfg_vendor_tiles_are_shape_gated():
    """The three 2026-10-23 arms (from the hipblaslt tuner's own kernels) are
    selected on an explicit (N, K) list, and the list is the whole point: all
    three tiles LOSE on shapes one step away, so a forgotten gate is a real
    regression rather than a missed optimisation.

      bn48  (16x48x256/ns2/t64/p1)  wins on 5 of the 6 decode shapes.  It loses
            on 10240x5120, the strongest cell of the table (0.82x hipblas),
            which keeps its bn32/bk256/ns1 tile.
      bn192 bm64 (64x192x128/ns2/t512/p1) wins on N = 34816 and 10240 only, and
            is 1.4-1.6x slower on N = 14336 and 6144.
      bn192 bm96 (96x192x64/ns2/t384/p0) wins on 34816x5120 only: at 10240x5120
            it is inside noise (69.60 vs 69.92us) and setprio=1 costs 2.5%, so
            that shape keeps its shipped bm96/bn96/bk64 tile.
    """
    # Selected shapes.
    for N, K in ((5120, 6144), (5120, 17408), (6144, 5120), (14336, 5120),
                 (34816, 5120)):
        c = decode_cfg(16, N, K)
        assert (c.block_M, c.block_N, c.block_K, c.num_stages, c.threads,
                c.setprio) == (16, 48, 256, 2, 64, 1), (N, K, c)
    for M in (33, 48, 64):
        for N, K in ((10240, 5120), (34816, 5120)):
            c = decode_cfg(M, N, K)
            assert (c.block_M, c.block_N, c.block_K, c.num_stages, c.threads,
                    c.setprio) == (64, 192, 128, 2, 512, 1), (M, N, K, c)
    # The M=96 band takes the bk64 form of the same tile -- at bm96, bn192 needs
    # (96+192)*128*2 = 73728 B of LDS, above the 65536 B cap -- and the one
    # shape where it wins, at setprio=0.
    c = decode_cfg(80, 34816, 5120)
    assert (c.block_M, c.block_N, c.block_K, c.num_stages, c.threads,
            c.setprio) == (96, 192, 64, 2, 384, 0), c

    # Excluded shapes keep the tile they had.  in_proj_qkv at M<=16 is bn32, not
    # bn48; qkv27/in_proj_z never take bn192 at any M; in_proj_qkv at M=96 keeps
    # bn96/bk64.
    c = decode_cfg(16, 10240, 5120)
    assert (c.block_N, c.block_K, c.num_stages) == (32, 256, 1), c
    for M in (48, 64):
        for N, K in ((14336, 5120), (6144, 5120)):
            c = decode_cfg(M, N, K)
            assert (c.block_M, c.block_N, c.block_K, c.num_stages) == \
                (64, 64, 128, 2), (M, N, K, c)
    c = decode_cfg(96, 10240, 5120)
    assert (c.block_M, c.block_N, c.block_K, c.num_stages) == (96, 96, 64, 2), c

    # K gates: bn48 needs bk256 and bn192 needs bk128/bk64, both of which divide
    # K only for some shapes.  A version of these shapes with K=8192 (a
    # non-multiple of 5120's neighbours) must fall back, not fail to validate.
    for M in (16, 48, 96):
        c = decode_cfg(M, 5120, 8192)
        assert c.block_N != 48 or M != 16, c
        _validate_cfg(c, M, 5120, 8192)


def test_validate_cfg_rejects_the_k8192_bk320_trap():
    """bk=320 divides K=5120 and K=25600 but NOT K=8192. A non-dividing block_K
    makes the pipelined T.copy read past the end of B and return silently wrong
    results (bk=192 on K=5120 gave rel=1.61 with no error)."""
    with pytest.raises(AssertionError, match="silent OOB copy"):
        _validate_cfg(DecodeCfg(16, 16, 320, 1, 64, 1), 1, 5120, 8192)
    # ...and the same config is fine where it does divide.
    _validate_cfg(DecodeCfg(16, 16, 320, 1, 64, 1), 1, 5120, 25600)


@pytest.mark.parametrize("N,K", _DECODE_SHAPES)
def test_every_chosen_decode_config_is_correct(N, K):
    """Run whatever ``decode_cfg`` actually returns, on every decode shape.

    This is the coverage that was missing: M=64 is (with the M>96 bn128 row below)
    one of only two configs that use ``num_stages > 1`` (bk128 ns2), and it was
    adopted on the strength of a hand-run measurement with no test pinning it. It is also the config most at
    risk -- until the 2026-10-17h fix, ``num_stages > 1`` was *silently wrong*
    (rel=2.4, or non-finite) for most other (block_M, block_N, block_K, threads)
    combinations, so a future edit to the M=64 row that "looks harmless" would
    have produced garbage rather than an error. See the doc's 2026-10-17f and
    2026-10-17h sections.
    """
    M = 64
    cfg = decode_cfg(M, N, K)
    assert cfg.num_stages > 1, f"this test is only meaningful for the ns>1 config, got {cfg}"
    A, B, scale_a, scale_b = _make_inputs(M, N, K)
    kernel = fp8_channelwise_gemm(M, N, K, cfg.block_M, cfg.block_N, cfg.block_K,
                                 cfg.num_stages, cfg.threads, setprio=cfg.setprio)
    C = kernel(A, B, scale_a, scale_b)
    ref = _ref_program(A, B, scale_a, scale_b)
    assert torch.isfinite(C).all(), f"decode_cfg(64, {N}, {K}) produced non-finite output"
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-3, f"decode_cfg(64, {N}, {K}) = {cfg} rel error {rel:.6f}"


@pytest.mark.parametrize("K", (8192, 25600))
def test_m128_wide_tile_is_exact(K):
    """M in (96,128] at N=5120 takes bm128/bn96/bk128/ns2/t512 (2026-10-21).

    It is a *newly chosen* ns>1 config at a tile where the pre-2026-10-17h code
    was silently wrong, so it gets the same correctness pin as the M=64 row:
    rel=6.1e-8 on down and 0 on o were measured by hand, not by the suite.  The
    tile is also in _UNROLL_TILES, so this pins the unrolled path (k_unroll->8),
    not just the tile choice."""
    M, N = 128, 5120
    cfg = decode_cfg(M, N, K)
    assert (cfg.block_M, cfg.block_N, cfg.block_K, cfg.num_stages,
            cfg.threads) == (128, 96, 128, 2, 512), cfg
    A, B, scale_a, scale_b = _make_inputs(M, N, K)
    kernel = fp8_channelwise_gemm(M, N, K, cfg.block_M, cfg.block_N, cfg.block_K,
                                 cfg.num_stages, cfg.threads, setprio=cfg.setprio)
    C = kernel(A, B, scale_a, scale_b)
    ref = _ref_program(A, B, scale_a, scale_b)
    assert torch.isfinite(C).all(), f"decode_cfg(128, {N}, {K}) produced non-finite output"
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-3, f"decode_cfg(128, {N}, {K}) = {cfg} rel error {rel:.6f}"


def test_wide_m_decode_cfg_is_shape_aware():
    """M in (128,256] -- the speculative-decode band (2026-10-23).

    Two structural facts, both of which the chooser would silently violate if
    someone widened a tile "for more parallelism":

    1. ``(bm + bn) * bk * ns <= 65536`` at bk128/ns2 is exactly ``bm + bn <= 256``.
       Band 160 therefore must take bn96 -- bm160/bn128 needs 73728 B.
    2. The N < 10240 and N >= 10240 groups differ in BOTH knobs, and neither
       difference is noise: setprio is worth +5-13% on the ns2 tile and costs
       11% on the ns1 tile (gate_up27 0.974 -> 1.078).
    """
    # N < 10240: the double-buffered, unrolled, prio-boosted tile.
    for M in (129, 160):
        for N, K in ((5120, 8192), (5120, 6144), (5120, 17408), (6144, 5120)):
            c = decode_cfg(M, N, K)
            assert (c.block_M, c.block_N, c.block_K, c.num_stages, c.threads,
                    c.setprio) == (160, 96, 128, 2, 256, 1), (M, N, K, c)
            _validate_cfg(c, M, N, K)
    # N >= 10240: one stage, no setprio.  gate_up27 is the only win here (0.974);
    # qkv32 measured 0.995 (inside the 2% gate) and gate_up32 1.179.
    for M in (129, 160):
        for N, K in ((34816, 5120), (10240, 5120), (51200, 5120)):
            c = decode_cfg(M, N, K)
            assert (c.block_M, c.block_N, c.block_K, c.num_stages, c.threads,
                    c.setprio) == (160, 96, 128, 1, 256, 0), (M, N, K, c)
            _validate_cfg(c, M, N, K)
    # Band 192 is the ns2 bn192 tile: (192 + 192) * 64 * 2 = 49152 B -> 4
    # blocks/CU.  It is the gate_up27 M=192 winner (3 clean runs 0.973-0.979);
    # decode_cfg serves it for every (N, K) and the *gate* (gemm_sweep) admits
    # only (34816, 5120) -- the other shapes' M=192 stay on hipblas there, the
    # same split as band 160 (which also serves a losing down32 tile).
    for M in (161, 192):
        for N, K in ((34816, 5120), (6144, 5120), (5120, 6144)):
            c = decode_cfg(M, N, K)
            assert (c.block_M, c.block_N, c.block_K, c.num_stages, c.threads,
                    c.setprio) == (192, 192, 64, 2, 384, 0), (M, N, K, c)
            _validate_cfg(c, M, N, K)
    # The band stops deliberately.  M=256's best arm is a 1.000 tie, and a ratio
    # of exactly 1.0 read at M >= 193 is the hipblas fallback (`band(M)` is
    # None), not a win -- so no config may be selected for it.
    for M in (193, 256):
        with pytest.raises(AssertionError, match="M > block_M"):
            _validate_cfg(decode_cfg(M, 5120, 8192), M, 5120, 8192)


@pytest.mark.parametrize("M,N,K", (
    (160, 5120, 6144),    # o27   -- ns2/sp1, u8
    (160, 6144, 5120),    # z27   -- ns2/sp1, 64 k-steps
    (160, 34816, 5120),   # gate_up27 -- ns1/sp0
    (192, 34816, 5120),   # band 192 -- bn192/ns2 (gate_up27 M=192 winner)
))
def test_wide_m_decode_configs_are_exact(M, N, K):
    """Band 160/192 are ns>1 (or bk64) tiles the pre-2026-10-17h code got wrong.

    bm160 > M is the normal case here (the band pads A), and band 192 uses bk64
    at ns=1, so this pins both the slice-form predicated copies and the stage
    stride at a tile nothing else in the suite covers.
    """
    cfg = decode_cfg(M, N, K)
    A, B, scale_a, scale_b = _make_inputs(M, N, K)
    kernel = fp8_channelwise_gemm(M, N, K, cfg.block_M, cfg.block_N, cfg.block_K,
                                  cfg.num_stages, cfg.threads, setprio=cfg.setprio)
    C = kernel(A, B, scale_a, scale_b)
    ref = _ref_program(A, B, scale_a, scale_b)
    assert torch.isfinite(C).all(), f"decode_cfg({M}, {N}, {K}) produced non-finite output"
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-3, f"decode_cfg({M}, {N}, {K}) = {cfg} rel error {rel:.6f}"


@pytest.mark.parametrize("N,K", _DECODE_SHAPES)
@pytest.mark.parametrize("M", (1, 16, 32))
def test_small_m_decode_configs_are_exact(M, N, K):
    """The M <= 32 rows of ``decode_cfg`` -- what every decode step actually runs.

    Two unrelated families live here: ``bn96/bk256/ns2/t192`` for N=5120 and
    ``bn32`` + ns1 for the wide-N shapes (qkv M=32 is ``bn32/bk128/t128``).
    Until now only their *config* was asserted, never their output, and the ns2
    family is exactly the one the packed-stage-stride defect lives in -- so the
    N=5120 rows are the cells that most need a numeric pin.
    """
    cfg = decode_cfg(M, N, K)
    A, B, scale_a, scale_b = _make_inputs(M, N, K)
    kernel = fp8_channelwise_gemm(M, N, K, cfg.block_M, cfg.block_N, cfg.block_K,
                                 cfg.num_stages, cfg.threads, setprio=cfg.setprio)
    C = kernel(A, B, scale_a, scale_b)
    ref = _ref_program(A, B, scale_a, scale_b)
    assert torch.isfinite(C).all(), f"decode_cfg({M}, {N}, {K}) produced non-finite output"
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-3, f"decode_cfg({M}, {N}, {K}) = {cfg} rel error {rel:.6f}"


@pytest.mark.parametrize("M,N,K", ((16, 5120, 5120), (16, 10240, 5120)))
def test_per_tensor_scale_a_is_exact(M, N, K):
    """``per_tensor_a=True`` must be the same GEMM with a 1-element ``scale_a``.

    It exists for the M == 1 decode step, where vLLM's ``scale_a`` is [1, 1]:
    declaring ``scale_a`` as ``(block_M,)`` would force the caller to
    materialise a band-length buffer, and on this card *one* DtoD copy node
    costs ~5 us of GPU time regardless of size.  Pinned by comparing against
    the per-row variant on a constant scale, so a wrong broadcast (``[0]`` vs
    ``[i]``, or a mis-sized read) shows up as a numeric error.
    """
    cfg = decode_cfg(M, N, K)
    A, B, scale_a, scale_b = _make_inputs(M, N, K)
    common = dict(block_M=cfg.block_M, block_N=cfg.block_N, block_K=cfg.block_K,
                  num_stages=cfg.num_stages, threads=cfg.threads, setprio=cfg.setprio)
    per_row = fp8_channelwise_gemm(M, N, K, **common)
    per_tensor = fp8_channelwise_gemm(M, N, K, per_tensor_a=True, **common)

    flat = scale_a[:1].reshape(1).contiguous()       # the M == 1 spelling
    C_ten = per_tensor(A, B, flat, scale_b)

    const = flat[0].expand(M).contiguous()           # makes the arms comparable
    C_row = per_row(A, B, const, scale_b)            # same GEMM, per-row spelling
    ref = _ref_program(A, B, const, scale_b)
    for name, C in (("per-row", C_row), ("per-tensor", C_ten)):
        assert torch.isfinite(C).all(), f"{name} produced non-finite output"
        rel = (C - ref).abs().max().item() / ref.abs().max().item()
        assert rel < 1e-3, f"{name} rel error {rel:.6f} at decode_cfg({M}, {N}, {K})"


@pytest.mark.parametrize("M,N,K", (
    (80, 10240, 5120), (96, 10240, 5120), (104, 10240, 5120),
    (112, 10240, 5120), (128, 10240, 5120),
    (80, 51200, 5120), (96, 51200, 5120), (128, 51200, 5120),
))
def test_wide_n_mid_band_is_exact(M, N, K):
    """The M in (64,128] wide-N rule (2026-10-21/21d) on both shapes it serves.

    Eight arms, and each one is a tile that was previously unshipped: qkv M=96
    takes bm96/bn96/ns2/U8, qkv M in (96,128] takes bm128/bn128/bk64/ns2/t512
    (2026-10-22f: the bk64 arm -- 2 blocks/CU -- over the whole band, not just
    above the old M=112 gate), and gate_up takes
    bm96/bn128/ns2/U8 at M=80/96 (2026-10-21d: bm96 deletes 48/128 and 32/128
    padded MMA rows at zero byte cost) and bm128/bn128/ns2/U8 at M>96.  All of
    them are ns>1 *or* a bm that does not
    equal M, so this pins the predicated copies and the unrolled path where the
    pre-2026-10-17h code was silently wrong.  M=104/112 are the interior points
    the old band boundary was never measured at.
    """
    cfg = decode_cfg(M, N, K)
    A, B, scale_a, scale_b = _make_inputs(M, N, K)
    kernel = fp8_channelwise_gemm(M, N, K, cfg.block_M, cfg.block_N, cfg.block_K,
                                 cfg.num_stages, cfg.threads, setprio=cfg.setprio)
    if cfg.num_stages > 1 and (N, M) != (51200, 64):
        assert "cp_async_wait<2>" in kernel.get_kernel_source(), (
            f"decode_cfg({M}, {N}, {K}) = {cfg} did not take the unrolled k-loop")
    C = kernel(A, B, scale_a, scale_b)
    ref = _ref_program(A, B, scale_a, scale_b)
    assert torch.isfinite(C).all(), f"decode_cfg({M}, {N}, {K}) produced non-finite output"
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-3, f"decode_cfg({M}, {N}, {K}) = {cfg} rel error {rel:.6f}"


@pytest.mark.parametrize("K", (8192, 25600))
@pytest.mark.parametrize("M", (65, 96, 112))
def test_unrolled_k_loop_is_exact(M, K):
    """The shipped decoder runs an explicitly unrolled, hand-barriered k-loop.

    Two things it must get right that the pipelined path did not exercise:
    ``block_M > M`` (M=65 into a 96-row tile, M=112 into a 128-row tile -- the
    copies are slice-form, so they must predicate), and the tail/remainder when
    ``(K/block_K - 1)`` is not a multiple of ``k_unroll`` (7 of 8 configurations
    here leave a remainder).  rel is pinned at 1e-3 like the other decode rows.
    """
    N = 5120
    cfg = decode_cfg(M, N, K)
    A, B, scale_a, scale_b = _make_inputs(M, N, K)
    kernel = fp8_channelwise_gemm(M, N, K, cfg.block_M, cfg.block_N, cfg.block_K,
                                 cfg.num_stages, cfg.threads, setprio=cfg.setprio)
    assert "cp_async_wait<2>" in kernel.get_kernel_source(), (
        f"decode_cfg({M}, {N}, {K}) = {cfg} did not take the unrolled k-loop")
    C = kernel(A, B, scale_a, scale_b)
    ref = _ref_program(A, B, scale_a, scale_b)
    assert torch.isfinite(C).all(), f"decode_cfg({M}, {N}, {K}) produced non-finite output"
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-3, f"decode_cfg({M}, {N}, {K}) = {cfg} rel error {rel:.6f}"


def test_decode_cfg_keeps_split_k_closed():
    """Split-K was built, measured, and closed (see the doc's 2026-10-17f
    section): every candidate was 10-21% slower than the non-split baseline on
    both o and down, and the ns=1 path was additionally numerically wrong. The
    chooser must never select it, and the builder no longer exists.

    This is a regression guard on the closure, not on the mechanism -- re-opening
    needs new evidence, not a config change."""
    import tilelang.hcu.op.gemm.gemm_fp8_channelwise as m

    assert not hasattr(m, "fp8_channelwise_gemm_splitk")
    for N, K in _DECODE_SHAPES:
        for M in (1, 16, 32, 64, 128):
            assert decode_cfg(M, N, K).split_k == 1


def test_warp_k_fullcolk_mls_is_correct_and_reduced():
    """FullColK (warp_k) on the MLS path: K is split across warps and the
    per-K-warp partials are summed back.

    This capability did not exist until 2026-10-19 -- the MLS ds_read templates
    ``static_assert(WarpK == 1)`` and both the Python macro generator and
    ``gemm.cc`` rejected the policy outright, so ``T.GemmWarpPolicy.FullColK``
    was unreachable on the fp8 decode path even though ``warp_k`` was already
    plumbed through ``inner_k_per_warp`` / ``_k_base`` / ``mmac``.

    The reduction is a contract, not an optimization: ``make_gemm_fragment_hcu``
    replicates the C fragment across the K warps and the copy-out guards to
    replica 0, so without ``T.reduce_sum_warp`` the kernel returns K-warp 0's
    partial with no error. The source assertions below pin both the K-split
    (A reads WarpM=1/WarpN=1/WarpK=4; Square reads 1/4/1) and the presence of
    the cross-warp sum, which is exactly the pair of facts that would silently
    regress.

    Measured verdict (2026-10-19, paired same-process, 9 rounds, min-of-rounds):
    warp_k is *not* a lever on o/down -- WK4 (bn16/bk256/t256/FullColK) vs the
    shipped bn16/bk256/t64/Square config is within +-3% with mixed sign across
    M in {1,16,32}, and it is a substitute for ``setprio`` rather than a
    complement (each recovers ~6-11% over the sp0 baseline, and they do not
    stack). So the capability is kept and tested, but ``decode_cfg`` must not
    select it. Do not re-open without new evidence.
    """
    import inspect

    import tilelang.language as T

    import tilelang.hcu.op.gemm.gemm_fp8_channelwise as m

    M, N, K = 16, 5120, 8192
    fck = T.GemmWarpPolicy.FullColK
    A, B, scale_a, scale_b = _make_inputs(M, N, K)
    ref = _ref_program(A, B, scale_a, scale_b)

    kern_wk = fp8_channelwise_gemm(M, N, K, 16, 16, 256, 1, 256, policy=fck)
    kern_sq = fp8_channelwise_gemm(M, N, K, 16, 16, 256, 1, 256)
    for tag, kern in (("FullColK", kern_wk), ("Square", kern_sq)):
        C = kern(A, B, scale_a, scale_b)
        assert torch.isfinite(C).all(), f"{tag} produced non-finite output"
        rel = (C - ref).abs().max().item() / ref.abs().max().item()
        assert rel < 1e-3, f"{tag} warp_k MLS rel error {rel:.6f}"

    wk_src, sq_src = kern_wk.get_kernel_source(), kern_sq.get_kernel_source()
    # A's ds_read takes (WarpM, WarpN, WarpK); B's takes (WarpM, WarpN, WarpK).
    assert "ds_read_format_tensor_a<tl::sequence<16, 256>, tl::sequence<16, 256>, " \
           "tl::sequence<16, 64>, 1, 1, 4, " in wk_src, "A is not K-split across 4 warps"
    assert "ds_read_format_tensor_a<tl::sequence<16, 256>, tl::sequence<16, 256>, " \
           "tl::sequence<16, 64>, 1, 4, 1, " in sq_src, "Square no longer N-splits A"
    assert "AllReduce<tl::SumOp, 256, 64, 0>" in wk_src, \
        "the cross-K-warp reduction is missing -- output would be K-warp 0's partial"
    assert "AllReduce" not in sq_src, "Square must not pay for a reduction it does not need"

    # Closure guard: the capability exists but the chooser must stay on Square.
    assert inspect.signature(m.fp8_channelwise_gemm).parameters["policy"].default is None
    assert "policy" not in DecodeCfg.__dataclass_fields__
    for N_, K_ in _DECODE_SHAPES:
        for M_ in (1, 2, 4, 8, 16, 32, 64, 128):
            _validate_cfg(decode_cfg(M_, N_, K_), M_, N_, K_)


def test_mls_packed_stride_is_shared_by_producer_and_consumer():
    """Pin the packed MLS LDS stride that the producer, the allocator and the
    ds_read consumer must all agree on.

    The 2026-10-17h defect: for a packed dst tile the C++ producer wrote stage s
    at ``s * packed`` while the Python consumer read it at ``s * logical``, so
    every ``num_stages > 1`` kernel whose tile took the packed branch read stage
    >= 1 from the wrong LDS address (rel ~1e4 at bm16/bn16/bk256/ns2/t64). The
    fix routes both through ``MlsPackedTileBytes`` / ``tl.MlsPackedTileElems``;
    this test is the tripwire if that ever forks again.

    The tile sizes are what ``ComputeMlsWarpPartition`` selects for each
    (block_mn, block_k, threads) -- fp8-trans uses (16, 128) at 64 threads and
    (16, 64) at 256, which is exactly why the bug tracked the thread count.
    """
    from tilelang.hcu.intrinsics.hcu_mmac_emitter_utils import mls_packed_tile_elems

    # fp8 trans, (16, 128), 64 threads.  A single 16x128 tile is 2048 B, which
    # does not fill a 4K slot, so block_K=128 keeps the conservative slot layout:
    # 4096 + 1*1024 = 5120 against a 2048-B logical tile.
    assert mls_packed_tile_elems(16, 128, 16, 128, True, "float8_e4m3fn") == 5120
    # From block_K=256 the tiles pack densely (2026-10-20).  A 16x256 tile is
    # exactly 4096 B -- the granule map gran = 64*(c>>2) + 4n + ((c&3+(n>>1))&3)
    # spans 0..4095 -- so dense storage *is* the contiguous size and the packed
    # stride collapses to 0.  Before this the 4K-slot accounting charged 6144 B
    # for that tile, and the 50% dead padding cost a resident block per CU on the
    # decode shapes (which is why bn96/bk256 only fits two stages).
    assert mls_packed_tile_elems(16, 256, 16, 128, True, "float8_e4m3fn") == 0
    assert mls_packed_tile_elems(16, 512, 16, 128, True, "float8_e4m3fn") == 0
    # 3 tiles = 6144 B is not a whole number of 4K slots, so the conservative
    # slot layout stays: 4096 + 3*1024 = 7168 against a 6144-B logical tile.
    assert mls_packed_tile_elems(16, 384, 16, 128, True, "float8_e4m3fn") == 7168
    # Same block, wider warp count -> the MLS tile is (16, 64), which takes no
    # packed branch at all. This is the config that was always correct.
    assert mls_packed_tile_elems(16, 256, 16, 64, True, "float8_e4m3fn") == 0
    # bf16 trans (16, 64) is the other packed branch, but at block_mn=64/block_k=128
    # its packed size lands exactly on the logical size (16 KB), so it reports 0.
    assert mls_packed_tile_elems(64, 128, 16, 64, True, "bfloat16") == 0
    # Non-packed regions must report 0 so the consumer falls back to the buffer's
    # own stride rather than inventing one.
    assert mls_packed_tile_elems(32, 256, 16, 128, True, "float8_e4m3fn") == 0


# (block_M, block_N, block_K, num_stages, threads, K, note) -- the configs the
# 2026-10-17h fix was verified against. The first two were 40/40 wrong before it;
# the rest were already correct and must stay correct (a fix that moves a working
# stride would be a regression, not a fix). K is per-row because K % (ns*block_K)
# must be 0, and 5120 is 2^10*5 -- so ns=3 needs a different K.
_NS_RACE = "ns>1 at bm16/bn16/t64 emits a wrong packed stage base (nondeterministic aliasing)"
_PACKED_STRIDE_CONFIGS = [
    (16, 16, 256, 2, 64, 5120, "was 40/40 WRONG"),
    (16, 16, 256, 2, 128, 5120, "was 40/40 WRONG"),
    (16, 16, 256, 2, 256, 5120, "packed branch never taken (tile_k=64)"),
    (16, 16, 128, 4, 64, 5120, "ns=4, tile_issue_k=1"),
    (16, 16, 64, 8, 64, 5120, "ns=8"),
    (16, 16, 512, 2, 128, 5120, "packed size == logical (tile_issue_k=4)"),
    (64, 64, 128, 2, 256, 5120, "shipped M=64 config"),
    # --- KNOWN BROKEN (2026-10-18). The 2026-10-17h packed-stride fix did NOT
    # close ns>1 at bm16/bn16/t64: these four are still wrong, and they are
    # wrong NON-DETERMINISTICALLY -- five runs on identical inputs give five
    # different rel values (measured 2026-10-18):
    #     bk128/ns2  3504 / 2898 / 2090 / 1980 / 1119   (K=25600)
    #     bk128/ns3 12123 / 11923 / 9552 / 10799 / 9271 (K=25600)
    #     bk256/ns3  8899 / 8709 / 8581 / 10180 / 6289  (K=25600)
    #     bk256/ns4  6597 / 5616 / 7074 / 7700 / 6511   (K=25600)
    # The cause is visible in the emitted source: the per-stage LDS base should be
    # stage * packed_size (5120 at bk128, 6144 at bk256). bk256/ns2 emits
    # (k&1)*6144 -- exact -- while bk128/ns2 emits (k&1)*1024 and the ns=4 forms
    # emit an affine-simplifier artifact, (((k&3)*24)&63)>>4, which over k=0..3
    # is 0,1,3,0: not even a permutation of the stages. Aliased stages corrupt
    # each other intermittently because the producer is async and the consumer
    # ds_reads independently -- hence the run-to-run variation. The irregular
    # bk/ns pattern is not a layout rule: bk256/ns2, bk128/ns4 and every
    # contiguous case (bk64, bk512) are exact and reproducible, which is why the
    # earlier "was 40/40 WRONG, now fixed" reading looked complete -- the old list
    # simply never covered these four. It is NOT K-dependent (bk128/ns2 is
    # already wrong at K=5120 / 40 k-steps, so a short K is not a safe escape).
    # No production exposure: fp8_channelwise_gemm asserts ns>1 only for
    # bm64/bn64, and all four decode shapes at M=16 and M=64 were re-verified
    # exact 5x each on 2026-10-18. Marked xfail so the region stays visible and
    # flips to xpass the moment the race is fixed -- do not delete these rows.
    pytest.param(16, 16, 128, 2, 64, 25600, "ns>1 race (known broken)", marks=pytest.mark.xfail(reason=_NS_RACE, strict=False)),
    pytest.param(16, 16, 128, 3, 64, 25600, "ns>1 race (known broken)", marks=pytest.mark.xfail(reason=_NS_RACE, strict=False)),
    pytest.param(16, 16, 256, 3, 64, 25600, "ns>1 race (known broken)", marks=pytest.mark.xfail(reason=_NS_RACE, strict=False)),
    pytest.param(16, 16, 256, 4, 64, 25600, "ns>1 race (known broken)", marks=pytest.mark.xfail(reason=_NS_RACE, strict=False)),
]


@pytest.mark.parametrize("block_M,block_N,block_K,num_stages,threads,K,note", _PACKED_STRIDE_CONFIGS)
def test_num_stages_gt_1_mls_is_correct(block_M, block_N, block_K, num_stages, threads, K, note):
    """End-to-end guard for the packed-stride fix: ns>1 MLS must be correct.

    Goes through all three consumers at once -- the C++ ``MatrixLoad`` producer,
    ``MergeSharedMemoryAllocations``, and the Python ds_read base pointer -- so
    it fails if any of them drifts. Written inline rather than through
    ``fp8_channelwise_gemm`` because that builder deliberately restricts ns>1 to
    bm64/bn64 (untuned elsewhere), which would make this untestable.
    """
    import tilelang
    import tilelang.language as T

    M, N = 16, 5120
    assert M <= block_M and K % (num_stages * block_K) == 0

    @T.prim_func
    def gemm(
            A: T.Tensor((M, K), "float8_e4m3fn"),
            B: T.Tensor((N, K), "float8_e4m3fn"),
            scale_a: T.Tensor((M,), "float32"),
            scale_b: T.Tensor((N,), "float32"),
            C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), "float8_e4m3fn")
            B_s = T.alloc_shared((block_N, block_K), "float8_e4m3fn")
            C_l = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_s, prefer_instruction="matrix_load")
                T.copy(B[bx * block_N, k * block_K], B_s, prefer_instruction="matrix_load")
                T.gemm(A_s, B_s, C_l, False, True, k_pack=1)
            for i, j in T.Parallel(block_M, block_N):
                C_l[i, j] = C_l[i, j] * scale_a[by * block_M + i] * scale_b[bx * block_N + j]
            T.copy(C_l, C[by * block_M, bx * block_N])

    kernel = tilelang.compile(gemm, out_idx=[4])
    A, B, scale_a, scale_b = _make_inputs(M, N, K)
    C = kernel(A, B, scale_a, scale_b)
    ref = _ref_program(A, B, scale_a, scale_b)
    assert torch.isfinite(C).all(), f"bm{block_M}/bn{block_N}/bk{block_K}/ns{num_stages}/t{threads} non-finite"
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-3, (f"bm{block_M}/bn{block_N}/bk{block_K}/ns{num_stages}/t{threads} "
                        f"({note}) rel error {rel:.6f} -- packed MLS stride diverged")


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
