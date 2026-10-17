# fp8 Channelwise decode GEMM — per-shape × per-M progress summary

*2026-10-17. Companion to the deep-dive in `fp8_decode_m_le_128.md`; this file
is the roll-up the user asked for ("不同 shape 以及不同 M 的测试数据 … 汇总一下
进展"). Absolute numbers below are from the clean per-shape sweep
(`/tmp/shape_tune.py`, 5 rounds × 30 iters, CUDA-graph replay, every candidate
correctness-verified with `isfinite` + `rel<1e-3` before timing). Ratios for
M∈{2,4,8,64,128} are from the paired 2026-10-17f matrix in the deep-dive.
M=64 is out of scope (structural optimum, documented separately).*

## TL;DR

- **We now have per-shape, per-M data** for all four qwen3-32B decode GEMMs
  (gate_up / qkv / o / down) at M∈{1,16,32} (clean sweep) and M∈{2,4,8,64,128}
  (paired 17f matrix).
- The shape-aware `decode_cfg` chooser **already picks the best (or a tied-best)
  config in 10 of the 12** (shape, M) cells. o and down already get their own
  correct configs — they do not share one with gate_up/qkv.
- **One real per-shape lever found this session: qkv M=32.** The chooser's
  `bm32/bn32/bk256/t256` is **rank 7**; a `bm32/bn32/bk128/t128` config is
  **~9% faster and flips that cell from a 1.08× loss to a 0.985× near-win.**
  **Not yet adopted** — the 9% is read off a 63-candidate process, which this
  repo's own timing rules say must not be trusted for a decision. It needs a
  clean 2-graph alternating A/B (bk128/t128 vs bk256/t256 vs hipblas, ~12
  rounds) before `decode_cfg` changes.
- One sub-noise gap: **o M=32**, where `bn32/t256` edges out the chosen
  `bn16/t256` by ~1% — inside the documented noise band, reported as-is.

## The per-shape sweep (M∈{1,16,32})

`ratio` = our time / hipblas time, **< 1 is a win**. `rank` is the CHOSEN
config's rank among all correctness-verified candidates in that (shape, M)
process (63 for the K=5120 shapes, 53 for o). `tied` = the "<- BEST" row is the
same kernel under its sweep label, differing from CHOSEN by ≤0.2 µs (noise) —
i.e. the chooser *is* the best.

| shape (B) | M | CHOSEN config | µs | rank | ratio | BEST config | µs | ratio |
|---|---|---|---|---|---|---|---|---|
| gate_up (262 MB) | 1  | bm16 bn32 bk256 t256 | 162.2 | 2/63 (tied) | 0.917 | same | 162.1 | 0.916 |
|  | 16 | bm16 bn32 bk256 t256 | 164.8 | **1/63** | 0.904 | same | 164.8 | 0.904 |
|  | 32 | bm32 bn32 bk256 t256 | 188.6 | 2/63 (tied) | 0.866 | same | 188.2 | 0.864 |
| qkv (52 MB) | 1  | bm16 bn32 bk256 t256 | 37.6  | 2/63 (tied) | 0.848 | same | 37.6 | 0.848 |
|  | 16 | bm16 bn32 bk256 t256 | 38.7  | **1/63** | 0.807 | same | 38.7 | 0.807 |
|  | 32 | bm32 bn32 **bk256 t256** | 57.3  | **7/63** | 1.082 | bm32 bn32 **bk128 t128** | **52.2** | **0.985** |
| o (42 MB) | 1  | bm16 bn16 bk256 t64 | 38.9 | 2/53 (tied) | 1.234 | same | 38.7 | 1.228 |
|  | 16 | bm16 bn16 bk256 t64 | 39.5 | 2/53 (tied) | 1.088 | same | 39.5 | 1.088 |
|  | 32 | bm32 **bn16** bk256 t256 | 47.2 | 3/53 | 1.152 | bm32 **bn32** bk256 t256 | 46.7 | 1.141 |
| down (131 MB) | 1  | bm16 bn16 bk256 t64 | 99.5  | 2/63 (tied) | 1.264 | same | 99.5 | 1.264 |
|  | 16 | bm16 bn16 bk256 t64 | 100.5 | **1/63** | 1.180 | same | 100.5 | 1.180 |
|  | 32 | bm32 bn16 bk256 t256 | 122.6 | 2/63 (tied) | 1.131 | same | 122.2 | 1.129 |

**Reading.** 10 of 12 cells the chooser is the best or a tied-best. The two
exceptions are both at M=32:

- **qkv M=32 is a real lever.** The top of the qkv M=32 list is
  `bk128 t128 = 52.2`, `bk128 t256 = 54.6`, … `bk256 t256 = 57.3` (CHOSEN) —
  two `bk128` variants both beat the chosen `bk256`, a coherent trend (smaller
  `bk` wins at this shape), not a single outlier. But it is a *ratio* read off a
  crowded 63-graph process; the doc's timing rules require a clean 2-graph A/B
  before a code change. **Action: verify, then either adopt a per-(N,K) key that
  gives qkv M=32 `bk128/t128`, or report the gap as-is.**
- **o M=32 is sub-noise.** `bn32/t256` = 46.7 vs CHOSEN `bn16/t256` = 47.2 is
  ~1%, inside the documented band. Report as-is; not worth a code path.

### Why the % column can exceed what looks "physical"

The sweep's `%` is N×K / time vs the **1800 GB/s nominal**. The measured pure-read
ceiling on this box is **~2016 GB/s** (the older 1646 / 91.5% figure is
retired — see the deep-dive correction). Read floor = N×K / 2016:

| shape | B | read floor (2016 GB/s) | our best M≤16 | % of 2016 |
|---|---|---|---|---|
| gate_up | 262 MB | 130.0 µs | 164.8 µs | 78.9% |
| qkv | 52 MB | 26.0 µs | 38.7 µs | 67.2% |
| o | 42 MB | 20.8 µs | 39.5 µs | 52.6% |
| down | 131 MB | 65.0 µs | 100.5 µs | 64.7% |

The gap to the floor is the overlap-quality residual, not a bandwidth wall —
see the deep-dive (M=64 LDS/overlap analysis, "the kernel is overlap-bound, not
producer-bound").

## Full ratio matrix (M=1..128, paired, 2026-10-17f)

`ratio` = ours/hipblas, **< 1 is a win**. M=1..32 rows are stable to ≤5% across
runs; M=64/128 move up to ~35% between runs (fewer blocks, shared box) —
treat those as indicative. M=64 is out of scope.

| shape | M=1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| gate_up | **0.93** | **0.78** | **0.87** | **0.88** | **0.91** | **0.86** | 1.08 | **1.00** |
| qkv | **0.85** | **0.82** | **0.83** | **0.84** | **0.85** | 1.13* | 1.05 | 1.15 |
| o | 1.19 | 1.14 | 1.11 | 1.11 | 1.13 | 1.19 | **0.89** | 1.83 |
| down | 1.27 | 1.23 | 1.08 | 1.18 | 1.18 | 1.11 | 1.09 | 2.23 |

\* qkv M=32 = 1.13 here is the *current* chooser (bk256/t256); the unverified
bk128/t128 candidate would put it at ~0.985 if it survives clean re-measurement.

**Wins (bold): 13 of 32** — gate_up M∈{1..32,128}, qkv M∈{1..16}, o M=64.

## What the per-shape tuning changed vs the old chooser

The old `_decode_cfg(M)` was shape-blind (gate_up-tuned tile sent to all four).
The current `decode_cfg`/`_validate_cfg` (`perf/gemm/fp8_channelwise_mls.py`)
is shape-aware:

- **N ≥ 10240 → bn32/t256** (gate_up, qkv); M≤16 → bm16, M=32 → bm32/t256.
- **N = 5120 → bn16** (o, down); `threads = 64 if bm≤16 else 256` (the M=32
  thread fix — a `bm32` tile at `bn16` has two 16×16 M-atoms, so one warp makes
  the block MMA-issue-bound; t256 recovers it, 1.60×/1.71× on o/down).
- **M=64 → (64,64,128,ns2,256,1)** structural optimum, out of scope.
- **M=128 → (128,128,128,ns1,512,0)**.

The per-shape sweep **confirms** these are already near-optimal (10/12) and
isolates the one remaining tuning lever (qkv M=32) — it does *not* show that o
and down need separate hand-tuning, because they already get distinct, correct
configs.

## Levers closed (do not re-open without new evidence)

- **Split-K** — the "block-starved" diagnosis was a confound; constant-`N·K`
  sweep shows 320 blocks is the best point, and a built split-K kernel measured
  **10-21% slower**. Removed from the codebase (`DecodeCfg.split_k` kept as a
  closure marker).
- **Tile widening on o/down** — monotonic catastrophe (bn16→bn128 at M=1:
  o 43→77 µs, down 118→221 µs). The small-N deficit is structural.
- **Pipeline depth `ns>1` outside bm64/bn64** — silently wrong (rel 0.75-580 or
  non-finite); guarded by `assert ns<=1 or (bm,bn)==(64,64)`.
- **B-direct-to-VGPR / MLS-atom swap** — both measured closed (deep-dive).

## Honest remaining gaps

- **o/down M≤32 (1.08-1.27×):** B-stream-bound, closed on every lever tried.
  Not a block-count / tile / pipelining / epilogue problem.
- **qkv M≥32 (1.05-1.15×):** the M=32 cell has the bk128 lever above (pending
  verification); M=128 is 40 blocks at bn128, unaddressed.
- **M=128 (o 1.83× / down 2.23×):** only 40 blocks at bn128 on N=5120 and `bn`
  cannot shrink without shrinking the grid further.

## Reproduce

```
HIP_VISIBLE_DEVICES=<idle 0/1/2/3> python perf/gemm/fp8_channelwise_decode_bench.py
# per-shape sweep (not in repo): /tmp/shape_tune.py <gate_up|qkv|o|down>
# qkv M=32 verification (TODO before any code change): clean 2-graph A/B
#   bm32/bn32/bk128/t128  vs  bm32/bn32/bk256/t256  vs  torch._scaled_mm
#   ~12 alternating rounds, only the 2-3 graphs resident.
```

**Timing rules (see deep-dive + `hcu-benchmark-gpu-is-shared`):** paired
same-process A/B only (absolutes drift ~15% between runs and orderings flip);
check `hy-smi` HCU% ≈ 0 first; a process holding many candidate CUDA graphs
inflates `torch._scaled_mm` ~17%, so quote ratios from a clean 2-graph A/B, not
from a sweep process.
