# fp8 Channelwise GEMM on HCU — the decode regime

**What this document is.** The results of the HCU (gfx938) fp8 channelwise GEMM
for LLM decode — `o` (N=5120, K=8192), `down` (N=5120, K=25600), `qkv` (N=10240)
and `gate_up` (N=51200), M from 1 to 128 — measured against hipblasLt
`torch._scaled_mm` (bf16 out, RowWise), plus the *path*: the bandwidth model the
kernel is scored against, why each era of it broke, and every lever that is
closed. Read it top-down.

The dated record — one section per experiment, with its arm tables and per-round
ratios — is the companion log,
`llm-adaptation/qwen3-32b-fp8-rowwise/docs/fp8_decode_gemm_log.md`. The
[history index](#history) maps every date to its one-line verdict and to its log
entry. Where a later result overturned an earlier one, the surviving statement
is the one here. Sections tagged *(compressed)* retain only their decisive
measurement; the full narrative was cut on 2026-10-21.

Follow-up to `fp8_fp8_channelwise_gemm.md` (the 200 → 300 → 432 TFLOPS three-stage
story). This note covers the **decode regime** (qwen3-32B per-token GEMMs:
M=1–128, K=5120, N=51200 gate_up), where the kernel is *bandwidth-bound*, not
compute-bound, and the goal is to beat hipblaslt's fp8 rowwise kernel on
effective bandwidth.

**Current shipped config** (`decode_cfg`,
`tilelang/hcu/op/gemm/gemm_fp8_channelwise.py`): N=5120 (`o`/`down`) at
M ≤ 32 → `bn96 / bk256 / ns2 / t192` with `setprio=1` (`bn16 / bk256 / t64` when
`K % 256 != 0`); M=64 → `bm64 / bn64 / bk128 / ns2 / t256` on every shape; M in
(64,128] at N=5120 → `bm96 / bn96 / bk128 / ns2 / t384` up to M=96 and
`bm128 / bn96 / bk128 / ns2 / t512` above it (log §2026-10-20e; the tile was
re-chosen by log §2026-10-21). At M ≤ 32, `qkv` (N=10240)
and `gate_up` (N=51200) keep `bn32` (the `qkv` M=32 point is `bn32 / bk128 /
t128`). M in (64,128] on those two shapes is decided by block count first and
MMA padding second: `ceil(N/128) >= 256` (gate_up) takes `bm96 / bn128 / bk128 /
ns2` with t512 up to M=80 and t256 up to M=96, then `bm128 / bn128 / bk128 /
ns2 / t512`; otherwise (qkv) M ≤ 96 takes `bm96 / bn96 / bk128 / ns2 / t384`, and
above it the o/down M>96 tile with `setprio=1` (log §2026-10-21b, §2026-10-21c,
§2026-10-21d).

Three arms added 2026-10-23 come from the hipblasLt **tuner's** own decode kernels
rather than from the in-tree search, and all three are gated on an explicit
`(N, K)` list — not a rule, because each tile loses on shapes one step away:
M ≤ 16 on five of the six 27B shapes → `bm16 / bn48 / bk256 / ns2 / t64` (tuner
`MT32x48x256`); M in (32,64] on N = 34816/10240 → `bm64 / bn192 / bk128 / ns2 /
t512` (tuner `MT128x192x64`); M in (64,96] on N=34816 only → `bm96 / bn192 /
bk64 / ns2 / t384`, the one arm in the table that wants `setprio=0`. The lists
live in `_BN48_M16_NK` / `_BN192_NK` / `_BN192_BM96_NK`. Worth 1–10% over the tile
each replaces, bit-identical to it (`relmax 0.0e+00`). `bn192` also **retires the
"`bn ≤ 128` is the hard cap" claim** that stood in the closed-lever index: the cap
is a property of the (bm, bk, ns, threads) combination, not of `bn`.

**Current standing** (paired same-process, idle card, `rel=0`; ratio =
ours/hipblas, **< 1 is a win**). Shipped `decode_cfg`, measured 2026-10-21:

| shape | M=1 | M=16 | M=32 | M=64 | M=80 | M=96 | M=128 |
|---|---|---|---|---|---|---|---|
| o | 1.082 | **0.995** | **0.967** | **0.956** | **0.939** | **0.933** | 1.037 |
| down | 1.092 | 1.028 | **0.875** | **0.978** | **0.891** | **0.889** | 1.036 |
| gate_up | **0.915** | **0.900** | **0.872** | 1.030 | **0.962** | **0.953** | **0.966** |
| qkv | **0.864** | **0.855** | **0.932** | 1.008 | 1.070 | 1.048 | 1.153 (was 1.178) |

gate_up M=80/96 are from the 2026-10-21d run that introduced `bm96/bn128` there
(285.6 / 284.6 µs); the unchanged M=112/128 tiles re-measured in the same run at
0.989 / 0.958, i.e. drift-level movement, not a change. Cell ratios come from
their own paired runs — compare µs, not ratios, across rows.

**18 of 28 cells win.** The M ≤ 96 band is largely ours; the `M=128` column is
where the work is (3 of 4 cells lose), and `qkv` is the worst row overall
(4 of 7 losses, worst cell 1.153). Full loss list: o M=1/128, down M=1/16/128,
gate_up M=64, qkv M=64/80/96/128. These numbers supersede the M ≤ 32-only table
that stood here (o 1.069/1.027/1.024, down 1.110/1.042/0.928).

The table above is the 2026-10-21 measurement and **predates the three 2026-10-23
arms**, which move the M ≤ 16 row (all shapes but N=10240), `M=64` on gate_up and
the gate_up `M=96` cell. The per-tile confirmation runs that justified those arms
are the authoritative numbers for them (1.03–1.16× over the tile each replaced,
see the paragraph above); the whole-table re-measurement against hipblas was
**drift-dominated** — the paired-ratio column moved 10 cells up and 12 down
against 2026-10-22 while the hipblas arm itself got 4–19% faster in the same
session, i.e. absolute time improved on 22 of 24 cells but the ratio did not
resolve. No dispatch-gate verdict flipped in that run.

## Contents

- [What actually won](#what-won)
- [Closed levers — index](#closed-levers)
- [The setup](#setup)
- [The bandwidth model](#bw-model)
- [What hipblas does at M=1](#hipblas-m1)
- [M ≤ 16 — the bn16/t64 era](#m-le-16)
- [Why M ≤ 16 was flat, and why M=32 dropped](#why-flat)
- [M ≥ 32 ≤ 128: the streaming kernel breaks down](#m-ge-32)
- [The LDS-capacity invariant](#2026-10-19-lds)
- [Correctness](#correctness)
- [Reproduce](#reproduce)
- [History index](#history)

<a id="what-won"></a>
## What actually won

Most of this document records closures. This is the short list of changes that
were **adopted**, with the measurement that justifies each, and the resulting
movement in the standing. Ratios are ours/hipblas (paired same-process, idle
card, `rel = 0`); lower is better.

| # | lever | measured effect | where |
|---|---|---|---|
| 1 | **Shape-aware chooser.** `decode_cfg` took only `M` and was tuned on gate_up, so o/down ran a tile chosen for a different shape. | o M=1 42.9 → 38.6 µs (1.188), o M=16 39.2 (1.129); down M=1 117.4 → 99.4 (1.273), down M=16 100.3 (1.187) | `decode_cfg` (log §2026-10-17f) |
| 2 | **qkv M=32 window** `bn32/bk128/t128`. | 52.3 vs 57.8 µs (+9-10%); qkv M=32 1.14 → 1.03 | `decode_cfg` (log §2026-10-17g) |
| 3 | **Dense MLS packing.** A 16×256 fp8-trans dst tile is exactly 4096 B, but the 4K-slot accounting charged 4096 (slot) + 2×1024 (payload) = 6144 B — 50% dead padding. `MlsPackedTileBytes` now returns −1 (default allocation) whenever the dense size is a whole number of slots. | 6144 → 4096 B per block-stage | `src/hcu/op/mls.cc` (log §2026-10-20) |
| 4 | **Fat tile** `bn96/bk256/ns2` for N=5120 — the widest tile that still fits two stages in one CU (`2·(16+96)·256 = 57344 ≤ 65536`). | 0.947/0.940/0.943 on o and 0.928/0.925/0.929 on down at M=1/16/32, vs the `bn16/t64` incumbent | `decode_cfg` (log §2026-10-20) |
| 5 | **Warp count 6 → 3** (t384 → t192) at bn96. A's k-loop loads carry no warp-id term, so A is re-read once per warp (N/16 warp-loads, an invariant of the tiling); three warps is the measured optimum of the U. | 0.952–0.962 vs t384 | `decode_cfg` (log §2026-10-20b) |
| 6 | **Unrolled k-loop** (`k_unroll`, U6/U8). `T.Pipelined` addresses the shared stage symbolically, and `(k+d) % 3` lowers to `v_mul_lo_u16`/`v_sub_u16`/**`v_readfirstlane_b32`** on the `matrix_load` address path — 13% of the kernel. Unrolling makes every stage index a constant and lets the next step's address hoist above the MMA. | down/o M=96 **195.8 → 155.2** / **71.4 → 58.7 µs** (0.79/0.82× of the pipelined loop); M=64 **160.3 → 127.0** / **57.0 → 49.6** (0.925×/0.951× hipblas, was 1.164/1.099); worth 2-3% down to M=1 | `fp8_channelwise_gemm(k_unroll=…)` (log §2026-10-21) |
| 7 | **M>96 tile back to bn96**, now with t512 and U8. 54 blocks of 64 CUs hide the per-step `cp_async` wait better than 40, enough to pay 17% more bytes. | down M=128 **214.7 → 182.2 µs**, o M=128 **71.0 → 67.8** (0.849/0.955×; 1.02× hipblas, was 1.20) | `decode_cfg` (log §2026-10-21) |
| 8 | **The wide-N mid band is decided by block count, then MMA padding.** gate_up (400 blocks at bn128) takes ns2+U8 at every M>64; qkv (80 blocks) instead shrinks to `bm96/bn96` while M ≤ 96. | gate_up M=96/128 1.020/1.002 → **0.990/0.966**; qkv M=80/96 1.202/1.246 → **1.070/1.048** | `decode_cfg` (log §2026-10-21b) |
| 9 | **gate_up M in (64,96]: `bm96/bn128`.** bm128 spent 25-37% of its MMA rows on padding that still issues mmac; bm96 is a legal MLS tile at bn128, and `A_issued = ceil(N/bn)·M·K` has no `bm` in it, so the bytes do not move. | gate_up M=80 **293.3 → 285.5 µs**, M=96 **296.0 → 286.7** (−2.7% / −3.1%); shipped-path ratios **0.962 / 0.953**. The qkv `bn96` tile *loses* here (303.2, 1.016) | `decode_cfg` (log §2026-10-21d) |

Net movement (ratio ours/hipblas; `—` = not measured at that stage):

| shape | M | shape-aware (10-17f) | fat tile (10-20) | + t192 (10-20b) | final (10-20c) |
|---|---|---|---|---|---|
| o | 1 | 1.188 | 1.117 | 1.069 | **1.069** |
| o | 16 | 1.129 | 1.064 | 1.020 | **1.027** |
| o | 32 | — | 1.137 | 1.000 | **1.024** |
| down | 1 | 1.273 | 1.184 | 1.110 | **1.110** |
| down | 16 | 1.187 | 1.100 | 1.053 | **1.042** |
| down | 32 | — | 1.061 | 0.935 | **0.928** |

The last column is a fresh re-measurement after the 2026-10-20c de-replication
attempt was reverted, so 10-20b → final is run-to-run noise (~1%), not a change.
Reading the `down` row: 27% behind at M=1, now **7.2% ahead** at M=32; `o` M=1
went from 18.8% behind to 6.9%.

The band the chooser had never won, M in (64,128] (2026-10-21, per-M, paired):

| shape | M=65 | M=80 | M=96 | M=112 | M=128 |
|---|---|---|---|---|---|
| o | **0.941** | **0.939** | **0.937** | 0.995 | 1.047 |
| down | **0.882** | **0.891** | **0.887** | 1.056 | 1.002 |

Before this round the same cells were 1.12-1.23× (pipelined loop) and
1.25-1.29× (2026-10-20e).

### The mechanisms worth remembering

1. **A is replicated per warp, not partitioned.** The SASS A-load address has no
   warp-id term, so A's total re-read is `blocks × warps = N/16` regardless of
   the tile — widening bn never cuts it, and the warp count at fixed bn is the
   real lever. (log §2026-10-20b)
2. **Prefetch depth is the currency; LDS is where we pay for it.** hipblas pays
   depth in VGPRs (free), we must pay in LDS, and both come out of the same
   budget — which is why every ns / split-K / bk sweep is neutral against
   `total = const × w × (L + C·ns)`. ([2026-10-19](#2026-10-19-lds))
3. **Making a per-warp load conditional silently shrinks the cp.async group
   count**, because `cp_async_wait<k>` is sized from the *unconditional* loads on
   the path. That is why producer A de-replication lost a prefetch level and
   1.04–1.36× despite saving 22% of per-step bytes. (log §2026-10-20c)
4. **The real DRAM read ceiling is 1725 GB/s, not 2016.** The earlier 2016 figure
   was a timing artifact; against 1725 the MLS B-stream is already at 97.5%, so
   the residual gap is exposed latency, not bandwidth. (log §2026-10-18b)
5. **Every loop construct that is not a constant is paid on the load-address
   path.** A `(k+d) % 3` shared-stage index costs 13% of the kernel
   (`v_readfirstlane_b32` serialises the wavefront), and the fix is unrolling the
   loop, not raising the stage count. Corollary: the `cp_async` counter is a FIFO,
   so "cover B two steps ahead" is inexpressible — the wait that would do it also
   forces A(k+2). (log §2026-10-21)

<a id="closed-levers"></a>
## Closed levers — index

One row per lever that has been measured and closed, with the decisive reason and
the dated section that decides it. "Closed" means measured negative, not merely
untried; do not re-open any of these without new evidence.

| lever | decisive reason | section |
|---|---|---|
| **split-K** | The motivating block-starvation diagnosis was a confound; measured **10-21% slower**, and the MLS K-slice offset is silently wrong for any nonzero base, so it is not expressible at all. | 2026-10-17f (log), 2026-10-18d (log) |
| **`num_stages > 1`** | At bn16, re-timed correctly, **29-45% slower**; where it is exact (`bk512/ns2`) it is worth ~1%, and the small-tile packed stage stride is a second, independent defect. | 2026-10-17i (log), 2026-10-18d (log) |
| **B-VGPR-direct / rolling-window producer** | B-direct is **2.1× slower even at `WarpM=1`**; the 16-consec-k rescue is mmac-incompatible and arithmetically insufficient (93.9 µs available vs a 267.6 µs gap). The per-lane rolling window is a multi-warp technique, measured closed on the producer axis. | B-direct (log §b-direct) |
| **MMA density / hipblas warp mapping** | `v_mmac` costs **0.5%** — the MMA itself is free. Copying hipblas's 54-block, 8-warp, N-tile-96 structure through MLS is worse at every cell (adding warps to a wide tile loses blocks/CU), and the largest remaining term is the transposing LDS read, **9.3%**. | [2026-10-19 (b)](#2026-10-19-hipblas-launch) |
| **MLS producer atom / partition swap** | The 128×16 atom is ~20% slower standalone than 64×32 (1600 vs 1892 GB/s), but swapping to the faster atom makes the full GEMM **15-16% slower** (287.2 vs 246.2 µs): the kernel is overlap-quality-bound, not producer-throughput-bound. | MLS producer (log §mls-producer) |
| **N-split / widening bn** | N-split is **2-3× worse** and A's LDS read scales with the warp count. At t=64, wider bn is 1.077/1.123 (bn32) and 1.191/1.340 (bn64) at o M=16 / down M=16: the loss is blocks, not A's duplicated LDS read. | N-split (log §n-split), 2026-10-18c (log) |
| **bn > 128** | bn160 is **not a legal MLS tile** (`No valid MLS tile config for trans: block_mn=160`); bn256 forces bk64 or ns1, measured **1.48-1.59×** slower. bn ≤ 128 is the hard cap of this path. **Partially retracted 2026-10-23:** `bn192` is a legal MLB tile at `bk128/t512` (bm64) and `bk64/t384` (bm96) and wins on the shapes it is gated to, so 128 is a cap of the *tried* combinations, not of `bn`. What stays closed is `bn ≥ 256`. | 2026-10-21b (log), 2026-10-23 (log) |
| **Wave packing (bm split, bn80)** | Buying whole waves by halving bm is **1.13-4.19×** worse (the MMA-efficiency loss dominates), and bn80 for an exact 1.00/2.00-wave grid is 1.03-1.09× worse (A's issued traffic is `N·M·K/bn`, so a narrower bn pays more than the tail saves). | 2026-10-21b (log) |
| **Layout A (out of scope)** | The 71.6% "70% target" is reached only via Layout A (blocked weight reorder), which is excluded; hipblas's own no-reorder baseline is **213.0 µs = 68.4%**. | [Layout A](#layout-a) |
| **Producer A de-replication** | Bit-exact but slower everywhere (**1.044-1.358×**). `cp_async_wait<1>` sizes its vmcnt constant from the unconditional async loads, so putting A behind `if (owner)` drops a prefetch level; A's duplicates are L2-served and were never on the critical path. | 2026-10-20c (log) |
| **M=1 GEMV structure** | Not expressible: the MMA output micro-tile is hardware-fixed at 16×16 and the MLS B atom floor is `tile_mn=16`; the reachable split-K equivalent was measured and rejected. | M=1 GEMV (log §gemv-m1) |
| **`warp_k` / `FullColK`** | Built and correct, but **0.97-1.03** vs shipped with mixed sign; it is a *substitute* for `setprio`, not a complement, and `decode_cfg` does not select it. | 2026-10-19 (log §2026-10-19-warpk) |
| **`__builtin_hcu_matrix_load_b8` (global→VGPR)** | The builtin exists, but its fixed per-lane layout is 4 consec-k × 4 n, not the mmac's 8 consec-k at one mn; bridging needs a full cross-lane permutation. | 2026-10-19 (log §2026-10-19-matrixload) |
| **A-direct-to-VGPR / `r_mls`** | A must round-trip through LDS: `r_mls` is **1.935×/2.592×** slower (o/down M=16), and the MMA fragment wants 8 consecutive K at one row, a 5120-byte lane stride. | 2026-10-18c (log) |
| **Block-count inflation / occupancy** | More blocks at constant bytes is worse, and **320 blocks is the best point** on both curves; the ~8× blocks that reach the ~0.98× o M=1 bound cannot be supplied, and 320 is exactly 5/CU against an 8-slot LDS residency limit. | 2026-10-17f (log), [2026-10-19](#2026-10-19-lds) |
| **Independent accumulator chains** | Splitting the single 16×16 chain is **1.539/1.524/1.644/1.641×** slower (o M=1/16, down M=1/16); a second accumulator over the same tile is neutral (**1.012/1.008/1.014/1.013×**). | 2026-10-17i (log), 2026-10-18b (log) |
| **Hand-written double buffering** | Correct (rel 0.00000) but **1.26-1.70× slower** with a symbolic stage index; the shipped `T.Pipelined` loop already emits the same two barriers. **The lever was the modulo, not the scaffolding**: unrolled so the index is constant, the same hand-written body is **0.79-0.82×** of the pipelined loop. | 2026-10-17h (log), 2026-10-21 (log §2026-10-21-d10x) |
| **Deeper cover (B two steps ahead, `ns>2`, split depth)** | **174.56 vs 155.20 µs** re-measured with the modulo removed, so the earlier result was not an artefact of stage-index arithmetic; `A3B2` 156.00, and an `A2B3` arm addressed `%2` (footprint isolated) 155.68. The `cp_async` counter is FIFO: a wait that covers B(k+1) also forces A(k+2), so B can never lead by more than one step. Footprint alone is free up to 65536 B. | 2026-10-21 (log §2026-10-21-d10x), 2026-10-20g (log) |
| **Tile widening** | Monotonically catastrophic in the early sweep (`bn64/128/256`), and `bn32` alone is **7-12% worse** at every (shape, M). At M=64 the whole neighbourhood loses to the shipped `bn64/bk128/ns2/t256`: depth (`ns3`/`ns4`) and width (`bn96`/`128`/`192`) are 1.13-1.87×, and the exactly-64-block `bn80` is 1.177×. `gate_up` M=64 reaches the same conclusion from 11 arms (1.059-1.070×), and its most instructive one is `bn128`: halving A's issued traffic drops co-residency from 2 blocks/CU to 1 (32768 → 49152 B) and loses **2.5%**. | 2026-10-17f (log), 2026-10-18d (log), 2026-10-20d (log), 2026-10-21d (log) |
| **Wave quantization (M=64)** | 80 blocks on 64 CUs is real but not the cost: the perfectly balanced 64-block point is *slower* (bn80 1.177×; N=4096 the slowest of 64/80/96/128), and the N-ladder is confounded by total bytes. No grid-shape knob exists to test co-residency cleanly. | 2026-10-20d (log) |
| **`k_pack=2`, `b_ds_read_alt=0`, `respect_layout_map`** | Hard-rejected by the MLS path / unsupported (only Alt1 exists) / exact no-ops respectively. | 2026-10-17h (log) |

<a id="setup"></a>
## The setup

- B is (N, K) fp8 = 51200 × 5120 × 1 B = **262 MB**; at the measured **1725 GB/s**
  read ceiling (log §2026-10-18b) a single read of B floors at
  ≈ 152 µs. (The earlier 1646 and 2016 GB/s figures are retired.)
- hipblaslt (`torch._scaled_mm`, rowwise, bf16 out) measured with the same
  graph-replay timing: M=1 174.9 µs (83.3%), M=2 68.1%, M=4 73.4%,
  M=8 75.2%, M=16 76.4%, M=32 65.4%.
- Our initial MLS GEMM (block 256×256×64, ns=1, t512): M=1 198.7 µs (73.3%),
  M=2 73.4%, M=32 67.5% — M=2 and M=32 beat hipblas, the others lost.

<a id="bw-model"></a>
## The bandwidth model

Every decision below was scored against one objective, and it is easy to get it
wrong: three different "bandwidth" numbers get quoted for the same kernel, and
only one of them binds.

```
B-only    N·K / t                        the repo bench's convention
DRAM-min  (M·K + N·K) / t                what memory *must* serve
issued    (ceil(N/bn)·M·K + N·K) / t     what the load path is *asked* for
```

The measured read ceiling on this part is **1725 GB/s** — the 1800 / 2016 / 2106
figures in older notes are retired (the 2016 one was ~17% too high; see the
history entry for 2026-10-18b).

- **Time tracks *issued* bytes**, at a flat ~1.5–2.0 TB/s across the whole
  standing table, while DRAM-min falls 2–3× from M=1 to M=128. The number that
  collapses with M is not the one the hardware is limited by, so "we are only at
  40% of DRAM" is not a deficit statement.
- **A's duplicates are L2-served.** They cost issue slots, not DRAM. hipblas's
  own DRAM utilisation falls the same way (o 78/70/63/48/40/39%, down
  97/90/70/58/44/44% over M=1..128) because it re-reads A per n-block too — 54×
  at M=16.
- **`bm` cancels out of `A_issued`.** With `A_issued = ceil(N/bn)·M·K`, shrinking
  `block_M` is byte-free and **`bn` is the only lever on A**. `bn ≤ 128` is a
  hard MLS cap (bn160 does not build), which is why the A-traffic story runs out
  of room early.
- **At M ≤ 16 we are at the ceiling** (gate_up M=1 is 97% of 1725 GB/s); what is
  left there is launch-fixed cost, not stream efficiency. At M=128 the remaining
  gap *is* the issue rate: down 1693 vs 1755 GB/s (3.7%), o 1463 vs 1535 (4.7%).

Two consequences that shaped the whole search: the lever list is short (tile
shape, co-residency, k-loop structure), and **the tile that wins is the one that
maximises co-residency at the smallest `bn` that still keeps the issued-byte
count down** — which is why the [LDS-capacity
invariant](#2026-10-19-lds) keeps coming back as the binding constraint.

<a id="hipblas-m1"></a>
## What hipblas does at M=1 *(compressed)*

An early disassembly of the M=1 kernel listed `buffer_load_dwordx4` **with
`slc`**, `s_setprio 1/0` around the MMA, the k+1 load issued at the top of the
loop, `s_waitcnt vmcnt(4)` before consuming a tile, and `ds_read_b128` LDS reads.

**All but one of those bullets were retracted** when the *native* kernels became
available: neither native kernel has `slc` on any load (the `slc` in the delivered
`.s` files was added by the delivery patch — see the provenance note in
[2026-10-19 (b)](#2026-10-19-hipblas-launch)), and the in-loop `vmcnt(4)` is
likewise a hand-edit of the native `vmcnt(0)`. What survives is `s_setprio`
(it is our single largest lever at bn16 — 7-11%, 2026-10-19 (log §2026-10-19-warpk))
and the load-ahead structure, which at M≥64 hipblas pays for in VGPRs, not LDS.

<a id="m-le-16"></a>
## M ≤ 16 — the bn16/t64 era *(compressed)*

> **2026-10-13 re-verification:** all utilization numbers in this section were
> re-measured with explicit CUDA-graph capture + replay, events, interleaved
> A/B/A/B, min/med of 5×60 (tilelang `do_bench_cudagraph` returns ms on this
> stack / single-launch events carry host overhead — both inflate utilization).

This section is kept for two things only: the tile search bottomed out at
`bn16/bk256/ns1/t64/sp1`, and that answer is the optimum of the in-tree search
space. Every "% of 1800 GB/s" figure below is retired — the measured read
ceiling is 1725 GB/s.

<a id="m-le-16-ceiling"></a><a id="m-le-16-streams"></a>
The two ceiling probes it once carried (1646 GB/s on a 512 MB read; 2016 GB/s on
5.12 GB) were both superseded (log §2026-10-18b). What they did
establish is that the kernel really streams all of B — at bn32/bk256/t256 the
latency scales linearly with B bytes (N = 12800/25600/51200/102400 → 68.8 /
96.6 / 186.7 / 376.3 µs at M=1, against hipblas 69.3 / 114.6 / 180.0 / 409.8).
The quantitative version of that is the bandwidth audit in
2026-10-21b (log).

<a id="m-le-16-h2h"></a>
### The final M=1 sweep (paired, idle card)

Full o/down sweep at M=1/M=16 (`/tmp/final_sweep.py`). o M=1, best first:

| config | µs | vs hipblas |
|---|---|---|
| **bn16 bk256 ns1 t64 sp1 (chosen)** | **38.6** | **1.048** |
| bn16 bk256 ns1 t128 sp1 | 38.7 | 1.052 |
| bn32 bk256 ns1 t128 sp1 | 41.0 | 1.113 |
| bn16 bk256 ns1 t64 **sp0** | 41.3 | 1.122 |
| bn16 bk512 ns1 t64 sp1 | 45.1 | 1.226 |
| bn16 bk128 ns1 t64 sp1 | 58.7 | 1.596 |

`setprio=1` is worth 6.5% and is required; `threads` 64→128 changes nothing (the
16×16 atom is one warp of work either way); bk 128 and 512 are both much worse.
**The chooser's answer is the optimum of the in-tree search space.**

`torch._scaled_mm` on o M=1 read 31.5, 36.8 and 32.5 µs across three runs while
our kernel moved only 38.9 → 38.6 — **hipblas is the noisy term**, and a crowded
process (18 resident CUDA graphs) inflates it by ~17%. A dedicated 12-round
alternating A/B with only two graphs resident is stable to ≤5%:
o M=1 = 1.188 [1.182, 1.197], down M=1 = 1.273 [1.270, 1.273]
(`/tmp/ratio_stability.py`). Quote those; do not read a ratio off a sweep process
holding many candidate graphs.

<a id="why-flat"></a>
## Why M ≤ 16 was flat, and why M=32 dropped *(compressed)*

- Grid is fixed (N=51200, bn=32) → M ≤ 16 only changes how many of the bm=16
  fragment rows are live, not the grid size or per-block bytes.
- At M=32, bm=32 halves the block count and doubles fragment work per block;
  MMA pressure starts to interfere with the streaming path (best measured
  config (32,32,256,1,256): 246 µs ≈ 59%).
- Both readings are superseded: at bn32 the M=32 loss was really the thread
  count (one warp per block — fixed with t=256, 2026-10-17f (log)),
  and the shape that "drops" is decided by the tile, not by M.

<a id="m-ge-32"></a>
## M ≥ 32 ≤ 128: the streaming kernel breaks down *(compressed)*

For M ≥ 32 the bm=pow2≥M tiles inflate: bm=64 → A tile is 64×256×1B = 16 KB
vs B's 32 KB (bn=32); bm=128 → 32 KB A vs 16 KB B per block. A is pure
redundant reads (A is 6 MB; rereading it 800× doesn't cost bandwidth but adds
LDS/MMA work), and MMA becomes the bottleneck: a 128×128×256 fp8 tile at
432 TFLOPS peak needs ~9.7 µs of MMA per block vs ~6.7 µs of B streaming —
compute-bound. Measured M=32 with the M≤16 winner shape: 246 µs ≈ 59%
(hipblas ~65%), i.e. the streaming win does not transfer. The A-reread reading
survives as "A costs LDS writes" ([m64-root-cause](#m64-root-cause)); the
LDS-*read* half of it does not (N-split (log §n-split)).

<a id="m64-root-cause"></a>
### M=64 root cause (measured 2026-10-16): A's LDS write traffic *(compressed)*

Pinned-address controls (same instruction stream and LDS traffic, only addresses
changed):

| variant | time |
|---|---|
| V0 baseline `bm64 bn64 bk256 ns1 t256 sp1` | 246.9 µs (59.0%) |
| V1 A staged once (A's per-iteration load removed) | **217.0 µs (67.1%)** |
| V8 A pinned to one 16 KB, so A stays L2-resident | 246.1 µs |
| V9 B pinned to one 16 KB, so B's 262 MB of DRAM reads vanish | 239.0 µs |
| W2 pure compute (A_s/B_s filled once, bare `T.gemm` loop) | 114.1 µs |
| hipblas `torch._scaled_mm` | 226–232 µs |

- **V8 == V0** → A's cost is not L2 capacity or L2 bandwidth.
- **V9 saves only 8 µs for 262 MB of DRAM reads** → the DRAM is essentially idle;
  262 MB / 246 µs = 1065 GB/s = 59%, and removing it entirely does not help.
- **V8 vs V1 differ only by A's `matrix_load` and its 262 MB of LDS writes** → 29 µs.
  262 MB / 29 µs = 9.0 TB/s, matching LDS write bandwidth
  (64 CU × 128 B/clk × 1.3 GHz = 10.6 TB/s).

A's LDS write traffic is `M·N·K / bn` — **independent of `bm` and of `bk`** — so
only `bn` (N-coverage per block) reduces it, and `bn=64` is the floor for the fast
MLS path: wider tiles change the MLS warp partition and raise LDS *read*
amplification from 2× to 3–4×, which costs more than the A writes saved (ten
wide-bn configs measured, all worse than the 245.9 µs baseline). There is no
register-destination `matrix_load`, so A cannot skip the LDS round trip via the
MLS path; the only A→VGPR route is `r_mls`, a proven 3.7× loss. This supersedes
the earlier split-K / fold-M suggestions for M=64, which target a bandwidth wall
that measurement shows is not there.

<a id="m64-paired"></a>
### M=64 is LDS-port-bound; the gap is overlap quality *(compressed)*

**Measure paired, or don't measure.** HCU[4] is shared with co-tenants, so
absolute times drift up to ~15% between invocations and the *ordering* of two
candidate configs flips. Build both kernels in one script, interleave the rounds,
report the ratio. Idle-card paired: hipblas `MT96x64x256`-class **224.2 µs
(64.9%)** vs ours **247.0 µs (58.9%)** — **10.2%, stable to ±0.2% over 8 rounds**
(the same comparison read 5.8% under co-tenant load, because contention hurts
hipblas's deeper queue more).

**LDS volume is not the differentiator — hipblas moves exactly the same bytes.**

| | LDS bytes / output / k-step |
|---|---|
| ours `bn64 bk256 t256` | 160 KB / (4096 x 256) = **0.15625** |
| ours `bn64 bk128 ns2 t256` | 80 KB / (4096 x 128) = **0.15625** |
| hipblas `MT128x64x256` | 320 KB / (8192 x 256) = **0.15625** |

Identical, and invariant across `bk`. The 4× read amplification over the tile's
unique bytes is entirely the per-warp operand duplication the mmac layout
requires: the `ds_read` volume (512 B/lane) exactly equals the mmac operand volume,
so **there is no recoverable waste**. Load-only arms: B stream alone 166.1 µs
(95.9% of the then-ceiling), A+B through LDS 196.0 µs (A's round trip costs
**+30 µs** for a 327 KB matrix), full kernel ~247 µs (~51 µs of compute exposed).
The load path is not the problem; A's `global → LDS → VGPR` round trip is the
largest overhead we own and is **not recoverable without the excluded weight
reorder**. That leaves **overlap quality** as the residual.

<a id="m64-sass"></a>
### hipblas M=64 k-loop vs ours — SASS *(compressed)*

Source of truth: the native, unpatched kernels, never the delivered `.s` files
(they carry Layout A / slc / vmcnt4 patches, and their README warns the patches
are per-M and cannot be copied across kernels). Extract with
`llvm-objdump -d --mcpu=gfx938 /opt/dtk/lib/hipblaslt/library_bw_nmz/FusedMatmulKernel_gfx938.co.bak_0916_1000`.

The native M=64 kernel (`MT96x64x256_TT6_2_WG16_32_1_MFWGS512_WGM8`) settles every
operand-role question:

```
buffer_load_dwordx4 v[96:99],   v[184:185], s[12:15], 0 idxen offen
...
v_mmac_f32_16x16x32_fp8_fp8 v[0:3],  v[32:33], v[96:97]
```

— the destination block `v[96:127]` of the `idxen` global loads **is literally the
third mmac operand**: B never touches LDS, A comes from `ds_read_b128`. The B
index is built from `lane%16` alone with **no M coordinate**, so the four waves
split **N**, the fetch is issued once (no redundancy), and one 8-byte B operand
feeds 18 mmacs. Ours reads both operands from `ds_read_*`. **That is the whole
trick: put all of M in one wave and split N across waves** — and it is the door
this repo measured shut twice (B-direct (log §b-direct)).

<a id="async-wait"></a>
### Async-wait granularity — the lever, measured (2026-10-17d) *(compressed)*

**Hypothesis (refuted):** the gap is hipblas's "fine-grained `vmcnt(4)` waits"
versus our coarser `cp_async_wait`. **Dead on the native SASS:** hipblas's k-loop
uses **only wait-all** synchronisation (`s_waitcnt lgkmcnt(0)` ×8,
`s_waitcnt vmcnt(0)` ×4 per k-tile); the `vmcnt(4)` in `down_readable.s` is the
hand-edit. So hipblas's in-loop waits are *coarser* than ours.

The knob that does exist is `num_stages`: `bk128/ns2` beats `bk256/ns1` by **1.2%
(243.5 vs 246.6 µs)**, and the gain is cross-k-step overlap, not wait granularity
(at fixed bk256, ns=2 is 1.6% *slower*). Guard found here: `bk=192` compiles and
runs but is **silently wrong** (`rel=1.61`) — `5120 % 192 = 128`, so the pipelined
`T.copy` runs 64 elements past the end of B. The builder now asserts
`K % block_K == 0` as well as `block_K % 64 == 0`.

<a id="layout-a"></a>
### The "70% target" is Layout A — which is out of scope *(compressed)*

| M | kernel | tile | 优化 | 需权重重排 | 基线 → 优化 |
|---|---|---|---|---|---|
| 64 | gate_up | MT128x64x256 | **Layout A + vmcnt4** | ✅ | 213.0 → **203.3 µs** |
| 64 | down | MT96x64x256 | slc + W2 | ❌ | 131.8 → 117.7 µs |

203.3 µs / 262.1 MB = 1289 GB/s = **71.6%** — that is the 70% figure, and it is
reached **only** via Layout A (blocked weight reorder), which is excluded by
standing instruction. Without the reorder hipblas's own baseline is **213.0 µs =
68.4%**, so the in-scope gap was ours 247.0 µs (59.0%) → 213.0 µs (68.4%): **34 µs**,
not 44 µs. "A staged once" (V1 = 217.0 µs) is not reachable — staging A two
k-steps at a time into two separate buffers gives 249.3 µs (same instruction
count, same LDS bytes; the cost is per-byte), and A's LDS write is `M·N·K/bn`,
invariant under `bm`, `bk`, split-K and A-hoisting. hipblas's tile shape does not
transfer either: `bm128 bn64 bk256 t512 ns1` copied into our DSL gives **589.8 µs**
(partition degrades to (4,2)).

<a id="2026-10-19-lds"></a>
## The LDS-capacity invariant

**1. The occupancy cliff** (same kernel, only the dynamic smem size varied), down
M=16, `<<<320,64,smem>>>`:

| smem | blk/CU | µs |
|---|---|---|
| **12288** | **5** | **93.7** |
| 14336 / 16384 | 4 | 168.9 / 169.1 |
| 20480 | 3 | 166.8 |
| 24576 / 32768 | 2 | 219.7 / 219.7 |

5→4 blocks/CU costs **80%**. The shipped config sits at exactly 320 blocks =
64 CU × 5 = **one wave**, on a knife edge.

**2. The invariant.** Normalizing every occupancy point by `waves × iters` gives a
per-warp iteration time that is essentially occupancy-independent (~0.88–1.35 µs):

```
LDS/block = (A_logical + B_logical) × w × ns        w = MLS packed / logical
blk/CU    = 65536 / LDS
waves     = (N/bn) / (64 × blk/CU)
total     = waves × (K/bk) × T,   T = L/ns + C
          = const × w × (L + C·ns)                 (bk and ns cancel)
```

- `L ≈ 0.79 µs` = one DRAM round trip per warp-iteration (1350 cycles @1.5 GHz).
- `C ≈ 0.15 µs` = the exposed `ds_read_matrix_trans_format ×8 → mmac ×8` chain.
- `w` is set by the 4096-byte MLS slot stride: **bk128 → 2.5, bk256 → 1.5,
  bk512 → 1.0**.

Predicted vs measured: bk64/ns1 282/292.5, bk128/ns1 156/144.4, bk256/ns1
94/93.7 — all ✓. **More stages ⟺ fewer blocks/CU ⟺ more waves, exactly
cancelling** — which is why `ns>1` and split-K were always neutral.

**3. SASS root cause of `C`.** The four `matrix_load_128x16_b8` producers are
emitted after the mmac and the next iteration opens with
`s_waitcnt vmcnt(0) lgkmcnt(0)`, so the whole global-load latency is exposed every
iteration. **4. Decomposition** (ratios to the same-run rotated kernel): rotate
producer above the mmac **0.965** (3.5% win, reproduced 3×); + drop mmac 0.906;
+ drop B's ds_read 0.926; + drop A's ds_read 0.983. Removing mmac alone (9.4%),
ds_read alone (9.3%), both (9.8%) ⇒ **one serialized chain, not two costs**.
**5. Closed this session:** `ns=2` at every bn (LDS doubles, blocks/CU collapses);
wider bn at ns=1 (103.4 → 114.4 → 140.6 → 173.9 for bn16/32/64/96); `bk128/ns2`
(the stages alias, and patching 1024→5120 in /tmp did not restore `rel`); warps
per block (t128/WK2 97.0 vs t64 99.2, t256/WK4 100.2 — within noise).

**6. The one landed-quality improvement** (not taken): rotating the MLS producer
above the `promote_prio`/mmac block is worth ~3% consistently, but the producer is
placed by the MLS lowering, not by a config flag.

<a id="2026-10-19-hipblas-launch"></a>
### What hipblas actually launches (2026-10-19b)

| N | M | K | µs | GB/s | util | kernel |
|---|---|---|---|---|---|---|
| 5120 | 16 | 25600 | **85.0** | 1463 | **86.0%** | `MT96x16x256_TT12_2_WG8_8_4_MFWGS256_WGM1` |
| 5120 | 1 | 25600 | **79.4** | 1621 | **91.7%** | `MT8x1x8192_MFWGS512_WGM1_ONLYN1` (GEMV) |
| 5120 | 32 | 25600 | 107.3 | 1221 | 68.5% | `MT96x64x256_TT6_2_WG16_32_1_MFWGS512` |
| 5120 | 64 | 25600 | 114.7 | 971 | 64.6% | `MT96x64x256_TT6_2_...` |

Name decode for the M=16 decode kernel: **N-tile 96, M-tile 16, K-chunk 256;
256 threads = 8 warps; ThreadTile 12×2; 4 threads cooperate along K.** Grid =
5120/96 → **54 blocks**, i.e. 0.84 blocks per CU — hipblas does *not* use
occupancy here. Consequences vs us (bn16/t64, 320 blocks, 1 warp each): A is read
**54× instead of 320×** (hipblas moves 153 MB through L2 per k-step, we move 262),
8 warps per block so one warp's compute overlaps another's loads, and at **M=1 it
abandons the MMA path entirely** for a GEMV kernel (N-tile 8 → 640 blocks, 512
threads) reaching 91.7%.

Copying that structure does not work through the MLS path: bn64/t64 135.2, t128
135.4, t256 161.8, bn32/t128 115.8 against the shipped 99.2/99.8 at down M=1/M=16.
Adding warps to a wide tile makes it *worse* — the packed MLS B tile costs 16384 B
at bn64/bk256, so the block is 22528 B → 2 blocks/CU, and the `(1 + 16/bn)`
A-amortization gain is smaller than the blocks/CU loss. hipblas can afford bn96
because it only launches 54 blocks.

**Where the residual actually is:** `v_mmac` costs **0.5%** — the MMA itself is
free; `ds_read_matrix_trans_format` costs **9.3%**, the single largest term; the
rest is exposed global-load latency.

<a id="correctness"></a>
## Correctness — vs hipblas fp8, not fp32

Because the M=1 kernel runs with bm=16 (only M=1 rows of the fragment are
live), correctness is verified directly against the hipblas fp8 rowwise
operator (`torch._scaled_mm`, same A/B/scales, bf16 out), not against an
fp32 reference:

```
M=1   rel vs hipblas fp8: 0.003099   (deterministic bit-exact across runs)
M=2   rel vs hipblas fp8: 0.002928
M=4   rel vs hipblas fp8: 0.002353
M=8   rel vs hipblas fp8: 0.002127
```

Both sides round the fp8 accumulator to bf16, so rel ~0.002–0.003 is
fp8-vs-fp8 accumulation-order noise, not an error. A streaming reduce over the
same 262 MB B runs at only 8.6% — a reduce does not saturate HBM, so it says
nothing about whether a streaming GEMM can. Every shipped `decode_cfg` cell is
pinned against a chunked fp32 reference at `rel = 0` in
`testing/python/hcu/test_hcu_gemm_fp8_channelwise.py` (28 of 28 cells).

<a id="reproduce"></a>
## Reproduce

```
HIP_VISIBLE_DEVICES=2 python perf/gemm/fp8_channelwise_decode_bench.py
```

Verified tables above from `perf/gemm/fp8_channelwise_decode_bench_r_mls.py`
(head-to-head, graph replay) and the re-verification probes in
`/tmp/verify_bw.py`, `/tmp/m1_careful.py`, `/tmp/final_head2head.py`
(graph-replay, interleaved). **Timing must use explicit CUDA-graph capture +
replay**: tilelang `do_bench_cudagraph` returns ms on this stack and
single-launch event timing carries host overhead; both give inflated
"utilization" numbers that exceed the measured read ceiling and are not physical.
The harness also carries ~10.3 µs of event-pair overhead in *every* arm including
hipblas, so read ratios and differences, never absolutes across processes.

<a id="history"></a>
## History index

One row per experiment, newest last. The row is the verdict; the log entry
(`fp8_decode_gemm_log.md` in the llm-adaptation repo, same anchor) is the
evidence — arm tables, per-round paired ratios, SASS. Rows tagged **adopted**
changed the shipped kernel; **closed** means the lever is done and the bound
that makes re-opening pointless is in the log.

| date | what it settled | verdict |
|---|---|---|
| <a id="h-b-direct"></a>2026-10-16 → 17 | B-direct-to-VGPR: 4× m-warp redundancy and the 8-consec-k half-sector penalty are independent and each fatal; the full GEMM is 2.5–3.2× slower where re-tested (o/down M=1 and M=16), including at `WarpM=1`. The native SASS does put B in VGPRs, so this is a structural difference, not a bug we could inherit. | closed |
| <a id="h-2026-10-17f"></a>2026-10-17f | Shape-aware chooser is a real win (o M=1 42.9→38.6 µs); split-K's block-starvation diagnosis was a confound and the lever is 10–21% slower; the M=32 thread count is worth 1.60–1.71×; `ns>1` is silently wrong outside `bm64/bn64`. | adopted + closed |
| <a id="h-2026-10-17g"></a>2026-10-17g | `qkv` M=32 wants `bn32/bk128/t128` (1.14× → 1.03×); earlier ratios had used a `setprio=0` baseline; `ns=2` at bn16 is 40/40 wrong or slower. | adopted |
| <a id="h-2026-10-17h"></a>2026-10-17h | The load path already beats hipblas on all four o/down cells (0.917–0.977), so the deficit is on the mmac side and scales with K; the packed-stage stride bug is root-caused and fixed. | refocus |
| <a id="h-2026-10-17i"></a>2026-10-17i | Re-timed, `ns=2` is 29–45% slower (the earlier "fast" candidates had been measuring garbage); independent accumulator chains are 1.5–1.6× slower; the deficit is the 16–20% where neither pipe is busy. | closed |
| <a id="h-2026-10-18"></a>2026-10-18 | N-split (widen `bn` at constant warp count) is closed at 2–3×; `ns>1` has a *second*, independent stage-index defect (rel ≈ 1119–3504 outside the four configs the old tests covered). | closed |
| <a id="h-2026-10-18b"></a>2026-10-18b | The "2016 GB/s ceiling" was ~17% too high — 1725 GB/s measured; accumulator splitting is neutral (1.012–1.014×), so the exposure is not the mmac chain. | corrected |
| <a id="h-2026-10-18c"></a>2026-10-18c | The exposure is volumetric and per-iteration (half gemm +4.0 µs, double +13.6 µs at o M=16); A-direct-to-VGPR is closed again at 1.9–2.6×. | closed |
| <a id="h-2026-10-18d"></a>2026-10-18d | Split-K is not merely slow but *inexpressible*: the MLS K-slice offset is silently wrong for any nonzero base. Double buffering is worth ~1%. | closed |
| <a id="h-2026-10-19-lds"></a>2026-10-19 | o/down's gap is an **LDS-capacity invariant**, not a missing knob — every `ns` / split-K / `bk` sweep is neutral against `total = const × w × (L + C·ns)`, and 320 blocks is the optimum because it is exactly the 5-blocks/CU residency limit. Carried forward as a standing section: [The LDS-capacity invariant](#2026-10-19-lds). | refocus |
| <a id="h-2026-10-19-warpk"></a>2026-10-19 | `warp_k`/`FullColK` is built and exact but 0.97–1.03× with mixed sign — a substitute for `setprio`, not a complement; the MLS `moffset` is a signed 10-bit immediate, so `block_K ≤ 512`. | closed |
| <a id="h-2026-10-19-matrixload"></a>2026-10-19 | `__builtin_hcu_matrix_load_b8` exists, but its per-lane layout is 4-consec-k × 4-n, not the mmac's 8-consec-k at one mn; bridging them needs a full cross-lane permutation. | closed |
| <a id="h-2026-10-20"></a>2026-10-20<br><a id="h-2026-10-20b"></a>2026-10-20b<br><a id="h-2026-10-20c"></a>2026-10-20c<br><a id="h-2026-10-20d"></a>2026-10-20d | The fat tile: `bn96/bk256/ns2` for N=5120 at M ≤ 32 (the first real gain on `o`/`down`). It did *not* cut A after all — warp count is what pays. Producer-side A de-replication is closed (making the loads conditional costs a `cp_async_wait` group of depth: 4–36% slower). The M=64 tile neighbourhood is closed at 11 arms. | adopted + closed |
| <a id="h-2026-10-20e"></a>2026-10-20e | M ∈ (64,128] had three defects at once; one tile change (`bm96/bn96/bk128/ns2/t384`) is 1.7×. | adopted |
| <a id="h-2026-10-20f"></a>2026-10-20f | The stream already beats hipblas in that band; what loses is unoverlapped compute, not load. | refocus |
| <a id="h-2026-10-20g"></a>2026-10-20g | The residual is a per-step *fixed* cost, not a per-byte one — so the lever is the k-loop, not the tile. | refocus |
| <a id="h-2026-10-20j"></a>2026-10-20j | "o's producer is 21% slower than down's" is amortisation; at M=1 the producer alone is the wall. Also retracts the "62 µs DRAM floor" (a DCE artifact). | corrected |
| <a id="h-2026-10-21-d10x"></a>2026-10-21 | Unrolling the k-loop: the per-step cost was an *address* chain (a symbolic stage index costs 13%). Worth 20–24%, deeper cover is falsified; it reverses the M>96 tile back to bn96 and takes the lever at M ≤ 64 without moving the tile. | adopted |
| <a id="h-2026-10-21b"></a>2026-10-21b | The bandwidth audit: time tracks *issued* bytes at a flat 1.5–2.0 TB/s, `bm` cancels, `bn` is the only A lever, `bn ≤ 128` caps it. Wide-N mid band splits by block count: gate_up takes `bm96/bn128`, qkv takes `bm96/bn96`. | adopted |
| <a id="h-2026-10-21c"></a>2026-10-21c | `qkv` M=80..128: the pursuit is closed — every remaining arm at that shape loses, and the wave-packing alternatives cost more A traffic than they save. | closed |
| <a id="h-2026-10-21d"></a>2026-10-21d | gate_up M=(64,96]: `bm96/bn128` removes 48/128 (M=80) and 32/128 (M=96) padded MMA rows at zero byte cost → 0.962 / 0.953. M=64 re-closed at 11 arms. | adopted |
| <a id="h-2026-10-23"></a>2026-10-23 | Three tiles lifted from the hipblasLt **tuner's** own decode kernels and gated on explicit `(N,K)` lists: `bm16/bn48/bk256` (5 of 6 shapes at M ≤ 16, 1.03–1.12× over the tile it replaces), `bm64/bn192/bk128` (N=34816/10240 at M ∈ (32,64], 1.04–1.16×), `bm96/bn192/bk64` (N=34816 at M ∈ (64,96], 1.03–1.09×). All bit-identical to the tile they replace. `bn192` is legal — it retires the `bn ≤ 128` cap for the (bm,bk,ns,threads) combos now in the table; `bn ≥ 256` stays closed. The e2e A/B is drift-dominated (10 cells up / 12 down vs 2026-10-22, hipblas itself 4–19% faster), but no dispatch-gate verdict flipped. | adopted |
