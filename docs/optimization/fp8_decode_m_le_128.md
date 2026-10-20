# fp8 Channelwise GEMM on HCU — decode regime (M ≤ 128)

**What this document is.** The measured record of the HCU (gfx938) fp8
channelwise decode GEMM: `o` (N=5120, K=8192), `down` (N=5120, K=25600),
`qkv` (N=10240) and `gate_up` (N=51200) at M ≤ 128, measured against hipblasLt
`torch._scaled_mm` (bf16 out, RowWise). It holds the paired ratios, the SASS
evidence, and the verdict for every lever that was tried. Sections are dated and
append-only; where a later section corrects an earlier one, it says so.

Follow-up to `fp8_channelwise_gemm.md` (the 200 → 300 → 432 TFLOPS three-stage
story). This note covers the **decode regime** (qwen3-32B per-token GEMMs:
M=1–128, K=5120, N=51200 gate_up), where the kernel is *bandwidth-bound*, not
compute-bound, and the goal is to beat hipblaslt's fp8 rowwise kernel on
effective bandwidth.

**Current shipped config** (`decode_cfg`, `perf/gemm/fp8_channelwise_mls.py`):
N=5120 (`o`/`down`) at M ≤ 32 → `bn96 / bk256 / ns2 / t192` with `setprio=1`;
`qkv` (N=10240) and `gate_up` (N=51200) keep `bn32` (the `qkv` M=32 point is
`bn32 / bk128 / t128`). M=64 and M=128 are unchanged (their earlier returns).
See [2026-10-20](#2026-10-20), [2026-10-20b](#2026-10-20b) and
[2026-10-20c](#2026-10-20c) below.

**Current standing** (paired same-process, idle card, `rel=0`; ratio =
ours/hipblas, **< 1 is a win**):

| shape | M=1 | M=16 | M=32 |
|---|---|---|---|
| o | 1.069 | 1.027 | 1.024 |
| down | 1.110 | 1.042 | **0.928** |

<a id="what-won"></a>
## What actually won

Most of this document records closures. This is the short list of changes that
were **adopted**, with the measurement that justifies each, and the resulting
movement in the standing. Ratios are ours/hipblas (paired same-process, idle
card, `rel = 0`); lower is better.

| # | lever | measured effect | where |
|---|---|---|---|
| 1 | **Shape-aware chooser.** `decode_cfg` took only `M` and was tuned on gate_up, so o/down ran a tile chosen for a different shape. | o M=1 42.9 → 38.6 µs (1.188), o M=16 39.2 (1.129); down M=1 117.4 → 99.4 (1.273), down M=16 100.3 (1.187) | `decode_cfg` ([2026-10-17f](#2026-10-17f)) |
| 2 | **qkv M=32 window** `bn32/bk128/t128`. | 52.3 vs 57.8 µs (+9-10%); qkv M=32 1.14 → 1.03 | `decode_cfg` ([2026-10-17g](#2026-10-17g)) |
| 3 | **Dense MLS packing.** A 16×256 fp8-trans dst tile is exactly 4096 B, but the 4K-slot accounting charged 4096 (slot) + 2×1024 (payload) = 6144 B — 50% dead padding. `MlsPackedTileBytes` now returns −1 (default allocation) whenever the dense size is a whole number of slots. | 6144 → 4096 B per block-stage | `src/hcu/op/mls.cc` ([2026-10-20](#2026-10-20)) |
| 4 | **Fat tile** `bn96/bk256/ns2` for N=5120 — the widest tile that still fits two stages in one CU (`2·(16+96)·256 = 57344 ≤ 65536`). | 0.947/0.940/0.943 on o and 0.928/0.925/0.929 on down at M=1/16/32, vs the `bn16/t64` incumbent | `decode_cfg` ([2026-10-20](#2026-10-20)) |
| 5 | **Warp count 6 → 3** (t384 → t192) at bn96. A's k-loop loads carry no warp-id term, so A is re-read once per warp (N/16 warp-loads, an invariant of the tiling); three warps is the measured optimum of the U. | 0.952–0.962 vs t384 | `decode_cfg` ([2026-10-20b](#2026-10-20b)) |

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

### The four mechanisms worth remembering

1. **A is replicated per warp, not partitioned.** The SASS A-load address has no
   warp-id term, so A's total re-read is `blocks × warps = N/16` regardless of
   the tile — widening bn never cuts it, and the warp count at fixed bn is the
   real lever. ([2026-10-20b](#2026-10-20b))
2. **Prefetch depth is the currency; LDS is where we pay for it.** hipblas pays
   depth in VGPRs (free), we must pay in LDS, and both come out of the same
   budget — which is why every ns / split-K / bk sweep is neutral against
   `total = const × w × (L + C·ns)`. ([2026-10-19](#2026-10-19-lds))
3. **Making a per-warp load conditional silently shrinks the cp.async group
   count**, because `cp_async_wait<k>` is sized from the *unconditional* loads on
   the path. That is why producer A de-replication lost a prefetch level and
   1.04–1.36× despite saving 22% of per-step bytes. ([2026-10-20c](#2026-10-20c))
4. **The real DRAM read ceiling is 1725 GB/s, not 2016.** The earlier 2016 figure
   was a timing artifact; against 1725 the MLS B-stream is already at 97.5%, so
   the residual gap is exposed latency, not bandwidth. ([2026-10-18b](#2026-10-18b))

<a id="closed-levers"></a>
## Closed levers — index

One row per lever that has been measured and closed, with the decisive reason and
the dated section that decides it. "Closed" means measured negative, not merely
untried; do not re-open any of these without new evidence.

| lever | decisive reason | section |
|---|---|---|
| **split-K** | The motivating block-starvation diagnosis was a confound; measured **10-21% slower**, and the MLS K-slice offset is silently wrong for any nonzero base, so it is not expressible at all. | [2026-10-17f](#2026-10-17f), [2026-10-18d](#2026-10-18d) |
| **`num_stages > 1`** | At bn16, re-timed correctly, **29-45% slower**; where it is exact (`bk512/ns2`) it is worth ~1%, and the small-tile packed stage stride is a second, independent defect. | [2026-10-17i](#2026-10-17i), [2026-10-18d](#2026-10-18d) |
| **B-VGPR-direct / rolling-window producer** | B-direct is **2.1× slower even at `WarpM=1`**; the 16-consec-k rescue is mmac-incompatible and arithmetically insufficient (93.9 µs available vs a 267.6 µs gap). The per-lane rolling window is a multi-warp technique, measured closed on the producer axis. | [B-direct](#b-direct) |
| **MMA density / hipblas warp mapping** | `v_mmac` costs **0.5%** — the MMA itself is free. Copying hipblas's 54-block, 8-warp, N-tile-96 structure through MLS is worse at every cell (adding warps to a wide tile loses blocks/CU), and the largest remaining term is the transposing LDS read, **9.3%**. | [2026-10-19 (b)](#2026-10-19-hipblas-launch) |
| **MLS producer atom / partition swap** | The 128×16 atom is ~20% slower standalone than 64×32 (1600 vs 1892 GB/s), but swapping to the faster atom makes the full GEMM **15-16% slower** (287.2 vs 246.2 µs): the kernel is overlap-quality-bound, not producer-throughput-bound. | [MLS producer](#mls-producer) |
| **N-split / widening bn** | N-split is **2-3× worse** and A's LDS read scales with the warp count. At t=64, wider bn is 1.077/1.123 (bn32) and 1.191/1.340 (bn64) at o M=16 / down M=16: the loss is blocks, not A's duplicated LDS read. | [N-split](#n-split), [2026-10-18c](#2026-10-18c) |
| **Layout A (out of scope)** | The 71.6% "70% target" is reached only via Layout A (blocked weight reorder), which is excluded; hipblas's own no-reorder baseline is **213.0 µs = 68.4%**. | [Layout A](#layout-a) |
| **Producer A de-replication** | Bit-exact but slower everywhere (**1.044-1.358×**). `cp_async_wait<1>` sizes its vmcnt constant from the unconditional async loads, so putting A behind `if (owner)` drops a prefetch level; A's duplicates are L2-served and were never on the critical path. | [2026-10-20c](#2026-10-20c) |
| **M=1 GEMV structure** | Not expressible: the MMA output micro-tile is hardware-fixed at 16×16 and the MLS B atom floor is `tile_mn=16`; the reachable split-K equivalent was measured and rejected. | [M=1 GEMV](#gemv-m1) |
| **`warp_k` / `FullColK`** | Built and correct, but **0.97-1.03** vs shipped with mixed sign; it is a *substitute* for `setprio`, not a complement, and `decode_cfg` does not select it. | [2026-10-19](#2026-10-19-warpk) |
| **`__builtin_hcu_matrix_load_b8` (global→VGPR)** | The builtin exists, but its fixed per-lane layout is 4 consec-k × 4 n, not the mmac's 8 consec-k at one mn; bridging needs a full cross-lane permutation. | [2026-10-19](#2026-10-19-matrixload) |
| **A-direct-to-VGPR / `r_mls`** | A must round-trip through LDS: `r_mls` is **1.935×/2.592×** slower (o/down M=16), and the MMA fragment wants 8 consecutive K at one row, a 5120-byte lane stride. | [2026-10-18c](#2026-10-18c) |
| **Block-count inflation / occupancy** | More blocks at constant bytes is worse, and **320 blocks is the best point** on both curves; the ~8× blocks that reach the ~0.98× o M=1 bound cannot be supplied, and 320 is exactly 5/CU against an 8-slot LDS residency limit. | [2026-10-17f](#2026-10-17f), [2026-10-19](#2026-10-19-lds) |
| **Independent accumulator chains** | Splitting the single 16×16 chain is **1.539/1.524/1.644/1.641×** slower (o M=1/16, down M=1/16); a second accumulator over the same tile is neutral (**1.012/1.008/1.014/1.013×**). | [2026-10-17i](#2026-10-17i), [2026-10-18b](#2026-10-18b) |
| **Hand-written double buffering** | Correct (rel 0.00000) but **1.26-1.70× slower**; the pipelined form is genuinely slower at this tile. | [2026-10-17h](#2026-10-17h) |
| **Tile widening** | Monotonically catastrophic in the early sweep (`bn64/128/256`), and `bn32` alone is **7-12% worse** at every (shape, M). At M=64 the whole neighbourhood loses to the shipped `bn64/bk128/ns2/t256`: depth (`ns3`/`ns4`) and width (`bn96`/`128`/`192`) are 1.13-1.87×, and the exactly-64-block `bn80` is 1.177×. | [2026-10-17f](#2026-10-17f), [2026-10-18d](#2026-10-18d), [2026-10-20d](#2026-10-20d) |
| **Wave quantization (M=64)** | 80 blocks on 64 CUs is real but not the cost: the perfectly balanced 64-block point is *slower* (bn80 1.177×; N=4096 the slowest of 64/80/96/128), and the N-ladder is confounded by total bytes. No grid-shape knob exists to test co-residency cleanly. | [2026-10-20d](#2026-10-20d) |
| **`k_pack=2`, `b_ds_read_alt=0`, `respect_layout_map`** | Hard-rejected by the MLS path / unsupported (only Alt1 exists) / exact no-ops respectively. | [2026-10-17h](#2026-10-17h) |

## Contents

Dated sections and the sections that support them, in file order; nested dated
subsections are indented.

- [What actually won](#what-won)
- [Closed levers — index](#closed-levers)
- [The setup](#setup)
- [What hipblas does at M=1](#hipblas-m1)
- [M ≤ 16: bn32 / bk256 / ns0 / t256](#m-le-16)
  - [The achievable ceiling](#m-le-16-ceiling)
  - [GEMM really streams B](#m-le-16-streams)
  - [Verified M=1..16 head-to-head](#m-le-16-h2h)
- [M=1..128 vs hipblas — all four decode shapes (2026-10-17e)](#2026-10-17e)
- [Correctness](#correctness)
- [Why M ≤ 16 is flat at ~78% and M=32 drops to ~59%](#why-flat)
- [M ≥ 32 ≤ 128: the streaming kernel breaks down](#m-ge-32)
  - [M=64 root cause (2026-10-16)](#m64-root-cause)
  - [M=64 is LDS-port-bound (paired)](#m64-paired)
  - [hipblas M=64 k-loop vs ours — SASS](#m64-sass)
  - [Async-wait granularity (2026-10-17d)](#async-wait)
  - [The "70% target" is Layout A](#layout-a)
- [o/down: shape-aware config, split-K closes, `ns>1` landmine (2026-10-17f)](#2026-10-17f)
- [Reproduce](#reproduce)
- [B-direct-to-VGPR (2026-10-16 → closed 2026-10-17c)](#b-direct)
- [qkv M=32 lever adopted; o/down levers exhausted (2026-10-17g)](#2026-10-17g)
- [2026-10-17h — o/down decomposed](#2026-10-17h)
  - [2026-10-17i — re-timed; the last two mechanisms close](#2026-10-17i)
- [2026-10-18 — N-split closed; a second `ns>1` defect](#2026-10-18)
- [2026-10-18b — the "2016 GB/s ceiling" is not real](#2026-10-18b)
- [2026-10-18c — the exposure is volumetric and per-iteration](#2026-10-18c)
- [2026-10-18d — split-K is not expressible; double buffering ~1%](#2026-10-18d)
- [2026-10-19 — `warp_k` / `FullColK` is not a lever](#2026-10-19-warpk)
- [2026-10-19 — `__builtin_hcu_matrix_load_b8` exists but is closed](#2026-10-19-matrixload)
- [2026-10-19 — o/down: the LDS-capacity invariant](#2026-10-19-lds)
  - [2026-10-19 (b) — what hipblas actually launches](#2026-10-19-hipblas-launch)
- [2026-10-20 — the fat tile is real](#2026-10-20)
  - [2026-10-20b — warp count, not A traffic](#2026-10-20b)
  - [2026-10-20c — producer A de-replication CLOSED](#2026-10-20c)
  - [2026-10-20d — M=64 tile neighbourhood CLOSED](#2026-10-20d)

<a id="setup"></a>
## The setup

- B is (N, K) fp8 = 51200 × 5120 × 1 B = **262 MB**; 1800 GB/s peak → floor
  ≈ 146 µs for a single read of B. **Measured achievable read on this box:
  ~2016 GB/s** (the older 1646 GB/s / 91.5% figure came from a too-short 512 MB
  probe and is retired), so the real floor is ≈ 130 µs.
- hipblaslt (`torch._scaled_mm`, rowwise, bf16 out) measured with the same
  graph-replay timing: M=1 174.9 µs (83.3%), M=2 68.1%, M=4 73.4%,
  M=8 75.2%, M=16 76.4%, M=32 65.4%.
- Our initial MLS GEMM (block 256×256×64, ns=1, t512): M=1 198.7 µs (73.3%),
  M=2 73.4%, M=32 67.5% — M=2 and M=32 beat hipblas, the others lost.

<a id="hipblas-m1"></a>
## What hipblas does at M=1 (from disassembly)

No split-K, no dedicated GEMV. A plain per-block streaming kernel:

- weight loads: `buffer_load_dwordx4` **with SLC** (scalar cache — bypasses L2
  on the read side so all 64 CUs pull their distinct slices of the 262 MB
  weight in parallel),
- **`s_setprio 1/0`** around the MMA region (wave priority: MMA waves get
  scheduling preference; load/store warps wait),
- the k+1 load is issued at the top of the loop (not in the tail),
- `s_waitcnt vmcnt(4)` (not 0) before consuming a loaded tile,
- `ds_read_b128` (scalar LDS reads — no VGPR pressure on the read path).

> **Correction (2026-10-17): the SLC bullet is not a hipblas property.** Checked
> against the two native, unpatched kernels now available: **neither has `slc` on
> any load.** In `MT256x128x128` all 45 `slc` occurrences are on the
> **`buffer_store_dwordx4` for D** (`glc slc // store D`), and the native M=64
> down kernel (`MT96x64x256`, extracted from the pristine
> `FusedMatmulKernel_gfx938.co.bak_0916_1000`) has **0** `slc` on loads — while
> the *delivered* `m64/src/down_readable.s` has **9** `buffer_load_dwordx4 … slc`
> that the patch **added** (the README calls that patch "slc + W2"). So `slc` on
> weights is a property of the delivery patch, not of hipblas. The M=1
> disassembly this bullet came from is not recorded in this repo, so it cannot be
> re-verified; treat the bullet as unconfirmed until it is re-sourced from a
> native `.s`.

<a id="m-le-16"></a>
## M ≤ 16: bn32 / bk256 / ns0 / t256 — ~78%, beats hipblas (except M=1)

> **2026-10-13 re-verification:** all utilization numbers in this section were
> re-measured with explicit CUDA-graph capture + replay, events, interleaved
> A/B/A/B, min/med of 5×60 (tilelang `do_bench_cudagraph` returns ms on this
> stack / single-launch events carry host overhead — both inflate
> utilization). Effective bandwidth = N×K / latency vs 1800 GB/s nominal.

<a id="m-le-16-ceiling"></a>
### The achievable ceiling (measured)

Pure HBM read on this box tops out well below the 1800 GB/s nominal:

| probe | time | BW | % of 1800 |
|---|---|---|---|
| pure READ sum (512 MB fp32) | 311.0 µs | **1646 GB/s** | **91.5%** |
| read+write copy (1024 MB moved) | 810.1 µs | 1264 GB/s | 70.2% |
| 2×read+write add (1536 MB) | 913.0 µs | 1682 GB/s | 68.1%* |
| write-only fill (512 MB) | 501.8 µs | 1020 GB/s | 56.7% |

\* add's per-op bytes move 3 streams; figure reported on total bytes.

So **~91.5% is the physical read ceiling** — any GEMM claim above it is
definitionally wrong unless B is partially served from L2 (it is not: 262 MB
>> 8.4 MB L2, and the kernel demonstrably streams all of B — see below).

> **Correction (2026-10-17): 1646 GB/s is not the read ceiling; ~2016 GB/s is.**
> A longer grid-stride read (5.12 GB, not 512 MB) measures **2016 GB/s**, and the
> exact B fetch pattern (64 rows × 256 B, rows 5120 B apart) reaches
> **1881-2016 GB/s** with a plain copy. The 512 MB probe above under-measures
> because it is too short to reach steady state. Two consequences:
>
> - **Our standing vs the ceiling is worse than the 91.5% framing implies.** "78%
>   of nominal" is 85% of 1646 GB/s but only **70% of 2016 GB/s**; M=64's 59% of
>   nominal is 52.6% of the real ceiling, not 64.5%. Absolute headroom is
>   correspondingly *larger* (2016 − 1061 = 955 GB/s at M=64, vs 585 GB/s).
> - **Nothing here is above the ceiling.** The correction moves the ceiling up,
>   so the "no GEMM claim above the read ceiling" argument still holds — it is now
>   just harder to violate.
>
> The gap to hipblas is unaffected: that is a paired same-process measurement, not
> a ratio against a nominal. See *"The MLS producer, not the fetch pattern, is the
> real bottleneck"* below for the probe details.

<a id="m-le-16-streams"></a>
### GEMM really streams B

Latency scales linearly with B bytes at M=1 (bn32/bk256/ns0/t256):

| N | B size | tilelang | hipblas `_scaled_mm` |
|---|---|---|---|
| 12800 | 66 MB | 68.8 µs | 69.3 µs |
| 25600 | 131 MB | 96.6 µs | 114.6 µs |
| 51200 | 262 MB | 186.7 µs | 180.0 µs |
| 102400 | 524 MB | 376.3 µs | 409.8 µs |

At the decode N=51200 the kernel moves 262 MB → **1403 GB/s ≈ 78%** — right
where it should be, just under the 91.5% ceiling.

<a id="m-le-16-h2h"></a>
### Verified M=1..16 head-to-head (current slc code, bm16/bn32/bk256/ns0/t256)

| M | tilelang (min/med) | % | hipblas (min/med) | % | TL vs hipblas |
|---|---|---|---|---|---|
| 1  | 185.6 / 189.8 µs | 78.5 | 174.9 / 180.2 µs | 83.3 | **+6.1% (lose)** |
| 2  | 186.1 / 186.7 µs | 78.3 | 213.9 / 214.1 µs | 68.1 | **−13.0% (win)** |
| 4  | 186.4 / 186.9 µs | 78.1 | 198.4 / 198.6 µs | 73.4 | **−6.0% (win)** |
| 8  | 186.4 / 186.6 µs | 78.1 | 193.6 / 193.8 µs | 75.2 | **−3.7% (win)** |
| 16 | 188.5 / 188.6 µs | 77.3 | 190.6 / 190.9 µs | 76.4 | **−1.1% (win)** |

The M=1..16 latency is flat ~186 µs because the grid and per-block bytes don't
change with M (only fragment rows go live). We beat hipblas at **M=2..16**
(M=2 by 13%, shrinking to 1% at M=16); **M=1 is the one case hipblas wins**
(+6.1%) — its kernel has a cheaper small-M path we haven't matched yet
(its buffer_load+SLC path beats our matrix_load stream at the single-row
edge; slc is already in on our side).

**Even the most favorable timing does not beat hipblas at M=1.** Chained
steady-state (n_calls unrolled in one graph, timed as a block — no sync
between calls):

| M | TL chained | % | HIP chained | % |
|---|---|---|---|---|
| 1  | 176.6 µs | 82.5 | 167.9 µs | 86.8 |
| 2  | 177.1 µs | 82.2 | 201.7 µs | 72.2 |
| 4  | 177.2 µs | 82.2 | 186.4 µs | 78.1 |
| 8  | 177.6 µs | 82.0 | 187.0 µs | 77.9 |
| 16 | 179.9 µs | 80.9 | 184.0 µs | 79.1 |

The chained per-call figure is 176.6 µs (82.5%) — the honest best-case
steady-state, still behind hipblas's 167.9 µs (86.8%) at M=1.

DSL shape ranking at M=1 (min/med of 5×30, graph replay):

| bn | bk | ns | thr | min/med |
|---|---|---|---|---|
| 32 | 256 | 0 | 256 | 185.6 / 185.8 µs (78.5%) |
| 32 | 256 | 1 | 256 | 186.1 / 186.4 µs (78.3%) |
| 64 | 256 | 0 | 512 | 202.7 / 203.5 µs (71.8%) |
| 64 | 256 | 1 | 512 | 205.3 / 205.6 µs (70.9%) |
| 32 | 256 | 1 | 512 | 251.7 / 251.8 µs (57.9%) |

Key: **bn=32 + t256**. bn=64 doubles per-block bytes and halves the block count
→ worse tail/latency amortization; t512 halves per-thread coalescing width on
a 32-wide bn tile. slc is in (already shipped); ns/prio make no further dent.

<a id="2026-10-17e"></a>
## M=1..128 vs hipblas — all four decode shapes (paired, 2026-10-17e)

> **The `ours` column below predates the shape-aware chooser (2026-10-17f).** It
> used `_decode_cfg(M)`, which is why o/down lose here at every M. Current
> numbers are in the 2026-10-17f section. Kept for the gate_up/qkv rows and for
> the record of the confounded diagnosis discussed at the end of this section.

One process, every kernel in its own CUDA graph, replayed round-robin, idle
GPU 2. `ours` = `_decode_cfg(M)`; `hip` = `torch._scaled_mm` (bf16 out, RowWise).
Both columns are **% of nominal 1800 GB/s**; `ratio` = ours/hip, so **<1 is a
win**. Note our epilogue writes fp32 where hipblas writes bf16 — that *favours
hipblas*, by 0.6% of B at M=128 and less elsewhere.

| shape | B | M | ours µs | ours % | hip % | ratio |
|---|---|---|---|---|---|---|
| gate_up | 262 MB | 1 | 160.0 | **91.0** | 84.6 | 0.929 |
| | | 2 | 159.8 | **91.1** | 70.1 | **0.770** |
| | | 4 | 160.5 | **90.8** | 79.1 | 0.872 |
| | | 8 | 161.4 | **90.2** | 79.1 | 0.877 |
| | | 16 | 163.2 | **89.2** | 80.1 | 0.898 |
| | | 32 | 188.5 | **77.3** | 66.9 | 0.866 |
| | | 64 | 243.7 | 59.8 | 64.5 | 1.079 |
| | | 128 | 309.4 | 47.1 | 46.9 | 0.996 |
| qkv | 52 MB | 1 | 37.8 | **77.1** | 65.0 | 0.843 |
| | | 2 | 38.2 | **76.2** | 62.3 | 0.818 |
| | | 4 | 38.4 | **75.9** | 62.3 | 0.822 |
| | | 8 | 39.0 | **74.6** | 62.3 | 0.836 |
| | | 16 | 39.4 | **74.0** | 62.8 | 0.848 |
| | | 32 | 57.6 | 50.6 | 57.1 | 1.129 |
| | | 64 | 64.8 | 44.9 | 46.6 | 1.036 |
| | | 128 | 87.7 | 33.2 | 38.1 | 1.146 |
| o | 42 MB | 1 | 44.6 | 52.2 | **73.2** | 1.402 |
| | | 2 | 44.8 | 52.0 | **68.4** | 1.315 |
| | | 4 | 44.6 | 52.2 | **65.9** | 1.262 |
| | | 8 | 45.0 | 51.8 | **66.2** | 1.277 |
| | | 16 | 45.1 | 51.6 | **66.5** | 1.288 |
| | | 32 | 48.3 | 48.2 | **59.2** | 1.228 |
| | | 64 | 57.4 | 40.6 | **45.2** | 1.115 |
| | | 128 | 117.1 | 19.9 | **36.0** | 1.812 |
| down | 131 MB | 1 | 118.2 | 61.6 | **92.5** | 1.502 |
| | | 2 | 118.6 | 61.4 | **89.8** | 1.462 |
| | | 4 | 118.2 | 61.6 | **78.3** | 1.272 |
| | | 8 | 118.6 | 61.4 | **84.6** | 1.377 |
| | | 16 | 119.0 | 61.2 | **84.8** | 1.385 |
| | | 32 | 127.0 | 57.3 | **67.1** | 1.171 |
| | | 64 | 161.3 | 45.1 | **53.7** | 1.189 |
| | | 128 | 377.1 | 19.3 | **39.8** | 2.060 |

**12 of 32 cells are wins, and they are all in gate_up (7/8) and qkv (5/8).**
On `o` and `down` we lose at **every** M — up to 2.06×.

**The pattern is not M, it is the block grid.** The k-loop of each block has no
cross-k-step overlap at `ns=1`, so the deficit is exposed exactly when there are
too few blocks to hide it. Grid = `ceildiv(N,bn) × ceildiv(M,bm)`, k-steps =
`ceildiv(K,bk)`:

| shape | blocks (M≤16) | k-steps (M≤16) | verdict |
|---|---|---|---|
| gate_up | 1600 (25/CU) | 20 | **win** |
| qkv | 320 (5/CU) | 20 | **win** |
| o | 160 (2.5/CU) | 32 | lose 1.29× |
| down | 160 (2.5/CU) | 100 | lose 1.39× |

20 k-steps → win; ≥32 k-steps → lose. `down` is the extreme: 160 blocks each
running a 100-step serial loop with a barrier per step. Our `down` time is
**flat at ~118 µs for M=1..16** — it does not depend on M at all, i.e. it is
pinned by the B stream through that starved grid, at ~1110 GB/s.

**Both obvious fixes are measured closed.**

- *Deeper pipelining does not transfer.* The `bk128/ns2` trick that won at
  gate_up M=64 was swept on o/down at M=1/16/32: `bk256 ns2` and `bk128 ns2` are
  **worse** everywhere (e.g. down M=1: 117.8 → 120.8 → 138.9 µs), and `ns=3` is
  much worse (164.2 µs). The gate_up M=64 win came from a *long* loop needing
  overlap; here the loop is already the bottleneck and more staging only costs
  LDS and sync.
- *It is not under-tuning — `bm16/bn32` is already optimal.* Widening the tile
  to add work per block (and thus *fewer* blocks) is catastrophic, monotonically:

  | config | o M=1 | down M=1 |
  |---|---|---|
  | bm16/bn32 (current) | **43.0** | **118.1** |
  | bm16/bn64 | 45.6 | 147.8 |
  | bm64/bn64 | 47.4 | 128.0 |
  | bm32/bn64 | 51.8 | 159.5 |
  | bm16/bn128 | 77.1 | 221.0 |
  | bm16/bn256 | 97.4 | 374.2 |
  | hipblas | 32.3 | 83.2 |

  Every alternative is worse, so the small-N deficit is **structural**, not a
  config miss.

**The "block-count deficit" reading above is a CONFOUND, and split-K is closed.**
The sweep behind it varied N at fixed K, so block count and total bytes moved
together — it cannot distinguish "more blocks" from "more bytes". Holding bytes
constant and varying only block count reverses the sign (2026-10-17f, below):
320 blocks is the *best* point on both the o and down curves. Split-K, which is
exactly that transformation, was then built and measured **10-21% slower** than
this baseline. See the 2026-10-17f section.

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
nothing about whether a streaming GEMM can. The GEMM's 78% sits just under the
measured 91.5% read ceiling, which is the right place.

<a id="why-flat"></a>
## Why M ≤ 16 is flat at ~78% and M=32 drops to ~59%

- Grid is fixed (N=51200, bn=32) → M ≤ 16 only changes how many of the bm=16
  fragment rows are live, not the grid size or per-block bytes.
- The ~22% gap to nominal is real: 91.5% ceiling, minus per-block overhead
  (barrier + wait + ds_read serialization) and a non-perfect stream.
- At M=32, bm=32 halves the block count and doubles fragment work per block;
  MMA pressure starts to interfere with the streaming path (best measured
  config (32,32,256,1,256): 246 µs ≈ 59%).

<a id="m-ge-32"></a>
## M ≥ 32 ≤ 128: the streaming kernel breaks down

For M ≥ 32 the bm=pow2≥M tiles inflate: bm=64 → A tile is 64×256×1B = 16 KB
vs B's 32 KB (bn=32); bm=128 → 32 KB A vs 16 KB B per block. A is pure
redundant reads (A is 6 MB; rereading it 800× doesn't cost bandwidth but adds
LDS/MMA work), and MMA becomes the bottleneck: a 128×128×256 fp8 tile at
432 TFLOPS peak needs ~9.7 µs of MMA per block vs ~6.7 µs of B streaming —
compute-bound. Measured M=32 with the M≤16 winner shape: 246 µs ≈ 59%
(hipblas ~65%), i.e. the streaming win does not transfer.

<a id="m64-root-cause"></a>
### M=64 root cause (measured 2026-10-16): LDS-bound, not bandwidth-bound

The A-reread intuition above is right in spirit but the mechanism is **LDS write
bandwidth**, and the kernel is not bandwidth-bound at all. Pinned-address
controls (same instruction stream and LDS traffic, only addresses changed):

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
only `bn` (N-coverage per block) reduces it. But `bn=64` is the floor for the fast
MLS path: wider tiles change the MLS warp partition and raise LDS *read*
amplification from 2× to 3–4×, which costs more than the A writes saved. Ten
wide-bn configs were measured, all worse than the 245.9 µs baseline:

- `bn128 bk256 t256` → 340.3 µs (VGPR 130 → 3 blk/CU; B flips to `axis::k`)
- `bn128 bk128 t256` → 281.4 µs — **same** 82 VGPR / 6 blk/CU as the best config,
  so this is *not* occupancy; the partition becomes (4,1)
- split-gemm `bn128` as 2×(64,bk) B buffers + 2 C fragments → 333.1 µs, identical
  to plain bn128, which rules out the B_local register cost
- `nt`-shared-A (2/4 sequential n-tiles sharing one A_s load) → 364–415 µs
- `bk512`, `k_pack=2`, `ns=2@bk256`, A hoisted into a (64,512) buffer → all worse

**Structural wall.** Every `__builtin_hcu_matrix_load_*` builtin is global→LDS;
there is no register-destination variant, so A cannot skip the LDS round-trip via
the MLS path. The only A→VGPR route is r_mls, which lowers to 8-byte
`hcu_buffer_load<fp8_t, 8>` — a proven 3.7× loss.

So `bm64 bn64 bk256 ns1 t256 sp1` (245.9 µs fp32 / 243.4 µs bf16) is a structural
local optimum. This supersedes the earlier split-K / fold-M suggestions for M=64,
which target a bandwidth wall that measurement shows is not there.

<a id="m64-paired"></a>
### M=64 is LDS-port-bound; the gap is 5.8% (measured paired)

**Measure paired, or don't measure.** HCU[4] is shared with co-tenants
(`rocm-smi --showpower` shows HCU[2]-[7] at 360-372 W with nothing of ours running),
so absolute times drift up to ~15% between invocations and the *ordering* of two
candidate configs flips. Repeated rounds inside one process agree to ~0.2us. Build
both kernels in one script, interleave the rounds, and report the ratio.

Doing that, 4 rounds, `torch._scaled_mm` (bf16 out) vs our `(64,64,256,1,256,1)`:

| | time (contended) | vs hipblas |
|---|---|---|
| hipblas `MT128x64x256` | 257.0us | — |
| **ours `bn64 bk256 ns1 t256 sp1`** | **272.0us** | **1.058x** |
| ours `bn64 bk128 ns2 t256 sp1` | 276.3us | 1.076x |

So the gap is **5.8%**, and `bk256/ns1` — the committed config — is the better one.
(Quiet-machine equivalents: ~231.7 vs ~247.0us, i.e. 62.9% vs 59.1%.)

> **Superseded (2026-10-17d).** Both halves of that conclusion have since been
> corrected. The 5.8% is a *contended* figure — on an idle card the paired gap
> is 10.2%, then 8.7%, and after the `bk128/ns2` change below, **7.3%**. And
> `bk128/ns2` now *beats* `bk256/ns1` by 1.2% (243.5 vs 246.6 µs); the
> `276.3 µs` for `bk128 ns2` in the table above was taken under contention and
> confounded `bk` with `ns`. See "Async-wait granularity — the lever, measured".

**Where the time goes** (load-only kernels, no mmac, so the load path is isolated):

| | time | note |
|---|---|---|
| B stream alone | **166.1us** | 1578 GB/s = **95.9%** of the 1646 GB/s achievable ceiling |
| A + B through LDS | **196.0us** | A's round trip costs **+30us** for a 327 KB matrix |
| full kernel | ~247us | ~51us of compute exposed |

The load path is not the problem. A's `global → LDS → VGPR` round trip is the largest
overhead we own, and it is **not recoverable without the excluded weight reorder**:

- **A→VGPR direct is refuted — and hipblas agrees.** The MMA fragment wants 8
  *consecutive-K* fp8 per lane at one row — `A_local[(ki*warp_rows + Mi)*8]` with
  lane → `m = Mi*16 + (lane&15)`, `k = ki*32 + (lane>>4)*8`. Loading that from
  global means 8-byte requests at a **5120-byte lane stride**. (Note: the *stride*
  is not what killed `b_vgpr_direct` — strided `buffer_load` matches `hipMemcpyD2D`
  bandwidth; that experiment died on the 4× m-warp redundancy, see below.) The
  native hipblas kernels stage A through LDS too (`offen … lds` → `ds_read_b128`),
  so the LDS round trip is the *price* of the fragment layout, not a mistake on
  our side.
- **`bn` cannot amortize it.** bn 128/256/512 → 253.6 / 275.4 / 382us: halving the
  redundant A re-read loses more on the B path than it saves.

**LDS volume is not the differentiator — hipblas moves exactly the same bytes.**
Per warp per k-step the committed config issues 32 `ds_read` of 16 B/lane = 32 KB read,
plus A(16 KB) + B(16 KB) = 32 KB written, over a 64x64 tile with k=256:

| | LDS bytes / output / k-step |
|---|---|
| ours `bn64 bk256 t256` | 160 KB / (4096 x 256) = **0.15625** |
| ours `bn64 bk128 ns2 t256` | 80 KB / (4096 x 128) = **0.15625** |
| hipblas `MT128x64x256` | 320 KB / (8192 x 256) = **0.15625** |

Identical, and invariant across `bk`. The 4x read amplification over the tile's unique
bytes is entirely the per-warp operand duplication that the mmac layout requires: the
`ds_read` volume (512 B/lane) exactly equals the mmac operand volume (32 mmac x 8 B A
+ 8 B B), so **there is no recoverable waste**. Total LDS traffic is 2.56 GB, which at
128-160 B/clk/CU is a 192-240us floor — the same order as our 247us, so LDS port
pressure is a plausible co-limiter, but it cannot explain a gap to a kernel that moves
the identical volume.

That leaves **overlap quality** as the residual: our load path alone runs at 95.9% of
the achievable ceiling, yet in the full kernel B's effective rate drops to ~1062 GB/s.
hipblas gets the same bytes to DRAM sooner via hand-tuned scheduling and 4 `s_setprio`
placements, not via any tile-shape or resource choice available from Python.

> **Correction (2026-10-17d): the "fine-grained `vmcnt(4)` waits" half of that
> sentence was an artifact of reading the *patched* deliverable.** Disassembling
> the **native** M=64 kernel (`.bak_0916_1000`, see the provenance note below)
> shows the k-loop contains **only wait-all** synchronisation — `s_waitcnt
> lgkmcnt(0)` ×8 and `s_waitcnt vmcnt(0)` ×4 per k-tile. There is **no partial
> `vmcnt(N>0)` wait in the loop at all**; the `vmcnt(23)→…→0` ladder and the
> `vmcnt(14)…(1)` cascade live in the *prefetch / scale / epilogue* region, not
> the k-loop. The in-loop `vmcnt(4)` in `down_readable.s` is precisely the
> hand-edit of the native `vmcnt(0)`. So hipblas's in-loop waits are *coarser*
> than ours, not finer. See "Async-wait granularity — the lever, measured"
> below for the experiment this prompted.

Occupancy is **not** the lever: 82 VGPR → 6 blocks/CU (24 warps) vs 50 VGPR → the
thread-capped 10 blocks/CU (40 warps) buys ~1%, and that does not survive re-measurement.

Our wait structure is already clean — the k-loop body has no `matrix_load`, no
`s_barrier` and no `vmcnt` wait; the `vmcnt(0); s_barrier` sits *after* the back-edge,
once, in the epilogue.

> **Contested (2026-10-17d).** Recompiling the cached device source for
> `bm64 bn64 bk256 ns1 t256 sp1` with the exact HCU `-mllvm` flags shows the
> opposite: the main-loop body *does* contain `s_waitcnt vmcnt(0) lgkmcnt(0)` at
> the top, an `s_barrier` on each side, and the 4 `matrix_load_128x16_b8` at the
> bottom, with the back-edge branching straight back to the wait. The claim
> above is retained because it was made from a *shipped* SASS dump while the new
> reading is a recompile of the cached device source (no gfx938 code object is
> embedded in the host `.so`), so neither is authoritative on its own. What is
> **not** in doubt is the *native hipblas* side, which comes straight from the
> vendor binary. The distinction matters: if our loop really does branch into a
> full drain, the next k-tile's loads have zero overlap with the current tile's
> compute — which is exactly the effect the `ns=2` sweep below then measures.

<a id="m64-sass"></a>
### hipblas M=64 k-loop vs ours — SASS structural comparison

The 5.8% figure above was measured under mild co-tenant contention. On a fully
idle card (2026-10-17, all 8 HCUs at 0% HCU utilization) the paired gap is
**10.2%, stable to ±0.2% over 8 rounds**: hipblas `MT96x64x256`-class kernel
**224.2us (64.9% of 1800 GB/s)** vs ours **247.0us (58.9%)**. Contention
disproportionately hurts hipblas (its deep queue needs sustained DRAM + full CU
quota), which is why the gap read as 5.8% under load and even as "parity"
under heavy load. **Only idle-card numbers are comparable to the 68.4%
historical baseline.**

Source: `/workspace/llm-adaptation/qwen3-32b-fp8-rowwise/deliverable_fp8_gemm/m64/src/down_readable.s`
(byte-exact reconstruction of the deployed hipblaslt M=64 fp8 rowwise kernel,
`Cijk_Alik_Bljk_F8BS_MT96x64x256_TT6_2_WG16_32_1_MFWGS512_WGM8_bias_channelwise`),
vs our generated SASS for `bn64/bk256/t256`.

> **Artifact provenance — read this before citing anything below.** The
> `deliverable_fp8_gemm/**/*.s` files are **patched** kernels (Layout A / slc /
> vmcnt4 hand-edits applied on top of the vendor binary), and the deliverable's
> own README warns at line 54: *"**operand 角色注意**：M=128 与 M=64 的权重/activation
> 走的 VGPR/LDS 路径不同，`.s` 是各自 kernel 独立 patch 的，**不能跨 M 复制**。"*
> The **native, unpatched** hipblaslt reference — the ground truth for "what does
> hipblas actually do" — is
> `/workspace/misc/Cijk_Alik_Bljk_F8BS_MT256x128x128_TT16_8_WG16_16_1_MFWGS512_WGM8_bias_channelwise.s`
> (Tensile-generated, readable). **And for the M=64 kernel specifically, the native
> binary is recoverable** — the pristine pre-patch library is kept at
> `/opt/dtk/lib/hipblaslt/library_bw_nmz/FusedMatmulKernel_gfx938.co.bak_0916_1000`
> (the Oct-12 `FusedMatmulKernel_gfx938.co` next to it is the *patched* one):
>
> ```
> /opt/dtk/aillvm/bin/llvm-objdump -d --mcpu=gfx938 clean.co > full.s
> # then slice the label
> #   0000000000c9bd00 <Cijk_Alik_Bljk_F8BS_MT96x64x256_TT6_2_WG16_32_1_MFWGS512_WGM8_bias_channelwise>:
> ```
>
> That yields the **native M=64 down kernel**, which is the decisive witness for
> every operand-role question below. Comparing it with the delivered file settles
> the contradiction in this document: native has **20 `idxen offen` dwordx4** into
> `v[96:127]` (B, global→VGPR) and **6 `offen … lds` dwordx4** (A, global→LDS),
> and **0 `slc`**; the delivered `down_readable.s` has the same 20 `idxen` but
> **9 added `slc`** loads. Several conclusions in this document were
> originally drawn from the patched files and are corrected below against the
> native one.

**Per k-iteration (hipblas does 2 × 32-k slices per iteration):**

| | hipblas (from its SASS) | ours (bn64/bk256/t256) |
|---|---|---|
| weight (B, 262 MB) | `buffer_load_dwordx4 ... idxen offen slc` → **direct global→VGPR** (4×16B/lane, v[96:127]); **never touches LDS** | `matrix_load_128x16_b8` (slc+lds) → LDS → `ds_read_matrix_trans_format` (256 B/lane) |
| activation (A) | `ds_read_b128` (12×16B/lane, LDS→VGPR) — A is the *small* operand staged via LDS | same shape (via `ds_read_format_tensor_a`) |
| LDS footprint | A only | A + B (2×32KB double-buffered) |
| barriers | one `s_barrier` per iteration (LDS reuse), B side barrier-free | `cp_async_wait<0>` + `__syncthreads()` per iteration |
| in-flight depth | **vmcnt ladder `vmcnt(23)→...→0`** at the loop back-edge — ~20+ dwordx4 in flight, drains **once per 2-chunk window** | waits to 0 **every iteration**; zero intra-block prefetch; latency hidden only by 800-block concurrency |
| mmac chain | 48 `v_mmac` per iteration, wrapped in `s_setprio 1/0` (we match this) | 32 `v_mmac` per k-iteration, `s_setprio` on |
| mmac operand wiring | `v_mmac v[acc], v[A-from-LDS], v[B-direct-VGPR]` — 2nd operand (A rows) from LDS, 3rd (B rows) direct | both operands from `ds_read_*` |

Note: the *weight* is the operand hipblas loads direct-to-register (slc-tagged),
and the *activation* goes through LDS. Our `b_vgpr_direct` experiment matched
this split (B direct, A via LDS) with 8-byte loads.

**Confirmed against the native kernel (2026-10-17).** The patched M=64 file above
is not the right witness for the operand-role question, because the Layout A /
vmcnt4 patch itself moves operands between paths (see the provenance warning).
The native kernel settles it — in
`/workspace/misc/Cijk_Alik_Bljk_F8BS_MT256x128x128_TT16_8_WG16_16_1_MFWGS512_WGM8_bias_channelwise.s`
(lines 2162-2165) the weight B is loaded straight into the MMA operand block:

```
buffer_load_dwordx4 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+3], v[vgprGlobalReadOffsetB+0:vgprGlobalReadOffsetB+1], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+7], v[vgprGlobalReadOffsetB+2:vgprGlobalReadOffsetB+3], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+11], v[vgprGlobalReadOffsetB+4:vgprGlobalReadOffsetB+5], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0
buffer_load_dwordx4 v[vgprValuB_X0_I0+12:vgprValuB_X0_I0+15], v[vgprGlobalReadOffsetB+6:vgprGlobalReadOffsetB+7], s[sgprSrdB:sgprSrdB+3], 0, idxen offen offset:0
```

— **no `lds` modifier: B never touches LDS.** A (activation) does, via
`buffer_load_dwordx4 v[vgprGlobalReadOffsetA+N], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0, lds`
(8 loads, lines 1930-1944) plus the `ds_write_b128` G→VGPR→LDS dance. So the table
row above is right, and the "no weights-bypass trick" claim further down is wrong
— it was an artifact of reading the *patched* gate_up kernel.

**Why it is free for hipblas: the B index has no M component.** Lines 1610-1695
derive the per-lane B address from `Serial` alone:

```
v4 = Serial & 63          (lane)          v3 = v4 >> 4        (lane/16)
v5 = Serial >> 6          (wave id)       v2 = v4 & 15        (lane%16)
v3 <<= 4                                  v5 <<= 5
v2 += v5                  -> groB1J_0 = (lane%16) + wave*32   (B **N** index)
v7  = v6 + 16             -> groB1J_1 = groB1J_0 + 16         (LSPB=16)
v9  = v3                  -> groBL_0  = (lane/16)*16          (B **K** index)
v18 = v9 + 64             -> groBL_1  = groBL_0 + 64          (LSCB=64)
```

There is **no M coordinate anywhere in it**. The 4 waves partition **N**: wave W
owns the N slab `[32W, 32W+32)`, and its tile is 256 M × 32 N. Each lane covers
2 N × 2 k-blocks × 16 B = 64 B, so the wave's 32 N × 128 k = 4096 B is covered
exactly once by 64 lanes × 64 B — **zero redundancy**, and B's reuse across the
whole 256-deep M extent happens **in the register file**: one 8-byte B operand
(`v[vgprValuB_X0_I0+0:+1]`) is consumed by **8 mmacs** with 8 different A operands;
the full 16-byte B block feeds 32 mmacs (`MAC_16x8_X0` = 4 B-operands × 8
A-operands = 32 mmacs → 128 C regs = 16 M × 8 N thread tile).

**That is the whole trick: put all of M in one wave and split N across waves.**

**The same is true of the actual M=64 kernel.** From the native
`MT96x64x256_TT6_2_WG16_32_1_MFWGS512_WGM8` extracted above:

```
buffer_load_dwordx4 v[96:99],   v[184:185], s[12:15], 0 idxen offen
buffer_load_dwordx4 v[100:103], v[186:187], s[12:15], 0 idxen offen
buffer_load_dwordx4 v[104:107], v[188:189], s[12:15], 0 idxen offen
buffer_load_dwordx4 v[108:111], v[190:191], s[12:15], 0 idxen offen
...
v_mmac_f32_16x16x32_fp8_fp8 v[0:3],  v[32:33], v[96:97]
v_mmac_f32_16x16x32_fp8_fp8 v[0:3],  v[34:35], v[98:99]
v_mmac_f32_16x16x32_fp8_fp8 v[0:3],  v[64:65], v[100:101]
```

— **the destination block `v[96:127]` of the `idxen` global loads is literally the
third mmac operand.** B never touches LDS. The B index is built from `lane%16`
alone (`v_mul_lo_u32 v184, s78, v6` with `v6 = Serial & 15`, `s78` = the B row
stride) — **no M coordinate** — and the four loads differ only in k (`v9`,
`v9+64`, `v9+128`, `v9+192`, `v9 = (lane%16)>>2`), i.e. **16 consecutive k per
lane**. A comes from `ds_read_b128`. One 8-byte B operand feeds **18 mmacs** here
(8 in the MT256 kernel). And the native file has **zero `slc`** — every `slc` in
the delivered file was added by the patch.

Cross-check on redundancy: the native M=64 down baseline is **131.8 µs for
131.1 MB = 995 GB/s**. An 8× or even 2× replicated B fetch would demand
7960 / 1990 GB/s — at or beyond the measured 2016 GB/s ceiling. So the native
kernel really does fetch B about **once**.

**Update (2026-10-17): the opaque-asm scheduler theory was wrong; the real cost is
4x m-warp redundancy.** Making 16 loads into one opaque asm block with `=&v`
(early-clobber — required because `=v` lets LLVM alias output VGPRs over later
input addresses, causing VMFaults at ≥3 loads) does reproduce hipblas's
"issue all, drain once" structure (SASS-verified: 16 distinct output reg pairs,
one `s_waitcnt vmcnt(0)`). Raw BW matches `hipMemcpyD2D` at ~1320 GB/s for both
contiguous and strided patterns — the hardware handles strided `buffer_load`
fine. The full GEMM is still ~72x slower than the LDS control because **4 m-warps
per n-warp each issue their own 16 loads of identical B data** → 4x global traffic
+ VGPR pressure (101 vs 82 VGPR) + register-shuffle overhead. LDS staging is what
coalesces those 4 reads into 1. The 8-consec-k fragment then adds another ~2x of
sector-inefficiency vs hipblas's 16-consec-k.

**Correction (2026-10-17c): the 4x is indeed a *partition* artifact — but removing
it is not enough, and both B-fetch levers are now measured closed.** The 4x exists
only because our M=64 kernel is `WarpM=4` — M is split across **4 separate
waves**, so four *independent* memory instructions fetch the same B. The native
hipblas kernel has no such term: its B index contains no M coordinate and its 4
waves split **N** (see the native-kernel block above), so the identical fetch is
issued **once**. That much is right, and it is exactly why `WarpM=1` was the
correct experiment — but the full GEMM then showed `WarpM=1` B-direct is still
**2.1× slower than baseline** (513.8 vs 246.2 µs), and the 16-consec-k fragment
cannot close the remaining gap (arithmetic bound: at most 93.9 µs/pass available
against a 267.6 µs gap). **See "B-direct-to-VGPR: the cost, the root cause, and
why it is closed" below.** The **MLS producer** lever is closed negative as well:
`matrix_load` does run at 79% of achievable bandwidth vs 99% for a plain copy, but
that standalone gap is already hidden by the pipeliner — swapping to the faster
64×32 atom makes the full GEMM **15–16% slower** (287.2 vs 246.2 µs). The partition
is shape-dependent and controlled by `ComputeMlsWarpPartition` (`src/hcu/op/mls.cc`)
+ `hoist_mls_resource.cc`; both levers require a C++ rebuild, and neither pays.

<a id="async-wait"></a>
### Async-wait granularity — the lever, measured (2026-10-17d)

**Hypothesis (now refuted).** The residual gap is overlap quality, and the doc
attributed hipblas's advantage to "fine-grained `vmcnt(4)` waits instead of our
coarser `cp_async_wait`". If true, the fix is to make our main-loop wait
finer-grained.

**The hypothesis is dead on the native SASS.** Disassembling the native M=64
kernel shows the k-loop uses **only wait-all** synchronisation: 8×
`s_waitcnt lgkmcnt(0)` (after 12 `ds_read_b128`, one per 24-MMA block) and 4×
`s_waitcnt vmcnt(0)` (after 4 `buffer_load_dwordx4`, one per 48 MMAs) per
k-tile. The `vmcnt(23)` ladder and the `vmcnt(14)…(1)` cascade are in the
prefetch/scale/epilogue region (`0xc9d80c`–`0xc9fc1c`), **not** the loop. The
in-loop `vmcnt(4)` of `down_readable.s` is the hand-edit of the native
`vmcnt(0)`. On the LDS axis we are already *finer* than hipblas (a rolling
`lgkmcnt(7)→(0)` cascade interleaved into the 32 MMAs vs its single coarse
`lgkmcnt(0)` per 24).

**The real difference is queue depth, and it is a *consequence* of where B
lives.** hipblas issues 4 `buffer_load_dwordx4` mid-block and waits one 24-MMA
block later, so the next B chunk is in flight while the current block computes
— with the queue held in **VGPRs** (no LDS footprint, hence the `vmcnt(23)`
ladder). Ours issues the 4 `matrix_load_128x16_b8` at the bottom of the body
and the back-edge branches straight into `s_waitcnt vmcnt(0) lgkmcnt(0)`: the
LDS buffer is single-buffered, so a full drain is *forced*.

**The knob is therefore `num_stages`, not the wait count.** `cp_async_wait<N>`
comes from the generic `InjectSoftwarePipeline`
(`tilelang/hcu/pipeline.py:36-42` keeps the HCU-native MLS waitcnt planner
disabled), and `N` is relaxed by `PipelinedRetainGroups = num_stages - 1`
(`src/transform/inject_pipeline.cc:1773-1781`, gated at `:2274`). `ns=1` ⇒
`wait<0>`; `ns=2` ⇒ `wait<1>`. LDS budget `(bm+bn)·bk·ns ≤ 65536` ⇒ at
bm=bn=64, `bk·ns ≤ 512`.

Generated wait sequences (source dump, `bk·ns ≤ 512`):

| config | LDS | `cp_async_wait` sequence |
|---|---|---|
| bk256 ns1 (old baseline) | 32 KB | `[0, 0]` |
| bk256 ns2 | 64 KB | `[1, 0]` |
| bk128 ns2 | 32 KB | `[1, 0]` |
| bk128 ns4 | 64 KB | `[3,2,1,0]` |
| bk64 ns4 | 32 KB | `[3,2,1,0]` |
| bk64 ns8 | 64 KB | `[7,6,5,4,3,2,1,0]` |

All measured same-process, CUDA-graph replay, interleaved rounds, idle card
(GPU 2, HCU% 0), paired against `torch._scaled_mm` (bf16 out). Within-config
round-to-round spread was ≤0.5 µs with no overlap between configs, so the
ordering is trustworthy; absolutes still move between runs.

| config | µs | GB/s | % | vs base | vs hipblas |
|---|---|---|---|---|---|
| bk256 ns1 t256 | 246.6 | 1063 | 59.1% | 1.000 | 1.087 |
| bk256 ns2 t256 | 250.7 | 1046 | 58.1% | **1.017** | 1.115 |
| bk128 ns1 t256 | 269.3 | 973 | 54.1% | 1.092 | 1.197 |
| **bk128 ns2 t256** | **243.5** | **1076** | **59.8%** | **0.988** | **1.073** |
| bk128 ns3 t256 | 336.0 | 780 | 43.3% | 1.363 | 1.494 |
| bk64 ns1 t256 | 313.6 | 836 | 46.4% | 1.272 | 1.394 |
| bk64 ns2 t256 | 329.0 | 797 | 44.3% | 1.334 | 1.462 |
| bk128 ns2 t512 | 263.2 | 996 | 55.3% | 1.067 | 1.160 |
| bk128 ns2 bn128 | 250.6 | 1046 | 58.1% | 1.016 | 1.104 |
| bk256 ns2 t512 | 287.7 | 911 | 50.6% | 1.167 | 1.268 |
| hipblas | 226.9 | 1155 | 64.2% | 0.920 | 1.000 |

**Reading.** The wait *count* is not the lever: at fixed `bk=256`, going
`ns=1→2` (i.e. `wait<0>`→`wait<1>`) makes it **1.6% slower**. The win is an
**interaction with `bk`**: at `bk=128` the `ns=1→ns=2` swing is **−25.8 µs**,
at `bk=256` it is **+4.1 µs**. That is a genuine cross-k-step-overlap effect —
double-buffering lets the next tile's loads run under the current tile's
compute, which is the *same mechanism* hipblas gets from its VGPR queue, just
implemented in LDS. It needs the smaller tile to pay off (bk128 ns1 alone is
23 µs *worse* than bk256 ns1, because it syncs twice as often with no
overlap), and `bk128 ns2` sits at 32 KB LDS so two CTAs stay resident per CU.
Everything past depth 2 is much worse (`bk128 ns3` 336 µs, `bk64 ns8` 430 µs):
deeper staging costs more LDS and more sync than the extra overlap returns.

**Adopted.** `_decode_cfg(64)` is now `(64, 64, 128, 2, 256, 1)` — **243.5 µs /
59.8%**, 1.2% faster than the old `bk256 ns1` and closing 3.5 µs of the ~20 µs
gap to hipblas (ratio 1.087 → 1.073). rel=0.00000 on all four decode shapes.
The 8192³ compute path is untouched (433.9 TFLOPS).

**Honest size of the win.** 1.2%, not the 5.8% the in-scope framing hoped for.
The overlap-quality story survives, but the *specific* lever named in this doc
(finer waits) does not exist: hipblas's fine-grained in-loop wait was a patched
artifact. What remains is that hipblas hides B's latency in a deep VGPR queue
while we must round-trip through single/double-buffered LDS — a structural
consequence of the mmac fragment layout, and the same wall the B-direct
experiments hit from the other side.

**Guard.** `bk=192` compiles and runs but is **silently wrong** (`rel=1.61`) —
`5120 % 192 = 128`, so the pipelined `T.copy` reads a whole `block_K` tile at
`k=4992` and runs 64 elements past the end of B. The builder only asserted
`block_K % 64 == 0`, which is necessary for the trans-B atom but does **not**
imply `K % block_K == 0`; the latter is now asserted too. (Checked that this is
really divisibility and not an atom-width rule: `bk=320 ns1` has
`K % 320 == 0`, is *not* a multiple of 128, and is **correct** at rel=0.000000.
`bk=320 ns2` and `bk=2560` fail only on the 64 KB LDS cap.)

<a id="layout-a"></a>
### The "70% target" is Layout A — which is out of scope

The reference deliverable
(`/workspace/llm-adaptation/qwen3-32b-fp8-rowwise/deliverable_fp8_gemm/`) states
its M=64 gate_up recipe in its own README:

| M | kernel | tile | 优化 | 需权重重排 | 基线 → 优化 |
|---|---|---|---|---|---|
| 64 | gate_up | MT128x64x256 | **Layout A + vmcnt4** | ✅ | 213.0 → **203.3 µs** |
| 64 | down | MT96x64x256 | slc + W2 | ❌ | 131.8 → 117.7 µs |

203.3 µs / 262.1 MB = 1289 GB/s = **71.6%** — that is the 70% figure, and it is
reached **only** via Layout A (blocked weight reorder). Without the reorder
hipblas's own baseline is **213.0 µs = 68.4%**. So the in-scope gap is
ours 247.0 µs (59.0%) → 213.0 µs (68.4%): **34 µs**, not 44 µs.

**"A staged once" (V1 = 217.0 µs) is not reachable.** Two measurements:

- Staging A for 2 k-steps at a time into two separate (64,256) buffers — every MLS
  atom keeps the proven `seq<64,256>`/(2,2)/`axis::mn` shape, B also
  double-buffered — gives **249.3 µs**, i.e. no gain. Same instruction count, same
  LDS bytes. The cost is per-byte, not per load group.
- A's LDS write is `M·N·K/bn`, invariant under `bm`, `bk`, split-K and A-hoisting.
  Only `bn` reduces it, and `bn>64` breaks the MLS partition.

**N-scaling** (`/tmp/m64_scale.py`) separates fixed from marginal cost:

| N | B MB | ours | GB/s | % | compute-only |
|---|---|---|---|---|---|
| 12800 | 65.5 | 80.3 µs | 816 | 45.3 | 42.9 µs |
| 25600 | 131.1 | 142.2 µs | 921 | 51.2 | 66.4 µs |
| 51200 | 262.1 | 247.0 µs | 1061 | 59.0 | 113.9 µs |
| 102400 | 524.3 | 462.7 µs | 1133 | 62.9 | 209.8 µs |

Slope **0.834 µs/MB = 1200 GB/s**, intercept 25.7 µs. hipblas has slope
0.793 µs/MB = 1261 GB/s, intercept 25.4 µs — the same structure, differing by 5% of
stream efficiency. Perfect overlap of the 114 µs of mmac with a 159 µs DRAM floor
would give 159 µs; we get 247 µs, so compute and stream are largely **serialised**.
Prefetch depth is not the fix: `ns=2` generates a correct `cp_async_wait<1>` with
double-buffered `A_s`/`B_s` and is still slower.

**hipblas's tile shape does not transfer.** Its kernel is
`Cijk_Alik_Bljk_F8BS_MT128x64x256_TT8_2_WG16_32_1_MFWGS512_WGM1_bias_channelwise`
— 512 threads on a (128,64) tile, vs our 256 on (64,64). Copied into our DSL,
`bm128 bn64 bk256 t512 ns1` → **589.8 µs** (partition degrades to (4,2)).

**What the patched gate_up kernel does** (from disassembling
`m64/src/gateup_layoutA_vmcnt4.s`): weights go **global→LDS** via
`buffer_load_dwordx4 ... slc lds`, exactly the path our `matrix_load` uses. The
`buffer_load_ubyte_d16` clusters are the channelwise bias/scale epilogue gather,
not weights. Its loop body is 64 mmac + 32 `ds_read_b128` + 8 `ds_write_b128` +
**1328 address VALU** (the price of Layout A's blocked addressing), with 149
`s_waitcnt` at fine-grained `vmcnt(1..14,0)` granularity. **slc is already in on
our side, and hipblas's is partial** — only 8 of its 20 weight→LDS loads carry it.

> **Correction (2026-10-17).** An earlier revision of this paragraph concluded
> from the file above that "there is no *weights bypass LDS into VGPR* trick".
> **That conclusion is wrong**, and it was wrong for a specific, traceable
> reason: `gateup_layoutA_vmcnt4.s` is a **patched** kernel (Layout A), and the
> deliverable README explicitly warns that operand roles are patch-specific and
> **cannot be copied across kernels** ("不能跨 M 复制"). The Layout A reorder is
> what routes this particular weight through LDS. The **native** hipblaslt
> kernel does the opposite: weight B goes **global→VGPR directly into the MMA
> operand registers with no `lds` modifier at all**, and only A is staged through
> LDS. Both statements are true of different artifacts; only the native one is
> evidence about hipblas's method. This is why the reference `.s` used for the
> M=64 comparison must be the unpatched one for structural questions.

The one in-scope lever left is the MLS partition heuristic
(`ComputeMlsWarpPartition`, `src/hcu/op/mls.cc:506`; `AnalyzeResourceAxis`,
`src/hcu/transform/hoist_mls_resource.cc:1615`): the reference's plain
`ds_read_b128` lets it hold an arbitrary tile shape, while our
`ds_read_format_tensor_a/b` pins the partition and degrades to (4,2)/(8,1) with
3–4× LDS read amplification as soon as the shape moves. That needs a C++ rebuild.

<a id="2026-10-17f"></a>
## o/down: shape-aware config wins, split-K closes, and `ns>1` is a landmine (2026-10-17f)

Target was to beat hipblas on `o` and `down` (qkv/gate_up already win; M=64 out of
scope). Full write-up in `handoff.md`; the load-bearing parts:

### The chooser was shape-blind

`_decode_cfg` took only `M`, so the gate_up-tuned tile went to all four shapes.
The deficit is a property of the *shape*: **N ≥ 10240 → bn32/t256; N = 5120 →
bn16/t64**. Paired, same-process, idle card:

| shape | M | before | after | hipblas | ratio |
|---|---|---|---|---|---|
| o | 1 | 42.9 µs | **38.6** | 32.5 | 1.188 |
| o | 16 | — | **39.2** | 34.7 | 1.129 |
| down | 1 | 117.4 | **99.4** | 78.1 | 1.273 |
| down | 16 | — | **100.3** | 84.5 | 1.187 |

Still losing, but −10% on o and −15% on down. The rule now lives in
`decode_cfg`/`_validate_cfg` (`perf/gemm/fp8_channelwise_mls.py`), which validates
every invariant with the shape in the message. `bn16` is the floor of the search
space — `block_N < 16` cannot be expressed (MMA output micro-tile fixed at 16×16,
`src/tl_templates/hcu/gemm.h:150`; MLS B atom floor `tile_mn=16`,
`src/hcu/op/mls.cc:408`).

<a id="split-k"></a>
### Split-K: the motivating diagnosis was a confound, and the lever is closed

The "o/down are block-starved" claim came from a sweep at fixed K varying N, which
moves block count and total bytes together. Split-K holds bytes constant and only
multiplies blocks, so the honest test is a constant-`N·K` curve
(`/tmp/b0_grid_scale.py`, existing kernel unmodified):

| N | K (o curve) | K (down curve) | blocks | k-steps | GB/s o | GB/s down |
|---|---|---|---|---|---|---|
| 5120 | 8192 | 25600 | 320 | 32 / 100 | **1053** | **1302** |
| 10240 | 4096 | 12800 | 640 | 16 / 50 | 986 | 1282 |
| 20480 | 2048 | 6400 | 1280 | 8 / 25 | 1036 | 1247 |
| 40960 | 1024 | 3200 | 2560 | 4 / — | 923 | — |

**More blocks at constant bytes is worse**, and 320 blocks is the best point on
both curves. Split-K was built anyway and measured directly (paired, 7 rounds,
all four o/down cells) — it loses monotonically, 10-21% slower than the
already-losing baseline:

| shape | M | base | split_k=2 | split_k=4 | hipblas |
|---|---|---|---|---|---|
| o | 1 | 38.9 (1.234×) | 49.1 (1.558×) | 54.6 (1.731×) | 31.5 |
| o | 16 | 39.5 (1.113×) | 51.4 (1.446×) | 59.8 (1.685×) | 35.5 |
| down | 1 | 100.0 (1.273×) | 110.9 (1.411×) | 117.3 (1.493×) | 78.6 |
| down | 16 | 100.6 (1.178×) | 113.0 (1.322×) | 122.1 (1.429×) | 85.4 |

Builder, runner and tests removed; `DecodeCfg.split_k` remains as an assertable
closure marker. **Do not re-open without new evidence.**

One structural note that outlives the experiment: `out_idx` outputs are allocated
with `torch.empty` (`tilelang/jit/adapter/tvm_ffi.py:280`) so there is no host
hook to zero them, and a split-K kernel *accumulates* onto C. `T.clear(C)` in the
kernel races across `bz` (no grid barrier on HCU), and a `bz == 0` guard is the
same race. The only correct form is C as a caller-owned input zeroed on the same
stream inside the timed closure.

<a id="gemv-m1"></a>
### The M=1 GEMV structure is infeasible, by hardware

hipblas uses `Cijk_..._MT8x1x8192_MFWGS512_WGM1_ONLYN1` (N-tile-1 GEMV) at M=1.
That is not expressible here: the MMA output micro-tile is hardware-fixed at 16×16
(`src/tl_templates/hcu/gemm.h:150-151`; fp8 atom
`__builtin_hcu_mmac_f32_16x16x32_fp8_fp8` at `:118`) and the MLS B atom floor is
`tile_mn = 16` (`src/hcu/op/mls.cc:408-412`, surfaced as `kMinNPerWarp =
TransB ? 16 : 32` at `src/tl_templates/hcu/gemm_mls.h:36`). Padding B_s to 16
rows, `gemm_s_mls`, and transposing to `Cᵀ = B·Aᵀ` all still hit 16 — and padding
adds no blocks. The reachable equivalent, maximal split-K at bn=16, is measured
and rejected above.

### `num_stages > 1` is silently wrong outside bm64/bn64

The `ns=2` candidates in the o/down sweep did not just lose — they returned
non-finite or `rel` 0.75–580 values. Reproduced on all four shapes, all M,
bisected in `/tmp/ns2_repro.py`, `/tmp/ns_threshold.py`, `/tmp/ns_threads.py`:

| config | LDS | result |
|---|---|---|
| bm16 bn16 bk128 ns2 t64 | 8192 | NON-FINITE |
| bm16 bn32 bk128 ns2 t64 | 12288 | rel=2.47 |
| bm16 bn64 bk128 ns2 t64 | 20480 | rel=0.98 |
| bm16 bn128 bk128 ns2 t64 | 36864 | rel=0.90 |
| bm16 bn16 bk512 ns2 t64 | 32768 | **ok** |
| bm16 bn16 bk128 ns2 t128 | 8192 | **ok** |
| bm16 bn32 bk256 ns2 t128 | 24576 | rel=0.80 |
| bm16 bn32 bk256 ns2 t256 | 24576 | **ok** |
| bm64 bn64 bk128 ns2 t256 | 32768 | **ok** |
| any ns=1, any tile | — | **ok** |

It is **not** an LDS-size threshold (32768 ok vs 36864 wrong), not a k-step count,
and not deterministic in `bk` — it tracks the thread count at small tiles. The
mechanism is not understood; it is a lowering/pipeliner defect in a region no
shipped config uses.

**The adopted M=64 config is nevertheless sound, and that was checked rather than
assumed** — if the mechanism were a race, "correct" would only mean "has not lost
yet". 300 trials × 4 shapes × setprio {on, off} = 2400 runs of `decode_cfg(64)`
(bk128 ns2 t256): **worst rel 0.00, zero non-finite** (`/tmp/ns2_stability.py`).

Guards: both `fp8_mls_channelwise` and `_validate_cfg` assert `num_stages <= 1 or
(block_M, block_N) == (64, 64)`, and
`test_every_chosen_decode_config_is_correct` now runs the chooser's M=64 output on
all four shapes — the only place `ns>1` is exercised, and previously untested.

### The residual, and why the ratio is quoted as a range

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

**Net: o/down are closed on all four levers** — shape-aware config (adopted, the
real win), split-K (measured, closed), tile widening (catastrophic, see the table
above), pipeline depth (the `ns>1` region is broken anyway and `ns=1` is the safe
optimum). The residual 1.19×/1.27× at M=1 and 1.13×/1.19× at M=16 is not
reachable by adding blocks within the MLS/MMA structure.

### M=32: a second real win, from the thread count (found by the full matrix)

Running the full matrix surfaced a 2.0× cliff at the band edges — a shape getting
*faster* as M grows and then falling off:

```
down: M=16 100.8us -> M=32 209.9us (2.08x) -> M=64 148.0us -> M=128 402.1us
o:    M=16  39.7us -> M=32  75.4us (1.90x) -> M=64  50.4us -> M=128 117.1us
```

A 2.0× step is the signature of doing twice the work with the same parallelism.
At `bn16` a `bm32` tile is two 16×16 M-atoms, and the M-only table gave it **one
warp** (`t=64`), so the block became MMA-issue-bound instead of B-stream-bound.
Shrinking `bm` does not fix it — `bm16` at M=32 doubles B traffic (the grid is
`N/16 × M/16`, so each B tile is read twice) and measured worse. The untested
axis was **threads**:

| shape | M=32, t=64 | t=128 | t=256 |
|---|---|---|---|
| o | 75.2 µs (1.911×) | 72.3 (1.837×) | **46.9 (1.191×)** |
| down | 209.9 µs (1.944×) | 203.0 (1.880×) | **122.4 (1.133×)** |

`t=256` is a **1.60× / 1.71×** win, adopted (`threads = 64 if bm <= 16 else 256`
on the N=5120 branch). `bm32/bn32/t256` on gate_up/qkv was already correct and is
unchanged (`bm32` is right there — measured 0.861/1.040 against `bm16`'s
1.531/1.214).

M=128 has the same shape of problem but no fix: the chooser's `bm128/bn128/t512`
beat `bm128/bn16` (1.83 vs 3.57 on o) and `bm128/bn32` (2.69) — the wide `bn` is
what keeps the grid non-trivial.

### The final matrix (paired, chooser output, idle card)

`ratio` = ours/hipblas, **< 1 is a win**. M=1..32 rows are stable to ≤5% across
runs; M=64/128 rows move up to ~35% between runs (fewer blocks, more sensitive to
the shared box), so treat those two rows as indicative.

| shape | M=1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| gate_up | **0.93** | **0.78** | **0.87** | **0.88** | **0.91** | **0.86** | 1.08 | **1.00** |
| qkv | **0.85** | **0.82** | **0.83** | **0.84** | **0.85** | 1.13 | 1.05 | 1.15 |
| o | 1.19 | 1.14 | 1.11 | 1.11 | 1.13 | 1.19 | **0.89** | 1.83 |
| down | 1.27 | 1.23 | 1.08 | 1.18 | 1.18 | 1.11 | 1.09 | 2.23 |

**13 of 32 cells win.** They are gate_up M∈{1..32, 128}, qkv M∈{1..16}, and o M=64.
o/down still lose at M≤32 by 1.08-1.27× and at M=128 by 1.8-2.2×; qkv loses at
M≥32 by 1.05-1.15×.

The remaining gaps, honestly:
- **o/down M≤32 (1.08-1.27×):** B-stream-bound at ~1300-1500 GB/s vs hipblas's
  1543-1670. Closed on every lever tried (above). Not a block-count, tile,
  pipelining or epilogue problem.
- **M=128 (1.83× o / 2.23× down, 1.15× qkv):** only 40 blocks at bn128 on N=5120,
  and `bn` cannot shrink without shrinking the grid further. Not addressed.
- **qkv M≥32 (1.05-1.15×):** 320 blocks at bn32; the M=32 thread fix does not
  apply (it already uses t=256) and `bm32/bn32/t256` is the measured best of its
  neighbourhood.

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
"utilization" numbers that exceed the measured ~2016 GB/s read ceiling and are
not physical. (The old "91.5% of 1800 GB/s" ceiling was wrong — see the
correction in the M=64 section.)

<a id="b-direct"></a>
## B-direct-to-VGPR: the cost, the root cause, and why it is closed (2026-10-16, corrected 2026-10-17, closed 2026-10-17c)

Goal of this round: make the weights (B) bypass LDS, hipblas-style. All figures
are same-process paired runs on an idle card (`hy-smi` HCU% 0), 800 blocks x 256
threads, 262 MB of B, 20 passes. Harnesses: `/tmp/bstream.cu`, `/tmp/mlsbw.cu`,
`/tmp/bktest.cu`.

**Headline (2026-10-17c, measured — this lever is CLOSED).** Two candidate
directions were carried to completion and both are negative.

1. **B-direct is now fully wired end-to-end and produces correct results
   (rel=0.00000) — and it is 3.7× slower than baseline even at `WarpM=4`
   (904.2 vs 246.2 µs), and still 2.1× slower at `WarpM=1` (513.8 µs).** So the
   earlier reading — "the 3.62× is a `WarpM=4` partition artifact; retry at
   `WarpM=1`" — is **refuted by the full GEMM**. The loader-only probe measured
   only the `buffer_load`s; the real kernel adds the mmac chain, the A-MLS path,
   the per-lane B fragment shuffle, and the half-sector pattern across the actual
   4-warp M-split, and the win does not survive them.
2. **A 16-consec-k fragment cannot rescue it** — the fp8 mmac B operand is
   hardware-fixed at 8 B/lane (`int32x2`, `gemm.h:109`), no fp8 `16x16x64` exists
   anywhere in the tree, and the 16-consec-k loader's per-lane (n,k) ownership is
   *provably not* the mmac's fragment (code-verified below), so it would need a
   cross-lane repack. Even crediting the entire loader-only gain, the arithmetic
   cannot reach baseline (bound below).
3. **The MLS producer lever is closed negative too** — measured, no change made.

Do not re-open either direction without new evidence. Details below.

### Three independent costs, measured

Loader-only kernels in `bstream.cu`, all using the same multi-load asm with
early-clobber outputs, same grid/block, same B bytes. Times are per pass
(us/20) and the ratios are between variants of that one harness:

| variant | fragment | B reads | us/pass | rate | vs 1x-dwordx2 |
|---|---|---|---|---|---|
| `d8b` | 8-consec-k, `buffer_load_dwordx2` (8 B/lane) | 4x (once per m-warp) | 789.2 | 332 GB/s | **3.62x** |
| `d16b` | 16-consec-k, `buffer_load_dwordx4` (16 B/lane) | 4x | 394.7 | 664 GB/s | **1.81x** |
| `d8b_1w` | 8-consec-k, `dwordx2` | **1x** | 217.8 | 1204 GB/s | 1.00x |
| `d16b_1w` | 16-consec-k, `dwordx4` | **1x** | **123.9** | **2115 GB/s** | **0.57x** |

1. **The 4x m-warp redundancy is the dominant cost.** `d8b_1w` is `d8b` with the
   loads predicated to one m-warp per n-warp: reading each B byte once, the
   identical pattern costs 3.62x less. Our block is `t256` = 2 n-groups x 4
   m-warps (`nbase = (warp_id / 4) * 32`), so every B byte is fetched 4x. That is
   exactly what LDS staging buys, and it is why the standalone B-direct GEMM
   measured **975.4 us vs 243.4 us** for the LDS control (4.0x) even after the
   VMFault and the scratch spill were fixed.
2. **The 8-consec-k fragment costs an independent 2x.** `d8b` and `d16b` issue
   identical bytes and differ by 2.0x: `dwordx2` makes a wave touch 16 rows x 32 B
   (half-sectors), `dwordx4` 16 rows x 64 B (full sectors). Same 2x at 1x
   redundancy (217.8 vs 123.9 us/pass, 1.76x).
3. **The inference from 1 and 2 — "so B-direct wins at 16-consec-k + WarpM=1" —
   is REFUTED by the full GEMM.** The reason is worth stating plainly: these
   loader-only numbers measure *only the B `buffer_load`s*. Everything the real
   kernel does around them — the mmac chain, the A-MLS/LDS path, the per-lane B
   fragment register shuffle, and the half-sector pattern across the actual
   4-warp M-split — sits outside the probe. The full GEMM is the arbiter, and it
   says B-direct loses.

The "4× is a partition artifact, not a fetch-pattern property" argument (true in
isolation: our `WarpM=4` splits M across 4 independent waves that each re-issue
the same B, while the native hipblas B index carries **no M coordinate** and its
4 waves split **N**) predicts that `WarpM=1` should recover the whole 4×. It does
recover about half — and lands nowhere near baseline. See below.

### The full GEMM refutes it (measured 2026-10-17c, same-process, idle card)

`/tmp/bdirect_probe.py`, M=64 gate_up (`A 64×5120`, `B 51200×5120` fp8 = 262.1 MB,
`C 64×51200` fp32), `bm64 bn64 bk256 ns1 t256 sp1`, CUDA-graph replay, stable
across runs. Every row verified **rel=0.00000**.

| config | µs | GB/s | vs baseline |
|---|---|---|---|
| baseline — Square, B-via-LDS | **246.2** | 1065 | 1.00x |
| FullCol, B-via-LDS (WarpM=1, no B-direct) | 273.9 | 957 | 1.11x |
| B-direct — Square (WarpM=4, 8-consec-k) | 904.2 | 290 | **3.67x** |
| B-direct — FullCol (WarpM=1, 8-consec-k) | 513.8 | 510 | **2.09x** |

`WarpM=1` does what the loader probe predicted — it removes the m-warp
redundancy and halves the B-direct cost (904.2 → 513.8) — but it lands **2.1×
above baseline**, not near it. `WarpM=1` by itself also costs 27.7 µs (273.9 vs
246.2) before any B-direct is involved.

### Why 16-consec-k cannot close it — two independent proofs

**(a) It cannot feed the mmac without a cross-lane repack.** The fp8 mmac B
operand is `int32x2 b_val` = 8 B/lane (`src/tl_templates/hcu/gemm.h:109`,
`__builtin_hcu_mmac_f32_16x16x32_fp8_fp8`). There is **no fp8 `16x16x64`** — the
only `16x16x64` builtins in the tree are fp4 (`scale_buffer.hpp:545`) and ROCm i8.
So 16-consec-k can only ever be a *load* shape, never an mmac operand shape. The
two loaders in `tilelang_ds_read_format.hpp` have provably different per-lane
(n,k) ownership:

- `b_vgpr_direct_load` (8-consec-k): `n = (lane&15) + panel*16`,
  `k0 = ki*32 + (lane>>4)*8` → a lane owns **16 n-rows × 4 k-offsets**, its
  consecutive k-chunks **32 apart**. Its header comment states it targets "the
  same `B_local` VGPR layout that `ds_read_format_tensor_b` produces (so the
  downstream mmac is **byte-identical** and unchanged)".
- `b_vgpr_direct_load16` (16-consec-k): `n_row = lane&31`,
  `kwin = (lane>>5)*4 + w` → a lane owns **32 n-rows × 64 contiguous k**. Its own
  comment says the ownership is "**DIFFERENT** from the `ds_read_format_tensor_b`
  / mmac fragment: … no lane owns the mmac's '8-consec-k at fixed n with k-groups
  32 apart' pattern".

The mmac hardware fixes which (n,k) each lane must supply, so the 16-consec-k
layout would have to be shuffled into the 8-consec one (an 8-byte exchange with
`lane^16` per 16 B chunk). That repack is untested — but (b) shows it cannot
matter.

**(b) The arithmetic bound: the swap is worth at most 93.9 µs/pass, and the gap
is 267.6 µs.** The 8→16-consec swap can only recover the sector-penalty component,
whose measured size is the loader-only delta `d8b_1w − d16b_1w` = 217.8 − 123.9 =
**93.9 µs/pass**. The B-direct FullCol gap to baseline is 513.8 − 246.2 =
**267.6 µs**. Since 93.9 < 267.6, **even under the most generous additive model —
crediting the entire loader gain and charging nothing for the repack — B-direct
lands at ~419.9 µs, still 173.7 µs above baseline.** This bound is independent of
the overlap model, so it settles the question without building the repack.

(Note the redundancy term does not transfer either: loader-only said 3.62×, the
full GEMM shows 1.76×. A large part of the B load *is* already hidden, so every
lever measured in isolation is smaller in the real kernel, not larger.)

**Conclusion: B-direct-to-VGPR is closed for this structure.** The residual gap to
hipblas does not live here — the M=64 section above already localizes it: the gap
is 5.8%, it is overlap quality, and LDS bytes/output is already identical to
hipblas's.

### WIP code state

The B-direct dispatch on branch `feat/hcu-smls-b-vgpr`
(`src/tl_templates/hcu/mls/tilelang_ds_read_format.hpp`:
`b_vgpr_direct_load`, `b_vgpr_direct_load_stage`, `b_vgpr_direct_load16`, plus the
`tl.hcu_b_vgpr_direct` annotation gate and its lowering branch) is complete and
correct but **not a winning path** — closed-experiment infrastructure, kept on the
branch, not merged. The 16-consec-k `"+v"` → `"=&v"` fix at
`tilelang_ds_read_format.hpp:557` is a genuine correctness fix regardless.

<a id="mls-producer"></a>
### The MLS producer lever is also CLOSED NEGATIVE (measured 2026-10-17c)

The "91.5% of 1646 GB/s" framing in earlier notes is wrong. A plain grid-stride
read of 5.12 GB measures **2016 GB/s**, and the exact B fetch pattern (64 rows x
256 B, rows 5120 B apart) reaches **1881-2016 GB/s** with a plain copy — 93% at a
64 B run per row, 99%+ at 128 B and up (`bktest.cu`, TPR sweep). The pattern is
not the problem.

What is: **`matrix_load_128x16_b8` is 1.25x slower at moving those bytes than a
plain copy** (`mlsbw.cu`, paired and interleaved in one process):

| producer (same tile, same bytes) | per pass | rate | of 2016 peak |
|---|---|---|---|
| MLS `matrix_load` global->LDS (our real path) | 164.0 us | 1599 GB/s | 79.3% |
| plain `buffer_load_dwordx4` copy | 131.0 us | 2002 GB/s | 99.3% |
| plain copy + 1 `__syncthreads()` per k-step | — | 2013 GB/s | 99.9% |
| MLS, 2-deep double-buffered (2x LDS) | — | 1456 GB/s | 72.2% |

164.0 us/pass reproduces the previously recorded "B stream alone 166.1 us".
Neither the per-k-step barrier (the plain copy with the same barrier still hits
2013 GB/s) nor pipeline depth (2-deep is *worse*) explains it — it is intrinsic to
`matrix_load_128x16_b8`, which emits 2 instructions per wave per k-step
(2 x 8 waves x 1024 B = 16 KB, one pass over the tile, no redundancy).

**This is a real ~21% standalone gap, but it does NOT translate end-to-end.** The
earlier reading — "33 us/pass of headroom against a 23 us total gap to hipblas" —
is falsified by the swap test. The 128×16 atom (what `ComputeMlsWarpPartition`
picks for the (64,256) fp8 trans tile → warp(2,2)) is intrinsically ~20% slower
**in isolation** than the 64×32 atom (warp(1,4)) — standalone probe
(`/tmp/mlsatoms.cu`): 128×16 = 1600 GB/s (79%) vs 64×32 = 1892 (94%) — and
`AnalyzeResourceAxis`'s axis choice makes no difference (1603 vs 1604).

But a true same-process A/B (distinct DSL sources, CUDA-graph replay, idle card,
both rel=0.000000) shows swapping to the *faster* 64×32 atom makes the full M=64
GEMM **15–16% SLOWER**: 246.2 µs baseline vs 287.2–287.5 µs. The atom swap changes
the warp→atom assignment at every k-step barrier, and the overlap/serialization
quality degrades more than the producer gain recovers.

**Conclusion: the kernel is overlap-quality-bound, not producer-throughput-bound.**
The 128×16 atom's ~21% standalone penalty is already (almost) fully hidden by the
pipeliner in the real kernel, so there is no 33 µs to reclaim by moving the
`matrix_load` atom / partition / axis. Do **not** pursue this lever. (Gotcha that
cost real time: after a C++ change you must clear
`/root/.tilelang/cache/<version_gitXXX>` or the in-process IR cache silently
serves the old partition; standalone HIP probes must build with
`/opt/dtk/bin/aicc`, not hipcc.)

**Both B-fetch levers are therefore closed** — the producer path (this section) and
the replacement path (B-direct, above) — and for the same underlying reason: the
B stream is already overlapped well enough in the real kernel that standalone
B-fetch gains do not survive into it. The remaining 5.8% to hipblas is overlap
quality (see the M=64 section), not B-fetch throughput.

### Fragility notes (these cost a day)

- Multi-instruction inline asm chaining `buffer_load_*` **must** use early-clobber
  outputs (`"=&v"`). With plain `"=v"` LLVM may allocate an output over a later
  input (the next load's address register); with no `s_waitcnt` between loads the
  in-flight return clobbers that address and the kernel VMFaults. Verified A/B in
  `/tmp/bld3.cu`, `/tmp/bld5.cu`: `"=v"` faults, `"=&v"` runs, same offsets,
  exact results. `hcu_buffer_load_impl_inline_asm` chained in an unrolled loop
  shows the same fault intermittently.
- A dynamic slice index (e.g. `&B_local[(k&1)*128]`) blocks register promotion and
  spills the array to scratch (ScratchSize 272 B -> 7.77x slowdown). A constant
  index gives ScratchSize 0.

<a id="2026-10-17g"></a>
## qkv M=32 lever adopted; o/down levers exhausted (2026-10-17g)

Target: make o/qkv/down beat hipblas at M<=32. Current standing, chooser vs
`torch._scaled_mm`, paired same-process round-robin, 10 rounds, min-of-rounds
(the o/down ratios move with hipblas's mode -- see the caveat below):

| shape | M=1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| gate_up | **0.917** | **0.757** | **0.866** | **0.873** | **0.909** | **0.847** |
| qkv | **0.868** | **0.757** | **0.832** | **0.770** | **0.822** | 1.031 (0.945-1.035) |
| o | 1.204 | 1.029 | **0.980** | 1.119 | 1.093 | 1.077 |
| down | 1.275 | 1.178 | 1.075 | 1.161 | 1.125 | 1.091 |

gate_up wins 6/6, qkv 5/6, o 1/6, down 0/6.

### First: every earlier o/down and M=1/16 ratio in this doc was measured against a wrong baseline

`decode_cfg` sets `setprio=1` for M<=32, but the sweep harnesses compared against
explicitly-constructed configs with `setprio=0`. That is not the chooser. It cost
10% on `o` M=32 alone (51.8us sp0 vs 46.7us sp1) and it made the o/down deficits
look ~8 points smaller than they are. Re-measured with `decode_cfg`'s exact
output, o M=1 is **1.204-1.235**, not the 1.048 the older table claims. Lesson:
compare against `decode_cfg(M,N,K)` itself, never a hand-copied config.

Also: **hipblas is the noisy term at o/down.** `_scaled_mm` read 31.4, 31.5, 32.5,
35.8 and 36.8 us across runs for o M=1 while our kernel moved 38.6 -> 38.9. The
ratio therefore has a ~20% run-to-run spread and must be quoted as a range. Our
absolute times are stable to <1%.

### The one real find: qkv (N=10240) at M=17..32 wants bk128/t128

At N=10240 with bm32, `bn32/bk128/ns1/t128/sp1` beats the `bn32/bk256/ns1/t256/sp1`
the chooser picked by **9-10%** (52.3 vs 57.8 us, paired, 16 rounds, both arms
sp1). That moves qkv M=32 from 1.14x to **1.03x** (best round 0.945) -- a loss to
parity, the largest single-cell gain of the session.

Evidence it is real, not a harness artifact:
- 11 independent paired measurements across 8 separate processes, all 0.84-0.91.
- Survives an L2 flush before every timed replay (0.852 warm -> 0.862 cold) while
  hipblas is bit-identical under the same flush (51.2 us both ways).
- Holds at N=10240 for K = 5120, 10240 and 20480 alike (0.852 / 0.843 / 0.892).

Evidence it is **not** a general rule, and the mechanism is unknown:
- At M=32/K=5120 with bn32, bk128/t128 **loses** by 17-38% at N = 11264, 15360,
  20480, 25600 and 51200. The next point up from 10240 already loses.
- Not a byte-count effect (K-independent), not an L2-capacity effect (L2 is 8 MB;
  a 52 MB B cannot be resident), not block count (N=15360 has *more* blocks and
  loses by 38%).

So it is written into `decode_cfg` as the one measured point (`bm == 32 and
N <= 10240 and K % 128 == 0` inside the `N >= 10240` branch), not extrapolated to
a range. `N=9216` was measured as a win too, but the chooser never reaches bn32
for N<10240, so it is not encoded.

### o/down: the last unexplored lever (`ns=2` at bn16) is now closed too

Every shipped decode config is `ns=1`, i.e. no overlap between the next k-step's
loads and the current k-step's MMA. o/down have the fewest blocks (320), so they
have the least inter-block parallelism to hide that with -- which made `ns=2` the
one structural lever left. The doc's ns2 table lists `bm16 bn16 bk128 ns2 t128` as
OK, so it was worth testing properly rather than trusting one pass.

`/tmp/ns2_odown.py` (builder's ns guard relaxed on purpose, /tmp only), 40 random
trials per config, o and down at M=16:

| config | correctness | o us | down us |
|---|---|---|---|
| **bn16 bk256 ns1 t64 (shipped)** | **0/40 bad** | **39.5** | **100.8** |
| bn16 bk256 ns2 t128 | **40/40 bad** (rel 3.8e4) | — | — |
| bn16 bk256 ns2 t64 | **40/40 bad** (rel 5.5e4) | — | — |
| bn16 bk128 ns2 t128 | 0/40 bad | 138.1 (3.49x) | 420.8 (4.18x) |
| bn16 bk512 ns2 t128 | 0/40 bad | 41.6 (1.05x) | 105.3 (1.04x) |
| bn16 bk256 ns2 t256 | 0/40 bad | 51.2 (1.30x) | 143.0 (1.42x) |

The configs that would be fast are silently wrong (40/40, rel ~1e4); every
correct ns2 variant is slower. This independently re-confirms the ns>1 landmine
with a clean 40/40 signal on both shapes, and closes the lever. The pattern
matches the earlier finding: at bk256 the correctness flips with the thread count
(t256 ok, t128/t64 wrong).

### Where o/down actually stand

Our absolute time is flat across M (o: 38.9 -> 46.7 us over M=1..32; down:
100.0 -> 122.4), so both are purely B-bandwidth-bound. Achieved bandwidth:
gate_up 1632 GB/s (81% of the 2016 ceiling), down 1311 (65%), qkv 1227 (61%),
o 1079 (53%) -- efficiency tracks B size, which is a fixed-overhead signature,
and hipblas reaches 1665 GB/s on down M=1.

Every in-tree lever for o/down is now measured and closed: bn16 is the hardware
floor (MMA atom 16x16, MLS B atom floor 16); bk 128/256/512 swept (256 best);
threads 64/128 identical; `setprio=1` required and already in the chooser;
split-K closed (10-21% slower); the M=1 GEMV structure is not expressible; and
`ns=2` closed above. The residual 1.08-1.28x is the structural difference between
hipblas's N-tile-1 GEMV (no LDS round-trip, 5120 tiny blocks) and a 16x16-atom
MLS GEMM, not a config that has not been tried.

Do not re-open o/down config search without a new mechanism.

<a id="2026-10-17h"></a>
## 2026-10-17h — o/down decomposed: the load path already wins, the mmac side is the whole deficit, and it is closed

The previous section closed every *configuration* lever for o/down. This one
answers the prior question — **which side of the kernel is actually losing?** —
and then closes the remaining *instruction-level* knobs.

### The load path already beats hipblas on all four o/down cases

`/tmp/odown_decomp.py` builds four arms per (shape, M) with the **same tiles and
the same access pattern**, all in one process, round-robin interleaved (7 rounds,
median-of-rounds, GPU 1 verified idle):

- `hip`   — `torch._scaled_mm`
- `full`  — the chooser kernel (`decode_cfg`)
- `Bonly` — only B copied global→LDS via MLS, no mmac, no ds_read
- `AB`    — A and B both copied, still no mmac

Because `Bonly`/`AB` use the *same* `(bn, bk, ns, t)` as `full`, the `full - AB`
delta is exactly what the ds_read + mmac + epilogue side adds on top of an
identical global→LDS stream.

| case | hip | AB (load only) | full | AB/hip | full/hip | mmac side adds |
|---|---|---|---|---|---|---|
| o M=1    | 37.3 | **35.7** | 43.5 | **0.957** | 1.167 | +7.8 us |
| o M=16   | 34.9 | **32.0** | 39.4 | **0.917** | 1.128 | +7.4 us |
| down M=1 | 82.7 | **80.8** | 99.5 | **0.977** | 1.203 | +18.7 us |
| down M=16| 88.8 | **84.2** | 100.3| **0.948** | 1.130 | +16.2 us |

**The load path is not the problem — it is already faster than hipblas on all
four.** The entire 1.13–1.20x deficit is the per-k-step ds_read/mmac/epilogue
cost, and it is *proportional to the work*, not a fixed intercept (it doubles
from o to down, tracking K). This kills the "o/down are block-starved / the grid
is too small" story from the outside: adding blocks cannot remove a cost that
scales with the k-loop.

### `num_stages=2` at bn16: the exact root cause, and it is a real repo bug

The ns=2 table in the previous section recorded *that* `bn16 bk256 ns2 t64/t128`
is 40/40 wrong. `/tmp/ns2_probe*.py` dumping the generated source shows *why*,
and it is a genuine lowering defect, not a race:

```c
void* A_s = buf_dyn_shmem + 0;
void* B_s = buf_dyn_shmem + 12288;          // 2 * 6144  <-- packed stride
...
_tl_mls_h_0.async_mls_load_asm<...>(&A_s[(((k+1)&1)*6144) + ...], ...);
ds_read_format_tensor_a<...>((&A_s[((k & 1) * 4096)]), ...);   // 4096 != 6144
```

At `bn16` with `t=64/128` the MLS dst selects the **packed** LDS layout
(`TryGetMlsDstActualSizeBytes`, `src/hcu/op/mls.cc:89` → `packed_bytes_for_4k_slots`,
6144 B for a (16,256) fp8 trans tile), and the multi-version stride follows it.
But the consumer's stage offset comes from `TryGetMlsPackedLeadingElemOffset`
(`src/hcu/op/ds_read_format.cc:639`), which only applies when the region rank is
> 2 — and it falls back to the logical `OffsetOf` stride **4096**. The producer
and consumer disagree on the stage stride, so stage 1 is read from the wrong
place. At `t=256` the packed branch is not taken, both sides use 4096, and ns=2
is correct (rel=0.00000). That is why correctness flips with the thread count.

This is worth fixing independently of o/down — it is the reason
`num_stages > 1` has to be asserted off outside `bm64/bn64`.

### Double buffering is closed even where it is correct

Where ns=2 *is* correct it does buy real overlap at bk512 — `ns2/ns1` =
0.888/0.927 (o M=1/16), 0.782/0.841 (down M=1/16) — but at **bk256 it hurts**
(1.130/1.168, and 1.048/1.136 for `bn32/bk256/t256`). And the best *absolute*
ns=2 config (`bn16/bk512/t128` = 40.6/42.7/98.2/106.9 us) is no better than the
shipped `ns1/bn16/bk256/t64` (38.4/39.7/99.4/100.3) — a 1.2% win on down M=1 is
the only thing it has, inside hipblas's own noise band. `ns=3` is 1.5–1.9x worse.

A **hand-written** two-buffer pipeline (`/tmp/manual_pipe2.py`: explicit
`T.ptx_wait_group(2)`/`(0)`, buffers alternated by hand) is *correct* (rel
0.00000) but 1.26–1.70x slower than the ns=1 baseline at the same tile. So the
overlap is not being left on the table by a missing config — the pipelined form
is genuinely slower here.

### More blocks do not help the full kernel (so split-K's premise is dead twice over)

`/tmp/grid_scale.py` at fixed K, varying grid size (load-only vs full):

| blocks | 320 | 640 | 1280 | 2560 |
|---|---|---|---|---|
| load-only GB/s | 1298 | 1417 | 1644 | 1743 |
| full GB/s | 1181 | 1200 | 1288 | 1323 |

The load path responds strongly to occupancy; the full kernel barely moves
(+12% for 8x the blocks). Combined with the split-K measurement (10–21% slower),
this closes the occupancy family for good: the limiter is per-k-step and scales
with the work, not with resident warps.

### The last instruction-level knobs, all measured closed

`/tmp/ann_sweep.py`, `bn16/bk256/ns1/t64/sp1` (the shipped tile), 5 rounds,
interleaved, paired vs hipblas:

| arm | o M=1 | o M=16 | down M=1 | down M=16 |
|---|---|---|---|---|
| base (shipped) | 38.7 (1.186) | 39.4 (1.123) | 99.7 (1.271) | 100.5 (1.189) |
| `tl.hcu_a_respect_layout_map=1` | 38.7 | 39.2 | 99.7 | 100.6 |
| `tl.hcu_b_respect_layout_map=1` | 38.7 | 39.5 | 99.5 | 100.5 |
| both | 38.7 | 39.4 | 99.7 | 100.5 |
| `policy=FullRow/FullCol` @ t128 | 39.2 | 39.8 | 100.6 | 101.3 |

- `k_pack=2` — **hard-rejected**: `Check failed: (gemm->kPack_ == 1) is false:
  MatrixLoad dst Gemm consumer must have kPack=1`. The MLS path forbids it.
- `tl.hcu_b_ds_read_alt=0` — `unsupported B interleave=0`; for b8-trans the
  dispatcher has only the Alt1 specialisation, which is already the default.
- `respect_layout_map` on A, on B, and on both — **exact no-ops** (within 0.3%).
- Warp policy: at bn16 with 2 warps `FullRow` and `FullCol` produce *identical*
  times, i.e. the partition does not change at this tile, and both are 1.2%
  slower than t=64. `bm32/bn16` at M<=16 is 1.8–2.6x worse, as the 16-row MMA
  atom floor predicts.

### Honest bound

Within the in-tree MLS path, **o and down cannot be made to beat hipblas by
configuration.** The measured decomposition says the load path already wins
(0.917–0.977) and a per-k-step compute cost of +7.4 to +18.7 us is what loses.
Every knob that could attack that cost — tiles, threads, bk, bn, ns, swizzle,
split-K, hand-written pipelining, warp policy, k_pack, ds_read alt, layout-map —
is now measured closed, and the two families that would remove it outright
(occupancy via more blocks; overlap via double buffering) are both measured
negative on their own terms.

A best-case estimate that transfers the *bk512* overlap fraction onto the shipped
tile gives o M=1 ~34.1 (1.046, still losing), o M=16 ~36.8 (1.051), down M=1
~77.7 (0.991), down M=16 ~84.4 (0.998) — parity on down at best, still losing on
o, and inside hipblas's own ~20% spread at these shapes. The remaining
structural difference is that hipblas runs an N-tile-1 **GEMV** at M=1 (no LDS
round-trip at all) and ~40 warps/CU at M=16, while the MLS path is pinned to a
16x16 MMA atom and its partition degrades as soon as warps are added at these
tile sizes.

**Do not re-open the o/down config search without a new mechanism.** The one
mechanism that would plausibly change the answer is the MLS partition heuristic
(`src/hcu/op/mls.cc:506`, `src/hcu/transform/hoist_mls_resource.cc:1615`), which
is a C++ lowering change and is out of scope under the current directive.

### The `ns>1` landmine is root-caused and FIXED (packed MLS LDS stride)

The 2026-10-17f section recorded that `num_stages > 1` was silently wrong outside
`bm64/bn64` and could not explain it. The mechanism is now proven, and it was a
real bug in the tree, not a tile-geometry limit.

**Root cause.** An MLS dst tile whose layout is *packed* occupies more LDS bytes
than its logical element count. `TryGetMlsDstActualSizeBytes` knew this; the
Python ds_read consumer did not. For a `(16, 128)` fp8-trans tile the packed size
is `4096 + tile_issue_k * 1024` bytes (4 KB slot header + a 1 KB payload per
k-tile): 5120 at `bk=128`, **6144 at `bk=256`**, 8192 at `bk=512` — where it
finally coincides with the logical 8192 and the bug disappears. So the producer
(C++ `MatrixLoad`) and the shared-memory allocation both advanced the stage
pointer by 6144, while the consumer (`retrieve_mls_lds_base_ptr`) advanced it by
`prod(shape[-2:])` = 4096. Every `ns > 1` kernel read stage ≥ 1 from the wrong
address. Dumped source at `bm16/bn16/bk256/ns2/t64`:

```
producer:  A_s[((k + 1) & 1) * 6144]      # C++ MatrixLoad, packed stride
consumer:  A_s[(k & 1) * 4096]            # Python ds_read, logical stride
```

**Why it tracked the thread count.** `ComputeMlsWarpPartition` picks the MLS tile
from `(block_mn, block_k, threads)`: at `t=64` fp8-trans gets `(16, 128)` — packed
— and at `t=256` it gets `(16, 64)`, which matches no packed branch and is
contiguous. `bk=512` lands on 8192 == logical by arithmetic accident. That is the
whole of the "it tracks the thread count" observation; nothing was
geometry-limited.

**Fix (three consumers, one source of truth).** The packed byte count was
extracted into `MlsPackedTileBytes` in `src/hcu/op/mls.cc` and exported over FFI
as `tl.MlsPackedTileElems`; `TryGetMlsDstActualSizeBytes` (allocator),
`TryGetMlsPackedLeadingElemOffset` (C++ producer) and the new
`mls_packed_tile_elems` in
`tilelang/hcu/intrinsics/hcu_mmac_emitter_utils.py` (Python consumer, via
`retrieve_mls_lds_base_ptr`) now all read it. `ldmatrix_mls_a` / `ldmatrix_mls_b`
pass `tile_mn`/`tile_k`/`mls_trans` through.

**Verification.** `bm16/bn16/bk256/ns2` at `t=64` and `t=128` went from 40/40
wrong (rel ≈ 1.6e4) to `rel = 0.00000`; every previously-correct configuration is
unchanged. Covered: o/down/gate_up at M=1 and M=16, `ns` up to 8, `bk` 64–512,
`bm`/`bn` 16–64, plus a bf16 MLS probe. `92 passed, 58 skipped` in
`test_tilelang_gemm_mls.py` + `test_hcu_gemm_fp8_mls_channelwise.py`.

**This does not re-open o/down.** The shipped `o`/`down` tiles are `bn16/t64` at
`ns=1`, and the decomposition above shows the deficit is the per-k-step
ds_read/mmac cost, which `ns` does not touch — `ns=2` at `bk512` was measured at
0.782–0.927 of `ns=1` *relative* but with no better absolute time. What the fix
buys is that the correctness landmine is gone, so a future `ns>1` tile is a
performance question rather than a silent-wrong-answer question. The
`num_stages <= 1 or (bm, bn) == (64, 64)` asserts in
`perf/gemm/fp8_channelwise_mls.py` were reworded accordingly: they now say
"untuned", not "incorrect". `test_num_stages_gt_1_mls_is_correct` and
`test_mls_packed_stride_is_shared_by_producer_and_consumer` pin the fix — the
former fails with rel ≈ 1.6e4 if the consumer's stride is forced back to 0.

<a id="2026-10-17i"></a>
### 2026-10-17i — the fix invalidated one measurement; re-timed, and the last two mechanisms close

**1. The `ns>1` fix invalidated the `bn16` ns=2 timings, so they were re-measured.**
The 2026-10-17g sweep found `bk256 ns2 t64/t128` at bn16 to be *the two fastest*
ns=2 candidates — but those are exactly the two the packed-stride defect broke
(40/40 wrong, rel ~1e4), so "fast" was a measurement of garbage. With the fix in
place they are correct and therefore, for the first time, validly timed. Paired,
one process, 6 resident graphs, 7 rounds, min-of-rounds, GPU 2 idle:

| o/down arm (bn16, setprio=1) | o M=1 | o M=16 | down M=1 | down M=16 |
|---|---|---|---|---|
| bk256 ns1 t64 (**shipped**) | 38.9 | 39.4 | 100.0 | 101.0 |
| bk256 **ns2** t64 | 1.383 | 1.386 | 1.406 | 1.401 |
| bk256 **ns2** t128 | 1.399 | 1.528 | 1.451 | 1.447 |
| bk256 ns2 t256 (always-correct control) | 1.292 | 1.419 | 1.336 | 1.425 |
| bk512 ns2 t64 | 1.053 | 1.171 | **0.989** | **1.000** |
| hipblas `_scaled_mm` | 31.4 | 39.7 | 78.6 | 85.4 |

(ratios are vs the shipped ns1 baseline; µs only for the baseline and hipblas.)
So the corrected ns=2 is **29–45% slower**, uniformly, and it agrees with the
always-correct `t256` control. The earlier "the fast configs are silently wrong;
the correct ones lose" reading survives, but for a weaker reason than it looked:
the fast ones were never fast. `bk512 ns2 t64` is the sole parity point, and
parity on a 1.19x deficit is not a win. **ns=2 at bn16 is closed, decisively.**

This also retires the hope that the fix would re-open o/down: the whole reason
ns>1 was worth re-testing was that the two fastest-looking candidates had been
invalid, and they are not fast.

**2. Independent accumulator chains — built and closed (1.52-1.64x slower).**
At `bn16/t64` there is exactly one 16x16 accumulator, so all 256 mmacs of the
k-loop form a single serial chain issued by a single wavefront. That was the last
un-tested mechanism for the measured `full - AB` deficit (the ds_read+mmac side,
+7.8us on o / +18.7us on down). `/tmp/split_acc.py` splits it in-tree — two MLS
buffers of `(16,128)` per operand, two fragments `C0`/`C1`, summed in the
epilogue — giving two chains of 128:

| o/down, M | baseline | 2 fragments | ratio |
|---|---|---|---|
| o M=1 | 38.9 | 59.8 | 1.539 |
| o M=16 | 39.7 | 60.5 | 1.524 |
| down M=1 | 99.8 | 164.2 | 1.644 |
| down M=16 | 100.6 | 165.1 | 1.641 |

The extra MLS stream (4 copies/stage instead of 2) and the doubled LDS cost far
outweigh any chain overlap. **Closed.**

**3. A second, pre-existing packed-layout defect, in an unreachable config.**
`bn16/bk128/ns2/t64` is wrong (rel 0.55) at every K — and it was wrong *before*
this fix too (rel 3e4 with the consumer on the logical stride), so the fix
reduces it without removing it. The emitted stage stride for the packed tile is
`idx*1024 + idx*64`, i.e. 1088 bytes, while the tile is 2048 bytes — the two
pipeline stages overlap in LDS. `bn16/bk128/ns4/t64` is *correct* (stage stride
2048), and `bn16/bk128/ns2/t128` is correct (tile is `(16,64)`, not packed). So
the break is specific to the packed branch at `tile_issue_k == 1`. `bk256`
(`tile_issue_k == 2`) is the case the fix repaired and it is exact.
**Not chased further**: `bn16/bk128/ns2` is unreachable from `decode_cfg`
(`_validate_cfg` restricts `ns>1` to `bm64/bn64`, and `bk128` at `bn16` was
measured 1.596x slower at `ns=1` anyway). Recorded so the next reader does not
mistake it for a regression from this fix, and so nobody re-opens it as a win.

**Where that leaves o/down.** Today's paired numbers, same process, idle card:
o M=1 1.080, o M=16 1.000 (a tie), down M=1 1.202, down M=16 1.180. Every
mechanism that could have hidden the `full - AB` deficit is now measured and
negative: pipeline depth (both before and after the fix), independent chains,
more warps at `bn16` (nothing for a second warp to compute), more blocks
(grid inflation +12%, short of the 1.19x needed), tile widening (catastrophic),
split-K, hand-written pipelining, warp policy, `k_pack`, ds_read alt, layout-map.
The load path still wins; the ds_read/mmac side is still the whole deficit, and
nothing in-tree hides it.

**4. The three-arm decomposition was redone, and it CORRECTS the diagnosis above.**
The `full - AB` framing above treats the ds_read/mmac side as an *additive* cost
that must be shrunk. That is wrong. Re-measured with a clean compute-only arm —
identical kernel, but every iteration reads one of 4 cycling L2-resident B tiles
instead of streaming (`/tmp/floor4.py`; the earlier arm re-read a *single*
address, which is slower than streaming and made compute-only exceed the full
kernel, hiding the effect) — the two sides overlap almost completely:

| arm (bn16/bk256/ns1/t64, setprio=1) | o M=1 | o M=16 | down M=1 | down M=16 |
|---|---|---|---|---|
| hipblas `_scaled_mm` | 31.5 | 35.0 | 78.6 | 85.3 |
| load-only | 31.2 (0.990) | 33.1 (0.945) | 82.1 (1.045) | 85.0 (0.996) |
| compute-only (4 cycling tiles) | 29.8 (0.944) | 30.4 (0.868) | 69.8 (0.888) | 70.4 (0.826) |
| **full** | 38.7 (1.228) | 39.5 (1.128) | 100.3 (1.277) | 101.0 (1.184) |
| floor = max(load, compute) | 31.2 | 33.1 | 82.1 | 85.0 |
| exposed = full − floor | **+7.5** | **+6.4** | **+18.2** | **+16.0** |

Two things follow, and both invert the earlier reading. **(a) `bn16` is
load-bound, not compute-bound**: the compute side is *cheaper* than the load side
in all four cases (29.8 < 31.2, 30.4 < 33.1, 69.8 < 82.1, 70.4 < 85.0), so the
single 16x16 accumulator chain is not the constraint — which is consistent with
independent-chains being 1.5-1.6x *worse*. **(b) Neither side is the deficit.**
For o M=16 and down M=16 *both* arms individually beat hipblas, and for o M=1 the
load arm is at 0.990. The entire deficit is the 16-20% of wall time in which
neither pipe is busy — i.e. overlap, not throughput.

**5. What actually caps the overlap: resident blocks per CU, and it is the grid,
not the LDS.** Inflating the grid by `g` with *distinct* B rows (`/tmp/grid_mult2.py`
— the first version duplicated `bx % 320`, and with ~512 blocks co-resident the
duplicates shared an L2 window, so it measured partly L2; distinct rows removes
that, and the effect survives) gives, per block, identical work and identical bytes:

| o M=1, bn16/bk256/ns1/t64 | 320 blocks | 640 | 1280 | 2560 |
|---|---|---|---|---|
| load-only, µs/block | 31.4 | 30.1 | 24.6 | 23.0 |
| load-only, GB/s | 1337 | 1394 | 1702 | 1827 |
| full, µs/block | 38.9 | 34.4 | 32.2 | 30.9 |
| full, GB/s | 1079 | 1219 | 1301 | 1357 |

So the memory side is **not** at its rate limit at 320 blocks — it is at 66% of
the 2016 GB/s ceiling and reaches 91% once there is backfill. And the full kernel
converges onto its own compute floor (30.9 at g=8 vs compute-only 29.8), i.e.
**~0.98x hipblas at o M=1 is reachable** — but only at ~8x the blocks.

320 blocks over 64 CUs is *exactly* 5 per CU, against an LDS residency limit of 8
at 8 KB/block (the 65536 B/block budget is confirmed by `bk128/ns8` failing to
launch at ~40 KB of packed stages). So three residency slots sit empty for the
entire kernel, and the grid cannot fill them: `N/bn = 5120/16 = 320`, and `bn=16`
is the hardware floor (MMA micro-tile and MLS B atom, `gemm.h:150`, `mls.cc:408`).

**6. And wavefront occupancy is NOT the missing lever — measured, not assumed.**
The natural reading of §5 is "5 wavefronts/CU out of 40 slots", so 4x the warps
should help. It does not. `bn16/bk256/ns1` at `t = 64/128/256` (`/tmp/warps.py`;
`ComputeMlsWarpPartition` picks tile `(16,128)` at t=64 and `(16,64)` at t>=128,
so the extra warps split K at *identical* bytes and LDS — no duplicated traffic):

| bn16/bk256/ns1 | warps | wavefronts/CU | o M=1 | o M=16 | down M=1 | down M=16 |
|---|---|---|---|---|---|---|
| t=64 (**shipped**) | 1 | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| t=128 | 2 | 10 | 1.011 | 1.008 | 0.971 | 1.013 |
| t=256 | 4 | 20 | 1.015 | 1.232 | 1.096 | 1.162 |

Flat to +1.5% at t=128 and worse at t=256. The reason is the same reason §5 works
and this does not: `wk`-split warps all work on *one* tile, so they add
instructions, not independent memory streams. What §5 adds is independent
*tiles*. The lever is block count, and block count is fixed at 320.

**7. Every remaining knob that could add resident blocks is measured closed.**
- **split-K** adds blocks but shortens each block's k-run (32 -> 8 steps at sk=4),
  and the clean constant-bytes experiment (2026-10-17f) found 1280 blocks/25
  k-steps *worse* than 320/100. The ramp-up cost of a short run exceeds the
  residency gain. Re-opening §5's ceiling as a split-K argument does not survive
  this: §5 holds the run length constant, split-K does not.
- **Deeper staging** (`/tmp/ns_full.py`, full arm, bn16/t64) is monotonically
  worse and tracks the LDS footprint: `bk256` ns1 1.000 / ns2 1.343 / ns4 1.812;
  `bk512` ns1 1.217 / ns2 0.984. The only free `ns=2` is `bk512`, and it is free
  because `tile_issue_k == 4` makes the packed LDS size coincide with the logical
  one, so the packed branch is never taken — the same branch whose `tile_issue_k == 1`
  case is broken (§3) and whose `tile_issue_k == 2` case costs 34%.
- **Smaller `bk`** (more, smaller loads) is worse: bk128 1.108x / bk64 2.0x on the
  load arm.
- **Wider `bn`** halves the block count (160 = 2.5 waves) — measured 1.08-1.14x
  worse, consistent with §5.
- **Independent accumulator chains**, **more warps at bn16**, **tile widening**,
  **warp policy**, **`k_pack`**, **ds_read alt**, **layout-map**: all closed above.

**The honest answer on o/down.** o M=1 sits 2% from a win whose only known route
is 8x the resident blocks at constant run length, which the fixed work cannot
supply — the two ways to manufacture blocks (split-K, wider tiles) both cost more
than the 25% they recover. o M=16 and down (1.13-1.28x) are further out: their
exposed fractions are 6.4-18.2 µs and the down load floor alone is already 1.045x.
So **o and down cannot be won in-tree**, and the two shapes that do win (gate_up
6/6, qkv 5/6) win for a reason consistent with §5: qkv runs 320 blocks x 2 warps
= 10 wavefronts/CU on a *different* tile geometry, and gate_up runs 1600 blocks.

*Caveat on the load-only arm's own precision:* the 2026-10-17h run put load-only
at 0.957/0.917/0.977/0.948 where this run has 0.990/0.945/1.045/0.996. Both are
same-process paired, and the card is shared, so treat the load arm as
"0.95-1.05 of hipblas", not as a point — it does not change any conclusion here
(the exposed fraction dominates the uncertainty), but it is why down M=1 flips
sign between runs.

<a id="2026-10-18"></a>
## 2026-10-18 — the N-split repartition is closed hard; `ns>1` has a second, independent defect

<a id="n-split"></a>
### `ComputeMlsWarpPartition` can N-split, and it is 2-3x worse

The 2026-10-17g note closed "more warps at bn16" on the grounds that the extra
warps split **K** (`w_major = min(slots_k, num_warps)`, decreasing). That is only
true when `slots_k > 1`. When `block_k == mls_tile_k` (i.e. `slots_k == 1`) the
heuristic is forced to `w_major = 1` and the warps split **N** instead — each warp
gets its own 16-wide N band, sharing one `A_s`. That is a genuinely different
configuration from anything measured before, and it is the only way to widen `bn`
while holding the total wavefront count at 320 (`N/bn * threads/64 = 320`):

| config (down, M=1 / M=16) | µs | vs hipblas | vs base |
|---|---|---|---|
| bn16/bk256/t64 (shipped baseline) | 99.5 / 100.3 | 1.267 / 1.190 | 1.000 |
| bn32/bk256/t128 | 113.3 / 125.1 | 1.442 / 1.484 | 1.14 / 1.25 |
| bn32/bk128/t128 (**N-split**) | 174.1 / 175.0 | 2.216 / 2.076 | 1.75 / 1.74 |
| bn64/bk256/t256 | 147.8 / 167.0 | 1.882 / 1.981 | 1.49 / 1.66 |
| bn64/bk128/t256 (**N-split**) | 205.4 / 208.0 | 2.615 / 2.467 | 2.06 / 2.07 |
| bn128/bk256/t512 | 173.4 / 172.2 | 2.208 / 2.042 | 1.74 / 1.72 |
| bn128/bk128/t512 (**N-split**) | 235.8 / 236.3 | 3.002 / 2.803 | 2.37 / 2.36 |

Cost scales ~linearly with `bn` (1.27 → 1.44 → 1.98 → 2.80), and it is the **A LDS
read** that scales with it: an N-split makes every warp of the block re-`ds_read`
the whole `A_s`, so A's LDS read traffic is multiplied by the warp count while B's
is not. At bn16 A and B are 50/50 of LDS read traffic; at bn128 A is 8/9 of it.
**So the per-warp operand duplication is the scarce resource, and widening `bn` to
amortize B over more N-atoms pays for it in A.** This supersedes the 2026-10-17g
"wider bn 1.08-1.14x worse" figure, which was measured at t=64 and understated the
penalty by 2-3x.

### `bk512/ns2` re-measured: real, but 1.6%

The staging table above lists `bk512` ns1/ns2 = 1.217/0.984, the one sub-1.0
self-relative number in the record. It reproduces, and it is the *only* ns>1
config whose packed size equals its logical size (so the packed branch is never
taken — see below):

| | o M=1 | o M=16 | down M=1 | down M=16 |
|---|---|---|---|---|
| bk256/ns1 (base) | 1.000 | 1.000 | 1.000 | 1.000 |
| bk512/ns1 | 1.162 | 1.163 | 1.214 | 1.214 |
| bk512/ns2 | 1.050 | 1.041 | **0.984** | 0.995 |
| bk256/ns2 | 1.387 | 1.380 | 1.402 | 1.406 |

So `bk512/ns2` is a 1.6% win on down M=1 and a 4-5% **loss** on o. Even taken at
face value it moves down M=1 from 1.272 to 1.252 — it is not the 25% lever, and
the record's "deeper staging is monotonically worse" line is imprecise (it is
monotone *except* at bk512/ns2) but not misleading.

### The `ns>1` landmine is **not** fixed — a second, independent defect remains

The 2026-10-17h entry above says the packed-stride defect "was root-caused and
fixed" and that `ns>1` outside bm64/bn64 is therefore "correct but unbenchmarked".
**That is wrong.** The packed-stride fix is real and landed (bk256/ns2 at bn16 is
now exact and reproducible), but at `bm16/bn16/t64` there is a *second* defect the
fix did not reach, and its symptom is nondeterministic — the same inputs give a
different result every run:

| config (bm16/bn16/t64, K=25600) | rel, 5 runs |
|---|---|
| bk128/ns2 | 3504 / 2898 / 2090 / 1980 / 1119 |
| bk128/ns3 | 12123 / 11923 / 9552 / 10799 / 9271 |
| bk256/ns3 | 8899 / 8709 / 8581 / 10180 / 6289 |
| bk256/ns4 | 6597 / 5616 / 7074 / 7700 / 6511 |
| bk256/ns2 | 0 / 0 / 0 / 0 / 0 (exact) |
| bk128/ns4 | 0 / 0 / 0 / 0 / 0 (exact) |
| bk64, bk512 (contiguous, all ns) | 0 (exact) |

The irregular bk/ns pattern is **not a layout rule** — which is exactly why it
looked fixed: the pre-2026-10-18 test list covered bk256/ns2, bk128/ns4, bk64/ns8,
bk512/ns2 and bm64, i.e. every config that happens to work, and none of the four
that do not. It is **not** K-dependent (bk128/ns2 is already wrong at K=5120 / 40
k-steps), so a short K is not a safe escape.

**The mechanism is a deterministic stage-index expression, not a race.** The
nondeterminism above is the *symptom*; the cause is in the emitted source. The
per-stage LDS base that both the MLS producer and the ds_read consumer use is
supposed to be `stage * packed_size` (5120 at bk128, 6144 at bk256):

| | emitted stage base | verdict |
|---|---|---|
| bk256/ns2 | `(k & 1) * 6144` | **== packed size — exact** |
| bk128/ns2 | `(k & 1) * 1024` | packed size is 5120; neither 1024 nor 2048 |
| bk128/ns4 | `((((k&3)*40)&63)>>4)*2048 + (k&1)*1024` | a stage map must not need `&63` |
| bk256/ns4 | `((((k&3)*24)&63)>>4)*4096 + (k&1)*2048` | ditto |

The ns=4 form is the giveaway: `(k*40) & 63) >> 4` is not a stage index, it is an
affine simplifier's garbage — for bk256 it evaluates to `0, 1, 3, 0` over
`k = 0..3`, i.e. it is not even a permutation of the four stages. Since the
producer is asynchronous (`async_mls_load_asm`) and the consumer `ds_read`s
independently, aliased stages corrupt each other *some of the time*, which is
exactly the observed run-to-run variation. Producer and consumer agree on the
expression, so this is not the 2026-10-17h stride fork reappearing — it is the
stage-index derivation itself being wrong once the packed branch and `ns > 1`
combine.

**Not root-caused further.** Isolating the exact aliasing (whether the 4096 B
4K-slot hole is applied by `async_mls_load_asm` and by `ds_read_format_tensor_a`
with the same convention) needs a read of both templates; the observable facts
above are enough to keep `ns > 1` closed at small tiles, which is the only
decision this doc has to support.

**No production exposure.** `fp8_mls_channelwise` asserts `num_stages <= 1 or
(block_M, block_N) == (64, 64)`, and all four decode shapes at M=16 and M=64 were
re-verified exact 5x each through the real builder on 2026-10-18 (bm64/bn64/bk128/ns2
is the only ns>1 config in `decode_cfg`). The action taken: the false "correct but
unbenchmarked" justification in `fp8_channelwise_mls.py` is replaced with the
measured race, and the four broken rows are added to
`test_num_stages_gt_1_mls_is_correct` as `xfail` so the region stays visible and
flips to `xpass` when someone fixes it. `17 passed, 4 xfailed`.

### Standing after this run

Nothing here re-opens o/down. The N-split was the last untested repartition and it
is closed by 2-3x, with the mechanism (A LDS read amplification ∝ warp count)
identifying *why* — and it also explains why every "more warps" variant in this
document failed. **o and down still cannot be won in-tree.**

<a id="2026-10-18b"></a>
## 2026-10-18b — the "2016 GB/s ceiling" is not real; the gap is A's redundancy + exposed compute

Everything above divides by a read ceiling of 2016 GB/s. That number has no
provenance in this repo and it is **too high by ~17%**. Three experiments replace
it, and they change which lever the o/down deficit belongs to.

**The ceiling, measured.** Contiguous fp32 streaming copy into shared, CUDA-graph
replay, idle card (`/tmp/ceiling.py`, `/tmp/nsload.py`):

| arm | bytes | time | GB/s |
|---|---|---|---|
| flat contiguous, G=320, ns=2 | 131.1 MB | 76.0 µs | 1725 |
| flat contiguous, G=1280, ns=2 | 136.3 MB | 75.5 µs | 1805 |
| flat contiguous, G=5120, ns=2 | 146.8 MB | 81.3 µs | 1806 |
| flat contiguous, G=320, **ns=1** | 131.1 MB | 76.0 µs | **1725** |
| flat contiguous, G=320, **ns=2** | 131.1 MB | 75.8 µs | **1728** |

So the achievable read bandwidth is **1721-1806 GB/s**, and it is flat in block
count from 320 to 5120 — which independently re-confirms that block count is not
the lever. Every "x% of ceiling" figure in this document is therefore understated;
the corrected denominators are 1725 (131 MB) and ~1806 (larger buffers).

**ns>1 buys nothing on the load side.** This matters because the whole case for
`ns=2` was "the k-loop drains its loads at the top of every iteration, so DRAM
latency is exposed; double-buffering would hide it". A *load-only* kernel pays no
LDS-residency price for depth, so it separates the two effects:

| arm | ns=1 | ns=2 |
|---|---|---|
| flat contiguous | 1725 GB/s | 1728 GB/s |
| MLS (16,256) bk256 | 1566 GB/s | 1569 GB/s |
| MLS (16,256) bk512 | 1162 GB/s | 1162 GB/s |
| MLS (16,256) bk128 | 1221 GB/s | 1215 GB/s |

`ns1 == ns2` to within 0.4% in every row. **The MLS load is throughput-limited by
its access pattern, not latency-limited by its buffering depth.** So the ns>1
regressions recorded above are pure LDS-residency cost with *zero* compensating
benefit on the load side, and "double-buffer to hide the exposed load" is closed
as a mechanism, not merely as a tuning result.

**The pattern is exonerated; the instruction is exonerated; A is the cost.** The
1566 GB/s figure above is the A+B arm. Reading B *alone* over the identical
strided (16, 256) region:

| arm | time | GB/s | vs ceiling |
|---|---|---|---|
| flat contiguous | 76.2 µs | 1721 | 1.000 |
| **MLS (16,256) strided, B only** | **78.1 µs** | **1679** | **1.025** |
| plain vectorized copy, same region | 79.8 µs | 1642 | 1.048 |

The 16-touch × 256 B × 25600 B-stride pattern costs **2.5%**, and the MLS path is
*faster* than a plain vectorized copy of the same region (1679 vs 1642). So
neither the row-major layout nor the MLS instruction is the problem. Adding A
costs **+5.6 µs** (78.1 → 83.7), because 320 blocks × 400 KB = **128 MB of L2
traffic is fetched for 400 KB of unique data** — a 320x redundancy, forced by
`bn = 16`. hipblas reads the same A with 53 blocks (21 MB) at M=16, and one row
(~0) at M=1. **At M=1, 15 of our 16 A rows are padding for the 16x16 mmac atom.**

bk is also settled as a *load* knob, and in the opposite direction to the
hypothesis that motivated it: if DRAM page locality were the cap, longer
row-touches would help. They do not — bk256 (1566) beats bk128 (1221) and bk512
(1162) by 22-31%.

**The k-loop, from SASS.** Compiled `/tmp/down_m16.cc` with the repo's own flags
and disassembled (`llvm-objdump -d`); the loop body is 41 instructions:

```
s_mov_b32 m0, 0
s_waitcnt vmcnt(0) lgkmcnt(0)        <-- FULL drain of global + LDS
4x ds_read_matrix_trans_format   (A) v[5:20]
4x ds_read_matrix_trans_format   (B) v[21:36]
s_setprio 1
s_waitcnt lgkmcnt(3); 2x v_mmac_f32_16x16x32_fp8_fp8  v[1:4]   <-- ONE accumulator
s_waitcnt lgkmcnt(2); 2x v_mmac ...
s_waitcnt lgkmcnt(1); 2x v_mmac ...
s_waitcnt lgkmcnt(0); 2x v_mmac ...
s_setprio 0
4x matrix_load_128x16_b8             <-- next k-step's loads issued LAST
s_cbranch_scc0
```

Two structural facts: the next k-step's global load is issued only *after* the
last mmac retires, and the 8 mmacs are a **single 8-deep dependent chain on
`v[1:4]`**. The chain depth is not tunable: the only fp8 atom is
`16x16x32` (`src/tl_templates/hcu/gemm.h:118`), and 8 mmacs per (16,256) tile is
forced by the hardware-fixed 8 B/lane operand mapping.

**Decomposition, and the best case.** At down, B streams at 98% of the ceiling;
the entire deficit is A (5.6 µs) + exposed compute (15-16.6 µs). hipblas's *whole
kernel* runs at the bandwidth our *load-only* arm achieves, i.e. hipblas hides
100% of its compute and we hide ~78%. The arithmetic that follows is the reason
this is closed rather than merely unsolved — capturing **100%** of the exposed
compute, which no available mechanism achieves, lands here:

| case | our load floor | hipblas | best case |
|---|---|---|---|
| o M=1 | 31.2 | 31.5 | 0.990 |
| o M=16 | 33.1 | 35.0 | **0.945** |
| down M=1 | 84.6 | 78.2 | 1.082 |
| down M=16 | 83.7-85.0 | 84.8-85.3 | ~0.99 |

So o M=16 is the only case with real headroom, down M=1 loses by 8% on A's
16-row over-read alone, and the rest are ~1% — inside this card's run-to-run
drift. **The one mechanism not yet tested** is splitting the 8-deep accumulator
chain into two independent chains *over the same tiles* (`_gemm_mls_mls` in
`gemm_hcu_mmac.py:848` holds one `C_buf` and emits `for ki in T.serial(0, 8)`).
The recorded "independent accumulator chains" closure does **not** cover it: that
experiment split the *MLS buffers* into two `(16,128)` tiles and doubled the
stream, whereas a second accumulator on one tile costs no LDS and no traffic.

### The accumulator split, measured — neutral, and the exposure is not the chain

Built and run the same-process paired A/B (`/tmp/accsplit.py`): two `T.gemm`
calls on the k-halves of the *same* `A_s`/`B_s`, each with its own accumulator,
summed in the epilogue. Numerically exact (rel 0.0), so the k-slice formulation
is sound. Timing:

| case | base | acc-split | split/base | vs hipblas |
|---|---|---|---|---|
| o M=1 | 38.7 µs | 39.2 µs | 1.012 | 1.191 → 1.206 |
| o M=16 | 39.4 µs | 39.7 µs | 1.008 | 1.125 → 1.134 |
| down M=1 | 99.4 µs | 100.8 µs | 1.014 | 1.271 → 1.289 |
| down M=16 | 100.3 µs | 101.6 µs | 1.013 | 1.183 → 1.198 |

(The base column reproduces this document's recorded ratios to within 0.5%, so
the pairing is valid.) Halving the chain depth 8 → 2x4 buys **nothing**. Combined
with the recorded t=64/128/256 sweep (1.000/1.011/1.015 at o M=1 — doubling the
warps per CU also buys nothing) and the constant-bytes block-count sweep, the
~15 µs of "exposed compute" is **not** limited by the mmac chain, the warp count,
the buffering depth, or the block count. It is the memory pipe going idle while a
block is in its compute phase, and no knob in this kernel changes how many
independent memory streams are in flight at once.

**Closed. The exposure has no remaining lever, and the load side is at 97.5% of a
ceiling that is 1725 GB/s rather than 2016.**

<a id="2026-10-18c"></a>
## 2026-10-18c — the exposure is volumetric and per-iteration; two real gaps closed

The previous section concluded the exposure "is the memory pipe going idle while
a block is in its compute phase" and that no knob changes it. That is right about
the *consequence* but wrong about the *shape*: it is not a fixed serialization
that a better-shaped compute would fall into for free. A dose-response settles it.

### The dose-response: exposure scales with compute VOLUME, per k-iteration

Same loads (identical bytes, identical MLS tiles), varying only the gemm. `half`
is one gemm on a 128-wide k-slice (0.5x mmacs and 0.5x ds_read), `dbl` is two
gemms on the full tile with two accumulators (2x). `/tmp/dose.py`, paired
same-process, idle card:

| arm | o M=16 | vs load-only | down M=16 | vs load-only |
|---|---|---|---|---|
| loadonly | 34.2 µs | — | 85.1 µs | — |
| half | 38.2 µs | +4.0 | 96.2 µs | +11.0 |
| base | 39.4 µs | +5.1 | 105.1 µs | +20.0 |
| dbl | 47.8 µs | +13.6 | 116.8 µs | +31.7 |

Monotone in volume, so the compute is genuinely additive and shrinking it is the
only route — which the accumulator split does not do (it changes the chain shape,
not the volume). The marginal cost is **~0.19 µs per k-iteration per block**:
bk256 pays it 100 times (+20.0 at down), bk512 pays it 50 times (+8.8: 121.6 full
− 112.8 load-only). It is roughly independent of how many mmacs are in the
iteration (bk512 issues 16 per iteration, bk256 issues 8), so it is **not** the
mmac chain and **not** the ds_read issue count. It is the single LDS
write→read round trip the loop cannot avoid with one buffer:

```
s_waitcnt vmcnt(0) lgkmcnt(0)     <- previous matrix_load's LDS write must land
ds_read_matrix_trans_format x8
v_mmac_f32_16x16x32 x8
matrix_load x4 (next k)
```

Per CU per iteration the arithmetic closes: memory ≈985 cycles, compute ≈5 warps
× 260 = 1300, measured 1365 = max(1300, 985) + 65. **The CU is compute-bound per
iteration, and compute and memory are already ~fully overlapped.** That is why
t=128 is neutral (10 warps × 130 = 1300, same total) and why the chain split is
neutral (same total issue). The exposure is the CU's own stall time, not idle
memory.

### Removing A from the LDS path — closed again, from a different direction

`gemm_hcu_mmac.py` has two dispatch arms the shipped kernel never reaches because
both operands are MLS-fed (`_gemm_mls_mls`, line 848): `_gemm_r_mls` (line 867,
A is a *fragment*, B via MLS — A never touches LDS at all) and `_gemm_s_mls`
(line 884, A is a plain shared buffer, read per-ki by `ldmatrix_a`). If the round
trip were the cost, `_gemm_r_mls` should delete half of it. `/tmp/ldspath.py`,
paired, same-process:

| arm | o M=16 | vs base | down M=16 | vs base |
|---|---|---|---|---|
| base (MLS A + MLS B) | 39.4 µs | 1.000 | 100.5 µs | 1.000 |
| smls (plain-shared A + MLS B) | 46.4 µs | 1.179 | 120.2 µs | 1.196 |
| rmls (fragment A + MLS B) | 76.2 µs | 1.935 | 260.5 µs | 2.592 |

`rmls` reproduces the recorded A-direct-VGPR closure (249.3 µs) independently.
A must go through LDS; the round trip is a property of the load path, not a
removable overhead.

Also measured and closed in the same run: a **per-block k-loop offset**
(`(k + bx*37) % nk`) to break phase lock between the 5 warps of a CU is
**neutral** (1.004 / 1.018). Note this arm only changes *addresses*, not the
*time* at which blocks issue — it does not actually desynchronize, so it is not
evidence against the lockstep story, only against an address-pattern one.

### Re-sweep of every remaining knob (post-fix), paired

| config | o M=16 | vs base | down M=16 | vs base |
|---|---|---|---|---|
| base bn16/bk256/ns1/t64/sp1 | 39.5 µs | 1.000 | 100.5 µs | 1.000 |
| sp2 | 44.0 µs | 1.114 | 100.5 µs | 1.000 |
| sp0 | 46.7 µs | 1.182 | 113.3 µs | 1.128 |
| swizzle(10) | 44.3 µs | 1.122 | 101.0 µs | 1.005 |
| bk512/ns2 | 41.4 µs | 1.048 | 106.1 µs | 1.056 |
| bk512/ns1 | 46.2 µs | 1.170 | 121.6 µs | 1.210 |
| bk256/ns2 | 54.2 µs | 1.372 | 145.4 µs | 1.447 |
| bk128/ns1 | 64.0 µs | 1.620 | 169.4 µs | 1.686 |

`setprio=1` is confirmed optimal (sp0 costs 12.8% at down — the mmac section
really does need the priority boost), swizzle is neutral at down and 12% worse at
o, and `bk256/ns1` is confirmed after the packed-stride fix.

### Two genuine gaps in the recorded closures, now filled

**1. Wider bn at t=64 — closed unconditionally.** Every recorded wider-bn test
used threads ≥ 128, where `ComputeMlsWarpPartition` is forced to `w_major = 1`
and the warps split **N**, so every warp re-reads the whole `A_s` and A's LDS
read is multiplied by the warp count. At t=64 there is exactly one warp, which
reads `A_s` once, so A's LDS read is **not** duplicated — the only cost is fewer
blocks. Never measured. It is worse anyway:

| config | o M=16 | vs base | down M=16 | vs base |
|---|---|---|---|---|
| bn16/bk256/t64 | 43.5 µs | 1.000 | 104.0 µs | 1.000 |
| bn32/bk256/t64 | 46.9 µs | 1.077 | 116.8 µs | 1.123 |
| bn64/bk256/t64 | 51.8 µs | 1.191 | 139.4 µs | 1.340 |

So the wider-bn penalty is **not** A's duplicated LDS read — that was the
recorded explanation and it is wrong for this arm. It is the loss of blocks
(160 / 80 instead of 320) and the resulting loss of memory parallelism. The
closure stands, on a different and stronger reason.

**2. Constant-run-length split-K — closed under the correct control.** The
recorded split-K test held `bk` fixed and therefore *shortened the run*
(nk 100 → 25), conflating "more blocks" with "less work per block"; the record
flags this itself. Holding the run length at the baseline value by halving `bk`
alongside `split_k` (`split_k=2, bk=128, nk=100`, 640 blocks, same total bytes)
is the clean test, and it also supplies the only real desynchronization available
— dynamic backfill, since 640 blocks over 64 CUs at 6 resident slots means blocks
start mid-stream. It loses:

| config | o M=16 | vs base | down M=16 | vs base |
|---|---|---|---|---|
| base 320 blocks, bk256, nk | 43.5 µs | 1.000 | 104.0 µs | 1.000 |
| sk2 bk128 t128, 640 blocks, nk | 53.4 µs | 1.228 | 117.1 µs | 1.126 |
| sk2 bk128 t64, 640 blocks, nk | 65.8 µs | 1.511 | 170.2 µs | 1.637 |

(These arms write to disjoint dummy slots, so the values are wrong; the traffic
and the run length are correct, which is all the timing needs.) **More resident
blocks does not help a fixed job even at constant run length**, which settles the
"block-starved" reading for good.

### Load-side: o and down are not the same problem

Measured with a load-only MLS kernel (`/tmp/loadprobe.py`), idle card:

| arm | o (B = 41.9 MB) | down (B = 131.1 MB) |
|---|---|---|
| MLS B only, G=320 | 30.24 µs → **1387 GB/s** | 78.08 µs → **1679 GB/s** |
| MLS A+B, G=320 | 31.36 µs (A costs +1.1) | 84.80 µs (A costs +6.7) |

Two things fall out. A's cost scales with A's **size** × 320 blocks: o's A is
131 KB (→ 42 MB of L2 traffic, +1.1 µs), down's is 400 KB (→ 128 MB, +6.7 µs),
so A's redundancy is a `down`-specific cost, not a general one. And **down's load
arm is at the 1725 GB/s ceiling while o's runs at 1387 GB/s** — o is ~5 µs above
its own byte floor (41.9 MB @ 1725 = 24.3 µs), which is the entire o M=16 gap to
hipblas. No mechanism is established for that 21% and no lever was found: the
`flat` contiguous control in the same script is contaminated by its reduction, so
it does not separate "not enough blocks" from "not enough bytes", and the G=640
MLS arms duplicate B rows (`bx % (N//BN)`) so they measure L2 contention rather
than parallelism. **This is the one open measurement in o/down.**

### Where the budget now stands (down M=16, all measured)

| term | µs | status |
|---|---|---|
| DRAM floor, 131 MB @ 1725 GB/s | 75.9 | at ceiling (B-only MLS = 1679 = 97.5%) |
| A's L2 redundancy, 320 blocks × 400 KB | +6.7 … +9.2 | needs fewer blocks or fewer rows; both closed |
| per-iteration LDS round trip, 100 iters × 0.19 | +20.0 | needs fewer iterations; bk512 halves it but costs 29 µs on the load |
| **total** | **105.1** | hipblas 85.3 |

Every term is now measured irreducible, and two of the three have had their
recorded *reasons* corrected. The structural reason the whole thing loses is
that **our load-only floor (85.1) is already equal to hipblas's entire kernel
(85.3)**: hipblas hides 100% of its compute, we hide 72%, and we have nothing to
hide it behind.

<a id="2026-10-18d"></a>
## 2026-10-18d — split-K is not merely slow, it is not expressible; and double buffering is worth ~1%

The 2026-10-18c section left one structural idea standing: the per-iteration LDS
round trip (~0.19 us x nk, 20.0 us at down M=16) is removable by double
buffering, but `ns=2` costs residency (12288 -> 24576 B = 5 -> 2 blocks/CU), and
the only way to pay for that residency is more blocks, i.e. split-K. The
recorded split-K closure was a *perf* argument on a confounded test. This
section replaces it with a correctness argument, and then measures the
double-buffer ceiling so the remaining direction can be costed rather than
guessed.

### The MLS K-slice offset is silently wrong for ANY nonzero base

`/tmp/skbase.py` — same tile, same bytes, four ways of expressing the K base of
the `T.copy(..., prefer_instruction="matrix_load")` source region, each compared
against the matching slice of `A@B.T` (down, bn16/bk256/ns1/t64, nk=8):

| base expression | diff vs reference slice |
|---|---|
| `0` | **0** (exact) |
| `512` (constant) | 12.19 |
| `1024` (constant) | 18.38 |
| `6400` (constant) | 23.10 |
| `bz * 6400` | bz0 exact; bz1 23.10; bz2/bz3 **NaN** (5135 NaNs) |

So a nonzero base is wrong even when it is a compile-time constant well inside
the signed-10-bit `moffset` range (512), and past ~12800 elements it produces
NaN. `MatrixLoadNode::Lower` (`src/hcu/op/mls.cc:718-729`) does take
`block_k_base = src_ranges[nr-1]->min` and passes it into `mls_load_tile`
(`:823`), so the plumbing exists — the value is simply not applied correctly.

**Consequence.** The plan's assumption ("the MLS lowering takes a dynamic K base
from the copy region, so a K-slice offset is legal") is false, and split-K
cannot be built in the MLS path at all. The earlier 10-21% timings
(`/tmp/final2.py`) stand as timings — the probe kernel read the right bytes — but
they were measuring a kernel whose partials were garbage, and no amount of
tuning reaches correctness. This is a stronger closure than the recorded one.

### Even if it were expressible, it would not win

`/tmp/skreal.py` builds the real thing (partial `(sk, M, N)` + a reduce kernel,
one process per shape, 9 rounds, paired vs hipblas). With the *broken* K offset,
i.e. the timing the probe would have got:

| arm | down M=16 | vs base |
|---|---|---|
| bn16/bk256/sk1/ns1 (shipped) | 103.36 | 1.000 |
| bn32/bk256/sk4/ns1 | 99.52 | 0.963 |
| bn32/bk256/sk4/ns2 | 93.92 | 0.909 |

The probe (`/tmp/skprobe2.py`, no reduce kernel) had these at 93.6 and 87.2, so
the reduce + partial costs ~6 us. Best case with a working offset and a perfect
reduce is therefore ~1.03-1.09 at down, and `o` does not improve at all
(probe best 36.0 vs base 38.9, plus a reduce). The epilogue cannot be removed:
atomics need a zeroed C, and the partial+reduce needs the second launch.

### `bn32` alone does not pay for the A-L2 it saves

The one mechanism the probe exercised that does *not* need a K offset is
halving A's L2 read traffic, `(N/bn) * 16 * K` — 128 MB at down, 42 MB at o,
inversely proportional to `bn` and independent of `bk`/`ns`/`sk`. The 2026-10-18c
note records `bn32/t64` at 1.077 at o M=16 against a shipped 1.123, suggesting
it might already win. Re-measured head-to-head, paired, one `(shape, M)` per
process (`/tmp/bnprobe.py`, all `rel=0`):

| shape | M | bn16 (shipped) | bn32 | bn64 |
|---|---|---|---|---|
| o | 16 | 39.36 | 42.24 (1.073) | 48.16 (1.224) |
| o | 32 | 69.60 | 74.72 (1.074) | 85.12 (1.223) |
| down | 1 | 99.52 | 111.20 (1.117) | 135.52 (1.362) |
| down | 16 | 100.64 | 112.48 (1.118) | 140.48 (1.396) |
| down | 32 | 191.20 | 213.12 (1.115) | 254.88 (1.333) |

`bn32` is 7-12% worse at **every** (shape, M) — the recorded 1.077 was relative
to a different baseline, not to the shipped config. Losing 160 of 320 blocks
costs more than halving A's L2 saves, which is the same "block supply" wall the
split-K probe was really climbing.

### Double buffering is worth ~1%, measured at the residency where it is exact

`/tmp/stagebase.py` establishes which small-tile `ns>1` configs are numerically
sound at bn16/t64, by checking the emitted per-stage LDS base is a permutation
of the packed size:

| tile | rel | emitted stage stride | correct? |
|---|---|---|---|
| bk128/ns2 | 5.35e-01 | `(k&1)*1024` (packed size is 5120) | no |
| bk256/ns2 | **0** | `(k&1)*6144` | yes |
| bk512/ns2 | **0** | `(k&1)*8192` | yes |
| bk256/ns3 | 7.80e-01 | no permutation emitted | no |
| bk128/ns4 | **0** | — | yes |

That is a genuine compiler defect (a wrong packed stage stride), and fixing it
would unlock `bk128/ns2` at 10240 B = **6 blocks/CU**, one more than the shipped
5, with the round trip removed. So it is worth costing. `/tmp/ns2probe.py`, the
exact configs only, paired vs hipblas, 9 rounds:

| arm | o M=16 | down M=16 | down M=1 |
|---|---|---|---|
| bn16/bk256/ns1 (shipped) | 39.36 | 100.80 | 99.52 |
| bn16/bk512/ns2 | 41.28 (1.049) | **100.64 (0.998)** | **98.24 (0.987)** |
| bn16/bk256/ns2 | 54.24 (1.378) | 141.12 (1.400) | 139.84 (1.405) |
| bn16/bk128/ns4 | 106.56 (2.707) | 315.52 (3.130) | 309.60 (3.111) |

`bk512/ns2` reproduces the record's 0.984/0.995 self-relative: **a 0.2-1.3% win
at down and a 4.9% loss at o.** The other exact arms are catastrophic. So the
double-buffer mechanism, measured where it already works, is worth about 1% —
not the 12% needed. Fixing the stage stride to reach 6 blocks/CU cannot plausibly
deliver the difference, and the fix is therefore **not** worth taking on
expected value. (It remains a real silent-wrong-answer bug; the shipped configs
avoid it because all four decode shapes use `ns=1` at M<=32, and `bm64/bn64/bk128/ns2`
is the only `ns>1` config in the table.)

Note also that `bk256/ns2` at 1.40 with a *smaller* LDS footprint than
`bk512/ns2` at 0.998 (24576 vs 32768 B, both 2 blocks/CU) rules out the
"residency is the whole story" model: `bk512` is the one config that takes the
contiguous (unpacked) LDS branch, so the padded layout is costing something
beyond its bytes.

### Where this leaves the budget

Unchanged, and now with every term having a measured reason:

| term | down M=16 | status |
|---|---|---|
| B stream at the 1725 GB/s ceiling | 78.1 | at 97.5% of ceiling |
| A's redundant L2 read (`(N/bn)*16*K`) | +7.0 | `bn32` halves it and costs 12% |
| exposed compute (LDS round trip / mmac issue) | +20.0 | `ns2` buys ~1%, split-K inexpressible |
| **total** | **~105** | hipblas 85-89 |

The remaining 12-18% is not reachable by tuning the in-tree search space: the
load side is at its ceiling, A's redundancy can only be reduced by `bn`, which
costs more in blocks than it saves, and the compute exposure is not removable by
the only mechanism available (`ns>1`) at any residency that keeps the CUs fed.

<a id="2026-10-19-warpk"></a>
## 2026-10-19 — `warp_k` / `FullColK` is now implemented on the MLS path, and it is **not** a lever

This was the last untested structural axis: the one place where the kernel's
shape differed from hipblas's in a way that *might* have been the gap. It was
built, it is correct, and it does not win. Closed with the bound.

### It was genuinely unreachable, not merely unused

Three independent guards rejected the policy on the fp8 MLS path, so this was a
real capability gap rather than a config the search had skipped:

- `tilelang_ds_read_format.hpp` — both the A and B ds_read templates carried
  `static_assert(WarpK == 1)`, and the A template could not even *recover*
  `warp_k`, because the warp-id traversal order is `warp_m -> warp_n -> warp_k`
  (`gemm.h:474-475,566`), so `warp_id / WarpM` yields `warp_n` once `n_warp > 1`.
- `hcu_mmac_macro_generator.py` — `raise ValueError` in both `ldmatrix_mls_a`
  and `ldmatrix_mls_b`.
- `gemm_hcu_mmac.py` / `gemm.cc` — a `warp_k != 1` guard and
  `ICHECK(!use_mls) << "... (FullColK policy)"`.

`inner_k_per_warp()`, `local_elems_a/b(full_k=True)`, `_k_base`, `_warp_index`
and `mmac` were already warp_k-aware, so the fix was the ds_read partition and
the guards: A's template gained a `WarpN` parameter threaded through
`gemm_mls.h`, `ds_read_format.cc`, `hcu_mmac_emitter_utils.py` and the macro
generator, and the A/B indices became
`warp_k_idx = warp_id / (WarpM * WarpN)`. With `WarpK == 1` both formulas
collapse to the old behaviour, so no existing path moves.

The reduction is a **contract, not an optimization**: `make_gemm_fragment_hcu`
replicates the C fragment across the K warps and the copy-out guards to replica
0, so `T.reduce_sum_warp(C_l, C_l, clear=False)` is mandatory or the kernel
returns K-warp 0's partial with no error. Verified in the emitted source:
`tl::AllReduce<tl::SumOp, 256, 64, 0>` (butterfly at offsets 128 then 64,
lane-preserving), copy-out `pred = (threadIdx.x >> 6) == 0`.

Correct at `rel = 0.000000` across bn16/bn32/bn64 x t128/t256/t512 x
bk256/bk512. Regressions green: `test_tilelang_gemm_mls.py` 75 passed / 58
skipped, `test_hcu_gemm_fp8_mls_channelwise.py` 18 passed / 4 xfailed (including
a new `test_warp_k_fullcolk_mls_is_correct_and_reduced` pinning both the K-split
template args and the presence of the reduction), `test_tilelang_test_gemm_hcu.py
-k FullColK` 6 passed.

### Measured: `warp_k` and `setprio` are substitutes, not complements

5-arm paired run, one process, all arms + hipblas as CUDA graphs, 9 rounds,
min-of-rounds, idle card (µs):

| shape | M | hipblas | BASE sp1 | WK4 sp1 | BASE sp0 | WK4 sp0 |
|---|---|---|---|---|---|---|
| o | 1 | 32.3 | **38.4** | 39.7 | 41.1 | 39.2 |
| o | 16 | 34.7 | **39.0** | 40.2 | 41.8 | 39.7 |
| o | 32 | 39.0 | 47.4 | **47.2** | 52.8 | 47.5 |
| down | 1 | 82.6 | **102.4** | 102.7 | 114.4 | 101.6 |
| down | 16 | 90.1 | **103.2** | 104.6 | 115.2 | 103.7 |
| down | 32 | 110.4 | 126.2 | **122.2** | 142.6 | 122.4 |

(`BASE` = the shipped `decode_cfg` config, `WK4` = bn16/bk256/ns1/t256/FullColK.
`sp0`/`sp1` = `setprio`. Bold = best non-hipblas arm.)

Reading it:

- **`WK4 sp1` vs `BASE sp1`** — 1.03, 1.03, 1.00, 1.00, 1.01, 0.97. Mixed sign,
  mean ~1.007. One real cell (`down` M=32, −3.2%); nothing else outside noise.
- **`BASE sp0 -> BASE sp1`** is −6.6 / −6.7 / −10.2% on o and −10.5 / −10.4 /
  −11.5% on down. `setprio` is the real lever, and it is already shipped.
- **`WK4 sp0 ≈ BASE sp1`** and **`WK4 sp0 ≈ WK4 sp1`**: either mechanism
  recovers the ~7-11% and they do not stack. The sp0/t64 loss is the mmac chain
  delaying the next load issue within a single wave; splitting the K work over 4
  warps removes that serialization just as effectively as raising the wave's
  scheduling priority. Best ratios vs hipblas are unchanged at **1.107-1.240**.

This also retires the stale claim in the builder docstring that `setprio` "only
pays off at M=64" — at the shipped bn16/t64 structure it is worth 7-11% at every
M in {1,16,32}.

### A hard architectural cap: the MLS `moffset` is a signed 10-bit element immediate

The hipblas-shaped direction — coarse `block_K` so the wait amortises over many
elements — is not expressible through MLS at this tile. `block_K = 1024` fails to
compile:

```
tilelang_mls_base.hpp:328: static assertion failed due to requirement
'moffset_value <= 511': MLS moffset exceeds its signed 10-bit element range
```

The MLS producer's per-access offset is a signed 10-bit *element* immediate, so
**`block_K <= 512` at the bn16 fp8 tile**. hipblas's 2048-element K chunk uses
`buffer_load_dwordx4` with 32-bit addressing; that is structurally unreachable
from the MLS path, not a tuning choice. And the shapes that *do* compile lose
badly anyway: `bk512_t256_k4` 46.6 vs 39.2 at o M=16, `bk1024_t512_k8` 111.2 vs
100.5 at down M=16 — the 16-32 KB LDS tile costs more residency than the coarser
wait buys.

### Verdict

`warp_k` is kept as a correct, tested capability (it removes an artificial
limitation and fixes a latent B-path warp-id decomposition), but `decode_cfg`
does **not** select it for any shape and the `policy` parameter stays unset. The
o/down gap of 1.11-1.24 has no remaining lever in the in-tree search space:
every partition axis (warp_m/n/k, policy), every tiling axis (bn, bk, threads,
residency), every pipelining axis (ns, setprio) and every producer axis (MLS vs
plain shared vs rolling window vs B-direct-VGPR) has now been measured. Re-opening
any of them needs new evidence, not a config change.

---

<a id="2026-10-19-matrixload"></a>
## 2026-10-19 — `__builtin_hcu_matrix_load_b8` exists; its micro-layout is why it is still closed

The 2026-10-17d closure of "lever 2" rested on a factual claim that is **wrong**:
that there is no global→VGPR *matrix* load. There is one. But measuring it does
not re-open the lever, for a different and now-measured reason.

### The builtin is real and unused

`/opt/dtk/aillvm/include/clang/Basic/BuiltinsHCU.def:337-338`

```
TARGET_BUILTIN(__builtin_hcu_matrix_load_b8,  "V4iV2iiibbbbb", "t", "gfx938-b8b16-mls-Insts|gfx936-mls-insts")
TARGET_BUILTIN(__builtin_hcu_matrix_load_b16, "V4iV2iiibbbbb", "t", "gfx938-b8b16-mls-Insts|gfx936-mls-insts")
```

`V4i` return, `V2i rsrc` (a plain 64-bit pointer, not the 4-dword `mls_resource`),
and no LDS pointer argument — unlike the `*_lds` twins at `:335-336`, which take
`c*3i`. It is the VGPR-destination sibling of the LDS instructions we already
use, it compiles on gfx938 with no extra feature flag, and it appears **nowhere**
in this repo. `stride` and `offset` are `ImmArg` (`IntrinsicsHCU.td`, class
`HCUMMC`), so both must be compile-time constants.

Emitted code for `matrix_load_b8(rsrc, stride, offset, 0,0,0,0,0)`:

```
matrix_load_b8 v[0:3] s[0:1] s6  moffset:50152
s_waitcnt vmcnt(0)
```

### Measured per-lane layout (the decisive fact)

Probe: one warp, `B[n][k] = n` and `B[n][k] = k` over a 64×512 fp8 buffer,
`stride = 512` (a clean, non-aliased pitch), one flag combination per process on
an idle card. `t = 0` and `t = 1` produce **byte-identical** output, so the
distribution below is hardware-fixed and not a flag choice.

The instruction reads an **8 rows × 128 contiguous bytes** tile with row pitch
`stride`. `stride = 512` with a 512-wide buffer puts the four dwords of lane 0 at
`n = 0, 2, 4, 6`, i.e. byte offsets 0 / 1024 / 2048 / 3072 — a pitch of exactly
512, and the instruction has no pitch information other than `stride`. The
`stride = 64` probe is the cross-check: the second 64-byte half of each 128-byte
row window aliases into the next row, so lanes 16-31 land on `n = 1,3,5,7` and
lanes 48-63 on `n = 2,4,6,8`, exactly as the model predicts. The 1024 bytes are
distributed as 16 bytes/lane, 4 dwords:

| | mapping |
|---|---|
| dword `d` (0..3) → row | `n = (lane >> 5) + 2*d` |
| byte `b` (0..3) of a dword → column | `k = ((lane >> 4) & 1) * 64 + (lane & 15) * 4 + b` |

So per lane: **4 consecutive k, at 4 different n** (rows 0/2/4/6 for lanes 0-31,
1/3/5/7 for lanes 32-63).

### Why that is not the mmac operand

`ds_read_format_traits` (`tilelang_ds_read_format.hpp:106-114`) fixes
`MmacKSize = 32` for `RegBits == 8`, hence `VecSize = 8` elements, and the
consumption index is `(ki * warp_rows + Mi) * vec_size`. `gemm.h:109` reads that
as `int32x2` — **8 bytes per lane, holding 8 consecutive k at one mn**. That is
independently confirmed by `b_vgpr_direct_load`, which produces the same layout
with `buffer_load` and is *numerically correct* (only slow).

| | `matrix_load_b8` | fp8 mmac operand |
|---|---|---|
| per lane | 4 consec-k × 4 n | 8 consec-k × 1 mn |
| lane → n | `(lane >> 5) + 2*d` | `lane & 15` |
| lane → k | `((lane >> 4)&1)*64 + (lane&15)*4 + b` | `(lane >> 4)*8 + [0..7]` |
| operand granularity | 4 bytes (1 dword) | 8 bytes (2 dwords) |

The two 4-byte halves of a mmac operand sit at rows `r` and `r+2` under
`matrix_load_b8` but must be 8 consecutive k at one mn. **No choice of `stride`,
`offset` or `t` changes the per-lane ownership** — it is fixed by the opcode.

### Verdict

The only bridge is a full cross-lane permutation. Bouncing through LDS (4 ×
`ds_write_b32` per lane, with a bank-conflict-prone pitch because all 8 rows of a
`stride`-paced tile share a bank at `pitch % 128 == 0`) is strictly more work than
the direct `matrix_load_64x16_b8` → LDS → `ds_read_format_tensor_b` path we
already run. There is no version of this that wins.

So "lever 2" stays **closed** — but the recorded reason is corrected: not
"there is no such builtin", rather **the builtin exists and its per-lane
micro-layout is mmac-incompatible, with the tile geometry (8n × 128k) also
different from the 16n × 64k the producer atom uses.**

The same reasoning retires the rest of the unused-builtin list for the B path:
`matrix_load_b16` is the same opcode family, and `buffer_load_lds*`,
`ds_mbarrier_*` and `ds_read_m32x32_u8_alt2` / `ds_read_m64x16_u8_alt4` do not
change the operand ownership either. At one warp per block there is no barrier to
remove, so the mbarrier primitives have nothing to buy.

---

<a id="2026-10-19-lds"></a>
## 2026-10-19 — o/down: the gap is an LDS-capacity invariant, not a missing knob

Driven by "我们和hipblas的差距没有找到吗 明明是有差距的". The gap **is** now
located, at SASS level, and it is arithmetically closed — not a config we failed
to find. All numbers below are paired same-process, idle card, min-of-7.

### 1. The occupancy cliff (same kernel, only the dynamic smem size varied)

`K_orig` at down M=16/N=5120/K=25600, `<<<320,64,smem>>>`, loop+event timing:

| smem | blk/CU | µs |
|---|---|---|
| **12288** | **5** | **93.7** |
| 14336 | 4 | 168.9 |
| 16384 | 4 | 169.1 |
| 20480 | 3 | 166.8 |
| 24576 | 2 | 219.7 |
| 32768 | 2 | 219.7 |

5→4 blocks/CU costs **80%**. The shipped config sits at exactly 320 blocks =
64 CU × 5 = **one wave**, on a knife edge.

### 2. The invariant

Normalizing every occupancy point by `waves × iters` gives a per-warp iteration
time that is essentially occupancy-independent (~0.88–1.35 µs). That yields

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
cancelling.** This is why `ns>1` and split-K were always neutral: they move work
between `waves` and `T` without changing the product.

### 3. SASS root cause of `C`

Shipped loop body (0x1960–0x1A58): the four `matrix_load_128x16_b8` producers are
emitted **after** the mmac, immediately before the branch, and the next iteration
opens with `s_waitcnt vmcnt(0) lgkmcnt(0)`. Nothing but the branch sits between
load issue and the full wait, so the whole global-load latency is exposed every
iteration. hipblas instead issues `buffer_load_dwordx4` *inside* its compute
region and waits with `s_waitcnt vmcnt(6)`.

### 4. Decomposition (all ratios to the same-run rotated kernel)

| arm | ratio |
|---|---|
| rotate producer above the mmac | **0.965** (3.5% win, reproduced 3×) |
| + drop mmac | 0.906 |
| + drop B's ds_read | 0.926 |
| + drop A's ds_read | 0.983 |
| + drop both ds_read and mmac | 0.902 |

Removing mmac alone (9.4%), ds_read alone (9.3%), both (9.8%) ⇒ **one serialized
chain, not two costs**. The pure load path is at ~1.02× hipblas, i.e. our load
machinery is already at the ceiling; the entire gap is this chain.

### 5. Levers measured closed this session

- `ns=2` at every bn: 103.4→141.1 (bn16), 114.4→124.5 (bn32), 140.6→149.4 (bn64).
  LDS doubles, blocks/CU collapses.
- Wider bn at ns=1: 103.4 (bn16) → 114.4 (bn32) → 140.6 (bn64) → 173.9 (bn96).
- **bk128/ns2**: the emitted stage stride is `(k&1)*1024` against a packed size of
  5120 — the two stages alias, so there is no double-buffering at all. Confirmed
  by generating it: the bk256/ns2 source (numerically exact) uses stride 6144 =
  its own packed size. Patching 1024→5120 in /tmp did **not** restore `rel`, and
  independently the 200-iteration count makes it 178 µs = 2× — per-iteration cost
  is ~0.9 µs regardless of bk, so halving bk never pays.
- bk512/ns1 125.4, bk512/ns2 97.8 (ties 99.4 within drift, worse on `o`).
- Warps per block: t128/WK2 97.0 vs t64 99.2 (down M=1), t256/WK4 100.2 —
  within noise. More warps per CU buys nothing; the machine is already saturated.
- split-K: neutral by the invariant (grid × iters is constant when LDS/block is
  constant).

### 6. The one landed-quality improvement

**Rotate the MLS producer above the `promote_prio`/mmac block.** The compiler
inserts the required `s_waitcnt lgkmcnt(0)` before the `matrix_load`, so it is
not racy. Worth ~3% consistently (down M=16 1.124→1.084, o M=16 1.11→1.08 in the
session's paired runs). It requires an emitter/pass change: the producer is
placed by the MLS lowering, not by a config flag.

<a id="2026-10-19-hipblas-launch"></a>
### 2026-10-19 (b) — what hipblas actually launches, decoded

From the hipblaslt shape table (columns are **N, M, K**):

| N | M | K | µs | GB/s | util | kernel |
|---|---|---|---|---|---|---|
| 5120 | 16 | 25600 | **85.0** | 1463 | **86.0%** | `MT96x16x256_TT12_2_WG8_8_4_MFWGS256_WGM1` |
| 5120 | 8 | 25600 | 85.8 | 1498 | 85.1% | same |
| 5120 | 1 | 25600 | **79.4** | 1621 | **91.7%** | `MT8x1x8192_MFWGS512_WGM1_ONLYN1` (GEMV) |
| 5120 | 16 | 5120 | 25.0 | 1026 | 58.8% | `MT96x16x256_TT12_2_...` |
| 5120 | 32 | 25600 | 107.3 | 1221 | 68.5% | `MT96x64x256_TT6_2_WG16_32_1_MFWGS512` |
| 5120 | 64 | 25600 | 114.7 | 971 | 64.6% | `MT96x64x256_TT6_2_...` |

Name decode for the M=16 decode kernel: **N-tile 96, M-tile 16, K-chunk 256;
workgroup 8×8×4 = 256 threads = 8 warps; ThreadTile 12×2 (24 accumulators per
thread); 4 threads cooperate along K.** Grid = 5120/96 → **54 blocks**, i.e.
0.84 blocks per CU — hipblas does *not* use occupancy here at all.

Consequences vs us (bn16/t64, 320 blocks, 1 warp each):

- A is read **54× instead of 320×**: per k-step hipblas moves B 131 MB + A 22 MB
  = 153 MB through L2; we move 131 + 131 = 262 MB. (1.71×)
- 8 warps per block vs our 1, so one warp's compute overlaps another's loads.
- At **M=1 it abandons the MMA path entirely** for a GEMV kernel
  (`MT8x1x8192`: N-tile 8 → 640 blocks, 512 threads) reaching 91.7%.

### Copying that structure does not work through the MLS path (measured)

| config | down M=1 | down M=16 |
|---|---|---|
| ship bn16/t64 | **99.2** | **99.8** |
| bn64/t64 | 135.2 | 138.9 |
| bn64/t128 | 135.4 | 145.9 |
| bn64/t256 | 161.8 | 169.8 |
| bn32/t128 | 115.8 | 152.2 |

Adding warps to a wide tile makes it *worse*, not better. Mechanism: the MLS
producer writes a packed tile with a 4096-byte slot header per 4 slots, so a
bn64/bk256 B tile costs 16384 B and the block 22528 B → 2 blocks/CU, and the
`(1 + 16/bn)` A-amortization gain is smaller than the blocks/CU loss. hipblas
can afford bn96 because it only launches 54 blocks.

### Where the residual 15–20% actually is

The decomposition (ratios to the same-run rotated kernel) says:

- **`v_mmac` costs 0.5%** — the MMA itself is free.
- **`ds_read_matrix_trans_format` costs 9.3%** — the format-transposing LDS read
  of A_s and B_s (8 × 1024 B per iteration). This is the single largest term.
- The rest is exposed global-load latency, ~3.5% of which the loop rotation
  recovers.

So the remaining work is *not* about the tile shape or the number of blocks — it
is about that transposing LDS read. The two known escapes are both closed:
`b_vgpr_direct_load` (already in the tree, `tilelang_ds_read_format_hip.hpp:400`)
skips the round trip but reads 8 consecutive k at a fixed n, so 16 lanes hit 16
different rows 25600 B apart — ~4× read amplification, measured 2.09× slower; and
double-buffering to overlap it costs LDS, which the occupancy cliff above turns
into a net loss.

<a id="2026-10-20"></a>
## 2026-10-20 — the fat tile is real: bn96/bk256/ns2 for N=5120

Two changes, both landed in the working tree:

1. **`src/hcu/op/mls.cc` packs fp8-trans `(16,128)` dst tiles densely.** A 16x256
   tile is exactly 4096 B (`gran = 64*(c>>2) + 4n + ((c&3+(n>>1))&3)` spans
   0..4095), but the 4K-slot accounting charged 4096 (slot) + 2x1024 (payload) =
   6144 B, i.e. 50% dead padding. `MlsPackedTileBytes` now returns -1 (contiguous
   / default allocation) whenever the dense size is a whole number of 4K slots.
   bk=128 (2048 B/tile) and bk=384 (3 slots) keep the conservative layout.
2. **`decode_cfg` selects `bn96/bk256/ns2/t384` for `N == 5120 and K % 256 == 0`.**
   `2*(16+96)*256 = 57344 <= 65536`, so two stages still fit one CU; this is the
   widest tile that does. qkv/gate_up keep bn32 and are unchanged.

Measured on a cleared kernel cache, idle card 0, one process, paired, 5 rounds:
`ship/old` = 0.947/0.940/0.943 on o and 0.928/0.925/0.929 on down at M=1/16/32,
all `rel = 0.0`. Standing vs hipblas is now o 1.117/1.064/1.137 and down
1.184/1.100/1.061 — improved, still behind.

Two operational notes. The disk kernel cache is keyed without codegen, so entries
predating an emitter change are served verbatim (source *and* binary); clear
`~/.tilelang/cache/<version>/linux-x86_64/kernels/*` after touching lowering or
headers. And dense packing incidentally makes `bm16/bn16/bk256/ns4/t64` exact
(its stage stride is now the logical stride); bk128 remains wrong at 5120 vs 2048.

<a id="2026-10-20b"></a>
### 2026-10-20b — the wide tile did not cut A after all; warp count does

The 2026-10-20 landing assumed bn 16 -> 96 would cut A's re-read from 320x to
54x. The SASS says otherwise: in the k-loop A's two `matrix_load_64x16_b8` take
their address from the stage base and k offset only — **no warp-id term** — while
B's loads compute `warp % 6` and offset by `warp * 2048`. Every warp therefore
loads the same A, so A's total re-read is blocks x warps = N/16 warp-loads, an
invariant of the tiling. 54 blocks x 6 warps is exactly the old 320 x 1. The fat
tile's 5-7% came from prefetch depth and block count, not traffic.

That makes the warp count, at fixed bn, a real lever (bn96 admits {1,2,3,6}):

| warps | 6 (t384) | **3 (t192)** | 2 (t128) | 1 (t64) |
|---|---|---|---|---|
| vs t384 | 1.000 | **0.952-0.962** | 0.971-0.983 | 1.23-1.29 |

U-shaped: 3 warps is the balance between A re-read and latency hiding. `t192` is
now what `decode_cfg` returns for N=5120. Versus hipblas at M=1/16/32:
o 1.069/1.020/**1.000**, down 1.110/1.053/**0.935** — o ties and down wins at
M=32, and every arm is exact.

The residual is still per-step (c = 0.810 vs 0.664 us/step; b = 15.9 vs 17.5, a
wash). The next structural lever is to stop replicating A in the producer
(`src/hcu/op/mls.cc`): A is ~1/3 of per-step per-block bytes, so loading it once
per block at 6 warps bounds the gain at ~28% of per-step traffic.

<a id="2026-10-20c"></a>
### 2026-10-20c — producer A de-replication: built, measured, CLOSED

The lever above (stop replicating A in the MLS producer) was implemented in
`tilelang_mls_base::{async_mls_load_asm_impl_ct,_impl_rt}` as an issue gate —
only the canonical owner of each source coordinate issues, every other warp
skips (`cluster_idx[ii] < EffectiveWarpCluster[ii]` per axis, so it covers both
full and partial replication). Bit-exact everywhere (rel = 0.0), and slower
everywhere. Paired same-process, idle card:

| shape / M | bn96/t384 base -> new | bn96/t192 base -> new |
|---|---|---|
| o M=16 | 37.44 -> 42.88 (**1.145**) | 36.16 -> 45.76 (**1.265**) |
| o M=1 | 36.64 -> 38.24 (**1.044**) | 35.20 -> 45.28 (**1.286**) |
| down M=16 | 93.28 -> 111.04 (**1.190**) | 88.64 -> 119.36 (**1.347**) |
| down M=1 | 92.48 -> 99.04 (**1.071**) | 87.04 -> 118.24 (**1.358**) |

**Why (SASS).** `cp_async_wait<1>` is lowered to a `vmcnt` constant sized from
the async loads on the *unconditional* path. A inside `if (owner)` leaves that
count, so the wait becomes one group too strict — t192 `vmcnt(8) -> vmcnt(4)`
(depth zero), t384 `vmcnt(4) -> vmcnt(2)`. The 22% traffic saving never pays
for the lost prefetch level.

**Two corrections this measurement forced.** (a) t384's A handle is
`WarpMN=3, WarpK=2`: the K axis is already split, so A is 3x replicated there,
not 6x — the first (fully-replicated) rule emitted byte-identical SASS at t384.
(b) A's duplicates are served from L2 and change no DRAM traffic, so they were
never on the step-time critical path; the "A is 1/3 of the bytes" bound was
about the wrong resource.

Header change reverted; the t192 landing of 2026-10-20b stands. Standing
re-measured after the revert (paired, 7 rounds, idle card): o 1.069/1.027/1.024
and down 1.110/1.042/**0.928** at M = 1/16/32, all exact.

<a id="2026-10-20d"></a>
### 2026-10-20d — M=64: the tile neighbourhood is closed; neither wave quantization nor co-residency explains it

**1. The M=64 neighbourhood is a measured optimum.** At `bm64`, paired
same-process, every arm `rel = 0` (hipblas 136.64 µs on down and 52.32 µs on o
in the same process):

| config | down M=64 /hip | o M=64 /hip |
|---|---|---|
| **ship `64x64x128 ns2 t256`** | **1.178** | **1.083** |
| `bn128 ns2 t256` | 1.280 | 1.128 |
| `bn96 ns2 t256` | 1.219 | 1.183 |
| `bn96 ns3 t256` | 1.304 | 1.235 |
| `bn64 bk256 ns2 t256` | 1.311 | 1.294 |
| `bn96 ns2 t384` | 1.371 | 1.303 |
| `bn128 ns2 t512` | 1.527 | 1.443 |
| `bn96 bk256 ns1 t256` | 1.481 | 1.459 |
| `bn64 ns4 t256` | 1.650 | 1.547 |
| `bn192 ns2 t384` | 1.823 | 1.700 |
| `bn64 ns3 t256` | 1.867 | 1.716 |

Depth (`ns3`/`ns4`) and width (`bn96`/`128`/`192`) both lose, and `ns3` is worse
than `ns4` at the same tile. So the M=64 gap is not LDS volume, consistent with
[2026-10-16c](#m64-paired).

**2. Wave quantization — tested, not supported.** At bn64 the block is 32768 B of
LDS, so 2 blocks fit per CU, and `N/bn` = 80 blocks land on 64 CUs: 16 CUs run
two blocks, 48 run one. The natural hypothesis is a 62.5%-efficient ragged wave.
Neither test supports it:

- *Direct*: `bn80` gives exactly 64 blocks (5120/80), one per CU — **1.177x
  slower** on down (190.08 vs 161.44 µs). Confounded: bn80 admits only 5 warps
  (t320), which changes A's replication factor, so it is not single-variable.
- *Indirect*: at the ship tile, vary only N so the block count is 64/80/96/128.
  Throughput is **780 / 821 / 920 / 1060 GB/s** — monotone in block count, with
  the perfectly balanced 64-block point the *slowest*. That refutes "ragged
  waves are the cost", but it carries the confound the 2026-10-17f retraction
  named: N moves block count and total bytes together, and a fixed-overhead
  amortisation alone explains the rise (a two-point fit gives a 71-96 µs
  intercept, which is not physical as a launch floor — so the linear model is
  wrong too; all that can be read is the monotonicity).

**Verdict.** Wave quantization is not the M=64 gap, and without a grid-shape knob
there is no clean experiment separating co-residency from fixed overhead. Do not
re-open on this evidence.

**3. Where M=64 actually sits.** `N·K / t` against the measured 1725 GB/s read
ceiling (B is >99% of the DRAM read stream at these M):

| | M=1 | M=16 | M=32 (est) | M=64 |
|---|---|---|---|---|
| o | 69.1% | 67.2% | ≈60.7% | **42.9%** (740 GB/s) |
| down | 87.3% | 85.8% | ≈75.9% | **47.2%** (814 GB/s) |

M=64 moves the same B bytes as M=16 but takes 1.57x (o) and 1.82x (down) the
time; the extra work is A (re-read once per block — 80x at bn64) plus 4x the
mmac volume, and it is not hidden behind the B stream. At 47% of the read
ceiling and ~24% of fp8 peak (104 TFLOPS on down), M=64 is neither
bandwidth- nor compute-bound: it is the same exposure/overlap deficit recorded
at M≤16, no longer covered by B.
