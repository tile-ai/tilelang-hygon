# fp8 Channelwise GEMM on HCU — decode regime (M ≤ 128)

Follow-up to `fp8_channelwise_gemm.md` (the 200 → 300 → 432 TFLOPS three-stage
story). This note covers the **decode regime** (qwen3-32B per-token GEMMs:
M=1–128, K=5120, N=51200 gate_up), where the kernel is *bandwidth-bound*, not
compute-bound, and the goal is to beat hipblaslt's fp8 rowwise kernel on
effective bandwidth.

## The setup

- B is (N, K) fp8 = 51200 × 5120 × 1 B = **262 MB**; 1800 GB/s peak → floor
  ≈ 146 µs for a single read of B. **Measured achievable read on this box:
  ~1646 GB/s (91.5%)**, so the real floor is ≈ 160 µs and ~91.5% is the
  hard ceiling for any single-read streaming kernel.
- hipblaslt (`torch._scaled_mm`, rowwise, bf16 out) measured with the same
  graph-replay timing: M=1 174.9 µs (83.3%), M=2 68.1%, M=4 73.4%,
  M=8 75.2%, M=16 76.4%, M=32 65.4%.
- Our initial MLS GEMM (block 256×256×64, ns=1, t512): M=1 198.7 µs (73.3%),
  M=2 73.4%, M=32 67.5% — M=2 and M=32 beat hipblas, the others lost.

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

## M ≤ 16: bn32 / bk256 / ns0 / t256 — ~78%, beats hipblas (except M=1)

> **2026-10-13 re-verification:** all utilization numbers in this section were
> re-measured with explicit CUDA-graph capture + replay, events, interleaved
> A/B/A/B, min/med of 5×60 (tilelang `do_bench_cudagraph` returns ms on this
> stack / single-launch events carry host overhead — both inflate
> utilization). Effective bandwidth = N×K / latency vs 1800 GB/s nominal.

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

## Why M ≤ 16 is flat at ~78% and M=32 drops to ~59%

- Grid is fixed (N=51200, bn=32) → M ≤ 16 only changes how many of the bm=16
  fragment rows are live, not the grid size or per-block bytes.
- The ~22% gap to nominal is real: 91.5% ceiling, minus per-block overhead
  (barrier + wait + ds_read serialization) and a non-perfect stream.
- At M=32, bm=32 halves the block count and doubles fragment work per block;
  MMA pressure starts to interfere with the streaming path (best measured
  config (32,32,256,1,256): 246 µs ≈ 59%).

## M ≥ 32 ≤ 128: the streaming kernel breaks down

For M ≥ 32 the bm=pow2≥M tiles inflate: bm=64 → A tile is 64×256×1B = 16 KB
vs B's 32 KB (bn=32); bm=128 → 32 KB A vs 16 KB B per block. A is pure
redundant reads (A is 6 MB; rereading it 800× doesn't cost bandwidth but adds
LDS/MMA work), and MMA becomes the bottleneck: a 128×128×256 fp8 tile at
432 TFLOPS peak needs ~9.7 µs of MMA per block vs ~6.7 µs of B streaming —
compute-bound. Measured M=32 with the M≤16 winner shape: 246 µs ≈ 59%
(hipblas ~65%), i.e. the streaming win does not transfer.

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

**Where the time goes** (load-only kernels, no mmac, so the load path is isolated):

| | time | note |
|---|---|---|
| B stream alone | **166.1us** | 1578 GB/s = **95.9%** of the 1646 GB/s achievable ceiling |
| A + B through LDS | **196.0us** | A's round trip costs **+30us** for a 327 KB matrix |
| full kernel | ~247us | ~51us of compute exposed |

The load path is not the problem. A's `global → LDS → VGPR` round trip is the largest
overhead we own, and it is **not recoverable without the excluded weight reorder**:

- **A→VGPR direct is refuted.** The MMA fragment wants 8 *consecutive-K* fp8 per lane at
  one row — `A_local[(ki*warp_rows + Mi)*8]` with lane → `m = Mi*16 + (lane&15)`,
  `k = ki*32 + (lane>>4)*8`. Loading that from global means 8-byte requests at a
  **5120-byte lane stride**, the same strided pattern that killed the `b_vgpr_direct`
  experiment. The LDS round trip is the *price* of the fragment layout.
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
hipblas gets the same bytes to DRAM 5.8% sooner via hand-tuned scheduling (fine-grained
`vmcnt(4)` waits instead of our coarser `cp_async_wait`, and 4 `s_setprio` placements),
not via any tile-shape or resource choice available from Python.

Occupancy is **not** the lever: 82 VGPR → 6 blocks/CU (24 warps) vs 50 VGPR → the
thread-capped 10 blocks/CU (40 warps) buys ~1%, and that does not survive re-measurement.

Our wait structure is already clean — the k-loop body has no `matrix_load`, no
`s_barrier` and no `vmcnt` wait; the `vmcnt(0); s_barrier` sits *after* the back-edge,
once, in the epilogue.

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

**What the reference does and does not do** (from disassembling
`m64/src/gateup_layoutA_vmcnt4.s`): weights go **global→LDS** via
`buffer_load_dwordx4 ... slc lds`, exactly the path our `matrix_load` uses — there
is no "weights bypass LDS into VGPR" trick. The `buffer_load_ubyte_d16` clusters
are the channelwise bias/scale epilogue gather, not weights. Its loop body is
64 mmac + 32 `ds_read_b128` + 8 `ds_write_b128` + **1328 address VALU** (the price
of Layout A's blocked addressing), with 149 `s_waitcnt` at fine-grained
`vmcnt(1..14,0)` granularity. **slc is already in on our side, and hipblas's is
partial** — only 8 of its 20 weight→LDS loads carry it.

The one in-scope lever left is the MLS partition heuristic
(`ComputeMlsWarpPartition`, `src/hcu/op/mls.cc:506`; `AnalyzeResourceAxis`,
`src/hcu/transform/hoist_mls_resource.cc:1615`): the reference's plain
`ds_read_b128` lets it hold an arbitrary tile shape, while our
`ds_read_format_tensor_a/b` pins the partition and degrades to (4,2)/(8,1) with
3–4× LDS read amplification as soon as the shape moves. That needs a C++ rebuild.

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
"utilization" numbers that exceed the measured 91.5% read ceiling and are
not physical.
