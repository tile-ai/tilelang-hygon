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

The M ≤ 16 win transfers only as far as the per-block B tile stays the
bottleneck. For M ≥ 32 the fix is to stop pretending it is a streaming GEMM:

1. **split-K**: each block streams a K-slice (chunk = K / split_k), so the
   grid grows to 800·split_k blocks and MMA fits under the stream; the
   partial sums are reduced with an atomic epilogue.
2. **fold M into the grid** (M_blocks × N_blocks), each block still streams
   one B tile — the natural M ≤ 16 approach taken to higher M.

Both are next steps beyond this note.

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
