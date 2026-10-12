# fp8 Channelwise GEMM on HCU — decode regime (M ≤ 128)

Follow-up to `fp8_channelwise_gemm.md` (the 200 → 300 → 432 TFLOPS three-stage
story). This note covers the **decode regime** (qwen3-32B per-token GEMMs:
M=1–128, K=5120, N=51200 gate_up), where the kernel is *bandwidth-bound*, not
compute-bound, and the goal is to beat hipblaslt's fp8 rowwise kernel on
effective bandwidth.

## The setup

- B is (N, K) fp8 = 51200 × 5120 × 1 B = **262 MB**; 1800 GB/s peak → floor
  ≈ 146 µs for a single read of B.
- hipblaslt (`torch._scaled_mm`, rowwise, bf16 out) measured: M=1 168.9 µs
  (86.2%), M=2 72.0%, M=4 77.8%, M=8 77.7%, M=16 78.8%, M=32 67.0%.
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

## M ≤ 16: bn32 / bk256 / ns1 / t256 — ~94%, beats hipblas

DSL-level sweep for M=1 with bm=16 (fixed grid 800 blocks, each block streams
its 327 KB of B):

| bn | bk | ns | thr | util |
|----|----|----|-----|------|
| 32 | 256 | 0 | 256 | **94.8%** |
| 32 | 256 | 1 | 256 | 93.8% |
| 64 | 256 | 1 | 512 | 84.7% |
| 64 | 256 | 0 | 512 | 84.5% |

Key: **bn=32** (bn=64 doubles per-block bytes and halves the block count →
worse tail/latency amortization) and **t256** (t512 halves per-thread
coalescing width on a 32-wide bn tile). The M=1 edge over hipblas is the
*small-tile streaming shape*, not slc/setprio.

Full M=1–32 sweep (bn ∈ {16,32,64}, bk ∈ {128,256}, ns ∈ {0,1,2},
thr ∈ {256,512}, bm = min(16, M)):

```
M= 1 best bn32 bk256 ns0 t256:  153.9us  94.6%  vs hipblas  86.2%  BEAT
M= 2 best bn32 bk256 ns1 t256:  154.6us  94.2%  vs hipblas  72.0%  BEAT
M= 4 best bn32 bk256 ns1 t256:  154.7us  94.1%  vs hipblas  77.8%  BEAT
M= 8 best bn32 bk256 ns1 t256:  155.0us  94.0%  vs hipblas  77.7%  BEAT
M=16 best bn32 bk256 ns1 t256:  161.8us  90.0%  vs hipblas  78.8%  BEAT
M=32 best bn32 bk256 ns1 t256:  193.3us  75.4%  vs hipblas  67.0%  BEAT
```

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
fp8-vs-fp8 accumulation-order noise, not an error. `torch.abs().sum()` over
the same 262 MB B runs at only 8.6% — a streaming reduce does not saturate
HBM; the GEMM's 94% is not suspicious relative to that.

## Why M ≤ 16 is flat at ~94% and M=32 drops to 75%

- Grid is fixed at 800 blocks (N=51200, bn=64) → M ≤ 16 only changes how many
  of the bm=16 fragment rows are live, not the grid size or per-block bytes.
- The residual ~6% is per-block overhead (barrier + cp_async_wait + ds_read
  serialization) — the same 6% that limits every M.
- At M=32, bm=32 halves the block count (400) and doubles fragment work per
  block; MMA pressure starts to interfere with the streaming path.

## M ≥ 32 ≤ 128: the streaming kernel breaks down

For M ≥ 32 the bm=pow2≥M tiles inflate: bm=64 → A tile is 64×256×1B = 16 KB
vs B's 8 KB; bm=128 → 32 KB A vs 4 KB B per block. A is pure redundant reads
(A is 6 MB; rereading it 800× doesn't cost bandwidth but adds LDS/MMA work),
and MMA becomes the bottleneck: a 128×128×256 fp8 tile at 432 TFLOPS peak
needs ~9.7 µs of MMA per block vs ~6.7 µs of B streaming — compute-bound.

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
