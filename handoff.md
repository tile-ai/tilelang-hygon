# HCU fp8 Channelwise Decode GEMM — Handoff

Status: **2026-10-16c** (M=64 is LDS-port-bound; gap to hipblas is 5.8% paired) /
2026-10-16b / 2026-10-16 / 2026-10-15 / 2026-10-13. Branch `feat/hcu-smls-b-vgpr`.

## 2026-10-16c — M=64 IS LDS-PORT-BOUND, AND THE GAP IS 5.8%. Read this first.

### 0. Methodology: GPU 4 is shared, so only paired ratios count

`rocm-smi --showpower` shows HCU[2]-[7] drawing 360-372 W with no process of ours
running, **including HCU[4]**, which is the GPU we benchmark on. Co-tenant load moves
the clocks: the same kernel measures 244us in one invocation and 281us in the next,
and the *ordering* of two candidate configs flips between runs.

Within one process, repeated rounds agree to ~0.2us — that part is fine. Across
invocations the whole curve shifts by up to ~15%. **Never compare a number measured
now against one measured in an earlier run or session.** To compare two kernels, build
both in one script and interleave the rounds; the paired *ratio* was stable to ~0.1%
while absolutes moved 15%. Treat any gain under ~3% as unproven unless it is a paired
ratio. (This bit us: a `bn64/bk128/ns2` config looked 1% faster in one run, was 2%
slower in the next, and is genuinely slower — see below.)

### 1. The gap, measured properly

Paired and interleaved in one process, 4 rounds, `torch._scaled_mm` (bf16 out) vs our
`_decode_cfg(64)` = `(64,64,256,1,256,1)`:

| | time (contended) | vs hipblas |
|---|---|---|
| hipblas `MT128x64x256` | 257.0us | — |
| **ours `bn64 bk256 ns1 t256 sp1`** | **272.0us** | **1.058x** |
| ours `bn64 bk128 ns2 t256 sp1` | 276.3us | 1.076x |

So the gap is **5.8%** — the user's "6%" is exactly right — and `bk256/ns1` is the
better of the two. The quiet-machine equivalents are ~231.7 vs ~247.0us (62.9% vs
59.1%). Note the reproducible hipblas baseline is bf16-out; the deliverable's 213.0us
uses fp16-out + bias and `torch._scaled_mm` rejects fp16 out for rowwise scaling.

### 2. Where the 247us goes (all measured, load-only kernels with no mmac)

| | time | note |
|---|---|---|
| B stream alone | **166.1us** | 1578 GB/s = **95.9%** of the 1646 achievable ceiling |
| A + B through LDS | **196.0us** | A's round trip costs **+30us** for a 327 KB matrix |
| full kernel | ~247us | ~51us of compute exposed |

The load path is *not* the problem. A's global→LDS→VGPR round trip is the largest
overhead we own, and it is **not recoverable without a weight reorder**:

- **A→VGPR direct is refuted.** The MMA fragment wants 8 *consecutive-K* fp8 per lane at
  one row (`A_local[(ki*warp_rows + Mi)*8]`, lane → `m = Mi*16 + (lane&15)`,
  `k = ki*32 + (lane>>4)*8`). Loading that from global is 8 bytes at a **5120-byte lane
  stride** — the exact strided pattern that killed the `b_vgpr_direct` experiment. The
  LDS round trip is the *price* of the fragment layout, not an implementation defect.
- **`bn` cannot amortize it.** bn 128/256/512 → 253.6 / 275.4 / 382us. Halving the
  redundant A re-read loses more on the B path than it saves.

### 3. LDS volume is identical to hipblas — the residual is overlap quality

Per warp per k-step the committed config issues 32 `ds_read` of 16 B/lane = 32 KB read,
plus A(16 KB) + B(16 KB) = 32 KB written, over a 64x64 tile with k=256:

| | LDS bytes / output / k-step |
|---|---|
| ours `bn64 bk256 t256` | 160 KB / (4096 x 256) = **0.15625** |
| ours `bn64 bk128 ns2 t256` | 80 KB / (4096 x 128) = **0.15625** |
| hipblas `MT128x64x256` | 320 KB / (8192 x 256) = **0.15625** |

Identical, and invariant across `bk`. The 4x read amplification over the tile's unique
bytes is entirely the per-warp operand duplication the mmac layout requires — the
`ds_read` volume (512 B/lane) exactly equals the mmac operand volume (32 mmac x 8 B A
+ 8 B B), so there is **no recoverable waste**. Total LDS traffic is 2.56 GB, a
192-240us floor at 128-160 B/clk/CU: the same order as our 247us, so LDS port pressure
is a plausible *co*-limiter, but it cannot explain a gap to a kernel moving the identical
volume.

So the residual 5.8% is **overlap quality**. Our load path alone runs at 95.9% of the
achievable ceiling, yet in the full kernel B's effective rate falls to ~1062 GB/s.
hipblas gets the same bytes to DRAM 5.8% sooner through hand-tuned scheduling
(fine-grained `vmcnt(4)` waits vs our coarser `cp_async_wait`, 4 `s_setprio`
placements), not through any tile-shape or resource choice reachable from Python.

Occupancy is **not** the lever: 82 VGPR → 6 blocks/CU (24 warps) vs 50 VGPR → the
thread-capped 10 blocks/CU (40 warps) buys ~1%, and it does not survive re-measurement.

Our wait structure is already clean: the k-loop body contains no `matrix_load`, no
`s_barrier` and no `vmcnt` wait — the `vmcnt(0); s_barrier` sits *after* the back-edge,
once, in the epilogue. So there is nothing to relax there.

### 4. Housekeeping

- An uncommitted `gemm_hcu_mmac.py` edit that moved `set_prio` to *before* the LDS reads
  was a regression (it is the sp2 variant, 281us) and has been **reverted**.
- The four `b_vgpr` WIP files remain uncommitted and are a confirmed dead end
  (backup at `/tmp/b_vgpr_deadend_full.patch`).
- The only untried in-scope lever is the MLS partition heuristic
  (`src/hcu/op/mls.cc:506`, `src/hcu/transform/hoist_mls_resource.cc:1615`), which
  needs a C++ rebuild.

## 2026-10-16b — THE 70% TARGET IS "LAYOUT A". Read this before the section below.

The "hipblas reaches ~70% at M=64 after slc-tagging" premise does not survive
contact with the reference deliverable. Its own README
(`/workspace/llm-adaptation/qwen3-32b-fp8-rowwise/deliverable_fp8_gemm/README.md`)
states the M=64 gate_up recipe and what it costs:

| M | kernel | tile | 优化 | 需权重重排 | 基线 → 优化 |
|---|---|---|---|---|---|
| 64 | gate_up | MT128x64x256 | **Layout A + vmcnt4** | ✅ | 213.0 → **203.3us** |
| 64 | down | MT96x64x256 | slc + W2 | ❌ | 131.8 → 117.7us |
| 128 | gate_up | MT128x128x256 | Layout A + vmcnt4 | ✅ | 310.4 → 288.0us |

203.3us / 262.1 MB = 1289 GB/s = **71.6%** — that is the 70% figure. It is reached
**only** by Layout A (blocked weight reorder, `reorder_weights.py`), which the user
explicitly excluded ("Layout A 不要看了先"). **Without the reorder, hipblas's own
baseline is 213.0us = 68.4%.** So the in-scope gap is ours 247.0us (59.0%) →
213.0us (68.4%), i.e. **34us**, not 44us to a 71.6% that requires reordering.

**The "A staged once" direction (V1 = 217.0us) is NOT actionable.** Two independent
proofs, both measured:

- **Grouping does nothing.** Staging A for 2 k-steps at a time into two separate
  (64,256) buffers (every MLS atom keeps the proven `seq<64,256>`/(2,2)/`axis::mn`
  shape; B also double-buffered so the pipeline planner accepts it) → **249.3us**,
  i.e. no gain and 3us slower than the 246.4us baseline. Total instructions and
  total LDS bytes are unchanged. (`/tmp/m64_astage.py`.) The cost is per-byte of
  LDS write, not per load group.
- **The byte count is invariant.** A's LDS write = `M·N·K/bn` — unchanged by `bm`,
  by `bk`, by split-K (each block keeping one A K-chunk resident still totals
  `(N/bn)·S × 64·(K/S)`), and by A-hoisting. Only `bn` reduces it, and `bn>64`
  breaks the MLS partition (section below).

**N-scaling separates the two costs** (`/tmp/m64_scale.py`, same shape
bm64/bn64/bk256/ns1/t256 at N = 12800/25600/51200/102400):

| N | B MB | ours | GB/s | % | W2 (compute only) |
|---|---|---|---|---|---|
| 12800 | 65.5 | 80.3us | 816 | 45.3 | 42.9us |
| 25600 | 131.1 | 142.2us | 921 | 51.2 | 66.4us |
| 51200 | 262.1 | 247.0us | 1061 | 59.0 | 113.9us |
| 102400 | 524.3 | 462.7us | 1133 | 62.9 | 209.8us |

Slope **0.834 us/MB = 1200 GB/s** marginal, intercept **25.7us**. hipblas
(`torch._scaled_mm`, same timing harness) has slope 0.793 = 1261 GB/s and intercept
25.4us — **the same structure; the whole difference is 5% of stream efficiency.**
Perfect overlap of the 114us mmac with a 159us DRAM floor would give 159us; we get
247us, so compute and stream are largely *serialised*, and prefetch depth is not
the fix (`ns=2` generates a correct `cp_async_wait<1>` + double-buffered loop —
verified in `/tmp/src_64_128_2_128.cu` — and is still slower).

**hipblas's tile shape does not transfer.** From the reference .s header, its kernel
is `Cijk_Alik_Bljk_F8BS_MT128x64x256_TT8_2_WG16_32_1_MFWGS512_WGM1_bias_channelwise`
— 512 threads on a (128,64) tile vs our 256 on (64,64). Copied into our DSL:
`bm128 bn64 bk256 t512 ns1` → **589.8us** (partition degrades to (4,2)); `bk128 ns2`
→ 479.0us. So the shape is not the lever by itself.

**What the reference actually does — and what it does *not* do** (verified by
disassembling `m64/src/gateup_layoutA_vmcnt4.s`):

- Weights go **global→LDS**, exactly like ours: `buffer_load_dwordx4 v128, s[8:11],
  0 offen slc lds` (16 B/lane, direct to LDS). There is **no** "weights bypass LDS
  into VGPR" trick to copy.
- The `buffer_load_ubyte_d16`/`_d16_hi` + `v_lshlrev 8` + `v_or` + `v_cndmask`
  clusters are the **channelwise bias/scale gather** in the epilogue, not weights.
- Its main loop body is 64 `v_mmac_f32_16x16x32_fp8_fp8` + 32 `ds_read_b128` +
  8 `ds_write_b128` + **1328 address VALU** (`v_add_co_u32`/`v_lshlrev`/`v_cndmask`/
  `v_cmp_lt`/`v_or`/`v_and`) — Layout A is expensive to address; it buys HBM
  contiguity. 149 `s_waitcnt` with a fine-grained `vmcnt(1,2,4,5,6,8..14,0)` spread
  (vs our whole-group `cp_async_wait<0>`).
- **slc is already in on our side and hipblas's is partial**: only **8 of the 20**
  weight→LDS loads carry slc in that kernel. Nothing to fix here.

**Bottom line.** Without Layout A the ceiling for this shape is ~68.4% and we are at
59.0%; the remaining 34us is not A's staging (proven above). The reference's
`ds_read_b128` (plain 16 B/lane LDS reads) is what lets it hold an arbitrary tile
shape, whereas our `ds_read_format_tensor_a/b` pins the warp partition and degrades
to (4,2)/(8,1) with 3–4× LDS read amplification as soon as the shape moves. That
heuristic — `ComputeMlsWarpPartition` (`src/hcu/op/mls.cc:506`) and
`AnalyzeResourceAxis` (`src/hcu/transform/hoist_mls_resource.cc:1615`) — is the
only in-scope lever left, and it needs a C++ rebuild.

## 2026-10-16 — M=64 gap ROOT-CAUSED: it is A's LDS round-trip, not bandwidth

All numbers GPU 4, graph-replay, same process, M=64 gate_up (N=51200, K=5120,
B = 262.1 MB). Scripts: `/tmp/m64_decomp.py`, `/tmp/m64_adiag.py`,
`/tmp/m64_w2.py`, `/tmp/m64_bn128.py`, `/tmp/m64_ntile2.py`, `/tmp/m64_final.py`.

**The kernel is not bandwidth-bound.** Two pinned-address controls keep the exact
instruction stream and LDS traffic and change only the addresses:

| variant | what changes | time |
|---|---|---|
| V0 baseline | — | 246.9us (59.0%) |
| V1 A staged once | A's per-iteration `matrix_load` removed | **217.0us (67.1%)** |
| V8 A pinned to one 16 KB | A stays L2-resident; identical instrs/LDS | 246.1us |
| V9 B pinned to one 16 KB | B's 262 MB of DRAM traffic disappears | 239.0us |
| W2 pure compute | A_s/B_s filled once, bare `T.gemm` loop, zero global traffic | 114.1us |
| hipblas `torch._scaled_mm` | — | 226–232us |

- V8 == V0 → A's cost is **not** L2 capacity/bandwidth.
- V9 saves only 8us for 262 MB of DRAM reads → **the DRAM is essentially idle; the
  kernel is not bandwidth-limited at all.** (262 MB / 246 us = 1065 GB/s = 59%.)
- V8 vs V1 differ *only* by A's `matrix_load` and its 262 MB of LDS writes → **29us**.
  262 MB / 29 us = 9.0 TB/s ≈ LDS write bandwidth (64 CU × 128 B/clk × 1.3 GHz
  = 10.6 TB/s). **A's LDS write traffic is the whole gap.**

**A's LDS write traffic = M·N·K / bn** — independent of `bm` and of `bk`. Only `bn`
(N-coverage per block) reduces it, and `bn=64` is the floor for the fast MLS path:
every wide-bn config is slower because the MLS warp partition changes shape and
LDS read amplification grows. 10 wide-bn configs measured, all worse than the
245.9us baseline:

- `bn128 bk256 t256` → 340.3us (VGPR 130, 3 blk/CU; B flips to `axis::k`)
- `bn128 bk128 t256` → 281.4us (**same** 82 VGPR / 6 blk/CU as the best config, so
  it is *not* occupancy; partition becomes (4,1) → 3× LDS read amplification)
- split-gemm `bn128` as 2×(64,bk) B buffers + 2 C fragments → 333.1us, identical to
  plain bn128 (proves the B_local register cost is not the issue)
- `nt`-shared-A (2/4 sequential n-tiles sharing one A_s load) → 364–415us
- `bk512`, `k_pack=2`, `ns=2@bk256`, A-hoisted-into-a-(64,512)-buffer → all worse

**Structural wall**: every `__builtin_hcu_matrix_load_*` builtin is global→LDS;
there is no register-destination variant, so A cannot skip the LDS round-trip via
the MLS path. The only A→VGPR route is r_mls, which lowers to 8-byte
`hcu_buffer_load<fp8_t, 8>` — already a proven 3.7× loss.

**Where that leaves us**: `bm64 bn64 bk256 ns1 t256 sp1` (245.9us fp32 / 243.4us
bf16) is a structural local optimum. V1 (217.0us) proves the design would beat
hipblas by ~6% if A's staging were free, but that is unreachable — see the
2026-10-16b section above for the two measurements that close it off (grouping A's
loads changes nothing; the byte count is invariant under every restructuring).
The in-scope target is hipblas's **68.4% baseline (213.0us)**, not the 71.6% that
requires the excluded Layout A weight reorder.

## Overnight findings (2026-10-15) — SASS-verified, same-process head-to-head

- **M=16/32 win CONFIRMED with fresh same-process hipblas** (`/tmp/h2h_sweep_mls.py`,
  graph-replay timing, GPU 4): M=16 MLS 164.3us (88.6%) vs hipblas 186.6us (78.1%)
  **→ +11.9%**; M=32 MLS 199.8us (72.9%) vs hipblas 218.4us (66.7%) **→ +8.5%**.
  rel=0.00000 both. Best config unchanged: bm{M} bn32 bk256 ns1 t256.
- **Plan B (`b_vgpr_direct`, B global→VGPR direct) is a 4× regression — dead end.**
  The b_vgpr pipeline (this branch's 5 uncommitted files, gated OFF by default so
  the MLS baseline is unaffected) runs 860–910us at M=32 (vs MLS 199.8us / hipblas
  218us). SASS-verified (`/tmp/pipe_upfront.s`): even with all B slice loads issued
  up-front + one loop-bottom fence, the compiler keeps only **~4 8B loads in flight
  (rolling `s_waitcnt vmcnt(3)`) and re-drains before each mmac consumer**. A 4-mmac
  consumer group is far too short to hide ~500-cycle VMEM latency. Config sweep
  (bn32/64/128, t256/512, M-padded to 128, setprio) — all 16–237% *worse* than hipblas.
  M-padding to hipblas's 128-row tile is worse still (1786us): the b_vgpr fragment
  per n-warp grows and the short chain still can't hide latency.
- **Real hipblas shape decoded** from `/workspace/misc/Cijk_Alik_Bljk_F8BS_MT256x128x128_...WGM8_bias_channelwise.s`
  (MT256x128x128, TT16x8, 512 thr, **255 VGPR**, 64KB LDS, 288 mmac = 4×72/loop):
  B = **4× `buffer_load_dwordx4` (16B/lane) per 32-wide k-slice into a rotating
  register buffer (X0/X1)**, 8 slices = 32 dwordx4 in flight; A = **16× `ds_read_b128`
  from LDS** (double-buffered via 2 `__syncthreads` at the loop seam); **72-mmac
  chain** per 32-wide slice as the latency window; **one** `s_waitcnt vmcnt(0)` per
  2-slice (128-k) chunk. Its 72-mmac window is ~4.5× ours (16/slice), so its single
  drain hides full VMEM latency where ours cannot. **The 16B dwordx4 (half the
  instruction count, 2× bytes/inst) + 72-mmac window is the actual win** — neither is
  reachable through the MMAC templating without a hand-rolled kernel.
- **M=64/128 breakthrough (2026-10-15, round 8)** — the C++ levers + a config sweep
  that had never been run (earlier M=64/128 sweeps only tried M-*splitting*, which
  re-reads B and always loses) close the gap to near-parity. Winning levers:
  (a) **slc on MLS b8 atoms** (already committed) and (b) **s_setprio(1) around the
  MLS mmac loop** — new, wired via the `tl.hcu_mmac_setprio` annotation in
  `gemm_hcu_mmac.py` (header-only `promote_prio`/`restore_prio`, no rebuild). The
  `s_setprio` only pays off combined with a **never-tried tile shape**: for M=64/128
  a *single* block that spans the full M (bm=64/128) with **bn64/bn128 + 512 thr**
  (not the bm=M/bn32/t256 shape used for M≤32). Fresh same-process head-to-head
  (gate_up 51200×5120, GPU 4, graph-replay, all rel=0.00000):

  | M | committed (old cfg) | new best cfg | new lat | hipblas | new gap |
  |---|---------------------|--------------|---------|---------|---------|
  | 16 | +11.9% | bm16 bn32 bk256 ns1 t256 **sp1** | 164.6us | 186.7us | **+11.8%** |
  | 32 | +8.5%  | bm32 bn32 bk256 ns1 t256 **sp1** | 188.3us | 217.8us | **+13.5%** |
  | 64 | −89%   | bm64 bn64 bk256 ns1 **t256 sp1** | 246.4us | 229.9us | −7.2% |
  | 128 | −225%  | bm128 bn128 bk128 ns1 t512 sp0 | 309.4us | 311.0us | **+0.3%** |

  M=64/128 went from **losing 2–4×** to **parity / slight win**. M=16/32 keep the
  clear win (and sp1 pushes M=32 to +13.5%). M=64 is the only M still short
  (−7.2%); it needs the 72-mmac window hipblas has (see Plan B / real-shape note)
  to fully close. Scripts: `/tmp/h2h_sweep64_128_v3.py`, `/tmp/h2h_final_confirm.py`.
- **M-splitting confirmed wrong** (still): bm=32/by=2 for M=64, bm=16/by=4 for
  M=128 re-read B and are worse than the single-full-M block above.

qwen3-32B decode GEMM bandwidth utilization (M=1..128, K=5120, N=51200 gate_up)
on Hygon DCU BW1101 (gfx938, 64 CU, warp 64, HBM peak ≈ 1800 GB/s).

## Current standing vs hipblas (M=32/64/128, gate_up 51200x5120)

effective BW = N×K/latency; % of 1800 GB/s peak. B = 262 MB, floor ≈ 146 µs.

| M | hipblaslt | tilelang best (config) | gap |
|---|-----------|------------------------|-----|
| 1..16 | ~64-67% | ~67% (bm16 bn32 bk256 ns1 t256) | **beaten** |
| 32 | 66.8% | 53.8% (bm32 bn128 bk64 ns4 t256) | −13 pp |
| 64 | 62.2% | 49.7% (bm64 bn128 bk64 ns2 t256) | −12.5 pp |
| 128 | 46.0% | 47.9% (bm128 bn128 bk64 ns2 t256) | **beaten** |

8192³ compute peak: tilelang MLS 432 TFLOPS vs hipblaslt ~560 — but decode is
bandwidth-bound, so 432 is not the blocker here.

All timings from `do_bench_cudagraph` on GPU 4 (pin it! device 0 is busy).

## What wins at M<=16 (already committed as the decode path)

`perf/gemm/fp8_channelwise_decode_bench.py` — bm16 bn32 bk256 ns1 t256 MLS
path beats hipblas on effective BW at M=1..16. That kernel was extended to
M=32..128 by growing block_M (bm=32/64/128), which is where we lose.

## Root cause of the M=32..128 gap (verified in generated CUDA)

hipblaslt decode kernel (`qwen3-32b-fp8-rowwise/deliverable_fp8_gemm/m64/src/gateup_layoutA_vmcnt4.s`):
- **A** → VGPR via `buffer_load_dwordx4` (16 B/thread, 4 per group), **no A LDS,
  no barrier**, `s_waitcnt vmcnt(4)` software pipeline.
- **B** → LDS + `ds_read_b128`, weight loads carry `slc`, `s_setprio 1` wraps the
  MMA, `s_setprio 0` after, `s_barrier` only at tile boundary.

tilelang MLS path (current best, `fp8_channelwise_mls.py` / `_gemm_mls_mls`):
- A and B both global→LDS via `matrix_load` (async, hidden), then
  `ds_read_format_tensor_a/b` LDS→VGPR + `__builtin_hcu_mmac_f32_16x16x32_fp8_*`.
- Costs: A round-trips through LDS and needs a barrier per K tile. At small M the
  A tile is tiny so the extra latency/barrier is amortized badly.

**r_mls (A→fragment VGPR) path is a proven failure** — see
`perf/gemm/fp8_channelwise_decode_bench_r_mls.py` (committed). A per-K iteration
lowers to 16 blocking 8-byte `hcu_buffer_load<fp8_t, 8>` (no slc, no vmcnt
pipeline): M=64 → 8.5-16%, M=128 → ~3%. Why: 8-byte granularity, no
`dwordx4`, no slc, exposed latency. This is exactly the *wrong* hipblas
emulation. The *right* one needs 16-byte VGPR loads (N=16) + slc + non-blocking.

## Authorized optimization levers (user-approved: may edit C++/pass, M<=64 may use pass changes)

**Status 2026-10-16 — levers 1, 2 and 4 are settled:**

1. **slc on MLS matrix_load atoms — DONE, and it was already correct.**
   The 7th (trailing) argument of `__builtin_hcu_matrix_load_*_b8` *is* the slc
   bit; the b8 atoms already pass it as `true`. Proof: flipping
   `", false, true);"` → `", false, false);"` in `tl_mls_atom_gfx938.hpp` and
   recompiling changes the SASS opcode `E44A2000` (slc) → `E4482000`, Δ = 0x20000
   = bit17. An A/B on/off test is **within noise** (246.1 vs 245.9us). No change
   needed; the header was restored (tree clean for that file).
2. **s_setprio around MMA — DONE** (committed, `tl.hcu_mmac_setprio` annotation,
   `gemm_hcu_mmac.py:786`).
3. **A direct-to-VGPR with real dwordx4 — still the only real lever, still open.**
   It is the one thing that would remove A's 29us of LDS write traffic (see the
   2026-10-16 section). `hcu_buffer_load_impl` at
   `src/tl_templates/hcu/core/arch/amd_buffer_addressing.hpp:1352` already has a
   `coherence` (glc/slc) template param and N=16 gives i32x4 = dwordx4; the r_mls
   path currently emits N=8. Blocked by the mmac A-fragment layout: the A operand
   is `int32x2` = 8 consecutive k per lane per (ki, i), so no lane owns 16
   consecutive k bytes. Getting dwordx4 needs a wider per-lane A fragment layout
   (and a matching operand permute), not just a bigger N.
4. **pipeline depth does NOT help — CONFIRMED** (ns3/4 no better than ns2 at
   M=64/128; ns4 marginally helps M=32). bk32 is illegal (MLS trans-B b8 tile
   needs block_K % 64 == 0). Consistent with the LDS-bound diagnosis: a
   bandwidth wall does not respond to more prefetch depth.
5. **Wide bn does NOT help either** — see the 2026-10-16 section. 10 configs
   measured, all worse.

## Environment / build (for the other machine)

- Repo: `/workspace/tilelang-hygon`, branch `feat/hcu-mqa-fp8`.
- Build: see memory `build-tilelang-hcu.md` — tvm submodule needs the mirror
  workaround (`git submodule update` fails on the official URL; use the local
  mirror / SSH swap first). Rebuild after any C++/hpp change:
  `pip install -e .` or the repo's build script (check `README`/memory).
- **GPU pinning is mandatory**: `CUDA_VISIBLE_DEVICES=4` or
  `torch.cuda.set_device(4)`; device 0 is busy.
- `tilelang.compile(gemm, out_idx=[4])` — use the **default target**
  (auto_detect from torch gcnArchName adds mcpu). Do NOT pass
  `target="hcu"`: `Check failed: (TargetHasMcpu(target)) is false`.
- Disk: `/` is overlay (root/tmp/var) and fills up; `/workspace` is separate.
- Other machines' absolute paths may differ — keep scripts runnable from repo
  root (`sys.path` insert is already in the perf files).

## Key files

- `perf/gemm/fp8_channelwise_mls.py` — production builder `fp8_mls_channelwise`
  (MLS-MLS, block_K % 64 == 0). Reference for any pass/C++ change.
- `perf/gemm/fp8_channelwise_decode_bench.py` — decode shape sweep.
- `perf/gemm/fp8_channelwise_decode_bench_r_mls.py` — committed r_mls-vs-mls
  head-to-head + measured table (don't re-run the r_mls path for perf).
- `src/tl_templates/hcu/mls/tl_mls_atom_gfx938.hpp` — MLS load atoms (slc site).
- `src/tl_templates/hcu/core/arch/arch.hpp:242` — s_setprio intrinsics.
- `src/tl_templates/hcu/core/arch/amd_buffer_addressing.hpp:1352` —
  `hcu_buffer_load_impl` (coherence / dwordx4 knob).
- `src/tl_templates/hcu/copy.h:295` — `hcu_buffer_load<T,N,oob>` front end.
- `tilelang/hcu/op/gemm/gemm_hcu_mmac.py:784-930` — `_gemm_mls_mls` /
  `_gemm_r_mls` / `_gemm_s_mls` lowering (pass-side edit points).
- hipblas assembly: `qwen3-32b-fp8-rowwise/deliverable_fp8_gemm/m64/src/gateup_layoutA_vmcnt4.s`.

## Correctness & verify

- Compare against hipblaslt fp8 `torch._scaled_mm` (bf16 out, then float); rel
  ≤ 5e-3. Do NOT use fp64 reference.
- Quick local ref (no hipblas): `C = A.float() @ B.float().T * sa[:,None] * sb[None,:]`.

## Next steps (recommended order)

Levers 1, 2, 4 and 5 are closed (see the levers section). What is left is
structural, not a config search:

1. **A wide-bn MLS partition that keeps the fast shape.** This is the highest-value
   compiler-side fix. `ComputeMlsWarpPartition` (`src/hcu/op/mls.cc:506`) and
   `AnalyzeResourceAxis` (`src/hcu/transform/hoist_mls_resource.cc:1615`) pick
   (warp_mn, warp_k) and the mn-vs-k resource axis from the tile shape. For
   `B(64,256)` at 4 warps they land on the optimal (2,2)/`axis::mn`; for
   `B(128,256)` the axis flips to `k`, and for `bk=128` the partition becomes
   (4,1). Both raise LDS read amplification from 2× to 3–4×, which costs more than
   the A-write traffic that wide bn saves. Forcing (2,2)/`axis::mn` for
   `B(128,256)` would cut A's LDS writes in half — worth ~15us if the read side
   holds. Measure with `__builtin_hcu_matrix_load_*` unchanged; this is a pure
   heuristic change.
2. **A global→VGPR matrix load.** There is no such builtin today (all
   `__builtin_hcu_matrix_load_*` are global→LDS). If the ISA has one, it removes
   A's LDS round-trip outright and V1 (217.0us) becomes the real kernel.
3. Only after 1 or 2: lock configs into `perf/gemm/fp8_channelwise_decode_bench.py`
   `_decode_cfg` and commit with a writeup. `_decode_cfg` already returns
   `(64,64,256,1,256,1)` for M<=64 — the measured optimum — so no change is
   pending there.

Do **not** re-run the wide-bn or nt-shared-A sweeps; they are settled negative.

## Repo hygiene

- Keep `/tmp` experiments out of the repo unless they become a committed bench.
- Commit message footer: `Co-Authored-By: Claude Code <noreply@anthropic.com>`.
