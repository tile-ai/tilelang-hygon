# HCU fp8 Channelwise Decode GEMM — Handoff

This is the entry point for a new session. Status comes first, then what is open,
then the closed levers with their decisive reasons, then the history index. The
full measured record — every table, ratio, SASS excerpt and correction — is in
`docs/optimization/fp8_decode_m_le_128.md`; links below point into it. Branch:
`feat/hcu-smls-b-vgpr`.

## 1. What is shipped right now

`decode_cfg` (`perf/gemm/fp8_channelwise_mls.py`) returns:

- **N=5120 (`o`/`down`) at M ≤ 32:** `bn96 / bk256 / ns2 / t192`, `setprio=1`,
  unrolled k-loop U8. Standing (unroll in parentheses): o 1.05 / 0.99 / 0.96 /
  0.99 / **1.01** at M=1/2/4/8/16, **0.97** at M=32; down 1.08 / 1.04 / **0.91**
  / 1.01 / 1.04 / **0.87**.
- **N=5120 at M=64:** `bm64 / bn64 / bk128 / ns2 / t256`, `setprio=1`, U8 —
  **0.925× down / 0.951× o** (was 1.164/1.099). The neighbourhood is closed, and
  the unroll did *not* reopen it (bn96 54.6 µs, bn128 60.2, bm96 58.6 against
  bn64's 49.6): a bm96 tile wastes 50% of the MMA rows at M=64. Docs
  [2026-10-20d](docs/optimization/fp8_decode_m_le_128.md#2026-10-20d).
- **N=5120 at M in (64,128]:** M ≤ 96 → `bm96 / bn96 / bk128 / ns2 / t384`
  (54 blocks, **0.88× down / 0.94× o**); above → **`bm128 / bn96 / bk128 / ns2
  / t512`** (`setprio=1`, 54 blocks, 1.00-1.06×). Both run the unrolled k-loop
  (`k_unroll` `"auto"` → U6 / U8), which is what produced those numbers: the
  pipelined loop was 1.12-1.23× at the same tiles. Docs
  [2026-10-20e](docs/optimization/fp8_decode_m_le_128.md#2026-10-20e),
  [2026-10-21](docs/optimization/fp8_decode_m_le_128.md#2026-10-21-d10x).
- **`qkv` (N=10240) / `gate_up` (N=51200):** `bn32`, `setprio=1`; the `qkv` M=32
  point is `bn32 / bk128 / t128`.

Standing (paired same-process, idle card, `rel=0`; ratio = ours/hipblas, **< 1 is
a win**):

| shape | M=1 | M=16 | M=32 |
|---|---|---|---|
| o | 1.069 | 1.027 | 1.024 |
| down | 1.110 | 1.042 | **0.928** |

These are the post-2026-10-20b (`t192`) numbers, re-measured after the 2026-10-20c
producer A de-replication attempt was reverted. All four decode shapes are exact.

How it got here: the five adopted levers and the per-stage ratio movement are in
the doc's [What actually won](docs/optimization/fp8_decode_m_le_128.md#what-won).
In short — `down` M=1 went 1.273 → 1.110 and `down` M=32 1.061 → 0.928 between the
2026-10-17f shape-aware chooser and now; `o` M=1 went 1.188 → 1.069. Everything
else in the search space is a measured closure (§3).

## 2. What is open

- **Nothing in the `decode_cfg` search space.** Every partition axis (warp_m/n/k,
  policy), tiling axis (bn, bk, threads, residency), pipelining axis (ns,
  setprio) and producer axis (MLS, plain shared, rolling window, B-direct-VGPR)
  has a measured closure. Anything further needs a new *mechanism* — a producer
  that is not MLS, or an ISA path that avoids the LDS round trip.
- **The one open measurement in o/down:** o's load-only arm runs at 1387 GB/s
  where down's runs at 1679 GB/s (ceiling 1725 GB/s); no mechanism is established
  for that 21% and no lever was found. (docs
  [2026-10-18c](docs/optimization/fp8_decode_m_le_128.md#2026-10-18c))
- **M in (64,128], the standing gap:** 0.88-0.94× at M ≤ 96 (we win), and
  1.00-1.06× at M=112/128. What is left is the exposed part of the per-step
  `cp_async_wait`: at o M=96 the floor is 49 µs and we run 59 µs, at 54 blocks of
  64 CUs. The lever that would act on it is *more blocks*, and both directions
  are measured closed (bn128 → 40 blocks 0.955× worse; bn64 → 80 blocks 1.53×
  worse), as is deeper cover (A2B3 174.6 vs 155.2), more warps (t1024 88.5),
  and removing a barrier (the shipped loop already has the same two). Docs
  [2026-10-21](docs/optimization/fp8_decode_m_le_128.md#2026-10-21-d10x).
- **M ≤ 64:** now 0.87-0.99× at M ≥ 2 except down M=8 (1.008) and down M=16
  (1.038). The only real gap left is **M=1 (1.05× o / 1.08× down)**, which is the
  launch-fixed ~5-6 µs of [2026-10-20j](docs/optimization/fp8_decode_m_le_128.md#2026-10-20j),
  untouched by the unroll. M=64 is 0.925×/0.951× and its tile neighbourhood is
  still closed (docs [2026-10-20d](docs/optimization/fp8_decode_m_le_128.md#2026-10-20d)).
- **`qkv` M≥32:** 1.05-1.15×; `bm32/bn32/t256` is the measured best of its
  neighbourhood. (docs [2026-10-17f](docs/optimization/fp8_decode_m_le_128.md#2026-10-17f))
- **Not taken:** rotating the MLS producer above the `promote_prio`/mmac block is
  worth ~3% (reproduced 3×) but needs an emitter/pass change, not a config flag.
  (docs [2026-10-19 §6](docs/optimization/fp8_decode_m_le_128.md#2026-10-19-lds))
- **Known correctness defects, unreachable from `decode_cfg`:** the small-tile
  packed stage stride (`bn16/bk128/ns2/t64` emits 1088 B for a 2048 B tile) and
  the MLS K-slice offset (any nonzero base is silently wrong). Both are
  documented, not fixed. (docs [2026-10-18](docs/optimization/fp8_decode_m_le_128.md#2026-10-18),
  [2026-10-18d](docs/optimization/fp8_decode_m_le_128.md#2026-10-18d))

## 3. What is closed, and why

The full one-row-per-lever index is in the doc's
[closed-levers index](docs/optimization/fp8_decode_m_le_128.md#closed-levers).
The load-bearing closures:

| lever | decisive reason |
|---|---|
| split-K | 10-21% slower, and the MLS K-slice offset is silently wrong for any nonzero base, so it is not expressible. |
| `num_stages > 1` | 29-45% slower at bn16; ~1% at best where exact (`bk512/ns2`). |
| B-VGPR-direct / rolling-window producer | 2.1× slower even at `WarpM=1`; 16-consec-k is mmac-incompatible and arithmetically insufficient (93.9 vs 267.6 µs). |
| MMA density / hipblas warp mapping | The MMA itself is 0.5% of time; adding warps to a wide tile through MLS is worse, and the loss is the transposing LDS read (9.3%). |
| MLS producer atom / partition swap | The faster 64×32 atom makes the full GEMM 15-16% slower. |
| N-split / widening bn | 2-3× worse; A's LDS read scales with warp count. |
| Layout A | The 71.6% target requires the excluded weight reorder; hipblas's no-reorder baseline is 68.4%. |
| Producer A de-replication | Bit-exact but 1.044-1.358× slower: gating A out of `cp_async_wait<1>` drops a prefetch level. |
| M=1 GEMV structure | Not expressible: 16×16 MMA atom and `tile_mn=16` MLS floor. |
| `warp_k` / `FullColK` | 0.97-1.03 vs shipped, mixed sign; a substitute for `setprio`, not a complement. |
| `__builtin_hcu_matrix_load_b8` | Exists, but its per-lane layout is not the mmac operand layout; needs a cross-lane permutation. |
| A-direct-to-VGPR / `r_mls` | 1.9-2.6× slower; A must round-trip through LDS. |
| Block-count inflation | 320 blocks is the best point; the ~8× blocks that reach the ~0.98× o M=1 bound cannot be supplied. |
| Independent accumulator chains | 1.52-1.64× slower; a second accumulator on one tile is neutral. |
| Hand-written double buffering | Correct but 1.26-1.70× slower. |
| Tile widening | Catastrophic; `bn32` alone is 7-12% worse at every (shape, M). |
| `k_pack=2`, `b_ds_read_alt=0`, `respect_layout_map` | Hard-rejected / unsupported / exact no-ops. |
| Prefetch depth > `ns=2` at `(bm+bn)=224, bk128` | 65536 B of LDS caps it; `bk64/ns3` 1.509× and `bk64/ns4` 1.427× because a (16,64) fp8-trans tile pays a 4096 B slot for 1024 B of payload (4× waste). |
| Split pipeline depth (B deep, A shallow) | Inexpressible: `inject_pipeline.cc:2584` rejects a consumer reading buffers produced at two different async stages, so both copies share one buffer count (`ns=3` ⇒ 86016 B). |
| `bm < M` to buy stages | Re-reads the whole weight stream per M-block: `bm64/bn96/bk128/ns3` 1.491×, `bm32/...ns3` 2.582×, `bm16/...ns4` does not build. |
| A's DRAM/L2 footprint (D1) | Forcing A to a 2-tile index is 1.019×: the footprint costs nothing (request instructions/bytes unchanged), so "A's traffic" is not the lever. |
| B's DRAM latency (D2) | Forcing B's tile index to `(k&1)` (same instructions/bytes, L2-served) is 1.007×: the 62.6 µs of exposure is **not** a memory stall, so "more cover" was the wrong target. |
| Pipeline issue order | Moving the MMA ahead of the copies is 1.34-1.49× worse; B-before-A is 1.002 (null). With one step of cover, issue order is the only thing keeping the stream alive. |
| `setprio > 1` | sp0 1.036, sp2 1.000, sp3 1.001 — above the top level the ISA honours it is an exact no-op. |
| `bk256/ns2` (halve the step count) | Launch-infeasible: 114688 B > 65536 B/CU (`hipModuleLaunchKernel` fails), and `(128+bn)·512 ≤ 65536` has no legal `bn`. The two fitting alternatives already lose (bn32 1.960×; bm32 4× B re-read). The residual is a per-step fixed cost (~0.48 µs), so fewer steps is the *only* lever and LDS blocks it. |
| Second per-step barrier (single-barrier body) | Removing the post-`cp_async_wait` barrier is worth **2.4 µs** at bn128/t256, not the ~44 µs 2026-10-20g implied: the two barriers are adjacent, so the second is already aligned away. The 32.0 µs belongs to the WAR sync at the top of the step, which needs more buffers, not a body rewrite. |
| `bn` between 96 and 128 at M>96 (the frontier) | Measured *pre-unroll* at all five legal multiples of 16: bn80 1.018× (64 blocks, 341 MB, 1496 GB/s), bn96 1.026×, bn112 1.219× (also a 5-warp-equivalent partition), bn128 1.000× — flat-topped, no interior optimum. `bn80/t256` and `bn112/t256` do not build (no valid trans MLS atom at 4 warps). **Partly falsified 2026-10-21**: unrolled, bn96/t512 beats bn128/t512 by 15% (down 182.2 vs 214.7), so the frontier has an interior optimum at 54 blocks once the modulo is out of the address path. bn80 is still not buildable at bm128. |
| Two blocks/CU at M ≥ 96 | Arithmetic, not a measurement gap: 2 blk/CU needs `(bm+bn)·bk·2 ≤ 32768` ⇒ `bm+bn ≤ 128`; with `bm = M ≥ 96` that leaves `bn ≤ 32`, whose A traffic is 1.05 GB. So M ≥ 96 is pinned at 1 blk/CU — and with `bm = 128` the ≥3 LDS stages that would buy prefetch slack need `bn ≤ 42` (1.3 GB of A). Deep slack and the byte minimum are mutually exclusive at every tile. |
| `bn < 128` at M>96, N=5120 | **Reversed 2026-10-21.** The byte law (`total = N·K·(M/bn + 1)`: bn96 308 MB vs bn128 262 MB) is arithmetically right and was the wrong objective: per step, the bound is `max(bytes/1725 GB/s, exposed wait)`, and 54 blocks of 64 CUs hide the wait 87% against 40 blocks' 68%. bn96/u8/t512 is 0.849× of bn128 on down and 1.022× hipblas. Only *narrower than 96* is closed (bn64 → 104 µs). |
| Deeper cover / second-barrier removal (re-measured with the unroll) | A2B3 (B two steps ahead, unrolled so no modulo) is 174.56 vs 155.20 µs; A3B2 156.00; a `%2`-addressed A2B3 that isolates the LDS footprint 155.68. The `cp_async` counter is FIFO, so a wait that covers B(k+1) also forces A(k+2) — B can never lead by more than one step. Removing the post-wait barrier is worth 2.4 µs, and the shipped `T.Pipelined` loop already emits exactly the two barriers of a hand-written double buffer, so there is nothing to remove at the pass layer. |

## 4. History index

One row per dated record, with its decisive measurement. The full prose for each
is in `docs/optimization/fp8_decode_m_le_128.md` at the linked anchor; this table
is the map, not the record. Rows are historical: numbers are as recorded at the
time, and several early readings were later corrected (the native operand roles,
the read ceiling, the `ns>1` mechanism) — the doc says where.

| date | what it decided | doc |
|---|---|---|
| 2026-10-15 | M≤16 was already a win (~78% of nominal) and a single full-M block + `setprio` took M=64/128 from −89%/−225% to −7.2%/+0.3% (the exact numbers are handoff-only, below). | [link](docs/optimization/fp8_decode_m_le_128.md#m-le-16-h2h) |
| undated (early) | Pre-MLS standing on effective BW: behind at M=32 (53.8% vs 66.8%) and M=64 (49.7% vs 62.2%), ahead at M=128 (47.9% vs 46.0%). | [link](docs/optimization/fp8_decode_m_le_128.md#m-ge-32) |
| undated (early) | `bm16/bn32/bk256/ns1/t256` is the M≤16 winner; growing `block_M` to 32/64/128 is where the kernel loses. | [link](docs/optimization/fp8_decode_m_le_128.md#m-le-16) |
| undated (early) | First generated-code reading of the M=32..128 gap (hipblas A→VGPR, B→LDS) — later refuted: the native kernel is the reverse. | [link](docs/optimization/fp8_decode_m_le_128.md#layout-a) |
| 2026-10-16 | M=64's root cause is A's LDS write traffic, not DRAM: A staged once is 217.0 vs 246.9 µs, and pinning B saves only 8 µs for 262 MB. | [link](docs/optimization/fp8_decode_m_le_128.md#m64-root-cause) |
| 2026-10-16b | The 71.6% "70% target" needs the excluded Layout A reorder; hipblas's no-reorder baseline is 68.4% (213.0 µs), so the in-scope gap is 34 µs, and grouping A's loads (249.3 µs) does not close it. | [link](docs/optimization/fp8_decode_m_le_128.md#layout-a) |
| 2026-10-16c | M=64 is LDS-port-bound, not bandwidth-bound: LDS bytes/output is identical to hipblas (0.15625) and the paired gap is 5.8% contended / 7.3% idle. | [link](docs/optimization/fp8_decode_m_le_128.md#m64-paired) |
| 2026-10-17b | B-direct-VGPR dies on two independent costs: 4× m-warp redundancy (loader-only 3.62×) and the 8-consec-k half-sector penalty (~2×); the full GEMM is 975.4 vs 243.4 µs. | [link](docs/optimization/fp8_decode_m_le_128.md#b-direct) |
| 2026-10-17c | The native SASS reverses the operand roles (B→VGPR with no M coordinate, A→LDS), yet B-direct at `WarpM=1` is still 2.1× slower (513.8 vs 246.2 µs). | [link](docs/optimization/fp8_decode_m_le_128.md#b-direct) |
| 2026-10-17d | Both B-fetch levers close: the 16-consec-k swap is worth at most 93.9 µs against a 267.6 µs gap, and the faster 64×32 MLS atom makes the full GEMM 15-16% slower (287.2 vs 246.2 µs). | [link](docs/optimization/fp8_decode_m_le_128.md#b-direct) |
| 2026-10-17e | The "fine-grained `vmcnt(4)`" premise was a patched-file artifact; the real knob is `num_stages`, worth 1.2% (`bk128/ns2` 243.5 vs 246.6 µs), and `bk=192` is silently wrong (rel 1.61). | [link](docs/optimization/fp8_decode_m_le_128.md#async-wait) |
| 2026-10-17f | A shape-aware chooser is a real win (o M=1 42.9→38.6 µs); split-K is 10-21% slower and its block-starvation premise was a confound; the M=32 thread fix is 1.60-1.71×; `ns>1` is silently wrong outside bm64/bn64. | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-17f) |
| 2026-10-17g | `qkv` M=32 wants `bn32/bk128/t128` (9-10%, 1.14× → 1.03×), the earlier ratios had used a `setprio=0` baseline, and `ns=2` at bn16 is 40/40 wrong or slower. | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-17g) |
| 2026-10-17h | The load path already beats hipblas on all four o/down cells (0.917-0.977), so the deficit is the mmac side and scales with K; the packed-stage stride bug is root-caused and fixed. | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-17h) |
| 2026-10-17i | Re-timed `ns=2` is 29-45% slower (the "fast" candidates had been measuring garbage), independent chains are 1.52-1.64× slower, and the deficit is the 16-20% where neither pipe is busy — 0.98× needs ~8× the blocks. | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-17i) |
| 2026-10-18 | N-split is closed hard (2-3× worse) and `ns>1` has a second, independent stage-index defect (bk128/ns2 rel ≈ 1119-3504, exact in the four configs the old tests happened to cover). | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-18) |
| 2026-10-18b | The "2016 GB/s ceiling" was ~17% too high (1725 GB/s measured) and accumulator splitting is neutral (1.012-1.014×) — the exposure is not the mmac chain. | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-18b) |
| 2026-10-18c | The exposure is volumetric and per-iteration (half gemm +4.0 µs, double +13.6 µs at o M=16) and A-direct-to-VGPR is closed again (`r_mls` 1.935×/2.592× slower). | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-18c) |
| 2026-10-18d | Split-K is not merely slow but inexpressible — the MLS K-slice offset is silently wrong for any nonzero base — and double buffering is worth ~1%. | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-18d) |
| 2026-10-19 | `warp_k`/`FullColK` is built and correct but 0.97-1.03× with mixed sign: a substitute for `setprio`, not a complement; the MLS `moffset` is a signed 10-bit immediate, so `block_K ≤ 512`. | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-19-warpk) |
| 2026-10-19 | `__builtin_hcu_matrix_load_b8` exists, but its per-lane layout is 4 consec-k × 4 n, not the mmac's 8 consec-k at one mn; bridging needs a full cross-lane permutation. | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-19-matrixload) |
| 2026-10-19 | hipblas launches 54 blocks × 256 threads (N macro-tile = N/54), so A is read 54× not 320×; reproducing that structure through MLS is still slower. | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-19-hipblas-launch) |
| 2026-10-19 | The gap is an LDS-capacity invariant, `total = const × w × (L + C·ns)`: 5→4 blocks/CU costs 80% (93.7→168.9 µs) and `bk`/`ns` cancel; the producer rotation worth ~3% needs an emitter change. | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-19-lds) |
| 2026-10-20 | Dense MLS packing (6144→4096 B) lets `bn96/bk256/ns2` land, a stable 5-7% win, and the disk kernel cache is blind to codegen changes. | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-20) |
| 2026-10-20b | The wide tile never cut A's re-read (A's address has no warp-id term, so blocks × warps is invariant); warp count is the lever, U-shaped, and `t192` is 0.952-0.962× of t384. | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-20b) |
| 2026-10-20c | Producer A de-replication is bit-exact but 1.044-1.358× slower: gating A out of `cp_async_wait<1>` drops a prefetch level (t192 `vmcnt(8)→(4)`). | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-20c) |
| 2026-10-20d | The M=64 tile neighbourhood is closed (ship 1.178× down / 1.083× o; every `bn`/depth/width arm loses) and wave quantization is refuted — the exactly-64-block `bn80` is 1.177× slower. | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-20d) |
| 2026-10-20e | M in (64,128] was losing 1.8-2.3×; `bn96` (54 blocks, hipblas's own macro-tile) + `ns2` + a `bm96` arm for M=96 takes it to 1.30× down / 1.26× o (M=96: 1.12/1.14). `ns=1` is the serialized one, so "prefetch distance is zero" is scoped to M≤32. | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-20e) |
| 2026-10-20f | Decomposed the band: stream-only 167.2 µs (0.945× hipblas) + chain-only 109.8 µs (0.620×, 84% of usable MMA peak) = 229.8 µs, so only 43% of the compute overlaps and perfect overlap would win. More cover is unreachable (LDS caps `ns` at 2, split-depth is planner-blocked, `bm<M` re-reads B). | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-20f) |
| 2026-10-21 | **The unrolled k-loop is the per-step lever**: a symbolic stage index `(k+d)%3` costs 13% (`v_readfirstlane` on the `matrix_load` address path), and the shipped loop already had both barriers. U6 takes down M=96 195.8→155.2 µs and o 71.4→58.7 (0.79/0.82× pipelined, 0.88/0.94× hipblas); unrolled, the M>96 tile reverses to bn96/t512/u8 (down 182.2 vs 214.7), i.e. 1.00-1.06×. Deeper cover re-falsified (A2B3 174.6 vs 155.2); LDS footprint still free. Every N=5120 tile takes U8: M=64 down 160.3→127.0 (0.925×, was 1.164) and o 57.0→49.6 (0.951×); M=1 is 2-3% (the residual there is the launch-fixed cost, untouched). M=64's tile does *not* move — bm96 would pad 50% of its MMA rows. | [link](docs/optimization/fp8_decode_m_le_128.md#2026-10-21-d10x) |

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

- Repo: `/workspace/tilelang-hygon`, branch `feat/hcu-smls-b-vgpr`.
- Build: see memory `build-tilelang-hcu.md` — tvm submodule needs the mirror
  workaround (`git submodule update` fails on the official URL; use the local
  mirror / SSH swap first). Rebuild after any C++/hpp change:
  `pip install -e .` or the repo's build script (check `README`/memory).
- **Card selection is mandatory and is not fixed.** The box is shared: any card
  can be taken between runs (cards 0/1 were idle in the morning and at 100% by
  the afternoon of 2026-10-20). Check `hy-smi` for an HCU% ≈ 0 card and pin it
  with `HIP_VISIBLE_DEVICES=<idle>` (`CUDA_VISIBLE_DEVICES` also works); re-check
  before trusting a measurement, and keep a sanity band on the hipblas arm.
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

**The config search space is exhausted as of 2026-10-19.** Every axis has now
been measured and closed, and the last one — `warp_k` / `FullColK` on the MLS
path — was implemented (it had three separate hard guards) and turned out to be
neutral-to-negative. See the doc's 2026-10-19 section. The only movement since is
the 2026-10-20 fat tile (+5-7%, §1), which is a producer/allocator change, not a
config sweep.

Closed, each with a measured bound:

| lever | verdict |
|---|---|
| wide-bn MLS partition (was "lever 1") | closed 2026-10-18: N-split is 2-3x worse; A's LDS read scales with warp count |
| B-direct-VGPR via `buffer_load` (was "lever 2") | closed: 2.5-3.2x slower at every M; 16-consec-k is mmac-incompatible |
| **`__builtin_hcu_matrix_load_b8` (global->VGPR matrix load)** | **the builtin DOES exist** (correcting the 2026-10-17d note); measured 2026-10-19 and still **closed** — its fixed per-lane micro-layout is not the mmac operand layout. See the doc's 2026-10-19 section. |
| rolling LDS window | closed: an 8-warp occupancy technique, 3x slower at our 1-warp structure |
| split-K | closed twice: not expressible (MLS K-offset silently wrong for any nonzero base) and 10-21% slower |
| `ns>1` | closed: ~1% at best where it is exact; wrong in four configs |
| register operands, plain shared, bn widening, thread count | all measured negative |
| **`warp_k` / `FullColK`** | **built 2026-10-19, correct, 0.97-1.03 vs shipped with mixed sign; a *substitute* for `setprio`, not a complement** |

Also newly established: the MLS producer's per-access offset is a **signed 10-bit
element immediate** (`tilelang_mls_base.hpp:328`), so `block_K <= 512` at the
bn16 fp8 tile. hipblas's 2048-element K chunk uses 32-bit `buffer_load_dwordx4`
addressing and is structurally unreachable from the MLS path — that is a cap,
not a tuning choice.

What is left is **not** a config search:

1. Lock the already-shipped configs and commit with the writeup (nothing is
   pending in `decode_cfg`; the shipped rows are the ones in §1 —
   `bn96/bk256/ns2/t192` for N=5120, `setprio=1` throughout).
2. Anything further needs a new *mechanism* — a producer that is not MLS, or an
   ISA path that avoids the LDS round trip — not another sweep.

Do **not** re-run the wide-bn, nt-shared-A, split-K, `ns>1`, rolling-window or
B-direct sweeps; they are settled negative.

## Handoff-only notes

Facts the cut sections had that the record lacks; everything else is at the anchors above.

- **2026-10-15 M=64/128 breakthrough (pre-MLS).** One full-M block + `setprio` took
  M=64/128 from −89%/−225% to −7.2%/+0.3% (`bm64/bn64/bk256/ns1/t256/sp1` 246.4 vs
  229.9 µs; `bm128/bn128/bk128/ns1/t512/sp0` 309.4 vs 311.0); M=16/32 read +11.8%/
  +13.5% (164.6 vs 186.7, 188.3 vs 217.8). M-splitting (`by=2/4`) is worse.
- **Earlier M≤16/32 MLS head-to-head:** 164.3 µs (88.6%) vs 186.6 (78.1%) = +11.9%,
  and 199.8 (72.9%) vs 218.4 (66.7%) = +8.5% — a different run from the doc's M≤16
  table. Pre-MLS standing: M=32 53.8%/66.8%, M=64 49.7%/62.2%, M=128 47.9%/46.0%, and
  8192³ 432 TFLOPS vs ~560. The early `r_mls` arm read "8.5-16%"/"~3%" at M=64/128
  (units unstated).
- **Plan B `b_vgpr_direct`:** 860-910 µs at M=32 (MLS 199.8 / hipblas 218), 1786 µs
  with M padded to 128; SASS kept ~4 8-byte loads in flight (rolling `vmcnt(3)`), so a
  4-mmac group cannot hide VMEM latency.
- **MT256x128x128 native decode:** 255 VGPR, 64 KB LDS, 288 mmac = 4×72/loop; B as
  4 `dwordx4` per 32-wide k-slice into rotating X0/X1 (8 slices = 32 in flight), A as
  16× `ds_read_b128`, a 72-mmac window (~4.5× ours), one `vmcnt(0)` per 128-k.
- **Layout A leftovers:** our-DSL copy of hipblas's `bm128/bn64` at `bk128 ns2` is
  479.0 µs; the M=128 gate_up Layout A row is 310.4 → 288.0 µs (`reorder_weights.py`);
  B-direct WIP diff header +181, emitter +44, macro generator +118, `gemm_op.py` +6/-2.
- **2026-10-20 cache/test leftovers:** `TILELANG_DISABLE_CACHE=1` emitted
  `ds_read_format_tensor_a<..., 1, 4, 1, ...>` where the cached `.cu` had
  `..., 1, 1, ...` (cache `0.1.12_cuda_git0a92729d`); tests: 18 passed, 3 xfailed, 1 xpassed.
- **Harness paths not in the doc:** `/tmp/m64_decomp.py` (+`m64_adiag`, `m64_w2`,
  `m64_bn128`, `m64_ntile2`, `m64_final`), `m64_astage.py`, `h2h_sweep64_128_v3.py`,
  `h2h_final_confirm.py`, `pipe_upfront.s`, `src_64_128_2_128.cu`, `bdirect_bench.cu`
  (+`stride_test.cu`, `asm16ec.cu`), `/tmp/b_vgpr_deadend_full.patch`.

## Repo hygiene

- Keep `/tmp` experiments out of the repo unless they become a committed bench.
- Commit message footer: `Co-Authored-By: Claude Code <noreply@anthropic.com>`.
