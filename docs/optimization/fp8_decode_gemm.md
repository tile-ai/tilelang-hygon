# fp8 Channelwise GEMM on HCU — the decode regime

**What this document is.** The results of the HCU (gfx938) fp8 channelwise GEMM
for LLM decode — `o` (N=5120, K=8192), `down` (N=5120, K=25600), `qkv` (N=10240)
and `gate_up` (N=51200), M from 1 to 256 — measured against hipblasLt
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
`tilelang/hcu/op/gemm/gemm_fp8_channelwise.py`). M ≤ 128 is the decode band
proper; M in (128,256] is the speculative-decode band added 2026-10-23 (see
[M in (128,256]](#m-gt-128)). Within M ≤ 128: N=5120 (`o`/`down`) at
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

A fourth, disjoint band was added the same day: **M ∈ (128,256]**, the speculative
decode band (3 drafts + 1, so high concurrency lands at 128–256 rows). It is
served by `bm160 / bn96 / bk128 / ns2 / t256 / sp1` — the LDS budget `bm+bn ≤ 256`
at bk128/ns2 is what forces `bn96` — or, above N=10240, the same tile at
`ns1 / sp0`. Worth 10–13% over hipblas on five shapes (see
[M in (128,256]](#m-gt-128)). No `band 256`: its best arm only ties hipblas.

**Current standing** (paired same-process, idle card, `rel=0`; ratio =
ours/hipblas, **< 1 is a win**). Shipped `decode_cfg`, both arms in one process
per shape, bf16-fused output, CUDA-graph replay, min of 7 interleaved rounds,
re-measured end-to-end through the drop-in shim on 2026-10-21 (the run that also
shipped the band-96/band-128 bk64 tiles and fixed the `M > 112` tile test):

| shape | M=1 | M=16 | M=32 | M=64 | M=80 | M=96 | M=128 |
|---|---|---|---|---|---|---|---|
| o | 1.061 | **0.932** | **0.888** | **0.910** | **0.887** | **0.894** | 0.997 |
| down | 1.096 | 0.991 | **0.845** | 1.005 | **0.909** | **0.938** | 1.010 |
| gate_up | **0.940** | **0.880** | **0.856** | 1.012 | **0.888** | **0.943** | **0.955** |
| qkv | **0.806** | **0.773** | 0.998 | **0.967** | **0.948** | **0.950** | **0.950** |

**20 of 28 cells win**, and the losses are now concentrated: `M=1`/`M=128` on
o and down, and the `M=64` shape pair (down/gate_up, the LDS-bound point).
Full loss list: o M=1/128, down M=1/16/64/128, gate_up M=64, qkv M=32.

The dispatch gate (`tl_fp8_gemm.py`) holds one ratio per M *band*, not per
endpoint — a decode batch moves inside its band, so the gate prices the band
(a band-level number, not its best endpoint). That is why a band whose endpoint
cell wins here can still route to hipblas: `o` band 16 carries 0.99 in the gate
against 0.932 measured at M=16 (M=1 in the same band is 1.061), and `qkv` band
32 sits at 0.998 in both.

Absolutes drift on this shared box (an idle co-tenant moves single-run ratios by
up to 6%), so only each row's paired ratio is meaningful; cells are from their
own runs — compare µs, not ratios, across rows. Two gate rows were corrected
from this run: `gate_up` band 16 was recorded as 1.02 but measures 0.940/0.880,
`qkv` band 64 as 0.98 but measures 0.967-0.973 over four samples.

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
- [M in (128,256] — the speculative-decode band](#m-gt-128)
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
| 10 | **The `bn16` fallback for N in (5120, 10240).** Only 27B's `in_proj_z` (N=6144, K=5120) lands there, and it was the worst cell in the whole table: 384 blocks each re-reading the entire A tile (`A_issued = 384·32·5120 = 62.9 MB` + 31.5 MB of B ≈ 1970 GB/s, i.e. *above* the 1725 GB/s ceiling). It now takes the same fat tile as N=5120, at **t128** — the grid is exactly 64 blocks, so two warps is 128 warps on 64 CUs; at N=5120's 54 blocks t192 still wins, so this is the measured point, not a rule. `k_unroll` 8 at bm32, 1 at bm16. | M=24 **1.981 → 0.833**, M=32 **1.816 → 0.863**, M=2/4/8/16 0.823/0.825/0.823/0.833 (`rel = 0`); 49.1 → 23.0 µs. **Drop-in caveat (2026-10-22c):** those are *raw-kernel* ratios (`rel = 0`, slab A + slab scale, zero pad nodes). Through the real dispatch, where vLLM hands over a tight A and a tight per-token `scale_a`, the same cells measure **1.18 (M=8) / 1.15 (M=24)** — two 5 µs nodes on a 25 µs call — so the integration falls back to hipblas there. | `decode_cfg` (log §2026-10-22) |
| 11 | **`per_tensor_a`: the per-tensor `scale_a` stops being a pad node.** At M=1 vLLM's `scale_a` is `[1,1]`, and the drop-in path turned it into a band-length buffer — a DtoD copy node — on top of the band pad's own node. On this card *one* DtoD node is **5.0 µs of GPU time at any size** (profiled: a 4-byte `copy_`), i.e. 10.9 µs on a 22 µs kernel. The kernel now declares `scale_a: (1,)` and broadcasts `scale_a[0]` in the epilogue, so the scale node is gone entirely. | z27 M=1 **1.291 → 0.815**, qkv32 M=1 **1.072 → 0.801**, gate_up M=1 **0.995 → 0.933**; the drop-in path now costs *exactly* the raw kernel (`new_slab − raw = −0.01 µs`). With a tight A (storage = M rows) the pad node is still there: 1.073 / 0.938 / 0.957 | `fp8_channelwise_gemm(per_tensor_a=…)` + `GATE_NODE_FREE` (log §2026-10-22b) |

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
| o | **0.941** | **0.939** | **0.937** | 0.995 → **1.044** | 1.047 → **1.012** |
| down | **0.882** | **0.891** | **0.887** | 1.056 → **1.038** | 1.002 → **1.009** |

(The M=112/128 cells were re-measured paired on a busier card — see
[M ∈ (96,128] is closed](#2026-10-22-m96-128-n5120). The bold arrows are the
fresh numbers; the band loses by 1–4%, not 0.5%.)

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

### Cumulative standing with Layout A stacked on the shipped path (2026-10-21)

Shipped path (unrolled k-loop + band tiles) with Layout A (blocked weights) on
top, all four shapes, **A measured at the tile that actually ships for that
(M, shape)** — the previous version of this table measured Layout A at a
different tile for the M ≥ 80 rows, and those numbers are retracted: the shipped
down/o tiles above M=64 are `bn96` and **blocked-B cannot exist there at all**
(`N % bn ≠ 0`, see the `A–` tag).

Ratio ours/hipblas, lower is better; `+x%` = we are x% faster. The first number
in a cell is the shipped path alone, the second (`→`) is the same cell with
blocked B. Tags apply to the Layout A arm:

| tag | meaning |
|---|---|
| `A✓` | ≥1% faster, applied |
| `A~` | neutral (±1%), not applied |
| `A✗` | slower, row-major kept (ratio in parens) |
| `A–` | structurally excluded: `N % bn ≠ 0`, blocked-B impossible at this tile |

| M | down | gate_up | o | qkv | µs-weighted |
|---|---|---|---|---|---|
| 1 | 1.096 `A–` | **0.940** `A✗` | 1.061 `A–` | **0.806** `A–` | 0.971 (+2.9%) |
| 16 | 0.991 `A–` | **0.880** `A✗` | **0.932** `A–` | **0.773** `A✗` | 0.899 (+10.1%) |
| 32 | **0.845** `A–` | **0.856** `A✗` | **0.888** `A–` | 0.998 `A✓`→**0.968** | 0.871 (+12.9%) |
| 64 | 1.005 `A✓`→**0.934** | 1.012 `A✓`→**0.869** | **0.910** `A✗`(1.042) | 0.967 `A✓`→**0.924** | 0.899 (+10.1%) |
| 80 | **0.909** `A–` | **0.888** `A~`(1.004) | **0.887** `A–` | **0.948** `A–` | 0.900 (+10.0%) |
| 96 | **0.938** `A–` | **0.943** `A~`(0.974) | **0.894** `A–` | **0.950** `A–` | 0.938 (+6.2%) |
| 128 | 1.010 `A–` | **0.955** `A~`(0.999) | 0.997 `A–` | **0.950** `A✗`(1.631) | 0.974 (+2.6%) |

**µs-weighted** = `Σ(hipblas µs × cell ratio) / Σ(hipblas µs)` over the four
shapes at that M, using the same-process hipblas absolutes (down/gate_up/o/qkv =
73.2/165.8/26.4/39.3 µs at M=1 … 174.3/306.1/60.9/70.6 µs at M=128); it is the
number that predicts end-to-end decode gain, since `gate_up` is 3-5× the µs of
`o`. Values after stacking are 0.871 at M=32 (qkv band) and 0.899 at M=64
(down+gate_up+qkv); at every other M no cell qualifies, so stacked == shipped.

**Verdict on where Layout A is worth carrying.** Only the `bm64/bn64/bk128` tile,
and there only for down (0.929), gate_up (0.859) and qkv (0.956) — `o` is 4%
*slower* at the same tile, so it needs a per-shape gate even inside the band.
Every `bn96` tile (down/o at M≥80, qkv at M=80/96) is **structurally excluded**
because 5120 and 10240 are not multiples of 96 — there is no room to widen `bn`
either, `bn96` is the measured optimum at those bands. `qkv` at M=128
(`bn128/bk64`) is the one place blocked B is not just useless but catastrophic
(1.63×) : at `bk64` the block reads 128×64 B per step and the blocked stride
defeats the MLS gather. Landing the lever therefore costs a second blocked copy
of B (~487 MB over the four shapes) for a ~1pp end-to-end gain at two M values;
it stays a pre-reorder-time decision, not a default.

**Status 2026-10-23: dropped, not just out of scope.** The user retired Layout A —
the deployment cost (an offline reorder + disk cache of a second weight copy, plus
a `data_ptr` registry to route it) is not worth a ~1pp gain at two M values, and
the band is narrow (M ∈ (32,64]) with `o` structurally excluded. The three research
copies (`gemm_fp8_channelwise_chongpai.py`, `_aonly.py`, `_chongpai_ab.py`) were
removed from the tilelang tree (archived outside it); the shipped row-major
builder is the only path.

### 2026-10-22e — the lever becomes shippable: bf16 ABI + runtime dispatch

Layout A had two gaps between "measurable" and "deployable", and neither of them
was speed.

**1. The blocked builder only emitted fp32.** `fp8_channelwise_gemm_blocked` had
no `out_dtype`, while the shipped row-major path and hipblas both honour it (bf16
under vLLM), so a blocked layer would have needed a separate cast node. It now
takes `out_dtype` — only the `C` tensor changes, the accumulator and the scale
epilogue stay fp32 — and the change is free: at M=64 the bf16 blocked arm is
**bit-identical** to the row-major bf16 arm, and to its own fp32 result
re-rounded (`blk_b == blk_f.to(bf16)`), with `blk_b/blk_f` = 1.001 / 0.995 /
0.999 / 0.977 over the four shapes (copy-out stores half the bytes).

Re-measured at M=64 in bf16, all four shapes (paired same-process, graph replay,
7 rounds, min-of-rounds, every arm bit-exact), ratio ours/hipblas:

| shape (N,K) | shipped | blocked | change |
|---|---|---|---|
| gate_up (51200,5120) | 1.043 | **0.894** | hipblas's 4% win → **our 11% win** |
| down (5120,25600) | 1.070 | **0.908** | +15pp (0.85–0.95 under load, below) |
| qkv (10240,5120) | 0.962 | **0.932** | +3pp |
| o (5120,8192) | **0.904** | 0.943 | correctly refused (blocked 4% slower) |

This reproduces the 2026-10-21 row within the drift band (0.924 / 0.869 / 0.934
there) and pins the accounting: `gate_up` is the cell that pays for the whole
lever. `down` is the least stable number in the table — across two segments
blocked/shipped read 0.949 then 0.849 while the *shipped* arm moved +13% and the
blocked arm +1%, i.e. the shared card penalises the arm with more blocks (each
block re-reads A), so the honest quote is 0.85–0.95, not a point.

**2. Nothing connected the reorder to the kernel.** `pre_reorder` could build a
blocked weight and the builder could consume one, but the runtime had no way to
know a tensor was blocked. `tl_fp8_gemm` now carries a `data_ptr`-keyed registry
(`register_blocked` / `blocked_shapes` / `clear_blocked`), and both entry points
(`scaled_mm`, `hcu_backend`) route a hit to `tl_scaled_mm_blocked` ahead of the
gate. Keying on the pointer means the loader may keep handing over
`W.view(-1, K).t()`: the 2-D spelling of the same storage hits the registry.

**The contract is deliberately hard.** A blocked weight read as row-major returns
*wrong numbers*, not slow ones, so a hit with no blocked tile for `(N,K,M)`
**raises** instead of falling back to hipblas, naming the mismatch (`blocked
weight for 10240x5120 is bn64/bk128 but the M=96 tile is bn96/bk64`). Verified on
qkv32 against the real drop-in path: M=64 bit-exact through both entry points,
M=96/16/128 all raise, and after `clear_blocked()` the unregistered path is
bit-identical to before. An 8-cell gate regression through
`torch.ops.vllm.hcu_channel_fp8_target_triton_scaled_mm` shows identical routing
and identical ratios before and after the branch — the dispatch costs the
unregistered path nothing.

**It survives graph capture, which is where it actually runs.** The serving path
is not `scaled_mm` called from Python: it is
`torch.ops.vllm.hcu_channel_fp8_target_triton_scaled_mm` inside a captured
graph, where the registry lookup, the tile choice and the output allocation all
happen once at capture time. Capturing that op with a blocked weight registered
passes every guard: 5 eager warm calls + the capture = 6 `tl-blocked` hits, the
replays add none (a replay is pure GPU — the graph re-executes the kernel, it
does not re-dispatch), the replayed output is bit-identical to the row-major arm,
and a second replay reproduces it exactly (the `out_idx` output is allocated per
call, so nothing accumulates — which is precisely why the split-K zeroed-C
convention must not be imported here). Two negative controls carry the rest of
the claim: *before* registration the same blocked storage goes through the
row-major kernel and disagrees with the reference (routing is by `data_ptr`, not
by shape, so a same-`(N,K)` row-major weight can never be mistaken for a blocked
one), and an M=96 call with the weight registered raises *inside* capture instead
of being baked into a graph that would read permuted bytes with the wrong tile.
`clear_blocked()` restores both directions bit-for-bit.

The spelling a loader must produce: hand over B as `[K, N]` with stride
`(1, K)` over the blocked bytes (`W_blocked.view(N, K).t()`), which is what
vLLM already does with `weight.t()`. `tl_scaled_mm_blocked` then does `W = B.t()`
to recover the contiguous `[N, K]` view, and the `data_ptr` the op sees is the
registered one.

The deployment window is narrower than "M ≤ 64" and is now measured: a blocked
checkpoint serves **only M ∈ (32, 64]**. `decode_cfg` returns the bn64/bk128/ns2
tile in exactly that band. At M ≤ 32 the N=5120 shapes take
`bm32/bn96/bk256/ns2/t192` (or `bm16/...` below 32), and a blocked weight is a
*permutation of the storage*, so those tiles cannot read it —
`tl_scaled_mm_blocked` raises instead of going silently wrong.

The obvious escape is to force one bn64/bk128 tile across all of M ≤ 32 so that a
single blocked weight serves the whole short-batch range. That was measured and
it is a 1.3-1.8x regression, not a trade (`/tmp/deploy_band_sweep.py`, paired,
graph replay, 7 rounds, min-of-rounds, `rel` 2.2-2.5e-3 = the bf16 floor; ratio =
ours/hipblas, < 1 wins):

| shape, M | shipped | bn64/bk128 row-major (t256 / t128) | bn64/bk128 blocked (t256 / t128) |
|---|---|---|---|
| o M=1 | **1.037** | 1.476 / 1.582 | 1.649 / 1.712 |
| o M=8 | **0.917** | 1.288 / 1.386 | 1.442 / 1.486 |
| o M=16 | **0.930** | 1.306 / 1.399 | 1.456 / 1.501 |
| o M=32 | **0.902** | 1.386 / 1.268 | 1.511 / 1.347 |
| down M=1 | **1.071** | 1.632 / 1.766 | 1.752 / 1.778 |
| down M=8 | **0.974** | 1.493 / 1.615 | 1.589 / 1.613 |
| down M=16 | **0.993** | 1.530 / 1.600 | 1.595 / 1.617 |
| down M=32 | **0.848** | 1.401 / 1.289 | 1.479 / 1.340 |

Two things follow. First, the wide bn96/bk256 tile is not a marginal preference at
these M — it is the difference between winning and losing by 50%+, so it cannot be
given up for a layout. Second, in **every** cell the blocked form is slower than
the identical tile row-major (1.5-10%) — the *opposite* sign from M=64, where
blocked wins by 4-10%. The reason is the mechanism behind the M=64 win itself:
blocked B only pays when `M == bn` makes `A_dup ≡ B`, so the permuted store turns
A's re-read into B's; at M ≤ 32 the tile is not square against M, so the reorder
buys no reuse and costs address arithmetic.

So the lever remains a `--max-num-seqs 64` decision, and even then a blocked
checkpoint would raise on every batch with M ≤ 32. A dual row-major + blocked
deployment (so prefill and short batches keep working) costs +26.6 GiB on 32B —
91% of the weights — which does not fit beside the KV cache on 96 GB. **Status:
Layout-A is an M=64-only, fixed-M-window lever, not a checkpoint format.**

<a id="2026-10-22g"></a>
### 2026-10-22g — "fill the CUs with a narrower N tile" is falsified at M ≤ 32 too

The last entry in the open list was o/down at `bn80`, on the theory that 5120/80 =
**64 blocks = exactly one block per CU** beats the shipped 54-block `bn96` grid.
It was recorded as needing "MMA warp policy 4x1x2", which is the wrong frame — the
constraint is `m_warp * n_warp * k_warp == num_warps` with `bn / n_warp >= 16`
(`kMinNPerWarp`). At `bm16` that is **impossible**: `bn80` means 5 warps in N, and
5 is prime, so no product of warp counts reaches it. The builder rejects
`bm16/bn80/t160` outright (`m_warp * n_warp * k_warp must equal num_warps`), and
`t160` is not among the sizes MLS will even consider. The only arithmetically
legal spelling is `m_warp=2 × n_warp=5`, i.e. **`bm32/bn80/bk256/ns2/t320`**,
which pads M=16 to 32 rows.

Measured, o (5120×8192) and down (5120×25600), paired same-process vs hipblas,
CUDA-graph replay, 7 rounds, min-of-rounds, `rel` ≤ 2.9e-3 (bf16 floor):

| arm | blocks | o M=16 | o M=32 | down M=16 | down M=32 |
|---|---|---|---|---|---|
| `bm16/bn96/bk256/ns2/t192` (shipped) | 53 | **27.98 µs · 0.931** | **30.90 · 0.901** | **80.39 · 0.994** | **87.10 · 0.844** |
| `bm32/bn80/bk256/ns2/t320` (64 blocks) | 64 | 33.13 · 1.105 | 39.22 · 1.142 | 94.09 · 1.164 | 112.44 · 1.090 |
| `bm32/bn64/bk256/ns2/t256` | 80 | 47.62 · 1.581 | 48.23 · 1.399 | 136.26 · 1.691 | 136.65 · 1.325 |
| `bm16/bn128/bk128/ns2/t256` | 40 | 40.03 · 1.344 | — | 135.64 · 1.683 | — |

Arm-vs-shipped: bn80 is **1.17–1.29×** slower, bn64 **1.56–1.70×**, and the
40-block bn128 **1.34–1.69×**. So one-block-per-CU is 18% *worse* than 54 blocks
covering 64 CUs, matching the M=128 finding (bn80 1.063/1.074 vs bn96) and the
N-split closure: narrowing `bn` buys blocks by re-reading A once more per block,
and A's LDS read scales with the warp count, so the trade is negative every time
it has been measured. Direction of the fix is therefore *wider* `bn` (bn96 at 53
blocks), not narrower.

**Status: o/down `bn80` closed.** This also closes the S5 M ≤ 128 decode table —
there is no configuration in it that has not now been measured.

<a id="2026-10-22f"></a>
### 2026-10-22f — a band boundary that was never measured: qkv M ∈ (96,112]

Refreshing the S5 table at M ∈ {96,112,128} produced a self-contradicting row:
qkv32's M=112 cell took **79.7 µs while M=128 took 67.9 µs** for 12.5% *more*
work. The cause is a guard, not a tile: `decode_cfg`'s M ∈ (64,128] branch tested
`if M > 112` before offering bk64, so M=97..112 fell through to
`bm128/bn128/bk128/**ns1**` — no pipeline stage, 1 block/CU, i.e. exactly the
configuration the bk64 arm exists to replace. The boundary was drawn when bk128
was the only `bk` in the table, and the bk64 arm was only ever measured *above*
it, so nothing had ever compared the two at M ≤ 112.

Four arms, qkv32 (10240×5120), paired same-process, CUDA-graph replay, 7 rounds,
min-of-rounds, all `rel == 0.0` (bit-exact), same hipblas 65.2 µs:

| arm | M=96 | M=112 | M=128 |
|---|---|---|---|
| `bm128/bn128/bk128/ns1/t512` (was shipped at 112) | — | 79.68 µs · 1.221 | — |
| `bm128/bn128/bk64/ns2/t512` (this fix) | 65.78 · 1.011 | **66.52 · 1.020** | 67.14 · 0.963 |
| `bm128/bn96/bk128/ns2/t512` | 78.15 · 1.199 | 78.67 · 1.206 | 78.27 · 1.122 |
| `bm128/bn128/bk128/ns2/t512` | 81.88 · 1.259 | 82.74 · 1.266 | 82.70 · 1.183 |
| shipped at M=96 (`bm96/bn96/bk64/ns2/t384`) | **61.30 · 0.944** | — | — |

So the arm is legal and correct over the whole band; M ≤ 96 keeps its bm96 tile
(it wins 61.30 against this tile's 65.78), and everything above 96 now takes
bk64. Re-checked *through the chooser* — not with hand-listed arms — at a
boundary and an interior point: qkv32 M=104 **1.014**, M=112 **1.020** (66.0 /
66.7 µs); qkv27 (14336×5120) M=104 **0.998**, M=112 1.034; both win at M=128.
The fix is 0.835× the old arm at M=112, and it removes the last losing cell in
the M ≤ 128 decode table for qkv32 except M=112's 1.02 (2% is inside the
shared-card drift).

The guard that has to survive: z27 (N=6144) is the one shape in this branch whose
**bn128 grid is only 48 blocks** — under 64 CUs, so the second resident LDS slot
would sit idle and bk64 costs 1.20-1.28×. The hoist therefore tests the bk64
arm's *own* block count (`ceil(N/128) > 64`), not M. Verified after the change:
z27 M=104/112 still `bm128/bn128/bk128/ns1`, z27 M=113/128 still
`bm128/bn96/bk128/ns2`, o/down and gate_up byte-identical, and
`test_decode_cfg_is_shape_aware` pins both sides of the boundary.

<a id="2026-10-22-m96-128-n5120"></a>
## The N=5120 band M ∈ (96,128] is closed — including the warp-count axis

`o` (5120×8192) and `down` (5120×25600) at M=112/128 ship
`bm128/bn96/bk128/ns2/t512` — 54 blocks, 57344 B of LDS, i.e. **one block per
CU**. Eight arms, paired same-process against `torch._scaled_mm`, 7 rounds,
min-of-rounds, `rel` 2.2–2.5e-3:

| arm (bm/bn/bk/ns/t) | o M=112 | down M=112 | o M=128 | down M=128 |
|---|---|---|---|---|
| **shipped** (ratio vs hipblas) | **1.0440** (60.70/58.14) | **1.0376** (174.71/168.39) | **1.0115** (60.57/59.88) | **1.0085** (174.76/173.29) |
| `t256` — 8 warps (m8/n1/k1) | 1.2703 | 1.3738 | 1.2676 | 1.3178 |
| `bk64` @ t512 | 1.7925 | 1.9118 | 1.7734 | 1.9283 |
| `bn128/bk64/t512` | 1.4286 | 1.7506 | 1.4018 | 1.8146 |
| `bn64/bk128/t512` | 2.1372 | 2.2289 | 2.1209 | 2.1786 |
| shipped + `k_unroll=4` | 1.0719 | 1.0680 | 1.0514 | 1.0404 |
| shipped + `k_unroll=16` | 1.1083 | 1.0529 | 1.0748 | 1.0204 |

(`bm128/bn96/bk64/t256` is not a legal arm: *no valid MLS tile config for trans:
block_mn=96 block_k=64 num_warps* — `bn/n_warp >= 16` needs 6 warps in N at
bn96, and 6 does not divide 8.)

Three things this fixes in the record:

1. **Warp count was the last unswept axis and it is the worst one.** A's load
   has no warp-id term, so A's LDS traffic is `blocks × warps` (measured
   separately: A is replicated in every warp, and gating its issue on owner warps
   costs `cp_async_wait` depth instead), so bm128 pins
   `m_warp=8`, so `t256` is the only legal step down. It is 1.22–1.37× slower,
   i.e. A's per-warp replication is **not** what binds on this band — the
   opposite of what the replication law alone would predict.
2. **`bk64` is block-count-conditional, not shape-conditional.** The same
   `bm128/bn128/bk64/ns2` tile that wins qkv32 (1.020 on 80 blocks, where two
   blocks co-reside per CU) is 1.40–1.81× slower here, because 40 + 40 blocks
   leave nothing to co-reside with and halving `bk` only halves the bytes in
   flight. Any future bk64 hoist must gate on the arm's own block count, exactly
   as the qkv hoist does.
3. **The band's honest standing is 1.01–1.04, not the 0.995–1.002** the older
   per-M table carries for M=112/128. Nothing has regressed; those two cells
   were measured on a quieter card. Use the fresh paired numbers.

`gate_up` (51200×5120) is the other shape the chooser routes through this band
(`ceil(N/128) = 400 >= 256` → `bm128/bn128/bk128/ns2/t512`, 400 blocks) and it
had never been measured above M=96. Paired arm/ship, same process:

| arm | M=112 | M=128 |
|---|---|---|
| shipped (ratio vs hipblas) | 288.14 µs · **0.9382** | 292.36 µs · **0.9546** |
| `bk64` | **0.9775** | 0.9969 |
| `k_unroll=16` | 0.9848 | 0.9892 |
| `k_unroll=4` | 0.9950 | 0.9963 |
| `bn96` (533 blocks) | 1.2251 | 1.2574 |
| `ns1` | 1.1009 | 1.0896 |
| `t256` | 1.1300 | 1.1248 |
| `bn256/ns1` (200 blocks) | 1.5484 | 1.5314 |

So the shipped tile stands: `bk64` and `k_unroll=16` are 0.3–2.2% and their sign
flips between the two M points, i.e. under the gate's own 2% threshold and
inside paired noise — **not adopted**. What does transfer from the qkv finding is
the *mechanism*: at 400 blocks (2 co-resident per CU) `bk64` is neutral-to-slightly
better here, whereas on o/down's 40 blocks it is 1.79× worse. `bn96`/`bn256`
lose for the usual reason (533 blocks of narrower bn re-read A; 200 blocks leave
CUs idle).

With this, every shape has a measured standing in M ∈ (96,128]:
`gate_up` **0.94–0.95 (win)**, `qkv` 1.02 → ~0.98, `o` 1.04/1.01, `down` 1.04/1.01.

<a id="m-gt-128"></a>
## M in (128,256] — the speculative-decode band (2026-10-23)

The band `decode_cfg` never had. It exists because speculative decoding with 3
drafts and a concurrency cap lands the batch at M ∈ (128,256], and the M ≤ 128
table simply refused those calls (`assert M <= block_M`) so they all went to
hipblas. Re-opened after the earlier "M>128 is closed" verdict, which had only
ever tested *big-M tiles on the M ≤ 128 kernel*, not the band's own geometry.

**The governing law: `bm + bn ≤ 256` at bk128/ns2.** The LDS budget is
`(bm+bn)·bk·ns ≤ 65536`, which at bk128/ns2 is exactly `bm+bn ≤ 256`. So a
single-block-row M=160 kernel can keep the k-loop double-buffered and unrolled,
but only at `bn96` — `bm160/bn128` needs 73728 B and is rejected. The
alternative, dropping to bk64/ns2 to buy bn back, measures worse (o27 M=256:
bk64 1.11 vs bk128 1.00). This is the same `(L + C·ns)` budget from
[the invariant](#2026-10-19-lds), now strong enough to decide a tile on its own.

Second constraint: **wide bn only pays while the grid stays above 64 CUs.**
`bn96` at bm160 gives `ceil(N/96)` blocks — 54 at N=5120, 64 at N=6144, 107+ at
N≥10240 — so the tile transfers from the M ≤ 128 band unchanged. At N=51200
(`gate_up32`) 534 blocks of 20480 B is 3 blocks/CU and the shape loses outright.

Shipped, all paired same-process, per-round ratios, dup-hipblas control in
[0.998,1.001], bit-exact against an fp32 reference on every arm
(`/tmp/final256h.log`, `/tmp/final256i.log`, `/tmp/down32_m.log`):

| shape | (N,K) | M=160 tile | ratio | |
|---|---|---|---|---|
| down27 | (5120, 17408) | `bm160/bn96/bk128/ns2/t256/sp1/u8` | **0.873** | win |
| o32 | (5120, 8192) | same | **0.882** | win |
| z27 | (6144, 5120) | same | **0.887** | win |
| o27 | (5120, 6144) | same | **0.900** | win |
| gate_up27 | (34816, 5120) | `bm160/bn96/bk128/ns1/t256/sp0/u1` | **0.974** | win (marginal) |
| qkv32 | (10240, 5120) | ns1 tile above | 0.995 | under the 2% gate → hipblas |
| gate_up32 | (51200, 5120) | ns1 tile above | 1.179 | hipblas |
| down32 | (5120, 25600) | bk128 | 1.05–1.10 | **closed**, see below |

**`setprio` is not a free default in this band.** On the N<10240 tile it is worth
5–13% (down27 1.008→0.873, o27 0.924→0.900), while on the N≥10240 ns1 tile it
*loses* ~11% (gate_up27 0.974→1.078). The two groups get different values for
that reason, not from tuning noise.

**K=25600 kills the unroll, and down32 is closed.** `down32` has the same tile
as `down27` but `k_unroll=8` at K=25600 is bimodal *within one process* —
3.000 / 0.919 / 2.971 at M=144/152/160 — and `u4` measures 3.96. With `u1` the
cell is 1.069/1.046/1.066 (ctl 1.001) and the drop-in agrees at 1.065–1.072, so
it is a genuine loss either way. It falls back to hipblas; `_UNROLL_TILES`
carries a `(160,96,128,2,256,25600): 1` entry only so a future arm does not
silently inherit `u8` from the row above. (This retracts an earlier 0.946 on
this cell, which was the tail of the u8 arm's instability.)

**M=256 is not served.** Best arm is `bm256/bn96/bk64/ns2/t512/sp1/u8` at 1.000
— a tie — and every bk128 arm is ≥1.21. So `BANDS` stops at 192 and `band(256)`
returns `None`, i.e. the call falls back rather than carrying a compiled kernel
that never wins.

**The gate entries are exact-M, and the interior was measured rather than
extrapolated.** A band-level ratio does not transfer: at the interior (M=144/152,
band-sized A so zero pad nodes) z27/o27/down27 hold at 0.87–0.91 but `o32` reads
**1.047** and `gate_up27` 0.979–0.984; with a tight A (pad node paid) `o32` is
1.193/1.196. Hence `GATE_EXACT[(N,K)][160]` only, and `band()` still pads — the
endpoint is where the win is real. `gate_up27` M=192 is *disputed* (0.968 in one
clean run, 1.026/1.032 in two others) and is not shipped.

Mechanism note: none of this contradicts the M ≤ 128 closures. `bk64`, `ns>1`,
split-K and warp-count sweeps all stay closed — the wide band wins by taking the
*same* bk128/ns2 invariant to a `bm` the old table refused, not by finding a new
knob.

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
| <a id="h-2026-10-22"></a>2026-10-22 | The `bn16` fallback is retired: N ∈ (5120,10240) — only 27B's `in_proj_z` (6144×5120) lives there — takes the same fat tile at `t128` because its grid is exactly 64 blocks, and goes from the worst cell in the table (1.98× at M=24) to 0.83–0.88. The `t128` point is bracket-checked against an N=5120 control in the same process, and `k_unroll` is measured per band (8 helps bm32 by 4%, hurts bm16 by 1.5%). | adopted |
| <a id="h-2026-10-22b"></a>2026-10-22b | The M=1 loss is *node count*, not bytes: a per-replay profiler names two `Memcpy DtoD` nodes at ~5 µs each on a 22 µs kernel (the per-tensor `scale_a` broadcast, then the band pad), and `copy_` costs 5.0 µs even at 4 bytes. Retracts the "`_prep_scales` broadcasts to N elements" explanation. A `per_tensor_a` kernel variant (declare `scale_a: (1,)`, multiply by `scale_a[0]`) deletes the scale node; the drop-in path then equals the raw kernel to 0.01 µs. Remaining M=1 losses (o 1.06, down 1.09) are the kernel itself, closed. | adopted |
| <a id="h-2026-10-22c"></a>2026-10-22c | S3 drove the swap end-to-end: 20 (shape, M) cells through `torch.ops.vllm.hcu_channel_fp8_target_triton_scaled_mm` in one process, zero `FALLBACK`, `rel ≤ 2.8e-3` vs an fp32 slice, and every cell reproducing its gate-table value — the swap itself adds no measurable overhead. The same run pinned the *spelling* vLLM uses (tight A + tight per-token `scale_a`), which makes `GATE_NODE_FREE` dead code under `vllm serve` and exposed three wrong gate entries: `down` band 64 **1.07 → 0.945** (three runs through the dispatch: 0.9162/0.9452/0.9456 — a shipped 4–8% win was going to hipblas, stale since the k-loop unroll), its band-64 interior 1.13 → 0.99 (verdict unchanged; the band loses at M ≤ 47 and wins from 48, and a band-keyed table can only ship one number), and z27 bands 16/32 **0.823/0.849 → 1.18/1.15** (slab-backed numbers that were shipping a 15–18% *loss* at M=2..31). | corrected |
| <a id="h-2026-10-22d"></a>2026-10-22d | S4 could not run a real engine (the per-process VRAM ceiling is ~1.5 GiB), so the swap was costed as a **whole-model GEMM sum** instead: real dispatch, both arms in one process, per-shape `µs` composed over each model's own layer list. 32B (64 layers, 31.2 GB fp8) gains **+3.1% / +10.5% / +12.5% / +1.3%** of decode GEMM time at M = 1/16/32/64; 27B (48 linear-attn + 16 full-attn layers, 24.3 GB) gains **+5.4% / +11.4% / +17.4% / +6.8%** at M = 1/16/24/64, its best cell being `gate_up27` at M=24 (195→138 µs). Two things this pins down: the gain is **not monotone in M** (best at M≈16–32, because hipblas's block layout is worst there), and the gate is load-bearing at the ends — forcing tilelang everywhere is a **3.3% loss** at 32B M=1 and a 0.8% loss at M=64, i.e. the shipped mix is the only arm that wins at those two points. | measured |
| <a id="h-2026-10-22e"></a>2026-10-22e | Layout A becomes deployable rather than merely measurable, via two non-speed gaps. The blocked builder had no `out_dtype`, so it emitted fp32 where the drop-in contract (and hipblas) returns bf16 — fixed, and free: blocked-bf16 is bit-exact against row-major-bf16 and against its own fp32 re-rounded output, `blk_b/blk_f` = 0.977–1.001. The runtime had no way to recognise a blocked tensor — a `data_ptr`-keyed registry plus a branch ahead of the gate now routes it, **raising** rather than falling back when `(N,K,M)` has no blocked tile (a row-major read of blocked storage is wrong, not slow). Re-measured in bf16 at M=64: `gate_up` **1.043 → 0.894** (hipblas's 4% win becomes our 11% win), `down` 1.070 → 0.908 (0.85–0.95 across segments), `qkv` 0.962 → 0.932, `o` correctly refused. An 8-cell gate regression shows the unregistered path is bit-identical and identically routed. Deployment bound measured, not assumed: the band is **M ∈ (32, 64]** — `bn96`/`bn32` govern M ≤ 32, and forcing a uniform bn64/bk128 tile down to those M costs 1.29–1.78x hipblas against the shipped 0.85–1.04x (8 cells, blocked 1.5–10% slower than its own row-major twin in every one), so the window cannot be widened and this is a `--max-num-seqs 64` decision that still raises below 33 rows. The route is also validated where it will actually run — inside a captured CUDA graph through the vLLM custom op: 6 blocked hits across warm-up + capture, replays re-execute the kernel without re-dispatching and stay bit-exact and idempotent, an unregistered blocked tensor stays on the row-major kernel (dispatch is by `data_ptr`, so no shape-based false positive), and M=96 raises *inside* capture instead of baking a wrong kernel into a graph. | adopted (opt-in) |
| <a id="h-2026-10-22f"></a>2026-10-22f | A refresh of the M≥96 table found qkv32 at M=112 costing **79.7 µs where M=128 cost 67.9** for *more* work: the M ∈ (64,128] branch tested `if M > 112` before offering bk64, so M=97..112 kept `bk128/**ns1**` (no pipeline stage, 1 block/CU) — a boundary drawn when bk128 was the only bk in the table, with the bk64 arm only ever measured above it. Hoisting the bk64 arm above the test over the whole band and gating it on its own block count (`ceil(N/128) > 64`) takes M=112 **79.68 → 66.52 µs (1.221 → 1.020)**, 0.835× the old arm, all arms bit-exact; re-verified through the chooser at an interior point too (M=104 1.014 qkv32 / 0.998 qkv27). z27's 48-block grid still falls through as before, and o/down/gate_up are byte-identical after the change. | adopted |
| <a id="h-2026-10-22g"></a>2026-10-22g | The last open cell — o/down `bn80` "one block per CU" — turns out to be **inexpressible** at `bm16`: bn80 is 5 warps in N, 5 is prime, and the builder requires `m_warp*n_warp*k_warp == num_warps` with `bn/n_warp >= 16`, so `bm16/bn80/t160` is rejected and the only legal spelling is `bm32/bn80/bk256/ns2/t320` (which pads M=16 to 32 rows). Measured anyway in 4 cells: bn80 is 1.17–1.29x *slower* than the shipped 53-block `bm16/bn96/bk256/ns2/t192`, bn64 (80 blocks) 1.56–1.70x, bn128 (40 blocks) 1.34–1.69x — so filling 64 CUs with 64 small blocks is worse than covering them with 54 wide ones, the same sign as M=128 (1.063/1.074) and as the N-split closure: narrowing bn re-reads A once per block and A's LDS read scales with warp count. Direction is *wider* bn, not narrower. The S5 M ≤ 128 decode table now has no unmeasured cell. | closed |
| <a id="h-2026-10-22h"></a>2026-10-22h | The M ∈ (96,128] N=5120 neighbourhood is closed: 8 arms × 4 cells (o/down × M=112/128), nothing beats the shipped `bm128/bn96/bk128/ns2/t512`, and the closest misses are `k_unroll` neighbours at +2–8%. **Warp count** — the one axis never swept on this band — is the worst direction: `t256` (the only legal step down, 8 warps) is 1.22–1.37× slower, so A's per-warp replication is not what binds here. `bk64` is 1.73–1.93× slower *at bn96*, which pins the qkv bk64 win as **block-count-conditional**: the same `bm128/bn128/bk64` tile is 1.020 on qkv (80 blocks, 2 blocks/CU co-reside) and 1.40–1.81× on o/down (40 blocks, nothing co-resides, so halving `bk` only halves the bytes in flight). The band's real standing is 1.01–1.04, not the 0.995–1.002 the older per-M table carried. | closed |
| <a id="h-2026-10-22i"></a>2026-10-22i | The last unmeasured M ∈ (96,128] cell — `gate_up` (51200×5120, 400 blocks, shipped `bm128/bn128/bk128/ns2/t512`) — verifies out at **0.9382 (M=112) / 0.9546 (M=128)** vs hipblas, and no neighbour beats it: `bk64` 0.9775/0.9969, `k_unroll=16` 0.9848/0.9892, both under the 2% gate threshold *and* flipping sign between the two M points, so not adopted; `bn96` 1.23–1.26×, `ns1` 1.09–1.10×, `t256` 1.12–1.13×, `bn256/ns1` 1.53–1.55×. This also confirms the block-count rule on a *third* regime: `bk64` is ~neutral at 400 blocks (like qkv's 80), 1.79× worse at o/down's 40. Every shape now has a measured standing in this band: gate_up 0.94–0.95 (win), qkv 1.02→~0.98, o 1.04/1.01, down 1.04/1.01. | closed |
| <a id="h-2026-10-23"></a>2026-10-23 | Three tiles lifted from the hipblasLt **tuner's** own decode kernels and gated on explicit `(N,K)` lists: `bm16/bn48/bk256` (5 of 6 shapes at M ≤ 16, 1.03–1.12× over the tile it replaces), `bm64/bn192/bk128` (N=34816/10240 at M ∈ (32,64], 1.04–1.16×), `bm96/bn192/bk64` (N=34816 at M ∈ (64,96], 1.03–1.09×). All bit-identical to the tile they replace. `bn192` is legal — it retires the `bn ≤ 128` cap for the (bm,bk,ns,threads) combos now in the table; `bn ≥ 256` stays closed. The e2e A/B is drift-dominated (10 cells up / 12 down vs 2026-10-22, hipblas itself 4–19% faster), but no dispatch-gate verdict flipped. | adopted |
| <a id="h-2026-10-23b"></a>2026-10-23b | The M ∈ (128,256] band (speculative decode, 3 drafts + 1) is opened and won on 5 of 8 shapes at band 160: `bm160/bn96/bk128/ns2/t256/sp1/u8` gives down27 **0.873**, o32 **0.882**, z27 **0.887**, o27 **0.900**, and the N≥10240 `ns1/sp0` variant gives gate_up27 0.974. The deciding constraint is `bm+bn ≤ 256` at bk128/ns2 (the LDS budget), which forces `bn96`; `setprio` splits by grid size (+5–13% below N=10240, −11% above). `down32` (K=25600) is **closed** — `u8` there is bimodal within one process (3.000/0.919/2.971) and `u1` still loses 1.05–1.10 — and M=256 is not served (best arm is a 1.000 tie). Gate entries are exact-M: the band interior re-measures at 0.87–0.91 on three shapes but 1.047 on o32. | adopted + closed |
