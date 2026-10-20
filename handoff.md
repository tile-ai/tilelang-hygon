# HCU fp8 Channelwise Decode GEMM — Handoff

This is the entry point for a new session. Status comes first, then what is open,
then the closed levers with their decisive reasons, then the detailed dated
record. The full measured record — every table, ratio, SASS excerpt and
correction — is in `docs/optimization/fp8_decode_m_le_128.md`; links below point
into it. Branch: `feat/hcu-smls-b-vgpr`.

## 1. What is shipped right now

`decode_cfg` (`perf/gemm/fp8_channelwise_mls.py`) returns:

- **N=5120 (`o`/`down`) at M ≤ 32:** `bn96 / bk256 / ns2 / t192`, `setprio=1`.
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
- **M=128:** o 1.83×, down 2.23×, qkv 1.15×. Only 40 blocks at bn128, and bn
  cannot shrink without shrinking the grid further. Not addressed. (docs
  [2026-10-17f final matrix](docs/optimization/fp8_decode_m_le_128.md#2026-10-17f))
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

## 4. Detailed history

The sections below are in file order, not chronological; the newest are
2026-10-20b/20c near the end. They are historical: wording and numbers are as
recorded at the time, and several early readings were later corrected (the
native-kernel operand roles, the read ceiling, the `ns>1` mechanism). Read the
dated corrections and closures in the later sections.

## 2026-10-17i — the fix invalidated one measurement; re-timed, last two mechanisms close

**1. `ns=2` at bn16, re-timed correctly for the first time.** The 2026-10-17g sweep
called `bk256 ns2 t64/t128` the two fastest ns=2 candidates — but those are exactly
the two the packed-stride defect broke, so "fast" measured garbage. Corrected and
re-timed (paired, one process, 6 resident graphs, 7 rounds, min-of-rounds, GPU 2):
they are **29-45% SLOWER** than the shipped `ns=1`, matching the always-correct
`t256` control (1.29-1.42). Ratios vs `bn16/bk256/ns1/t64`: o M=1 1.383, o M=16
1.386, down M=1 1.406, down M=16 1.401 (t64); t128 1.399/1.528/1.451/1.447; t256
1.292/1.419/1.336/1.425; `bk512 ns2 t64` 1.053/1.171/**0.989**/**1.000**.
**ns=2 at bn16 is closed** — and this retires the hope that the fix re-opened
o/down, since the only reason to retest ns>1 was that the fast-looking candidates
had been invalid.

**2. Independent accumulator chains — closed (1.52-1.64x slower).** At bn16/t64
there is one 16x16 accumulator, so all 256 k-loop mmacs are one serial chain from
one wavefront — the last untested mechanism for the measured `full - AB` deficit.
`/tmp/split_acc.py` splits it in-tree (two `(16,128)` MLS buffers per operand, two
fragments, summed in the epilogue): o M=1 1.539, o M=16 1.524, down M=1 1.644,
down M=16 1.641 vs the baseline. The extra MLS stream and doubled LDS outweigh any
chain overlap.

**3. A second, pre-existing packed-layout defect — found, unreachable, not fixed.**
`bn16/bk128/ns2/t64` is wrong (rel 0.55) at every K, and was wrong before the fix
too (rel 3e4 on the logical stride), so the fix reduces it without removing it.
The emitted packed stage stride is `idx*1024 + idx*64` = 1088 B while the tile is
2048 B, so the two stages overlap in LDS; `bk128/ns4/t64` (stride 2048) and
`bk128/ns2/t128` (tile `(16,64)`, not packed) are both correct. Break is specific
to the packed branch at `tile_issue_k == 1`. Not chased: unreachable from
`decode_cfg` (`_validate_cfg` restricts `ns>1` to bm64/bn64, and bk128 at bn16 is
1.596x slower at ns=1 anyway). Recorded so it is not mistaken for a regression.

**4. The three-arm decomposition was redone and it CORRECTS the 2026-10-17h
diagnosis.** That section treats the ds_read/mmac side as an *additive* cost to be
shrunk. It is not. With a clean compute-only arm (identical kernel, every iteration
reading one of 4 cycling L2-resident B tiles — the earlier arm re-read a single
address, which is slower than streaming and made compute-only exceed the full
kernel), the two sides overlap almost completely. `/tmp/floor4.py`, bn16/bk256/ns1/t64:

| arm | o M=1 | o M=16 | down M=1 | down M=16 |
|---|---|---|---|---|
| hipblas | 31.5 | 35.0 | 78.6 | 85.3 |
| load-only | 31.2 (0.990) | 33.1 (0.945) | 82.1 (1.045) | 85.0 (0.996) |
| compute-only | 29.8 (0.944) | 30.4 (0.868) | 69.8 (0.888) | 70.4 (0.826) |
| **full** | 38.7 (1.228) | 39.5 (1.128) | 100.3 (1.277) | 101.0 (1.184) |
| floor = max | 31.2 | 33.1 | 82.1 | 85.0 |
| exposed = full − floor | **+7.5** | **+6.4** | **+18.2** | **+16.0** |

**(a) bn16 is LOAD-bound, not compute-bound** — the compute side is cheaper than
the load side in all four cases, so the single accumulator chain is not the
constraint (consistent with independent chains being 1.5-1.6x *worse*).
**(b) Neither side is the deficit.** For o M=16 and down M=16 both arms beat
hipblas individually; for o M=1 the load arm is 0.990. The whole deficit is the
16-20% of wall time when neither pipe is busy.

**5. What caps the overlap is resident blocks per CU, and it is the grid, not the
LDS.** Inflating the grid with *distinct* B rows (`/tmp/grid_mult2.py`; the first
version duplicated `bx % 320` and the duplicates shared an L2 window, so it
measured partly L2 — distinct rows removes that and the effect survives), identical
per-block work and bytes, o M=1:

| | 320 blocks | 640 | 1280 | 2560 |
|---|---|---|---|---|
| load-only µs/block | 31.4 | 30.1 | 24.6 | 23.0 |
| load-only GB/s | 1337 | 1394 | 1702 | 1827 |
| full µs/block | 38.9 | 34.4 | 32.2 | 30.9 |

The memory side is at 66% of the 2016 GB/s ceiling at 320 blocks and 91% with
backfill; the full kernel converges onto its own compute floor (30.9 vs
compute-only 29.8), i.e. **~0.98x hipblas at o M=1 is reachable** — at ~8x the
blocks. 320 blocks over 64 CUs is *exactly* 5/CU against an LDS residency limit of
8 (8 KB/block; `bk128/ns8` fails to launch at ~40 KB of packed stages, confirming
the 65536 B budget), so 3 slots sit empty for the whole kernel, and `N/bn = 320`
with `bn=16` at the hardware floor cannot fill them.

**6. Wavefront occupancy is NOT the missing lever — measured.** `/tmp/warps.py`,
bn16/bk256/ns1, t=64/128/256 (`ComputeMlsWarpPartition` picks `(16,128)` at t=64
and `(16,64)` at t>=128, so extra warps split K at *identical* bytes and LDS):
ratios vs t=64 are 1.000/1.011/1.015 (o M=1), 1.000/1.008/1.232 (o M=16),
1.000/0.971/1.096 (down M=1), 1.000/1.013/1.162 (down M=16). `wk`-split warps work
on *one* tile, so they add instructions, not independent memory streams. §5 adds
independent *tiles*. The lever is block count, and it is fixed at 320.

**7. Every remaining knob that could add resident blocks is closed.**
split-K adds blocks but shortens the run (32 -> 8 k-steps at sk=4), and the clean
constant-bytes experiment (2026-10-17f) found 1280/25 *worse* than 320/100 — the
ramp cost of a short run exceeds the residency gain; §5 holds the run constant,
split-K does not, so §5 does not license it. Deeper staging (`/tmp/ns_full.py`) is
monotonically worse and tracks LDS: bk256 ns1 1.000 / ns2 1.343 / ns4 1.812;
bk512 ns1 1.217 / ns2 0.984 — the only free ns=2 is bk512, free because
`tile_issue_k == 4` makes the packed size coincide with the logical one so the
packed branch is never taken. Smaller bk is worse (1.108x / 2.0x). Wider bn halves
the block count (160 = 2.5 waves) and is 1.08-1.14x worse. Plus independent chains,
more warps, tile widening, warp policy, `k_pack`, ds_read alt, layout-map.

**Standing after this:** o M=1 1.080, o M=16 **1.000**, down M=1 1.202, down M=16
1.180 (paired, same process, idle card). o M=16 is a tie, not a loss. Every
mechanism that could hide the ds_read/mmac deficit is now measured negative — and
the deficit is no longer the mmac side: it is the 16-20% of wall time where
neither pipe is busy, capped by 5 resident blocks/CU against an 8-slot limit that
the fixed grid cannot fill. **o and down cannot be won in-tree.** (Load-only arm
precision: 2026-10-17h had 0.957/0.917/0.977/0.948, this run 0.990/0.945/1.045/0.996
— shared card; treat it as "0.95-1.05 of hipblas", which is why down M=1 flips
sign between runs. It does not change any conclusion.)

## 2026-10-17h — o/down: the load path already WINS; the mmac side is the whole deficit, and it is closed

This answers the question the previous sections assumed: **which side of the
kernel is losing?** Full writeup in `docs/optimization/fp8_decode_m_le_128.md`
("2026-10-17h"). Summary:

**1. The load path already beats hipblas on all four o/down cases.**
`/tmp/odown_decomp.py`: four arms per (shape, M) with the *same* tiles and
access pattern, one process, round-robin interleaved, 7 rounds.

| case | hip | AB (load only) | full | AB/hip | full/hip | mmac side adds |
|---|---|---|---|---|---|---|
| o M=1    | 37.3 | **35.7** | 43.5  | **0.957** | 1.167 | +7.8 us |
| o M=16   | 34.9 | **32.0** | 39.4  | **0.917** | 1.128 | +7.4 us |
| down M=1 | 82.7 | **80.8** | 99.5  | **0.977** | 1.203 | +18.7 us |
| down M=16| 88.8 | **84.2** | 100.3 | **0.948** | 1.130 | +16.2 us |

The deficit doubles from o (K=8192) to down (K=25600), i.e. it is proportional
to the k-loop, not a fixed intercept. **Any story that o/down are "block-starved"
is dead** — you cannot fix a cost that scales with the work by adding blocks.

**2. `ns=2` at bn16 root-caused — it is a real lowering bug.** Dumped source:
the MLS producer writes stage 1 at `A_s[((k+1)&1)*6144]` (the *packed* LDS
stride, `TryGetMlsDstActualSizeBytes`, `src/hcu/op/mls.cc:89`) while the ds_read
consumer reads `A_s[(k&1)*4096]` (the *logical* stride, from
`TryGetMlsPackedLeadingElemOffset`'s fallback — `src/hcu/op/ds_read_format.cc:639`
only applies the packed offset when region rank > 2). At t=256 the packed branch
is not taken, both sides use 4096, and ns=2 is correct. That is the whole
explanation for "correctness flips with the thread count". Worth fixing on its
own merits — it is why `num_stages>1` is asserted off outside bm64/bn64.

**3. Double buffering closed even where correct.** Where ns=2 is right it does
overlap at bk512 (ns2/ns1 = 0.782-0.927) but at bk256 it *hurts* (1.13-1.17),
and the best absolute ns2 config (bn16/bk512/t128 = 40.6/42.7/98.2/106.9) is no
better than the shipped ns1/bn16/bk256/t64 (38.4/39.7/99.4/100.3) bar a 1.2% win
on down M=1. A hand-written two-buffer pipeline (`/tmp/manual_pipe2.py`, explicit
`T.ptx_wait_group`) is correct but 1.26-1.70x slower.

**4. Grid inflation confirms it.** Fixed K, varying blocks: load-only 1298 → 1417
→ 1644 → 1743 GB/s (320 → 2560 blocks) while the full kernel goes 1181 → 1200 →
1288 → 1323. The load path loves occupancy; the full kernel does not care.

**5. The last instruction-level knobs, all closed** (`/tmp/ann_sweep.py`):
`tl.hcu_a/b_respect_layout_map=1` are exact no-ops; `policy=FullRow`/`FullCol` at
t=128 are identical to each other and 1.2% worse than t=64 (the partition does
not change at bn16); `k_pack=2` is hard-rejected by the MLS path
(`MatrixLoad dst Gemm consumer must have kPack=1`); `tl.hcu_b_ds_read_alt=0` is
`unsupported B interleave=0` (only Alt1 exists for b8-trans, and it is default);
`bm32/bn16` at M<=16 is 1.8-2.6x worse, as the 16-row atom floor predicts.

**Bound.** Best case, transferring the bk512 overlap fraction to the shipped
tile: o M=1 ~34.1 (1.046, losing), o M=16 ~36.8 (1.051), down M=1 ~77.7 (0.991),
down M=16 ~84.4 (0.998) — parity on down at best, inside hipblas's own ~20%
spread. The structural difference is hipblas's N-tile-1 GEMV at M=1 (no LDS
round-trip) and ~40 warps/CU at M=16, vs a 16x16 MMA atom pinned MLS partition.
**Do not re-open the o/down config search without a new mechanism.**

**6. The `ns>1` landmine is root-caused and FIXED.** (Added after the above; the
2026-10-17f section could only say "silently wrong outside bm64/bn64, mechanism
unknown".) An MLS dst tile whose layout is *packed* takes more LDS bytes than its
logical element count — `(16, 128)` fp8-trans is `4096 + tile_issue_k*1024`, i.e.
**6144** at `bk=256`. The C++ producer (`MatrixLoad`) and the shared-memory
allocation both advanced the stage pointer by 6144; the Python ds_read consumer
(`retrieve_mls_lds_base_ptr`) advanced it by the logical `prod(shape[-2:])` =
4096. Dumped source at `bm16/bn16/bk256/ns2/t64` showed exactly that split. It
tracked the thread count only because `ComputeMlsWarpPartition` picks the MLS tile
from `(block_mn, block_k, threads)` — `t=64` → `(16, 128)` packed, `t=256` →
`(16, 64)` contiguous — and `bk=512` coincides with the logical size by
arithmetic accident.

Fix: the packed byte count is now `MlsPackedTileBytes` in `src/hcu/op/mls.cc`,
exported as the `tl.MlsPackedTileElems` FFI and consumed by the new
`mls_packed_tile_elems` in `hcu_mmac_emitter_utils.py`, so all three consumers
share one source of truth. `bm16/bn16/bk256/ns2` at `t=64` and `t=128` went from
40/40 wrong (rel ≈ 1.6e4) to `rel = 0.00000`; everything previously correct is
unchanged (o/down/gate_up, M=1 and M=16, ns up to 8, bk 64-512, bf16 probe).
`92 passed, 58 skipped`. New tests:
`test_num_stages_gt_1_mls_is_correct` (fails with rel ≈ 1.6e4 if the consumer's
stride is forced back to 0) and
`test_mls_packed_stride_is_shared_by_producer_and_consumer`.

**This does not re-open o/down** — the shipped tiles are ns=1 and the deficit is
the ds_read/mmac side, which ns does not touch. The `num_stages <= 1 or
(bm, bn) == (64, 64)` asserts in `perf/gemm/fp8_channelwise_mls.py` were reworded
from "incorrect" to "untuned", which is now the true reason.

## 2026-10-17g — qkv M=32 lever adopted (1.14x → 1.03x); o/down levers exhausted

Full standing, chooser vs hipblas, paired 10 rounds: gate_up **6/6 wins**
(0.757-0.917), qkv **5/6** (0.757-0.868, M=32 now parity 1.031), o 1/6, down 0/6.

### Two measurement errors found and corrected

1. **Every earlier M<=32 and o/down ratio in this doc used a wrong baseline.**
   `decode_cfg` sets `setprio=1` for M<=32, but the sweep harnesses compared
   against hand-built configs with `setprio=0`. `o` M=32 is 51.8us at sp0 vs
   46.7us at sp1 — a 10% error. Re-measured against `decode_cfg(M,N,K)` itself,
   o M=1 is **1.204-1.235**, not the 1.048 the older table claims. Always compare
   against the chooser's actual output.
2. **hipblas is the noisy term at o/down** (31.4-36.8us at o M=1 across runs while
   our kernel moved 38.6→38.9). Quote the ratio as a range.

### The find: qkv (N=10240) at M=17..32 wants bk128/t128

`bn32/bk128/ns1/t128/sp1` beats the chosen `bn32/bk256/ns1/t256/sp1` by **9-10%**
(52.3 vs 57.8us, paired, 16 rounds, both sp1) → qkv M=32 goes 1.14x → **1.03x**
(best round 0.945). 11 independent paired measurements across 8 processes, all
0.84-0.91; survives an L2 flush before every replay (0.852 warm → 0.862 cold)
while hipblas is bit-identical under the same flush; holds at N=10240 for
K=5120/10240/20480.

It is **not** a general rule and the mechanism is unknown: at M=32/K=5120 with
bn32 it *loses* 17-38% at N = 11264, 15360, 20480, 25600, 51200. Not byte count
(K-independent), not L2 capacity (L2 = 8 MB vs a 52 MB B), not block count
(N=15360 has more blocks and loses). Encoded as the single measured point in
`decode_cfg` (`bm == 32 and N <= 10240 and K % 128 == 0`), not as a range.

### o/down: the last lever (ns=2 at bn16) is closed, and it is a correctness result

Every shipped config is ns=1 (no k-step overlap); o/down have the fewest blocks
(320), so ns=2 was the one structural lever left. `/tmp/ns2_odown.py`, 40 random
trials per config, o and down at M=16:

- `bn16 bk256 ns2 t128` and `... t64`: **40/40 WRONG** (rel ~1e4) — the ns>1
  landmine reproduced cleanly on both shapes.
- `bn16 bk128 ns2 t128`: correct but **3.5x / 4.2x slower**.
- `bn16 bk512 ns2 t128`, `bn16 bk256 ns2 t256`: correct but 1.04-1.42x slower.

The fast configs are silently wrong; the correct ones are slower. Lever closed,
and it independently re-confirms that at bk256 correctness flips with the thread
count (t256 ok, t128/t64 wrong).

### o/down standing: structural, not an untried config

Our time is flat across M (o 38.9→46.7us, down 100.0→122.4us) → purely B-bound.
Achieved BW: gate_up 1632 GB/s (81%), down 1311 (65%), qkv 1227 (61%), o 1079
(53%); hipblas hits 1665 on down M=1. Every in-tree lever is measured and closed:
bn16 is the hardware floor, bk swept, threads identical, setprio already on,
split-K 10-21% slower, GEMV structure not expressible, ns=2 closed above. The
residual 1.08-1.28x is hipblas's N-tile-1 GEMV (no LDS round-trip, 5120 tiny
blocks) vs a 16x16-atom MLS GEMM. **Do not re-open without a new mechanism.**

## 2026-10-17f — o/down: the shape-aware chooser wins, split-K closes, and ns>1 is a landmine

Goal was to beat hipblas on `o` and `down` (qkv/gate_up already win; M=64 out of
scope). Three results, two of them negative, one of them a correctness finding
that outranks the performance question.

### 1. Step A landed: the chooser was shape-blind, and fixing it is a real win

`_decode_cfg` took only `M`, so the gate_up-tuned tile was applied to all four
decode shapes. The deficit is a property of the *shape*: **N ≥ 10240 → bn32/t256;
N = 5120 → bn16/t64**. Same-process paired, idle card:

| shape | M | before | after | hipblas | ratio after |
|---|---|---|---|---|---|
| o | 1 | 42.9 us | **38.6** | 32.5 | 1.188 |
| o | 16 | — | **39.2** | 34.7 | 1.129 |
| down | 1 | 117.4 | **99.4** | 78.1 | 1.273 |
| down | 16 | — | **100.3** | 84.5 | 1.187 |

Still losing, but o M=1 is -10% and down M=1 is -15%. gate_up (0.93) and qkv
(0.86) are unchanged and still win. The chooser now lives in
`perf/gemm/fp8_channelwise_mls.py` as `decode_cfg`/`_validate_cfg` (single source
of truth, importable by the test) and validates every invariant *with the shape
in the message* before returning.

### 2. The block-starvation diagnosis was a CONFOUND, and split-K is closed

The doc's justification for split-K was a sweep at fixed K varying N (160 blocks
→ 832 GB/s, 1600 → 1598) — which moves **block count and total bytes together**.
It cannot tell "more blocks" from "more bytes". Split-K holds bytes constant and
only multiplies blocks, so the honest test is a constant-`N·K` curve
(`/tmp/b0_grid_scale.py`, existing kernel unmodified):

| N | K | blocks | k-steps | GB/s (o curve) | GB/s (down curve) |
|---|---|---|---|---|---|
| 5120 | 8192 / 25600 | 320 | 32 / 100 | **1053** | **1302** |
| 10240 | 4096 / 12800 | 640 | 16 / 50 | 986 | 1282 |
| 20480 | 2048 / 6400 | 1280 | 8 / 25 | 1036 | 1247 |
| 40960 | 1024 / 3200 | 2560 | 4 / — | 923 | — |

**More blocks at constant bytes is worse, not better**, and 320 blocks is the
*best* point on both curves. The original sweep was measuring "more total bytes
amortizes the per-block fixed cost", not block starvation.

Split-K was built anyway (`fp8_mls_channelwise_splitk`) and measured directly,
paired, 7 rounds, all four o/down cells. It loses monotonically:

| shape | M | base | split_k=2 | split_k=4 | hipblas |
|---|---|---|---|---|---|
| o | 1 | 38.9 (1.234x) | 49.1 (1.558x) | 54.6 (1.731x) | 31.5 |
| o | 16 | 39.5 (1.113x) | 51.4 (1.446x) | 59.8 (1.685x) | 35.5 |
| down | 1 | 100.0 (1.273x) | 110.9 (1.411x) | 117.3 (1.493x) | 78.6 |
| down | 16 | 100.6 (1.178x) | 113.0 (1.322x) | 122.1 (1.429x) | 85.4 |

10-21% *slower* than the already-losing baseline, growing with `split_k`. B0
predicted exactly this (the transformation costs ~5% before the memset and the
atomic epilogue are added). **Pre-registered rule R3 fires: split-K is closed for
o/down.** The builder, its runner, and its tests were removed; `DecodeCfg.split_k`
survives as an assertable closure marker.

Two things learned on the way, both still true:
- **`out_idx` outputs are `torch.empty`** (`tilelang/jit/adapter/tvm_ffi.py:280`),
  so there is no host hook to zero them. A split-K kernel accumulates onto C, so
  C must be a caller-owned input, zeroed on the same stream inside the timed
  closure (the zero is captured into the graph, so replay N does not stack on
  N-1). `T.clear(C)` in-kernel races across `bz` — no grid barrier exists on HCU.
- The ns=1 split-K path was additionally numerically wrong (`rel=nan`); see below.

### 3. The M=1 GEMV structure is infeasible, by hardware

The user asked for a separate GEMV-structure kernel at M=1 (what hipblas does:
`Cijk_..._MT8x1x8192_MFWGS512_WGM1_ONLYN1`). It is not expressible in the MLS
path: the MMA output micro-tile is hardware-fixed at 16×16
(`src/tl_templates/hcu/gemm.h:150-151`, fp8 atom
`__builtin_hcu_mmac_f32_16x16x32_fp8_fp8` at `:118`) and the MLS B atom floor is
`tile_mn = 16` (`src/hcu/op/mls.cc:408-412`, surfaced as `kMinNPerWarp =
TransB ? 16 : 32` at `src/tl_templates/hcu/gemm_mls.h:36`). Padding B_s to 16
rows, `gemm_s_mls`, and transposing to `Cᵀ = B·Aᵀ` all still hit 16 — and padding
adds *no* blocks, which was the entire point. The reachable GEMV-equivalent was
maximal split-K at bn=16; measured, rejected (above).

### 4. NEW: `num_stages > 1` is silently wrong outside bm64/bn64

Found while sweeping: the `ns=2` candidates in the o/down sweep did not merely
lose, they returned **non-finite or `rel` 0.75–580** values. Reproduced on all
four shapes, all M, and bisected (`/tmp/ns2_repro.py`, `/tmp/ns_threshold.py`,
`/tmp/ns_threads.py`):

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
| everything ns=1, any tile | — | **ok** |

It is **not** an LDS-size threshold (32768 ok vs 36864 wrong), **not** a k-step
count, and **not** deterministic in `bk`. It tracks the thread count at small
tiles. The mechanism is not understood; it is a lowering/pipeliner defect in a
region no shipped config uses.

**The adopted M=64 config is nevertheless sound**, and that was checked directly
rather than assumed, because if the mechanism were a race then "correct" would
only mean "has not lost yet": 300 trials × 4 shapes × setprio {on, off} = 2400
runs of `decode_cfg(64)` = bk128 ns2 t256, **worst rel = 0.00, zero non-finite**
(`/tmp/ns2_stability.py`). M=128 (ns=1) likewise.

**Guards added:** both `fp8_mls_channelwise` and `_validate_cfg` now assert
`num_stages <= 1 or (block_M, block_N) == (64, 64)`, and
`test_every_chosen_decode_config_is_correct` runs the chooser's M=64 output on all
four shapes — coverage that was missing, and the only place `ns>1` is exercised.

### 5. What is left on o/down — nothing in the cheap knob space

Full o/down sweep at M=1/M=16 (`/tmp/final_sweep.py`, paired vs hipblas in one
process, best-first). o M=1:

| config | µs | vs hipblas |
|---|---|---|
| **bn16 bk256 ns1 t64 sp1 (chosen)** | **38.6** | **1.048** |
| bn16 bk256 ns1 t128 sp1 | 38.7 | 1.052 |
| bn32 bk256 ns1 t128 sp1 | 41.0 | 1.113 |
| bn16 bk256 ns1 t64 **sp0** | 41.3 | 1.122 |
| bn16 bk512 ns1 t64 sp1 | 45.1 | 1.226 |
| bn16 bk128 ns1 t64 sp1 | 58.7 | 1.596 |

`setprio=1` is worth 6.5% and is required. `threads` 64→128 changes nothing (the
16×16 atom means one warp of work either way). `bk` 128 or 512 are both much
worse. **The chooser's answer is the optimum of the in-tree search space.**

### Caveat on the ratio, and why it is quoted as a range

`torch._scaled_mm` on o M=1 read 31.5, 36.8 and 32.5 µs across three runs while
our own kernel moved only 38.9 → 38.6. hipblas is the noisy term, and a crowded
process (18 resident CUDA graphs) inflates it. A dedicated 12-round alternating
A/B with only two graphs in the process gives a **stable** ratio: spread ≤5%,
o M=1 = 1.188 [1.182, 1.197], down M=1 = 1.273 [1.270, 1.273]
(`/tmp/ratio_stability.py`). Those are the numbers to quote. Do not read a ratio
off a sweep process that holds many candidate graphs.

### 6. A second real win: the M=32 thread count

Running the full matrix surfaced a 2.0x cliff at the band edges — a shape getting
*faster* as M grows, then falling off (down: M=16 100.8 -> M=32 209.9 -> M=64
148.0). A 2.0x step is "twice the work, same parallelism": at bn16 a bm32 tile is
two 16x16 M-atoms, and the M-only table gave it ONE warp (t=64), making the block
MMA-issue-bound instead of B-stream-bound. Shrinking bm does not fix it (bm16 at
M=32 doubles B traffic — the grid is N/16 x M/16, so each B tile is read twice,
measured worse). The untested axis was threads:

| shape | M=32 t=64 | t=128 | t=256 |
|---|---|---|---|
| o | 75.2us (1.911x) | 72.3 (1.837x) | **46.9 (1.191x)** |
| down | 209.9us (1.944x) | 203.0 (1.880x) | **122.4 (1.133x)** |

Adopted: `threads = 64 if bm <= 16 else 256` on the N=5120 branch. **1.60x/1.71x.**
bm32/bn32/t256 on gate_up/qkv was already right and is unchanged.

### The final matrix, and what is still open

`ratio` = ours/hipblas, <1 is a win. M=1..32 stable to <=5%; M=64/128 move up to
~35% between runs (fewer blocks, sensitive to the shared box) — indicative only.

| shape | M=1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| gate_up | **0.93** | **0.78** | **0.87** | **0.88** | **0.91** | **0.86** | 1.08 | **1.00** |
| qkv | **0.85** | **0.82** | **0.83** | **0.84** | **0.85** | 1.13 | 1.05 | 1.15 |
| o | 1.19 | 1.14 | 1.11 | 1.11 | 1.13 | 1.19 | **0.89** | 1.83 |
| down | 1.27 | 1.23 | 1.08 | 1.18 | 1.18 | 1.11 | 1.09 | 2.23 |

**13 of 32 cells win** — gate_up M in {1..32, 128}, qkv M in {1..16}, o M=64.
Still open, honestly: o/down M<=32 (1.08-1.27x, B-stream-bound at ~1300-1500 GB/s
vs hipblas 1543-1670, closed on every lever tried), M=128 (1.83x/2.23x, only 40
blocks at bn128 and bn cannot shrink without shrinking the grid further), and qkv
M>=32 (1.05-1.15x, already at its measured neighbourhood optimum).

**Net for o/down: closed on all four levers** — shape-aware config (adopted), the
M=32 thread fix (adopted, +1.6-1.7x), split-K (measured, closed), tile widening
(catastrophic), pipeline depth (the ns>1 region is broken anyway, and ns=1 is the
safe optimum). The residual at M<=32 is not reachable by adding blocks within the
MLS/MMA structure. Do not re-open split-K without new evidence.

## 2026-10-17e — the async-wait lever: hypothesis refuted, a small real win found instead

The in-scope plan was to attack wait granularity (hipblas's "fine-grained
`vmcnt(4)`" vs our `cp_async_wait`). Two results.

**1. The hypothesis was wrong, and the doc had it from the patched file.**
Disassembling the **native** M=64 kernel
(`/opt/dtk/lib/hipblaslt/library_bw_nmz/FusedMatmulKernel_gfx938.co.bak_0916_1000`)
shows the k-loop uses **only wait-all** sync: 8× `s_waitcnt lgkmcnt(0)` and 4×
`s_waitcnt vmcnt(0)` per k-tile. **No partial `vmcnt(N>0)` in the loop at all** —
the `vmcnt(23)` ladder and the `vmcnt(14)…(1)` cascade live in the
prefetch/scale/epilogue region. The in-loop `vmcnt(4)` in `down_readable.s` is
the hand-edit of the native `vmcnt(0)`. On the LDS axis we are already *finer*
than hipblas (rolling `lgkmcnt(7)→(0)` interleaved into the 32 MMAs vs its single
coarse `lgkmcnt(0)` per 24). The doc's "fine-grained vmcnt(4) waits" sentence was
an artifact of reading the delivered `.s`.

**2. The real knob is `num_stages` (= cross-k-step overlap), and it is worth 1.2%.**
`cp_async_wait<N>` comes from the generic `InjectSoftwarePipeline`; `N` is relaxed
by `PipelinedRetainGroups = num_stages - 1`. `ns=1` ⇒ `wait<0>` (full drain);
`ns=2` ⇒ `wait<1>`. Sweep at M=64 gate_up, same-process, interleaved, idle GPU 2
(within-config spread ≤0.5 µs, no round-to-round overlap):

| config | µs | vs base |
|---|---|---|
| bk256 ns1 t256 (old baseline) | 246.6 | 1.000 |
| bk256 ns2 t256 | 250.7 | 1.017 |
| bk128 ns1 t256 | 269.3 | 1.092 |
| **bk128 ns2 t256 (ADOPTED)** | **243.5** | **0.988** |
| bk128 ns3 t256 | 336.0 | 1.363 |
| bk64 ns1 / ns2 | 313.6 / 329.0 | 1.27 / 1.33 |
| bk128 ns2 t512 | 263.2 | 1.067 |
| bk128 ns2 bn128 | 250.6 | 1.016 |
| hipblas | 226.9 | 0.920 |

**The wait count is not the lever** — at fixed bk=256, `wait<0>`→`wait<1>` is
1.6% *slower*. The win is an interaction with `bk`: at bk=128 the ns=1→ns=2 swing
is **−25.8 µs**, at bk=256 it is **+4.1 µs**. Double-buffering lets the next
tile's loads run under the current tile's compute — the same mechanism hipblas
gets from its VGPR queue, in LDS. It needs the smaller tile (bk128 ns1 alone is
23 µs worse than bk256 ns1 — it syncs twice as often with no overlap), and
`bk128 ns2` sits at 32 KB LDS so 2 CTAs stay resident. Depth >2 is much worse.

**Adopted:** `_decode_cfg(64)` = `(64, 64, 128, 2, 256, 1)`. rel=0.00000 on all
four decode shapes; 8192³ untouched (433.9 TFLOPS). Gap to hipblas 1.087 → 1.073.

**Also fixed:** the builder only asserted `block_K % 64 == 0`, which does not imply
`K % block_K == 0`. `bk=192` on K=5120 compiles, runs, and is **silently wrong**
(rel=1.61 — 5120 % 192 = 128, so the tile copy runs past the end of B). Now
asserted. `bk=320 ns1` is correct at rel=0, so this is divisibility, not an
atom-width rule.

**Honest size:** 1.2%, not the 5.8% the in-scope framing hoped for. The overlap
story survives but the specific named lever does not exist. The residual 7.3% is
hipblas hiding B's latency in a deep VGPR queue while we must round-trip through
double-buffered LDS — a structural consequence of the mmac fragment layout, and
the same wall B-direct hit from the other side.

## 2026-10-17d — both B-fetch levers closed; the native-kernel structure is right, but the win is not there

Two subagents ran the two open levers to completion. **Both are negative**, and
B-direct's is provably negative, not just measured-negative.

**B-direct: fully wired, correct, and 2.1× too slow.** The dispatch is end-to-end
now and produces rel=0.00000. Full M=64 gate_up, same-process, idle card,
CUDA-graph replay (`/tmp/bdirect_probe.py`):

| config | µs | vs baseline |
|---|---|---|
| baseline — Square, B-via-LDS | **246.2** | 1.00x |
| FullCol, B-via-LDS (WarpM=1) | 273.9 | 1.11x |
| B-direct — Square (WarpM=4, 8-consec-k) | 904.2 | **3.67x** |
| B-direct — FullCol (WarpM=1, 8-consec-k) | 513.8 | **2.09x** |

So `WarpM=1` does remove the m-warp redundancy (904 → 514) — the 2026-10-17c
partition reading was not wrong — but it lands **2.1× above baseline**. The
loader-only probe (which measured only the `buffer_load`s) overstated the win; the
real kernel adds the mmac chain, A-MLS, the per-lane B fragment shuffle, and the
half-sector pattern across the actual 4-warp M-split, and the gain does not
survive them.

**The 16-consec-k fragment cannot rescue it — two independent proofs.**

1. *It cannot feed the mmac without a cross-lane repack.* The fp8 mmac B operand
   is hardware-fixed at `int32x2` = 8 B/lane (`src/tl_templates/hcu/gemm.h:109`,
   `__builtin_hcu_mmac_f32_16x16x32_fp8_fp8`); **no fp8 `16x16x64` exists** (the
   only `16x16x64` builtins are fp4 `scale_buffer.hpp:545` and ROCm i8). So
   16-consec-k is a *load* shape only. The two loaders in
   `tilelang_ds_read_format.hpp` have provably different per-lane ownership:
   `b_vgpr_direct_load` (8-consec-k) → lane owns 16 n-rows × 4 k-offsets, chunks
   32 apart, and its comment says it targets the **byte-identical**
   `ds_read_format_tensor_b` layout; `b_vgpr_direct_load16` → lane owns 32 n-rows
   × 64 *contiguous* k, and its own comment says that is **DIFFERENT** from the
   mmac fragment. Converting one to the other needs an 8-byte `lane^16` exchange
   per 16 B chunk.
2. *The arithmetic bound settles it without building the repack.* The 8→16-consec
   swap can only recover the sector penalty = loader-only `d8b_1w − d16b_1w` =
   217.8 − 123.9 = **93.9 µs/pass**. The B-direct FullCol gap to baseline is
   513.8 − 246.2 = **267.6 µs**. 93.9 < 267.6 ⇒ best case ~419.9 µs, still
   **173.7 µs above baseline**. Independent of the overlap model.

**MLS producer: closed negative.** The 128×16 atom is ~20% slower than 64×32 in
isolation (1600 vs 1892 GB/s), but that gap is already hidden by the pipeliner —
swapping to the faster atom makes the full GEMM **15–16% slower** (287.2 vs
246.2 µs, both rel=0.000000). The kernel is **overlap-quality-bound, not
producer-throughput-bound**. No source change.

**Net:** the native-kernel reading from 2026-10-17c (B→VGPR global-direct, no M in
the B index, waves split N) is still *structurally* correct and worth keeping as
the description of what hipblas does — but it does **not** transfer into a win in
the current tilelang structure. The remaining gap to hipblas at M=64 is overlap
quality (see the M=64 section of the doc), not B-fetch throughput. Do not re-open
either lever without new evidence. *(2026-10-17e: the overlap lever was then
attacked — see above — worth 1.2%, and the "5.8%" quoted here is the contended
figure; idle-card it was 10.2%, now 7.3%.)*

**Code state:** the B-direct dispatch (`gemm_hcu_mmac.py` `_gemm_mls_b_direct` +
`tl.hcu_b_vgpr_direct` gate + the `pass` branch in `infer_layout`) and the three
loaders are complete and correct but non-winning — keep on this branch as
closed-experiment infrastructure, **do not merge**. The `"+v"` → `"=&v"` fix at
`tilelang_ds_read_format.hpp:557` is a real correctness fix.

## 2026-10-17c — B-direct re-opened: the 4x was the partition, not the fetch

> **SUPERSEDED by 2026-10-17d (above).** The native-kernel reading below is
> structurally correct and worth keeping, but its *actionable* conclusion ("the
> retry shape is WarpM=1 + 16-consec-k, so B-direct is the larger lever") is
> falsified by measurement: WarpM=1 B-direct is still 2.1× slower than baseline,
> and the 16-consec-k fragment is both mmac-incompatible without a cross-lane
> repack and arithmetically insufficient. Read this section as a *description of
> what hipblas does*, not as a plan.

Re-read the **native, unpatched** hipblaslt kernels (the `deliverable_*.s` files
are *patched* — Layout A / slc / vmcnt4 hand-edits — and the deliverable README
itself warns "operand 角色不能跨 M 复制"). Two native witnesses now, both showing
the same structure:

- **M=64 down** — extracted from the pristine pre-patch library
  (`/opt/dtk/lib/hipblaslt/library_bw_nmz/FusedMatmulKernel_gfx938.co.bak_0916_1000`
  via `llvm-objdump -d --mcpu=gfx938`): `MT96x64x256_TT6_2_WG16_32_1_MFWGS512_WGM8`.
- **MT256x128x128** — the readable Tensile `.s` in `/workspace/misc/`.

Both confirm: **weight B is loaded `buffer_load_dwordx4 … idxen offen` straight
into the MMA operand block (M=64: `v[96:127]`; MT256: `vgprValuB_X0_I0`), no
`lds` modifier — B never touches LDS. A is staged through LDS.** The B index is
built from `lane%16` alone (M=64: `v_mul_lo_u32 v184, s78, v6`, `v6=Serial&15`;
MT256: `v6=(lane%16)+wave*32`, `v9=(lane/16)*16`) — **no M coordinate** — so the
4 waves split **N**, each B byte is fetched **once** (cross-check: M=64 down
baseline 131.8 µs / 131.1 MB = 995 GB/s, impossible under 2×/4× replication), and
B's reuse across the whole M extent happens **in-register** (one 8-byte B operand
→ 8 mmacs in MT256, 18 in M=64).

**This reverses the 2026-10-17b conclusion.** Our measured 3.62× is a **warp-
partition artifact**, not a property of B-direct: our M=64 kernel is `WarpM=4`,
so 4 independent waves each issue their own identical B loads. hipblas has no such
term because its B index has no M coordinate. B-direct is the **larger** lever
(123.9 vs 164.0 µs/pass = 40 µs, vs the MLS producer's 33 µs). The retry shape is
**`bm64 bn64` with WarpM=1, WarpN=4, t128** (each warp = all 64 M × 16 N; B fetched
once; A redundant across n-warps but 327 KB / L2-resident and stays on LDS).
Winning still needs the **16-consec-k fragment** too (8-consec-k at 1× redundancy
still loses: 217.8 vs 164.0 µs/pass). Both are in-repo generator changes
(`ComputeMlsWarpPartition` `src/hcu/op/mls.cc:506`, `AnalyzeResourceAxis`
`src/hcu/transform/hoist_mls_resource.cc:1615`, + the mmac B-operand/A-index
wiring), **not** hand-written C++, and **not** blocked by the excluded weight
reorder (Layout A). The WIP `b_vgpr_direct_*` loaders on this branch are the
foundation for the retry, not dead weight.

Also corrected in the doc: the M=1 "SLC" bullet (0 native loads carry `slc`; the
9 in the delivered M=64 file were added by the patch) and the "91.5% read
ceiling" (the real ceiling is ~2016 GB/s, not 1646 — the 512 MB probe was too
short; our standing vs the ceiling is *worse* than the 78% framing implies, and
headroom is *larger*).

## 2026-10-17b — B-direct-VGPR is a dead end; root cause is 4x m-warp redundancy, not the scheduler

Confirmed today (reproduced from first principles in `/tmp/bdirect_bench.cu` +
`stride_test.cu` + `asm16ec.cu`):

1. **The opaque-asm "one big block" fix is real but not enough.** Multi-load
   inline asm VMFaults without `=&v` (early-clobber) because LLVM aliases an
   output VGPR pair over a later input (the next load's address). 2 loads work,
   3+ fault. With `=&v`, 16 loads issue correctly (SASS-verified: distinct output
   regs, one `s_waitcnt vmcnt(0)`). Raw BW matches `hipMemcpyD2D` at ~1319 GB/s
   for contiguous patterns and ~1318 GB/s for strided patterns — the hardware
   handles strided buffer_load fine, and the aillvm re-drain was specific to
   the tracked `hcu_buffer_load_impl` path (which we never need to use again).
2. **Why the full GEMM is still 71x slower than control:** the 4 m-warps per
   n-warp each issue their own 16 buffer_loads of identical B data → **4x
   global traffic** plus VGPR pressure (101 vs 82 VGPR) plus register-move
   overhead. LDS staging exists precisely to coalesce 4 global reads into 1.
   The 8-consec-k fragment then adds a further ~2x of sector inefficiency vs
   hipblas's 16-consec-k. Combined, B-direct cannot beat the MLS path — it can
   only lose by a lot.
3. **To get hipblas-class B-direct you need two things we don't have:**
   (a) a 16-consec-k B fragment (requires full MMAC operand rewrite, not just
   a loader swap), and (b) WarpM ≈ 1 so the m-warp redundancy cost goes away
   (requires a wider tile / more n-warps). Both are out of scope for "optimize
   within existing tilelang".

**Action: do not merge the B-direct WIP.** The 6 files of diff
(`tilelang_ds_read_format.hpp` + 3 emitter/gemm_op Python files + docs +
handoff) are the record of the attempt; the header additions are the
infrastructure for a future fragment-rewrite effort only. Don't turn the
annotation on.

### The real lever: the MLS producer (33 us/pass headroom)

From [[hcu-b-direct-vgpr-closed]]: `matrix_load_128x16_b8` global→LDS runs at
1599 GB/s (79% of achievable) vs 2002 GB/s (99%) for a plain copy of the same
tile. 33 us/pass of headroom, vs a 23 us total gap to hipblas. The MLS
partition heuristic (`ComputeMlsWarpPartition` in `src/hcu/op/mls.cc` +
`hoist_mls_resource.cc`) is the only in-scope lever left. Requires a C++
rebuild.

### The 5.8% was a contention artifact

`hy-smi` shows co-tenant load varies hugely between sessions. Measured on three
load levels:

| machine state | hipblas | ours | ratio |
|---|---|---|---|
| idle (2026-10-17, all HCU 0%) | **224.2us (64.9%)** | **247.0us (58.9%)** | **1.102, stable ±0.2% / 8 rounds** |
| mild (GPU4, 10-16c) | 257.0us | 272.0us | 1.058 |
| heavy (GPU2, co-tenant 50-100%) | 276-293us | 284us | ~1.00 ("parity" — false) |

Contention disproportionately hurts hipblas's deep in-flight queue, so the gap
SHRINKS as the machine gets dirtier. **Never conclude "gap closed" on a loaded
card. Check `hy-smi` (HCU% column must be ~0) before trusting absolute times or
even ratios.** Quiet-machine reference now: 224.2 / 247.0 (supersedes the
231.7 / 247.0 pair; hipblas's 213-226us range in the older docs was also
idle-machine).

### Where the 23us goes (idle-card, paired load-only decomposition)

- B-only load: 166.6us = 1574 GB/s = 87% of the 1800 GB/s ceiling — **the load
  path is not the bottleneck**.
- A+B load: 195.0us. A's cost ≈ 28us, of which ~13us is A loaded once and ~15us
  is A re-loaded per k-iteration (800 blocks × 20 iters re-reading the 327 KB
  L2-resident matrix; LDS-size/occupancy hypotheses refuted by a dummy-alloc
  control at −0.2us).
- Full kernel: 247.0us → compute (ds_read+mmac) exposes ~52us.
- hipblas total: 224.2us. Gap 22.8us.

### hipblas M=64 k-loop structure (SASS from the delivered `down_readable.s`)

See the table in `docs/optimization/fp8_decode_m_le_128.md` (section "hipblas
M=64 k-loop vs ours"). The three load-bearing differences: (1) the weight is
loaded **direct global→VGPR** with slc (16B dwordx4), never through LDS; (2)
a **vmcnt ladder (vmcnt(23)→0) drains once per 2-chunk window** → ~20+ loads
in flight; (3) only the small activation operand uses LDS, so the B side is
barrier-free. Our 10-15 `b_vgpr_direct` attempt matched (1) with 8B loads but
the aillvm backend re-drained the queue before each 4-mmac group (rolling
vmcnt(3)) → 4× regression.

### Outcome (2026-10-16): B-direct is closed by measurement, and the lever moved

The annotation-gated B-direct path was built and measured. It is **4.0x slower**
(975.4us vs 243.4us LDS control) even after two real bugs were fixed (VMFault
from missing early-clobber `"=&v"` on multi-load inline asm; a scratch spill from
a dynamic slice index). Isolated loader kernels show why, and it is arithmetic
rather than a backend failure: our block is t256 = 2 n-groups x **4 m-warps**
(`nbase = (warp_id / 4) * 32`), so every B byte is fetched 4x. Predicating the
loads to one m-warp per n-warp (1x reads) removes **3.62x** of the cost on the
identical pattern. LDS staging is not overhead — it is the mechanism that turns
four global fetches into one. Our 8-consec-k fragment then adds an independent
**2x** vs hipblas's 16-consec-k (half-sectors vs full sectors). See
`docs/optimization/fp8_decode_m_le_128.md`, "B-direct-to-VGPR: what it costs".

B-direct does win in exactly one configuration — 16-consec-k fragment (`dwordx4`)
**and** WarpM=1, measured at 123.9us/pass vs 164.0us/pass for the MLS producer —
but that needs a fragment rewrite plus a different warp arrangement (t128 4
n-warps, or BN=128 with 8), and A must stay on MLS/LDS since it goes redundant
across n-warps under WarpM=1.

### Where the 23us actually is

Not the fetch pattern, and not the barrier. The B tile fetch pattern (64 rows x
256 B, rows 5120 B apart) reaches 1881-2016 GB/s with a plain copy — and the
"1646 GB/s read ceiling" in earlier notes is wrong; a grid-stride read measures
**2016 GB/s**. The loss is inside the producer: `matrix_load_128x16_b8`
global->LDS delivers **164.0us/pass (1599 GB/s, 79.3%)** where a plain
`buffer_load_dwordx4` copy of the same tile delivers **131.0us/pass (2002 GB/s,
99.3%)**. Not the per-k-step barrier (plain + same barrier = 2013 GB/s), not
pipeline depth (2-deep double-buffered MLS is *worse*, 1456 GB/s).

**33us/pass of headroom against a 23us total gap.** The MLS partition/axis is
shape-dependent (`ComputeMlsWarpPartition`, `src/hcu/op/mls.cc:506`): bm64 bn64
bk256 gives `sequence<64,256>, sequence<16,128>, 2, 2, ... mn`, bk512 gives
`1, 4, ... k`, bn128 gives `2, 2, ... k`. That is the next lever, and it is a
C++ change. The WIP B-direct lowering code (header +181, emitter +44, macro
generator +118, `gemm_op.py` +6/-2) is now dead weight — revert or keep as
reference, but it does not go in.

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

**The search space is exhausted as of 2026-10-19.** Every axis has now been
measured and closed, and the last one — `warp_k` / `FullColK` on the MLS path —
was implemented (it had three separate hard guards) and turned out to be
neutral-to-negative. See the doc's 2026-10-19 section.

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
   pending in `decode_cfg`; the o/down rows are `bn16/bk256/ns1/t64` at M<=16 and
   `bn32/bk256/ns1/t256` at M=32, both with `setprio=1`).
2. Anything further needs a new *mechanism* — a producer that is not MLS, or an
   ISA path that avoids the LDS round trip — not another sweep.

Do **not** re-run the wide-bn, nt-shared-A, split-K, `ns>1`, rolling-window or
B-direct sweeps; they are settled negative.

## 2026-10-19 — the o/down gap, located and closed (supersedes "not a config search")

The gap to hipblas **has** been found and it is arithmetically closed, not an
unfound knob. Full writeup: `docs/optimization/fp8_decode_m_le_128.md`,
section "2026-10-19 — o/down: the gap is an LDS-capacity invariant".

Three facts to carry forward:

1. **Occupancy cliff.** Same kernel, only the dynamic smem size changed:
   12288 B → 5 blk/CU → **93.7 µs**; 14336 B → 4 blk/CU → **168.9 µs**. 5→4
   blocks/CU costs 80%. The shipped config is exactly 320 blocks = 64 CU × 5 =
   one wave, on a knife edge. Any LDS increase is catastrophic.
2. **The invariant.** `total = const × w × (L + C·ns)`, where `w` = MLS packed /
   logical (bk128 → 2.5, bk256 → 1.5, bk512 → 1.0), `L ≈ 0.79 µs` = one DRAM
   round trip per warp-iteration, `C ≈ 0.15 µs` = the exposed
   `ds_read_matrix_trans_format ×8 → mmac ×8` chain. **bk and ns cancel** — more
   stages ⟺ fewer blocks/CU ⟺ more waves. This is *why* `ns>1` and split-K are
   neutral, and it predicts bk64/bk128/bk256 within 10%.
3. **SASS root cause.** The four `matrix_load_128x16_b8` producers are emitted
   *after* the mmac and immediately before the branch, and the next iteration
   opens with `s_waitcnt vmcnt(0)`. The full load latency is exposed every
   iteration. Decomposition: dropping mmac saves 9.4%, dropping the ds_reads 9.3%,
   dropping **both** 9.8% — one serialized chain. The pure load path alone is at
   ~1.02× hipblas, so the load machinery is already at the ceiling.

**Only landed-quality change available: rotate the MLS producer above the
`promote_prio`/mmac block** (~3%, reproduced 3×; the compiler inserts the needed
`s_waitcnt lgkmcnt(0)`, so it is not racy). It needs an emitter/pass change — the
producer placement comes from the MLS lowering, not a config flag. Note this
only overlaps the mmac portion; the ds_read portion cannot be hoisted above the
load issue without a second buffer, and a second buffer is exactly what the LDS
capacity wall forbids.

## 2026-10-20 — fat-tile decode config landed; the kernel cache serves stale code

**Landed (working tree, uncommitted).**

1. `src/hcu/op/mls.cc` — the fp8-trans `(16,128)` packed branch now packs densely
   when the dst tile is a whole number of 4K slots. A 16x256 tile is exactly
   4096 B, so `MlsPackedTileBytes` returns -1 (contiguous) and the allocator
   stops reserving 6144 B for it. bk=128 keeps the conservative 5120-B slot
   layout; bk=384 is 3 slots and also keeps it.
2. `perf/gemm/fp8_channelwise_mls.py` — `decode_cfg` returns
   `bn96/bk256/ns2/t384` for `N == 5120 and K % 256 == 0`; both `ns>1` guards
   allow exactly that one wide config. qkv/gate_up keep bn32.

**Cold-cache re-validation** (cache cleared, card 0 idle, one process, paired,
5 rounds, medians; `old` = the pre-landing bn16 chooser):

| shape/M | old | ship | hipblas | ship/old | ship/hip |
|---|---|---|---|---|---|
| o 1 | 38.88 | 36.80 | 32.96 | 0.947 | 1.117 |
| o 16 | 39.84 | 37.44 | 35.20 | 0.940 | 1.064 |
| o 32 | 47.84 | 45.12 | 39.68 | 0.943 | 1.137 |
| down 1 | 100.00 | 92.80 | 78.40 | 0.928 | 1.184 |
| down 16 | 100.80 | 93.28 | 84.80 | 0.925 | 1.100 |
| down 32 | 125.44 | 116.48 | 109.76 | 0.929 | 1.061 |

All rel 0.0 (down M=32 5.9e-08). So the wide tile is a stable 5-7% win over the
old chooser; o/down are still 6-18% behind hipblas.

**The kernel disk cache is blind to codegen changes — this cost the afternoon.**
`~/.tilelang/cache/<version>/linux-x86_64/kernels/<hash>/` stores
`device_kernel.cu` + `executable.so` keyed on TIR/options only. After the
2026-10-19 emitter changes (`ds_read_format.cc` gained the WarpN template arg,
`hcu_mmac_*` Python emitters changed), old entries kept being served:
`test_warp_k_fullcolk_mls_is_correct_and_reduced` saw
`ds_read_format_tensor_a<..., 1, 1, tl::fp8_t, ...>` — a call the current
header cannot even compile — while `TILELANG_DISABLE_CACHE=1` emitted the
expected `..., 1, 4, 1, tl::fp8_t, ...`. Clearing
`~/.tilelang/cache/0.1.12_cuda_git0a92729d/linux-x86_64/kernels/*` fixed it.
**Rule: after touching any emitter/lowering/header, clear that directory before
trusting a test or a timing.** Purely numeric changes (LDS sizes) change the TIR
hash and invalidate themselves; format-string and header changes do not.

**Test status after the cache clear: 18 passed, 3 xfailed, 1 xpassed.** The
xpass is `bm16/bn16/bk256/ns4/t64`, now exact *because* dense packing makes the
stage stride equal the logical stride (bk128 is still broken: 5120-B packed
stride against a 2048-B tile). `test_mls_packed_stride_is_shared_by_producer_and_consumer`
was updated accordingly: `mls_packed_tile_elems(16, 256, 16, 128, True, fp8)` is
now 0 (dense == contiguous), and a bk=384 case pins the 7168-B slot fallback.

### 2026-10-20b — 3 wavefronts, not 6: the A producer replicates per warp

The wide tile's intended mechanism (cut A's re-read 320x -> 54x) never happened.
SASS shows the A `matrix_load` address has **no warp-id term** (B's does: warp id
mod 6, offset warp*2048), so every warp loads the same A. A's re-read is
blocks x warps = N/16 warp-loads **independent of the tile** — 54x6 is the same
as the old 320x1. That is why the fat tile bought 5-7% and not 1.7x.

Warp count at fixed bn96/bk256/ns2 (legal counts {1,2,3,6}; paired, 5 rounds,
idle card 0):

| warps | t384 | **t192** | t128 | t64 |
|---|---|---|---|---|
| vs t384 | 1.000 | **0.952-0.962** | 0.971-0.983 | 1.23-1.29 |

U-shaped: fewer warps means less A traffic but also less latency hiding.
Landed `decode_cfg` N==5120 -> t192 (was t384). Against hipblas, M=1/16/32:
o 1.069/1.020/**1.000**, down 1.110/1.053/**0.935** — o ties and down wins at
M=32. All rel 0.0 (down M=32 5.9e-08). Tests: 18 passed, 3 xfailed, 1 xpassed.

Next: de-replicate the producer in `src/hcu/op/mls.cc` (one warp loads A, or
partition its copy across warps) to keep 6 warps without paying 6x; A is ~1/3 of
per-step per-block bytes, so the bound is ~28% of per-step traffic.

### 2026-10-20c — that lever was built and is CLOSED

Implemented as an issue gate in `tilelang_mls_base::{async_mls_load_asm_impl_ct,
_impl_rt}` (canonical owner per axis: `cluster_idx[ii] < EffectiveWarpCluster[ii]`,
covering full *and* partial replication). Bit-exact, and slower everywhere
(paired, idle card, base = the t192 landing above):

| shape / M | t384 | t192 |
|---|---|---|
| o M=16 / M=1 | 1.145 / 1.044 | 1.265 / 1.286 |
| down M=16 / M=1 | 1.190 / 1.071 | 1.347 / 1.358 |

Root cause (SASS): `cp_async_wait<1>` sizes its `vmcnt` constant from the
*unconditional* async loads; putting A behind `if (owner)` drops it out of the
count, so the wait becomes one group too strict (t192 `vmcnt(8)->(4)`, t384
`(4)->(2)`) and the prefetch level the whole kernel depends on disappears.
22% less traffic never pays for that.

Two corrections: t384's A handle is `WarpMN=3, WarpK=2` (K already split) so A
is only 3x replicated there, not 6x — the fully-replicated rule emitted
byte-identical SASS at t384; and A's duplicates are L2-served, so they were
never on the step-time critical path. Header change reverted (SASS verified
identical to the pre-change build). Standing after revert: o 1.069/1.027/1.024,
down 1.110/1.042/**0.928** at M=1/16/32, all exact.

## Repo hygiene

- Keep `/tmp` experiments out of the repo unless they become a committed bench.
- Commit message footer: `Co-Authored-By: Claude Code <noreply@anthropic.com>`.
