# HCU fp8 Channelwise Decode GEMM — Handoff

Status: **2026-10-15** (overnight update below) / 2026-10-13. Branch
`feat/hcu-mqa-fp8`. Goal: beat hipblaslt fp8 on

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

1. **slc on MLS matrix_load atoms** — `src/tl_templates/hcu/mls/tl_mls_atom_gfx938.hpp`:
   every `__builtin_hcu_matrix_load_*` ends with `..., false, false` — the
   glc/slc bools. Flip to `slc=true`. (gfx946 twin
   `tl_mls_atom_gfx946.hpp` shows the store path already passes glc/slc, so the
   builtin signature supports it.) Check whether slc is legal on gfx938
   matrix_load or if it belongs on the plain buffer loads instead.
2. **s_setprio around MMA** — `src/tl_templates/hcu/core/arch/arch.hpp:242-261`
   already has `promote_prio()` (s_setprio(1)) / `restore_prio()` /
   `set_prio<Prio>()`. Wire into the mmac loop (`gemm_hcu_mmac.py`) or the atom.
3. **A direct-to-VGPR with real dwordx4** — if we keep the VGPR-A idea, the load
   must be 16 B (`hcu_buffer_load<fp8_t, 16>` → `i32x4`), non-blocking/pipelined,
   with slc — i.e. exactly `buffer_load_dwordx4`. `hcu_buffer_load_impl` at
   `src/tl_templates/hcu/core/arch/amd_buffer_addressing.hpp:1352` already has a
   `coherence` template param (glc/slc), and N=16 gives i32x4 = dwordx4.
   Currently unused for the r_mls path (comes out N=8).
4. **pipeline depth does NOT help** — ns3/4 measured no better than ns2 at
   M=64/128; ns4 marginally helps M=32. bk32 is illegal (MLS trans-B b8 tile
   needs block_K % 64 == 0).

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

1. **slc injection** on MLS atoms (lever 1) — cheapest, one header, rebuild,
   measure M=32/64. Watch for compile error on gfx938 builtin slc (fallback:
   plain buffer-load slc in the MLS path, or per-lever test).
2. **s_setprio around MMA** (lever 2) with the same rebuild/measure loop.
3. If still short: **dwordx4 direct-A** experiment (lever 3) — make the r_mls A
   load 16 B + slc + pipelined, re-run `fp8_channelwise_decode_bench_r_mls.py`
   to see if the VGPR-A hipblas shape finally wins.
4. Combine winners; lock configs into `fp8_channelwise_mls.py` +
   `fp8_channelwise_decode_bench.py`; commit with writeup.

## Repo hygiene

- Keep `/tmp` experiments out of the repo unless they become a committed bench.
- Commit message footer: `Co-Authored-By: Claude Code <noreply@anthropic.com>`.
