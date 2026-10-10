# MQA Logits — MLS (matrix_load) Optimization

Target: HCU gfx938 (BW1101, 64 CU, warp 64). Beat lightop in all scenes.

## Baseline (T.copy, before MLS)

### mqa_logits (tilelang vs lightop)
| Scene | S | SKV | H | tilelang | lightop | speedup |
|---|---|---|---|---|---|---|
| tiny | 128 | 256 | 32 | 0.097 | 0.122 | 0.79x |
| small | 128 | 2048 | 32 | 0.097 | 0.122 | 0.80x |
| small_h64 | 128 | 2048 | 64 | 0.096 | 0.216 | 0.44x |
| medium | 1024 | 4096 | 32 | 0.225 | 0.215 | 1.05x |
| medium_h64 | 1024 | 4096 | 64 | 0.364 | 0.394 | 0.92x |
| large | 4096 | 8192 | 32 | 2.131 | 0.725 | 2.94x |
| large_h64 | 4096 | 8192 | 64 | 4.173 | 1.287 | 3.24x |
| xl | 4096 | 32768 | 32 | 2.386 | 1.108 | 2.15x |

### paged_mqa_logits (tilelang vs lightop)
| Case | B | H | avg_ctx | tilelang | lightop | speedup |
|---|---|---|---|---|---|---|
| bs1_H32_4k | 1 | 32 | 4096 | 0.074 | 0.065 | 1.15x |
| bs64_H32_4k | 64 | 32 | 4096 | 0.187 | 0.136 | 1.37x |
| bs128_H32_4k | 128 | 32 | 4096 | 0.327 | 0.235 | 1.39x |
| bs64_H64_4k | 64 | 64 | 4096 | 0.343 | 0.163 | 2.11x |
| glm51_bs32_8k | 32 | 32 | 8192 | 0.183 | 0.119 | 1.54x |
| glm51_bs64_4k | 64 | 32 | 4096 | 0.188 | 0.135 | 1.39x |

## Key findings
1. MLS = the hardware instruction behind `T.matrix_load` (global→LDS async).
2. The gemm's LDS read only switches to the MLS-aware `ds_read_format` when BOTH
   gemm operands come from matrix_load. Single-operand MLS → gemm reads with the
   plain swizzled layout → wrong data. So Q must also be matrix_load.
3. paged: Q/KV must be passed as 2D views; a multi-dim dynamic leading index gets
   baked into the hoisted MLS constructor and fails codegen (`phys_block` undefined).
4. **Async-wait rule (the correctness bug):** MLS is async — `async_mls_load_asm`
   returns before the data lands in LDS. The pipeline pass only inserts
   `cp_async_wait<N>` for copies it manages. A **direct `T.matrix_load` inside
   `T.Pipelined` is NOT managed** → 0 waits → gemm reads stale/partial LDS → huge
   error (small scenes pass by luck, large scenes fail). The fix is to use
   `T.copy(..., prefer_instruction="matrix_load")` so the copy lowering marks it as
   a pipeline-managed async copy and the pass emits the wait groups.

## Correctness status (FIXED ✅)
- [x] mqa_logits: 14/14 pass (~6e-15).
- [x] paged_mqa_logits: 22/22 pass (~3e-5, incl. GLM5.1 200k context).

### Fix (MLS instruction kept unchanged)
1. **Pipelined path (num_stages≥1):** replace `T.matrix_load(...)` with
   `T.copy(..., prefer_instruction="matrix_load")` for both Q and K/KV loads, so the
   pipeline pass manages the async commit/wait groups (generated waits `[1,1,0]`).
2. **FullRow path (paged, num_stages=0, bs≤4):** not software-pipelined, so the
   pipeline pass inserts no wait. Add explicit `T.ptx_wait_group(0)` +
   `T.sync_threads()` after the K/KV load (matches the reference test's
   `num_stages==0` convention). Zero perf cost: the gemm must wait for the data
   anyway, and num_stages=0 has no compute to overlap with the load.

## MLS perf (CORRECT, same-device A/B on HCU 2)

### mqa_logits: baseline (T.copy) vs MLS
| Scene | S | SKV | H | baseline | MLS | speedup |
|---|---|---|---|---|---|---|
| medium | 1024 | 4096 | 32 | 0.228 | 0.193 | 1.18x |
| large | 4096 | 8192 | 32 | 2.134 | 1.706 | 1.25x |
| large_h64 | 4096 | 8192 | 64 | 4.174 | 3.002 | 1.39x |
| xl | 4096 | 32768 | 32 | 2.388 | 1.974 | 1.21x |

### mqa_logits: MLS vs lightop (still behind in compute-bound scenes)
| Scene | tilelang(MLS) | lightop | lightop/tl |
|---|---|---|---|
| medium | 0.189 | 0.216 | 0.88x (tl faster) |
| large | 1.696 | 0.711 | 2.38x (lightop faster) |
| large_h64 | 2.992 | 1.271 | 2.35x (lightop faster) |
| xl | 1.910 | 0.986 | 1.94x (lightop faster) |

### paged_mqa_logits: MLS vs lightop (roughly at parity)
| Case | B | H | avg_ctx | tilelang(MLS) | lightop | lightop/tl |
|---|---|---|---|---|---|---|
| bs1_4k | 1 | 32 | 4096 | 0.166 | 0.178 | 0.93x (tl faster) |
| bs64_4k | 64 | 32 | 4096 | 0.293 | 0.324 | 0.90x (tl faster) |
| bs128_4k | 128 | 32 | 4096 | 0.383 | 0.432 | 0.89x (tl faster) |
| bs64_h64_4k | 64 | 64 | 4096 | 0.399 | 0.367 | 1.09x (lightop faster) |
| glm51_bs1_16k | 1 | 32 | 16384 | 0.140 | 0.180 | 0.78x (tl faster) |
| glm51_bs32_8k | 32 | 32 | 8192 | 0.300 | 0.314 | 0.95x (tl faster) |

**Takeaway:** MLS is correct and gives ~1.2–1.4x over the T.copy baseline. paged is
roughly at parity with lightop (several scenes already beat it). mqa large scenes are
still ~2.3x behind lightop — that's the target for round 2.

## Round 2 — close the mqa large-scene gap

### Diagnosis
- mqa is **compute-bound** (causal time is independent of SKV; ~42 TFLOPS at the
  original block_N=32 config). lightop does `large` in 0.711ms ≈ 97 TFLOPS.
- Config sweep (block_N × block_Q × num_stages) shows **block_N is the dominant
  factor** (the MMA M-tile). block_N=128 only fits with num_stages=0
  (128·D·2 = 32KB for K alone).

### Fix 1: `_pick_block_config` priority
Old priority was block_Q > num_stages > block_N (picked block_N=32, ns=2). New
priority is **block_N > block_Q > num_stages**, which picks:
- H=32: block_N=128, block_Q=4, ns=0 (64KB)
- H=64: block_N=128, block_Q=2, ns=0 (64KB)

### Fix 2: ns=0 async wait (same rule as paged FullRow)
block_N=128 forces num_stages=0 → not software-pipelined → no `cp_async_wait`.
Added `T.ptx_wait_group(0)` + `T.sync_threads()` after the K load when
`num_stages == 0`. Correctness re-verified: 14/14 pass.

### Result (mqa, same device HCU 2)
| Scene | before (bn32,ns2) | after (bn128,ns0) | lightop | lightop/tl |
|---|---|---|---|---|
| medium | 0.189 | 0.147 | 0.219 | 0.67x (tl faster) |
| medium_h64 | 0.276 | 0.203 | 0.400 | 0.51x (tl faster) |
| large | 1.696 | 1.067 | 0.730 | 1.46x (lightop faster) |
| large_h64 | 2.992 | 1.898 | 1.288 | 1.47x (lightop faster) |
| xl | 1.910 | 1.321 | 1.111 | 1.19x (lightop faster) |

Gap in large scenes closed from ~2.3x to ~1.2–1.5x. mqa now at ~71–81 TFLOPS (was 42).
Small/medium scenes all beat lightop. Remaining target: large scenes.

### Why lightop is still faster in large scenes: it uses FP8
- lightop `mqa_logits` takes a `kv_scale` arg (gemmopt.py:679) and the paged variant
  has an explicit `packed_fp8_path` (gemmopt.py:766, `fused_kv_cache.dtype == uint8`,
  `shape[3] == 132`). **lightop quantizes Q/K to fp8 and runs the gemm in fp8.**
- lightop `large` ≈ 0.711ms ≈ 94 TFLOPS — *above* the bf16 MMA peak (~72 TF). That's
  only reachable with fp8 (fp8 MMA ≈ 2x bf16 on gfx938).
- Our bf16 kernel is already at ~71 TF (near bf16 peak), so config/policy tuning can't
  close the remaining gap. **The only way to beat lightop in the compute-bound large
  scenes is to convert the mqa gemm to fp8** (quantize Q/K, apply kv_scale, fp8 MMA).
- Policy (Square vs FullRow) makes no difference at block_N=128 (both ~71 TF).

### Round-2 decision point
- bf16 path: done, correct, ~1.2–1.5x behind lightop in large scenes (at bf16 peak).
- fp8 path: the lever to actually beat lightop in large scenes. Bigger change (Q/K
  quantization + scale + fp8 gemm) and lower precision — needs a tolerance check.
