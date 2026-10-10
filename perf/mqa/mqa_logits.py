#!/usr/bin/env python3
"""
Benchmark: tilelang mqa_attn_return_logits vs lightop (HIP ASM) mqa_logits.

Usage:
    python3 mqa_logits.py
    python3 mqa_logits.py --S 4096 --SKV 8192 --H 32 --D 128
    python3 mqa_logits.py --scene small
    python3 mqa_logits.py --scene all --check
    python3 mqa_logits.py --validate
    python3 mqa_logits.py --csv results.csv

Dependencies (GPU server):
    - tilelang
    - lightop
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import tilelang
from tilelang import language as T
import torch


# ============================================================================
# tilelang kernel: mqa_attn_return_logits (BF16)
# ============================================================================

# DCU shared memory limit: 64 KB
LDS_LIMIT = 64 * 1024


def _pick_block_config(heads: int, index_dim: int):
    """Pick block_N, block_Q, num_stages that fit within 64KB shared memory.

    No D-splitting.  Q loaded as [block_Q * heads, D], K as [block_N, D].

    Priority: larger block_N (the MMA M-tile — the dominant throughput factor on
    gfx938), then larger block_Q (fewer grid launches), then more num_stages.
    block_N=128 only fits with num_stages=0 (128*D*2 = 32KB for K alone), which is
    fine: the kernel is compute-bound, so the async-load overlap that pipelining
    provides is worth less than the bigger MMA tile.
    """
    D = index_dim
    H = heads

    best = None
    for bn in (128, 64, 32):
        for bq in (4, 2, 1):
            q_smem = bq * H * D * 2
            remaining = LDS_LIMIT - q_smem
            if remaining < 0:
                continue
            for ns in (2, 1, 0):
                k_smem = (ns + 1) * bn * D * 2
                if k_smem <= remaining:
                    total = q_smem + k_smem
                    if best is None:
                        best = (bn, bq, ns, total)
                    else:
                        bN0, bQ0, ns0, tot0 = best
                        if bn > bN0:
                            best = (bn, bq, ns, total)
                        elif bn == bN0 and bq > bQ0:
                            best = (bn, bq, ns, total)
                        elif bn == bN0 and bq == bQ0 and ns > ns0:
                            best = (bn, bq, ns, total)

    if best is None:
        best = (32, 1, 0, 0)

    block_N, block_Q, num_stages, total = best
    return block_N, block_Q, num_stages, total


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def mqa_attn_return_logits_tl(
    heads: int,
    index_dim: int,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 256,
    block_Q: int = 1,
    policy: str = "square",
):
    """TileLang BF16 MQA logits kernel.

    Architecture (matches paged_mqa_logits):
      1. Load Q [block_Q * heads, D] and weights [block_Q, heads] into smem/fragment
      2. For each block_N tile of KV:
         - T.clear accumulator, load K tile into smem (pipelined)
         - T.gemm: [block_N, D] × [block_Q * heads, D]^T → [block_N, block_Q * heads]
         - relu×weight, reduce_sum → [block_N, block_Q]
         - Write to global Logits
    """
    D = index_dim

    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32

    # k_pack=1 for RDNA (gfx936); k_pack=2 would be for CDNA
    K_PACK = 1

    _policy = T.GemmWarpPolicy.FullRow if policy == "full_row" else T.GemmWarpPolicy.Square

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    index_q_shape = [seq_len * heads, D]
    index_k_shape = [seq_len_kv, D]
    logits_shape = [seq_len, seq_len_kv]

    @T.prim_func
    def kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),
        IndexK: T.Tensor(index_k_shape, dtype),
        Logits: T.Tensor(logits_shape, accum_dtype),
        Weights: T.Tensor([seq_len, heads], accum_dtype),
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            q_smem = T.alloc_shared([block_Q * heads, D], dtype)
            k_smem = T.alloc_shared([block_N, D], dtype)
            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            logits_tile = T.alloc_fragment([block_N, block_Q], accum_dtype)
            w_frag = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s = T.alloc_var(index_dtype)
            cu_k_e = T.alloc_var(index_dtype)
            cu_k_s = CuSeqLenKS[seq_len_i]
            cu_k_e = CuSeqLenKE[seq_len_i]
            for bq_i in T.serial(1, block_Q):
                cu_k_s = T.min(cu_k_s, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
                cu_k_e = T.max(cu_k_e, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))

            # Q is the gemm B operand (transposed).  The MLS-aware gemm read
            # (ds_read_format) is only emitted when BOTH operands come from
            # matrix_load, so load Q via MLS too (one-time, in-range).
            # Use T.copy(prefer_instruction="matrix_load") so the pipeline
            # pass manages the async commit/wait groups (direct T.matrix_load
            # inside T.Pipelined is not covered by the async-group lowering).
            T.copy(IndexQ[seq_len_i * heads, 0], q_smem, prefer_instruction="matrix_load")
            T.copy(Weights[seq_len_i, 0], w_frag)

            for nbn_i in T.Pipelined(T.ceildiv(cu_k_e - cu_k_s, block_N), num_stages=num_stages):
                kv_row = nbn_i * block_N
                # MLS: async global->LDS matrix load, selected by the copy
                # lowering.  The last tile may run past seq_len_kv (dynamic
                # cu_k_s offset); out-of-range rows are masked by clean_logits.
                T.copy(IndexK[cu_k_s + kv_row, 0], k_smem, prefer_instruction="matrix_load")

                # num_stages=0 (block_N=128 path) is not software-pipelined, so
                # the pipeline pass inserts no async wait.  Drain the outstanding
                # MLS groups (Q + this K tile) before the gemm reads LDS.  Zero
                # perf cost: the gemm must wait for the data anyway, and there is
                # no compute to overlap with the load.
                if num_stages == 0:
                    T.ptx_wait_group(0)
                    T.sync_threads()

                T.clear(s)
                T.gemm(
                    k_smem, q_smem, s,
                    k_pack=K_PACK, transpose_B=True,
                    policy=_policy,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = T.max(s_reshaped[bn_i, bq_i, h_i], 0) * w_frag[bq_i, h_i]

                T.reduce_sum(s_reshaped, logits_tile, dim=2, clear=True)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    gkv = cu_k_s + kv_row + bn_i
                    Logits[seq_len_i + bq_i, gkv] = logits_tile[bn_i, bq_i]

    return kernel


@tilelang.jit
def clean_logits_tl(threads: int = 256, block_K: int = 4096):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]
            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx < cu_k_s or idx >= cu_k_e:
                        Logits[bx, idx] = -T.infinity(dtype)

    return kernel


# ============================================================================
# High-level interface (tilelang)
# ============================================================================

_kernel_cache = {}


def run_tilelang(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True):
    """Run the tilelang mqa_attn_return_logits kernel (BF16)."""
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]

    block_N, block_Q, num_stages, lds_bytes = _pick_block_config(heads, index_dim)
    cache_key = (heads, index_dim, block_N, block_Q, num_stages)
    if cache_key not in _kernel_cache:
        print(f"  [tilelang config] H={heads}, D={index_dim}, "
              f"block_N={block_N}, block_Q={block_Q}, num_stages={num_stages}, "
              f"LDS={lds_bytes / 1024:.1f}KB / {LDS_LIMIT / 1024:.0f}KB")
        _kernel_cache[cache_key] = mqa_attn_return_logits_tl(
            heads=heads, index_dim=index_dim,
            block_N=block_N, num_stages=num_stages, block_Q=block_Q,
        )
    logits_kernel = _kernel_cache[cache_key]

    logits = torch.empty([seq_len, seq_len_kv], device=q.device, dtype=torch.float32)
    logits_kernel(
        q.view(seq_len * heads, index_dim),
        kv,
        logits,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    if clean_logits:
        clean_kernel = clean_logits_tl()
        clean_kernel(logits, cu_seqlen_ks, cu_seqlen_ke)
    return logits


def mqa_logits(
    q: torch.Tensor,           # [S, H, D] BF16
    k: torch.Tensor,           # [SKV, D] BF16
    weights: torch.Tensor,     # [S, H] float32
    ks: torch.Tensor,          # [S] int32, cu_seqlen start
    ke: torch.Tensor,          # [S] int32, cu_seqlen end
    clean_logits: bool = True,
) -> torch.Tensor:             # [S, SKV] float32
    """TileLang BF16 MQA logits kernel.

    API-compatible with original bf16_mqa_logits_torch:
        q_slice, k_fp8, weights_slice, ks_slice, ke_slice
    """
    return run_tilelang(q, k, weights, ks, ke, clean_logits=clean_logits)


# ============================================================================
# High-level interface (lightop HIP ASM)
# ============================================================================

from lightop import op


def run_lightop(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True):
    """Run the lightop HIP ASM mqa_logits kernel (BF16)."""
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]

    return op.mqa_logits(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        seq_len,
        seq_len_kv,
        heads,
        index_dim,
        None,          # KV_scale (not used for BF16)
        clean_logits,
        None,          # D_out
    )


# ============================================================================
# PyTorch reference implementation for self-consistency validation
# ============================================================================

def ref_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
    """PyTorch reference: ReLU(K @ Q^T) * weights, summed over heads.

    q:     (S, H, D)   bfloat16
    kv:    (SKV, D)     bfloat16
    weights: (S, H)     float32
    cu_seqlen_ks: (S,) int32  start index per row
    cu_seqlen_ke: (S,) int32  end index per row (causal boundary)
    """
    S, H, D = q.shape
    SKV = kv.shape[0]

    # logits[i, j] = sum_h(ReLU(K_j · Q_{i,h}) * w_{i,h})
    # Q: (S, H, D) -> (S, H, D)
    # K: (SKV, D) -> (1, SKV, D)
    # scores: (S, H, SKV)
    q_f32 = q.float()
    k_f32 = kv.float()
    scores = torch.einsum('shd,kd->shk', q_f32, k_f32)  # (S, H, SKV)
    scores = torch.relu(scores)  # ReLU
    scores = scores * weights.float().unsqueeze(-1)  # (S, H, SKV)
    logits = scores.sum(dim=1)  # (S, SKV)

    # Apply causal mask
    for i in range(S):
        ks = cu_seqlen_ks[i].item()
        ke = cu_seqlen_ke[i].item()
        if ks > 0:
            logits[i, :ks] = float('-inf')
        if ke < SKV:
            logits[i, ke:] = float('-inf')

    return logits


# ============================================================================
# Correctness check helpers
# ============================================================================

def calc_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return 1 - correlation on finite positions only. 0 = identical.

    Both tilelang and lightop produce -inf for masked positions. Comparing
    -inf values leads to nan from inf/inf division, so we only compare
    positions where BOTH tensors are finite.
    """
    a_f32 = a.float()
    b_f32 = b.float()
    mask = torch.isfinite(a_f32) & torch.isfinite(b_f32)
    n = mask.sum().item()
    if n == 0:
        return 0.0
    a_valid = a_f32[mask].double()
    b_valid = b_f32[mask].double()
    norm_sum = (a_valid * a_valid + b_valid * b_valid).sum()
    if norm_sum == 0:
        return 0.0
    corr = (2 * (a_valid * b_valid).sum() / norm_sum).item()
    return 1.0 - corr


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Maximum absolute difference on positions where both tensors are finite."""
    mask = torch.isfinite(a) & torch.isfinite(b)
    if mask.sum() == 0:
        return 0.0
    return (a[mask].float() - b[mask].float()).abs().max().item()


# ============================================================================
# Benchmark helpers
# ============================================================================

def bench_event(fn, warmup=10, rep=50):
    """Benchmark using CUDA events for accurate GPU timing.  Returns avg ms."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


# ============================================================================
# Scene definitions
# ============================================================================

SCENES = {
    "tiny":      {"S": 128,  "SKV": 256,  "H": 32, "D": 128},
    "small":     {"S": 128,  "SKV": 2048, "H": 32, "D": 128},
    "small_h64": {"S": 128,  "SKV": 2048, "H": 64, "D": 128},
    "medium":    {"S": 1024, "SKV": 4096, "H": 32, "D": 128},
    "medium_h64":{"S": 1024, "SKV": 4096, "H": 64, "D": 128},
    "large":     {"S": 4096, "SKV": 8192, "H": 32, "D": 128},
    "large_h64": {"S": 4096, "SKV": 8192, "H": 64, "D": 128},
    "xl":        {"S": 4096, "SKV": 32768, "H": 32, "D": 128},
}


# ============================================================================
# Correctness validation (tilelang vs lightop only)
# ============================================================================

CORRECTNESS_CASES = [
    ("S128_H32_D128",   128,    256,    32,     128),
    ("S128_H64_D128",   128,    256,    64,     128),
    ("S256_H32_D128",   256,    512,    32,     128),
    ("S256_H64_D128",   256,    512,    64,     128),
    ("S512_H32_D128",   512,    1024,   32,     128),
    ("S512_H64_D128",   512,    1024,   64,     128),
    ("S1024_H32_D128",  1024,   2048,   32,     128),
    ("S2048_H32_D128",  2048,   4096,   32,     128),
    ("unalign_99",      99,     199,    32,     128),
    ("unalign_257",     257,    513,    32,     128),
    ("unalign_500",     500,    999,    64,     128),
    ("short_kv",        32,     16,     32,     128),
    ("short_kv2",       64,     32,     32,     128),
    ("single_q",        1,      64,     32,     128),
]


def run_correctness_check(device="cuda"):
    """Systematic correctness validation: tilelang vs lightop."""
    all_pass = True
    failed = []

    print("=" * 90)
    print("Correctness Validation: tilelang vs lightop")
    print("=" * 90)

    header = f"{'Case':<20} {'S':>5} {'SKV':>5} {'H':>4} {'D':>4}  {'tl_vs_ref':>11}  {'lo_vs_ref':>11}  {'tl_vs_lo':>11}  {'mask_match':>10}"
    print(header)
    print("-" * 90)

    for name, S, SKV, H, D in CORRECTNESS_CASES:
        torch.manual_seed(42)
        q_bf16 = torch.randn(S, H, D, device=device, dtype=torch.bfloat16)
        kv_bf16 = torch.randn(SKV, D, device=device, dtype=torch.bfloat16)
        weights = torch.randn(S, H, device=device, dtype=torch.float32)

        # Use simple cu_seqlens: full causal attention
        ks = torch.zeros(S, dtype=torch.int32, device=device)
        ke = torch.arange(1, S + 1, dtype=torch.int32, device=device)

        # PyTorch reference (CPU)
        logits_ref = ref_mqa_logits(q_bf16.cpu(), kv_bf16.cpu(), weights.cpu(), ks.cpu(), ke.cpu())

        # TileLang
        logits_tl = run_tilelang(q_bf16, kv_bf16, weights, ks, ke)

        # Lightop
        logits_lo = run_lightop(q_bf16, kv_bf16, weights, ks, ke)

        diff_tl_ref = calc_diff(logits_tl.cpu(), logits_ref)
        diff_lo_ref = calc_diff(logits_lo.cpu(), logits_ref)
        diff_tl_lo = calc_diff(logits_tl, logits_lo)

        tl_isfinite = torch.isfinite(logits_tl)
        lo_isfinite = torch.isfinite(logits_lo)
        mask_match = torch.all(tl_isfinite == lo_isfinite).item()

        print(f"{name:<20} {S:>5} {SKV:>5} {H:>4} {D:>4}  {diff_tl_ref:<11.2e}  {diff_lo_ref:<11.2e}  {diff_tl_lo:<11.2e}  {str(mask_match):>10}")

        if diff_tl_ref >= 1e-3 or diff_lo_ref >= 1e-3 or not mask_match:
            all_pass = False
            failed.append(name)

    print("-" * 90)
    if all_pass:
        print("ALL PASSED")
    else:
        print(f"FAILURES: {', '.join(failed)}")

    return all_pass


# ============================================================================
# Main benchmark logic
# ============================================================================

def run_benchmark(name, S, SKV, H, D, warmup, rep, check, device):
    print(f"\n{'=' * 70}")
    print(f"Config: {name}")
    print(f"  S={S}, SKV={SKV}, H={H}, D={D}")

    torch.manual_seed(0)
    q_bf16 = torch.randn(S, H, D, device=device, dtype=torch.bfloat16)
    kv_bf16 = torch.randn(SKV, D, device=device, dtype=torch.bfloat16)
    weights = torch.randn(S, H, device=device, dtype=torch.float32)

    # Simple causal cu_seqlens
    ks = torch.zeros(S, dtype=torch.int32, device=device)
    ke = torch.arange(1, S + 1, dtype=torch.int32, device=device)

    # ---- Correctness: tilelang vs lightop ----
    if check:
        tl_out = run_tilelang(q_bf16, kv_bf16, weights, ks, ke)
        lo_out = run_lightop(q_bf16, kv_bf16, weights, ks, ke)
        diff = calc_diff(tl_out, lo_out)
        max_abs_err = max_abs_diff(tl_out, lo_out)
        mask_match = torch.all(torch.isfinite(tl_out) == torch.isfinite(lo_out)).item()
        print(f"  tilelang vs lightop diff: {diff:.6e}  max_abs_err: {max_abs_err:.4e}  mask_match: {mask_match}")

    # ---- Benchmark tilelang ----
    tl_fn = lambda: run_tilelang(q_bf16, kv_bf16, weights, ks, ke)
    tl_ms = bench_event(tl_fn, warmup=warmup, rep=rep)
    print(f"  tilelang:  {tl_ms:8.3f} ms")

    # ---- Benchmark lightop ----
    lo_fn = lambda: run_lightop(q_bf16, kv_bf16, weights, ks, ke)
    lo_ms = bench_event(lo_fn, warmup=warmup, rep=rep)
    speedup = tl_ms / lo_ms
    print(f"  lightop:   {lo_ms:8.3f} ms")
    print(f"  speedup:   {speedup:8.2f}x  (lightop vs tilelang)")

    return {
        "name": name, "S": S, "SKV": SKV, "H": H, "D": D,
        "tl_ms": tl_ms, "lo_ms": lo_ms,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tilelang vs lightop HIP ASM mqa_logits (BF16)"
    )
    parser.add_argument("--scene", type=str, default=None,
                        choices=list(SCENES.keys()) + ["all"],
                        help="Preset scene (use 'all' for every scene)")
    parser.add_argument("--S", type=int, default=None, help="Q sequence length")
    parser.add_argument("--SKV", type=int, default=None, help="KV sequence length")
    parser.add_argument("--H", type=int, default=None, help="Number of heads")
    parser.add_argument("--D", type=int, default=None, help="Index dimension")
    parser.add_argument("--check", action="store_true", help="Enable correctness check per benchmark case")
    parser.add_argument("--validate", action="store_true", help="Run dedicated correctness validation suite")
    parser.add_argument("--benchmark", action="store_true", default=False, help="Force run benchmark in addition to --validate")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=50, help="Benchmark repetitions")
    parser.add_argument("--csv", type=str, default=None, help="Export results to CSV")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA device not available.")
        sys.exit(1)

    device = "cuda"
    prop = torch.cuda.get_device_properties(0)
    gcn = getattr(prop, 'gcnArchName', prop.name)
    print(f"Device: {prop.name} ({gcn})")
    print(f"tilelang:   available")
    print(f"lightop:    available")

    # ---- Validation mode ----
    if args.validate:
        run_correctness_check(device)
        if not args.benchmark:
            return

    # ---- Benchmark mode ----
    configs = []
    if args.scene == "all":
        for scene_name, params in SCENES.items():
            configs.append((scene_name, params))
    elif args.scene is not None:
        params = SCENES[args.scene]
        configs.append((args.scene, params))
    else:
        S = args.S if args.S is not None else 4096
        SKV = args.SKV if args.SKV is not None else 8192
        H = args.H if args.H is not None else 32
        D = args.D if args.D is not None else 128
        configs.append(("custom", {"S": S, "SKV": SKV, "H": H, "D": D}))

    results = []
    for name, params in configs:
        r = run_benchmark(
            name, params["S"], params["SKV"], params["H"], params["D"],
            args.warmup, args.rep, args.check, device,
        )
        results.append(r)

    # ---- Summary table ----
    print("\n" + "=" * 95)
    print("Summary")
    print("=" * 95)
    header = f"{'Scene':<14} {'S':>6} {'SKV':>6} {'H':>4} {'D':>4}  {'tilelang(ms)':>12}  {'lightop(ms)':>12}  {'speedup':>8}"
    print(header)
    print("-" * 95)
    for r in results:
        lo_str = f"{r['lo_ms']:.3f}"
        sp_str = f"{r['speedup']:.2f}x"
        print(f"{r['name']:<14} {r['S']:>6} {r['SKV']:>6} {r['H']:>4} {r['D']:>4}  "
              f"{r['tl_ms']:>12.3f}  {lo_str:>12}  {sp_str:>8}")

    # ---- CSV export ----
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Scene", "S", "SKV", "H", "D",
                             "tilelang_ms", "lightop_ms", "speedup"])
            for r in results:
                writer.writerow([
                    r['name'], r['S'], r['SKV'], r['H'], r['D'],
                    f"{r['tl_ms']:.6f}",
                    f"{r['lo_ms']:.6f}",
                    f"{r['speedup']:.4f}",
                ])
        print(f"\nResults saved to: {args.csv}")


if __name__ == "__main__":
    main()

