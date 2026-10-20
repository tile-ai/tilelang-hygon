# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Benchmark fp8 channelwise GEMM on qwen3-32B decode shapes (M <= 128).

Bandwidth-utilization sweep over the qwen3-32B per-token GEMM shapes
(gate_up 51200x5120, down 5120x25600, qkv 10240x5120, o 5120x8192). The tile
config now comes from the shape-aware `decode_cfg(M, N, K)` in
`perf/gemm/fp8_channelwise_mls.py` — it used to take only M, which is why
o/down (both N=5120) lost at every M. Reports effective bandwidth vs the
1800 GB/s nominal peak.

NOTE this prints our absolute times only, one kernel per M. It is not a
head-to-head: the hipblas ratio has to be measured paired in the same process,
and with only two graphs resident — a process holding many candidate graphs
inflates `torch._scaled_mm` by ~17%. See `docs/optimization/fp8_decode_m_le_128.md`.

NOTE on timing (2026-10-13): this stack's `do_bench_cudagraph` returns
**milliseconds** and divides n_repeat out of a single unrolled-graph replay,
which inflates utilization (it produced the old inflated figures, since
removed). Use `graph_replay_time` below (explicit per-call CUDA-graph capture
+ replay with event timing) — the box's measured pure-read ceiling is ~2016
GB/s, so any reported % above that is a timing artifact. (The older ~1646
GB/s / 91.5% figure came from an undersized 512 MB probe and is deprecated.)

Run: HIP_VISIBLE_DEVICES=2 python perf/gemm/fp8_channelwise_decode_bench.py
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from perf.gemm.fp8_channelwise_mls import DecodeCfg, decode_cfg, fp8_mls_channelwise

PEAK = 1800.0  # GB/s, HCU gfx938 HBM

SHAPES = {
    "gate_up": (51200, 5120),   # N, K
    "qkv":     (10240, 5120),
    "o":       (5120, 8192),
    "down":    (5120, 25600),
}


def graph_replay_time(fn, reps=3, iters=50):
    """Per-call latency via explicit CUDA-graph capture + replay, event timing.
    Returns min and median of `reps` medians-of-`iters` (us)."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    meds = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        lat = []
        for _ in range(iters):
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            lat.append(s.elapsed_time(e) * 1e3)
        lat.sort()
        meds.append(lat[len(lat) // 2])
    meds.sort()
    return meds[0], meds[len(meds) // 2]


def bench(name, N, K, M, bm=None, bn=None, bk=None, ns=None, t=None, sp=None):
    torch.manual_seed(0)
    A = (torch.randn(M, K, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    B = (torch.randn(N, K, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    sa = (torch.rand(M, device="cuda") + 0.5)
    sb = (torch.rand(N, device="cuda") + 0.5)
    ref = (A.float() @ B.float().T) * sa[:, None] * sb[None, :]

    cfg = decode_cfg(M, N, K)
    if bm is not None:  # explicit override wins over the chooser
        cfg = DecodeCfg(bm, bn, bk, ns, t, sp, cfg.split_k)
    assert cfg.split_k == 1, f"{name} M={M}: split_k={cfg.split_k} needs the split-K builder"
    kern = fp8_mls_channelwise(M, N, K, cfg.block_M, cfg.block_N, cfg.block_K,
                               cfg.num_stages, cfg.threads, setprio=cfg.setprio)
    C = kern(A, B, sa, sb)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-3, f"{name} M={M} rel={rel:.4f}"

    tmin, tmed = graph_replay_time(lambda: kern(A, B, sa, sb))
    eff = N * K / tmin * 1e-3  # GB/s
    print(f"{name:8s} M={M:>3} (bm={cfg.block_M:>3},bn={cfg.block_N:>3},bk={cfg.block_K:>3},"
          f"ns={cfg.num_stages},t={cfg.threads},sp={cfg.setprio}): "
          f"{tmin:7.1f}/{tmed:7.1f}us  {eff:6.0f} GB/s  {eff / PEAK * 100:5.1f}%  rel={rel:.5f}", flush=True)
    return tmin, eff


def main():
    for name, (N, K) in SHAPES.items():
        print(f"\n== {name} {N}x{K} ==", flush=True)
        for M in [1, 8, 16, 32, 64, 128]:
            try:
                bench(name, N, K, M)
            except Exception as ex:
                print(f"{name:8s} M={M:>3} ERR {str(ex)[:80]}", flush=True)


if __name__ == "__main__":
    main()
