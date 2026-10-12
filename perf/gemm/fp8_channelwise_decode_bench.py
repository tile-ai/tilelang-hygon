# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Benchmark fp8 channelwise GEMM on qwen3-32B decode shapes (M <= 128).

Bandwidth-utilization sweep over the qwen3-32B per-token GEMM shapes
(gate_up 51200x5120, down 5120x25600, qkv 10240x5120, o 5120x8192) with the
winning M<=16 decode config (block 16x32x256, num_stages=1, 256 threads)
extended to M=32..128 via block_M=pow2 tiles. Reports effective bandwidth
vs the 1800 GB/s peak.

Run: HIP_VISIBLE_DEVICES=2 python perf/gemm/fp8_channelwise_decode_bench.py
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from tilelang.profiler import do_bench_cudagraph
from perf.gemm.fp8_channelwise_mls import fp8_mls_channelwise

PEAK = 1800.0  # GB/s, HCU gfx938 HBM

SHAPES = {
    "gate_up": (51200, 5120),   # N, K
    "qkv":     (10240, 5120),
    "o":       (5120, 8192),
    "down":    (5120, 25600),
}


def bench(name, N, K, M, bm):
    torch.manual_seed(0)
    A = (torch.randn(M, K, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    B = (torch.randn(N, K, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    sa = (torch.rand(M, device="cuda") + 0.5)
    sb = (torch.rand(N, device="cuda") + 0.5)
    ref = (A.float() @ B.float().T) * sa[:, None] * sb[None, :]

    kern = fp8_mls_channelwise(M, N, K, bm, 32, 256, 1, 256)
    C = kern(A, B, sa, sb)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-3, f"{name} M={M} rel={rel:.4f}"

    lat = do_bench_cudagraph(lambda: kern(A, B, sa, sb))
    eff = N * K / lat * 1e-3  # GB/s
    print(f"{name:8s} M={M:>3} (bm={bm:>3}): {lat:8.1f}us  {eff:6.0f} GB/s  {eff / PEAK * 100:5.1f}%  rel={rel:.5f}", flush=True)
    return lat, eff


def main():
    for name, (N, K) in SHAPES.items():
        print(f"\n== {name} {N}x{K} ==", flush=True)
        for M in [1, 8, 16, 32, 64, 128]:
            bm = 16 if M <= 16 else (32 if M <= 32 else (64 if M <= 64 else 128))
            try:
                bench(name, N, K, M, bm)
            except Exception as ex:
                print(f"{name:8s} M={M:>3} ERR {str(ex)[:80]}", flush=True)


if __name__ == "__main__":
    main()
