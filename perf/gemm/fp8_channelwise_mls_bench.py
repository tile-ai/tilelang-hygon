# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Config sweep for the fp8 channelwise MLS GEMM on the 8192^3 compute shape.

This used to carry its own copy of the kernel (a pipelined loop with no unroll,
no setprio, no dense packing) and to time it with ``do_bench_cudagraph``.  Both
were stale: the copy silently drifted from the production builder, and that
timing helper returns *milliseconds* on this stack, so its TFLOPS were inflated.
It now sweeps the real builder -- ``tilelang/hcu/op/gemm/gemm_fp8_channelwise.py``
-- through the decode bench's ``graph_replay_time`` (explicit capture + replay
with event timing).

The decode shapes are NOT here: use ``fp8_channelwise_decode_bench.py`` and
``decode_cfg`` for those.

Run: HIP_VISIBLE_DEVICES=2 python perf/gemm/fp8_channelwise_mls_bench.py
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from perf.gemm.fp8_channelwise_decode_bench import graph_replay_time
from tilelang.hcu.op.gemm.gemm_fp8_channelwise import fp8_channelwise_gemm

FP8_PEAK = 450.0  # TFLOPS, HCU gfx938 -- anything above this is a timing artifact

# The validated 8192^3 point first, then its neighbourhood.
CONFIGS = [
    (256, 256, 64, 1, 512),
    (256, 256, 64, 1, 256),
    (256, 256, 64, 1, 1024),
    (128, 128, 64, 1, 256),
    (256, 128, 64, 1, 512),
    (128, 256, 64, 1, 512),
]


def main():
    M = N = K = 8192
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.rand(M, device="cuda") + 0.5
    sb = torch.rand(N, device="cuda") + 0.5
    ref = (A.float() @ B.float().T) * sa[:, None] * sb[None, :]

    print(f"{'config':<28}{'TFLOPS':>9}{'us':>10}{'rel':>10}")
    for bm, bn, bk, ns, thr in CONFIGS:
        label = f"{bm}x{bn}x{bk} ns{ns} t{thr}"
        try:
            kern = fp8_channelwise_gemm(M, N, K, bm, bn, bk, ns, thr)
            C = kern(A, B, sa, sb)
            rel = (C - ref).abs().max().item() / ref.abs().max().item()
            if rel > 1e-3:
                print(f"{label:<28} WRONG rel={rel:.4f}", flush=True)
                continue
            us = graph_replay_time(lambda: kern(A, B, sa, sb))[0]
            tf = 2.0 * M * N * K / us * 1e-6
            flag = "  > peak!" if tf > FP8_PEAK else ""
            print(f"{label:<28}{tf:>9.1f}{us:>10.1f}{rel:>10.5f}{flag}", flush=True)
        except Exception as e:  # noqa: BLE001 - illegal MLS tiles are expected here
            print(f"{label:<28} ERR {type(e).__name__}: {str(e)[:40]}", flush=True)


if __name__ == "__main__":
    main()
