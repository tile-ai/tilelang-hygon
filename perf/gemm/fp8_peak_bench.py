# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Measure the achievable fp8 (e4m3) GEMM throughput on HCU (gfx938).

Two kernels are timed at large square sizes:
  * ``fp8_async`` - the hand-written 4-stage async channelwise kernel
    (perf/gemm/async_copy_gemm_channelwise.py), our fastest fp8 path.
  * ``fp8_gemm``  - a plain ``T.gemm`` fp8 GEMM with the standard async-copy
    pipeline (compiler-driven), for reference.

Reference peak for the same hardware: lightop hipBLASLt w8a8 channelwise
= ~560 TFLOPS (measured @ 8192^3).

Run:  HIP_VISIBLE_DEVICES=2 python perf/gemm/fp8_peak_bench.py
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench_cudagraph

from perf.gemm.async_copy_gemm_channelwise import gemm_async_copy_channelwise


def _ref(A, B, sa, sb):
    return (A.float() @ B.float().T) * sa[:, None] * sb[None, :]


def bench_async_fp8(M, N, K, block_M, block_N, block_K):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.rand(M, device="cuda") + 0.5
    sb = torch.rand(N, device="cuda") + 0.5
    kernel = gemm_async_copy_channelwise(M, N, K, block_M, block_N, block_K,
                                         dtype="float8_e4m3fn")
    C = kernel(A, B, sa, sb)
    ref = _ref(A, B, sa, sb)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    lat = do_bench_cudagraph(lambda: kernel(A, B, sa, sb))
    tf = 2.0 * M * N * K / lat * 1e-9
    return tf, rel, lat


def _plain_fp8_gemm(M, N, K, block_M, block_N, block_K, num_stages):
    """Plain T.gemm fp8 with the standard (compiler-driven) async pipeline."""
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float8_e4m3fn"),
        B: T.Tensor((N, K), "float8_e4m3fn"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), "float8_e4m3fn")
            B_s = T.alloc_shared((block_N, block_K), "float8_e4m3fn")
            C_l = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_s)
                T.copy(B[bx * block_N, k * block_K], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True)
            T.copy(C_l, C[by * block_M, bx * block_N])

    return tilelang.compile(gemm)


def bench_gemm_fp8(M, N, K, block_M, block_N, block_K, num_stages):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    kernel = _plain_fp8_gemm(M, N, K, block_M, block_N, block_K, num_stages)
    kernel(A, B, C)  # C is the in-place output (out_idx not set)
    ref = (A.float() @ B.float().T)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    lat = do_bench_cudagraph(lambda: kernel(A, B, C))
    tf = 2.0 * M * N * K / lat * 1e-9
    return tf, rel, lat


def main():
    print(f"torch HCU arch: {torch.cuda.get_device_capability()}")
    print(f"{'kernel':<28}{'size':<14}{'block':<16}{'TFLOPS':>9}{'rel':>10}{'us':>10}")
    print("-" * 81)
    for size in (8192, 4096):
        M = N = K = size
        try:
            tf, rel, lat = bench_async_fp8(M, N, K, 256, 128, 32)
            print(f"{'fp8_async (4-stage)':<28}{str(size)+'^3':<14}{'256x128x32':<16}{tf:>9.1f}{rel:>10.5f}{lat*1e3:>10.1f}")
        except Exception as e:  # noqa: BLE001
            print(f"{'fp8_async (4-stage)':<28}{str(size)+'^3':<14}{'256x128x32':<16}ERR {type(e).__name__}: {str(e)[:40]}")
        for bm, bn, bk, ns in ((256, 128, 32, 3), (128, 128, 64, 3)):
            if K < 4 * bk:
                continue
            try:
                tf, rel, lat = bench_gemm_fp8(M, N, K, bm, bn, bk, ns)
                print(f"{'fp8 T.gemm':<28}{str(size)+'^3':<14}{f'{bm}x{bn}x{bk} ns{ns}':<16}{tf:>9.1f}{rel:>10.5f}{lat*1e3:>10.1f}")
            except Exception as e:  # noqa: BLE001
                print(f"{'fp8 T.gemm':<28}{str(size)+'^3':<14}{f'{bm}x{bn}x{bk} ns{ns}':<16}ERR {type(e).__name__}: {str(e)[:40]}")
    print("-" * 81)
    print("Reference: lightop hipBLASLt w8a8 channelwise ~560 TFLOPS @ 8192^3")


if __name__ == "__main__":
    main()
