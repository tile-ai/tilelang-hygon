# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Diagnostic for native ``T.gemm`` fp8: is the global->shared copy async
(cp.async)? What tilings/stages give the best TFLOPS?

Run: HIP_VISIBLE_DEVICES=2 python perf/gemm/fp8_native_diag.py
"""

import os
import re
import sys

_REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench_cudagraph


def native_fp8(M, N, K, block_M, block_N, block_K, num_stages, threads):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float8_e4m3fn"),
        B: T.Tensor((N, K), "float8_e4m3fn"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), "float8_e4m3fn")
            B_s = T.alloc_shared((block_N, block_K), "float8_e4m3fn")
            C_l = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_s)
                T.copy(B[bx * block_N, k * block_K], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True)
            T.copy(C_l, C[by * block_M, bx * block_N])

    return tilelang.compile(gemm, out_idx=[2])


def main():
    M = N = K = 8192
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
    ref = A.float() @ B.float().T

    print(f"{'block':<18}{'ns':>3}{'thr':>5}{'cpasync':>9}{'wait':>7}{'TFLOPS':>9}{'rel':>10}")
    print("-" * 61)
    for (bm, bn, bk), ns_list in {
        (256, 128, 32): [3, 4],
        (128, 128, 32): [3, 4, 5],   # lightop tiling
        (128, 128, 64): [3, 4],
        (128, 256, 32): [3],
        (256, 256, 32): [3],
    }.items():
        for ns in ns_list:
            for thr in (256, 512):
                try:
                    kern = native_fp8(M, N, K, bm, bn, bk, ns, thr)
                    src = kern.get_kernel_source()
                    cp = src.count("cp_async_commit")
                    waits = re.findall(r"cp_async_wait<(\d+)>", src)
                    C = kern(A, B)
                    rel = (C - ref).abs().max().item() / ref.abs().max().item()
                    if rel > 1e-3:
                        print(f"{f'{bm}x{bn}x{bk}':<18}{ns:>3}{thr:>5}{cp:>9}{str(sorted(waits))[:6]:>7}  WRONG(rel={rel:.3f})")
                        continue
                    lat = do_bench_cudagraph(lambda: kern(A, B))
                    tf = 2.0 * M * N * K / lat * 1e-9
                    print(f"{f'{bm}x{bn}x{bk}':<18}{ns:>3}{thr:>5}{cp:>9}{str(sorted(waits))[:6]:>7}{tf:>9.1f}{rel:>10.5f}")
                except Exception as e:  # noqa: BLE001
                    print(f"{f'{bm}x{bn}x{bk}':<18}{ns:>3}{thr:>5}    ERR {type(e).__name__}: {str(e)[:30]}")


if __name__ == "__main__":
    main()
