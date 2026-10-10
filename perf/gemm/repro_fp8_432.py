# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Self-contained reproduction of the 432-TFLOPS fp8 channelwise GEMM (HCU gfx938).

Run:
    HIP_VISIBLE_DEVICES=2 python perf/gemm/repro_fp8_432.py

It compiles the operator, checks correctness against an exact fp32 reference,
and reports TFLOPS at 8192^3. Expected: rel ~= 0.0 and TFLOPS ~= 430.
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


def fp8_mls_channelwise(M, N, K, block_M, block_N, block_K, num_stages, threads):
    assert block_K % 64 == 0, "fp8 MLS channelwise GEMM requires block_K % 64 == 0"

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float8_e4m3fn"),
        B: T.Tensor((N, K), "float8_e4m3fn"),
        scale_a: T.Tensor((M,), "float32"),
        scale_b: T.Tensor((N,), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), "float8_e4m3fn")
            B_shared = T.alloc_shared((block_N, block_K), "float8_e4m3fn")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(
                    A[by * block_M, k * block_K],
                    A_shared,
                    prefer_instruction="matrix_load",
                )
                T.copy(
                    B[bx * block_N, k * block_K],
                    B_shared,
                    prefer_instruction="matrix_load",
                )
                if num_stages == 0:
                    T.ptx_wait_group(0)
                    T.sync_warp()
                T.gemm(A_shared, B_shared, C_local, False, True, k_pack=1)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = C_local[i, j] * scale_a[by * block_M + i] * scale_b[
                    bx * block_N + j
                ]
            T.copy(C_local, C[by * block_M, bx * block_N])

    return tilelang.compile(gemm, out_idx=[4])


def main():
    M = N = K = 8192
    # Winning config: 256x256x64, 1 stage, 512 threads.
    block_M, block_N, block_K, num_stages, threads = 256, 256, 64, 1, 512

    torch.manual_seed(0)
    A = (torch.randn(M, K, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    B = (torch.randn(N, K, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    scale_a = (torch.rand(M, device="cuda") + 0.5).to(torch.float32)
    scale_b = (torch.rand(N, device="cuda") + 0.5).to(torch.float32)

    kernel = fp8_mls_channelwise(M, N, K, block_M, block_N, block_K, num_stages, threads)

    # Correctness vs exact fp32 reference (fp8->f32 is lossless).
    C = kernel(A, B, scale_a, scale_b)
    ref = A.float() @ B.float().T
    ref = ref * scale_a[:, None] * scale_b[None, :]
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    print(f"max_rel = {rel:.2e}")

    # Confirm the matrix-LDS read path is actually used.
    src = kernel.get_kernel_source()
    used_mls = ("ds_read_format_tensor_a" in src) and ("ds_read_format_tensor_b" in src)
    print(f"uses ds_read_format_tensor_a/b (matrix LDS read): {used_mls}")

    # Throughput.
    lat_us = do_bench_cudagraph(lambda: kernel(A, B, scale_a, scale_b))
    tflops = 2.0 * M * N * K / lat_us * 1e-9
    print(f"TFLOPS @ 8192^3 = {tflops:.1f}")

    assert rel < 1e-3, f"correctness failed: rel={rel}"
    assert used_mls, "expected the MLS matrix-LDS read path"
    print("OK")


if __name__ == "__main__":
    main()
