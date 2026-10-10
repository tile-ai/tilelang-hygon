# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Channelwise-scaled fp8 GEMM built on the native ``T.gemm`` fp8 MMAC path.

Computes ``C[m, n] = scale_a[m] * scale_b[n] * sum_k A[m, k] * B[n, k]`` with
A (M, K) fp8, B (N, K) fp8 (N-major, ``transpose_B=True``), fp32 per-token /
per-channel scales applied as an epilogue multiply on the fp32 accumulator.

This is the standard ``T.copy`` (auto-promoted to 16-byte ``cp.async``) +
``T.gemm`` (fp8 ``v_mmac_f32_16x16x32_fp8_fp8``) + software pipeline, with a
DSL channelwise epilogue — no hand-written async pipeline. The winning config
on HCU gfx938 is **block 256x256x32, num_stages=2, 1024 threads** (~315 TFLOPS
@ 8192^3), where the 1024-thread block (16 waves) is what unlocks occupancy.

Run (validate + bench):
    HIP_VISIBLE_DEVICES=2 python perf/gemm/native_fp8_channelwise.py
"""

import torch
import tilelang
import tilelang.language as T


def native_fp8_channelwise(
    M,
    N,
    K,
    block_M=256,
    block_N=256,
    block_K=32,
    num_stages=2,
    threads=1024,
    accum_dtype="float32",
):
    """Native ``T.gemm`` fp8 GEMM with a channelwise (per-token/per-channel) epilogue."""

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float8_e4m3fn"),
        B: T.Tensor((N, K), "float8_e4m3fn"),
        scale_a: T.Tensor((M,), "float32"),
        scale_b: T.Tensor((N,), "float32"),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), "float8_e4m3fn")
            B_s = T.alloc_shared((block_N, block_K), "float8_e4m3fn")
            C_l = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_s)
                T.copy(B[bx * block_N, k * block_K], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                C_l[i, j] = C_l[i, j] * scale_a[by * block_M + i] * scale_b[bx * block_N + j]
            T.copy(C_l, C[by * block_M, bx * block_N])

    return tilelang.compile(gemm, out_idx=[4])


def _ref(A, B, scale_a, scale_b):
    return (A.float() @ B.float().T) * scale_a[:, None] * scale_b[None, :]


def _validate_and_bench(size=8192):
    from tilelang.profiler import do_bench_cudagraph

    torch.manual_seed(0)
    A = torch.randn(size, size, device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn(size, size, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.rand(size, device="cuda") + 0.5
    sb = torch.rand(size, device="cuda") + 0.5

    kern = native_fp8_channelwise(size, size, size)
    C = kern(A, B, sa, sb)
    ref = _ref(A, B, sa, sb)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    lat = do_bench_cudagraph(lambda: kern(A, B, sa, sb))
    tf = 2.0 * size * size * size / lat * 1e-9
    print(f"native_fp8_channelwise {size}^3 (256x256x32 ns2 t1024): "
          f"{tf:.1f} TFLOPS, rel={rel:.6f}, {lat * 1e3:.1f} us")
    assert rel < 1e-3, f"rel error {rel} too large"
    return tf, rel


if __name__ == "__main__":
    _validate_and_bench()
