# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Channelwise-scaled fp8 GEMM via the MLS (matrix-LDS-read) producer path.

Computes ``C[m, n] = scale_a[m] * scale_b[n] * sum_k A[m, k] * B[n, k]`` with
A (M, K) fp8, B (N, K) fp8 (N-major), fp32 per-token / per-channel scales
applied as an epilogue multiply on the fp32 accumulator.

Both A and B are loaded with ``T.copy(..., prefer_instruction="matrix_load")``
(MLS, the async global->LDS matrix load). This makes ``T.gemm`` lower through
``_gemm_mls_mls`` -> ``ds_read_format_tensor_a/b`` — the HCU hardware matrix
LDS read (``ds_read_m32x32_b8``), the same read the hipblaslt fp8 kernel uses.
On HCU (gfx938) the winning config (block 256x256x64, num_stages=1, 512
threads) reaches **~432 TFLOPS @ 8192^3** (rel=0), vs ~332 for the scalar-LDS
``cp.async`` path and ~560 for lightop/hipblasLt.

Constraint: with ``trans_B=True`` the B MLS atom is the trans b8 tile
(tile_k 64/128), so ``block_K`` must be a multiple of 64.

Run (validate + bench):
    HIP_VISIBLE_DEVICES=2 python perf/gemm/fp8_mls_channelwise.py
"""

import torch
import tilelang
import tilelang.language as T


def fp8_mls_channelwise(
    M,
    N,
    K,
    block_M=256,
    block_N=256,
    block_K=64,
    num_stages=1,
    threads=512,
    accum_dtype="float32",
):
    """Channelwise fp8 GEMM using MLS (matrix-LDS-read) A/B producers.

    ``block_K`` must be a multiple of 64 (the trans-B fp8 MLS atom width).
    """
    assert block_K % 64 == 0, "fp8 MLS channelwise GEMM requires block_K % 64 == 0"

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
                T.copy(A[by * block_M, k * block_K], A_s, prefer_instruction="matrix_load")
                T.copy(B[bx * block_N, k * block_K], B_s, prefer_instruction="matrix_load")
                if num_stages == 0:
                    T.ptx_wait_group(0)
                    T.sync_warp()
                T.gemm(A_s, B_s, C_l, False, True, k_pack=1)
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

    kern = fp8_mls_channelwise(size, size, size)  # 256x256x64 ns1 t512
    C = kern(A, B, sa, sb)
    ref = _ref(A, B, sa, sb)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    lat = do_bench_cudagraph(lambda: kern(A, B, sa, sb))
    tf = 2.0 * size * size * size / lat * 1e-9
    print(f"fp8_mls_channelwise {size}^3 (256x256x64 ns1 t512): "
          f"{tf:.1f} TFLOPS, rel={rel:.6f}, {lat * 1e3:.1f} us")
    assert rel < 1e-3, f"rel error {rel} too large"
    return tf, rel


if __name__ == "__main__":
    _validate_and_bench()
