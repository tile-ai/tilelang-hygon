# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Head-to-head: r_mls (A in VGPR) vs mls_mls (A via MLS LDS) on qwen3-32B decode.

Both producers load B via MLS (``prefer_instruction="matrix_load"``). The
r_mls variant keeps A in per-thread registers exactly like the hipblaslt
``buffer_load_dwordx4`` decode kernel (no A LDS round-trip, no A barrier),
but the generated loads are currently 8-byte ``hcu_buffer_load<fp8_t, 8>``
(no slc, no vmcnt pipeline) instead of 16-byte dwordx4.

Measured (gfx938, GPU 4, 2026-10-13), effective BW % of 1800 GB/s peak:
    hipblas: M=32 66.8%  M=64 62.2%  M=128 46.0%
    r_mls bm64 bn64 bk256 ns2:    M=32 16.1%  M=64 8.5%   M=128 ~2.5%
    r_mls bm64 bn128 bk256 ns2:   M=32 24.0%  M=64 13.0%
    mls_mls bm64 bn128 bk64 ns2:  M=32 51.7%  M=64 49.6%  M=128 26.5%
    mls_mls bm128 bn64 bk64 ns2:  M=64 42.1%  M=128 36.0%

=> The VGPR-A path is far slower than the current MLS-A path, so M=32..128
   stays on mls_mls (block 64x128x64 ns2 t256). See handoff.md for the
   root-cause analysis and the slc/s_setprio/dwordx4 TODO list.
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

PEAK = 1800.0  # GB/s, HCU gfx938 HBM


def _gemm_r_mls(M, N, K, bm, bn, bk, ns, thr):
    """A -> per-thread fragment VGPR (hipblas-style), B -> MLS LDS."""
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float8_e4m3fn"),
        B: T.Tensor((N, K), "float8_e4m3fn"),
        sa: T.Tensor((M,), "float32"),
        sb: T.Tensor((N,), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=thr) as (bx, by):
            B_s = T.alloc_shared((bn, bk), "float8_e4m3fn")
            A_l = T.alloc_fragment((bm, bk), "float8_e4m3fn")
            C_l = T.alloc_fragment((bm, bn), "float32")
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, bk), num_stages=ns):
                T.copy(A[by * bm, k * bk], A_l)
                T.copy(B[bx * bn, k * bk], B_s, prefer_instruction="matrix_load")
                if ns == 0:
                    T.ptx_wait_group(0)
                    T.sync_warp()
                T.gemm(A_l, B_s, C_l, False, True, k_pack=1)
            for i, j in T.Parallel(bm, bn):
                C_l[i, j] = C_l[i, j] * sa[by * bm + i] * sb[bx * bn + j]
            T.copy(C_l, C[by * bm, bx * bn])

    return tilelang.compile(gemm, out_idx=[4])


def _gemm_mls_mls(M, N, K, bm, bn, bk, ns, thr):
    """Both A and B via MLS LDS (current best decode path)."""
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float8_e4m3fn"),
        B: T.Tensor((N, K), "float8_e4m3fn"),
        sa: T.Tensor((M,), "float32"),
        sb: T.Tensor((N,), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=thr) as (bx, by):
            A_s = T.alloc_shared((bm, bk), "float8_e4m3fn")
            B_s = T.alloc_shared((bn, bk), "float8_e4m3fn")
            C_l = T.alloc_fragment((bm, bn), "float32")
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, bk), num_stages=ns):
                T.copy(A[by * bm, k * bk], A_s, prefer_instruction="matrix_load")
                T.copy(B[bx * bn, k * bk], B_s, prefer_instruction="matrix_load")
                if ns == 0:
                    T.ptx_wait_group(0)
                    T.sync_warp()
                T.gemm(A_s, B_s, C_l, False, True, k_pack=1)
            for i, j in T.Parallel(bm, bn):
                C_l[i, j] = C_l[i, j] * sa[by * bm + i] * sb[bx * bn + j]
            T.copy(C_l, C[by * bm, bx * bn])

    return tilelang.compile(gemm, out_idx=[4])


def bench(name, N, K, M, builder, cfg):
    torch.manual_seed(0)
    A = (torch.randn(M, K, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    B = (torch.randn(N, K, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    sa = (torch.rand(M, device="cuda") + 0.5)
    sb = (torch.rand(N, device="cuda") + 0.5)
    ref = (A.float() @ B.float().T) * sa[:, None] * sb[None, :]

    kern = builder(M, N, K, *cfg)
    C = kern(A, B, sa, sb)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 1e-3, f"{name} M={M} rel={rel:.4f}"

    lat = do_bench_cudagraph(lambda: kern(A, B, sa, sb))
    eff = N * K / lat * 1e-3  # GB/s
    print(f"{name:8s} M={M:>3} {str(cfg):<30}: {lat:8.1f}us  {eff:6.0f} GB/s  {eff / PEAK * 100:5.1f}%  rel={rel:.5f}", flush=True)
    return lat, eff


def main():
    N, K = 51200, 5120  # gate_up
    for M in [32, 64, 128]:
        print(f"\n== M={M} (hipblas: 66.8% / 62.2% / 46.0%) ==", flush=True)
        for kind, builder, cfg in [
            ("r-mls", _gemm_r_mls, (64, 64, 256, 2, 256)),
            ("r-mls", _gemm_r_mls, (64, 128, 256, 2, 512)),
            ("mls", _gemm_mls_mls, (64, 128, 64, 2, 256)),
            ("mls", _gemm_mls_mls, (128, 64, 64, 2, 256)),
        ]:
            try:
                bench(kind, N, K, M, builder, cfg)
            except Exception as ex:  # noqa: BLE001
                print(f"{kind:8s} M={M:>3} {str(cfg):<30}: ERR {str(ex)[:80]}", flush=True)


if __name__ == "__main__":
    main()
