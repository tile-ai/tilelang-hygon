# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""fp8 channelwise GEMM via the MLS (matrix LDS read) producer path.

Both A and B are loaded with ``T.copy(..., prefer_instruction="matrix_load")``
so ``T.gemm`` lowers through ``_gemm_mls_mls`` -> ``ds_read_format_tensor_a/b``
(HCU ``ds_read_m32x32_b8``, the same hardware matrix LDS read the hipblaslt fp8
kernel uses). Constraint: with ``trans_B=True`` the B MLS atom is the trans
b8 tile (tile_k 64/128) so ``block_K`` must be a multiple of 64.

Run: HIP_VISIBLE_DEVICES=2 python perf/gemm/fp8_mls_channelwise_bench.py
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


def fp8_mls_cw(M, N, K, bm, bn, bk, ns, thr):
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


def main():
    M = N = K = 8192
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.rand(M, device="cuda") + 0.5
    sb = torch.rand(N, device="cuda") + 0.5
    ref = (A.float() @ B.float().T) * sa[:, None] * sb[None, :]

    print(f"{'config':<32}{'TFLOPS':>9}{'rel':>10}")
    for (bm, bn) in [(128, 128), (256, 256), (256, 128), (128, 256)]:
        for thr in [256, 512, 1024]:
            for ns in [0, 1, 2]:
                label = f"mls {bm}x{bn}x64 ns{ns} t{thr}"
                try:
                    kern = fp8_mls_cw(M, N, K, bm, bn, 64, ns, thr)
                    C = kern(A, B, sa, sb)
                    rel = (C - ref).abs().max().item() / ref.abs().max().item()
                    if rel > 1e-3:
                        print(f"{label:<32} WRONG rel={rel:.4f}", flush=True)
                        continue
                    lat = do_bench_cudagraph(lambda: kern(A, B, sa, sb))
                    tf = 2.0 * M * N * K / lat * 1e-9
                    print(f"{label:<32}{tf:>9.1f}{rel:>10.5f}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"{label:<32} ERR {type(e).__name__}: {str(e)[:40]}", flush=True)


if __name__ == "__main__":
    main()
