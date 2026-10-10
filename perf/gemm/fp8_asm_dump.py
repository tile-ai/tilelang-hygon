# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Dump the generated asm for our native fp8 T.gemm and grep the key patterns."""
import os, re, sys
_REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import tilelang, tilelang.language as T


def native_fp8(M, N, K, block_M, block_N, block_K, num_stages, threads):
    @T.prim_func
    def gemm(A: T.Tensor((M, K), "float8_e4m3fn"), B: T.Tensor((N, K), "float8_e4m3fn"),
             C: T.Tensor((M, N), "float32")):
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


if __name__ == "__main__":
    k = sys.argv[1] if len(sys.argv) > 1 else "256x256x32ns3t256"
    parts = k.split("ns")
    bm, bn, bk = [int(x) for x in parts[0].split("x")]
    ns = int(parts[1].split("t")[0])
    thr = int(parts[1].split("t")[1])
    kern = native_fp8(8192, 8192, 8192, bm, bn, bk, ns, thr)
    src = kern.get_kernel_source()
    with open("/tmp/our_fp8_gemm_src.txt", "w") as f:
        f.write(src)
    print("source lines:", len(src.splitlines()))
    for pat in ["ds_read_m32x32_b8", "ds_read_b32", "ds_read_format", "ds_read",
                "v_mmac_f32_16x16x32_fp8_fp8", "v_mmac", "cp_async_gs", "cp_async",
                "buffer_load_dwordx4", "buffer_load_dword", "buffer_load", "matrix_load",
                "promote_prio", "s_waitcnt", "lgkmcnt", "ds_swizzle", "s_sched_group"]:
        print(f"  {pat:32s} {len(re.findall(pat, src)):5d}")
