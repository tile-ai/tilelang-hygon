# HCU note: ``T.gemm`` FP8 uses OCP ``e4m3fn`` / ``e5m2`` (see ``hcu_example_utils.fp8_tl_dtype``).

import torch
import tilelang
import tilelang.language as T
from tilelang.utils import determine_fp8_type

from hcu_example_utils import (
    cast_gemm_fp8_result,
    fp8_tl_dtype,
    gemm_fp8_output_dtype,
    gemm_tile_config,
    jit_pass_configs,
)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


@tilelang.jit(out_idx=[-1], pass_configs=jit_pass_configs())
def matmul(M, N, K, block_M, block_N, block_K, dtype, accum_dtype=T.float32, num_stages=3):
    out_dtype = gemm_fp8_output_dtype(dtype, accum_dtype)

    @T.prim_func
    def gemm_fp8(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_fp8


def _gemm_tile_config():
    h = gemm_tile_config(block_M=128, block_N=128, block_K=64, num_stages=3)
    return h["block_M"], h["block_N"], h["block_K"], h["num_stages"]


def test_gemm_fp8(M, N, K, dtype):
    torch_dtype = T.dtype(dtype).as_torch()

    bm, bn, bk, stages = _gemm_tile_config()
    kernel = matmul(M, N, K, bm, bn, bk, dtype, num_stages=stages)

    a = torch.randn(M, K, dtype=torch.float16, device="cuda").to(dtype=torch_dtype)
    b = torch.randn(N, K, dtype=torch.float16, device="cuda").to(dtype=torch_dtype)

    c = cast_gemm_fp8_result(kernel(a, b), dtype)
    ref_c = (a.half() @ b.half().T).to(dtype=torch_dtype)

    print(c)
    print(ref_c)

    diff = calc_diff(c, ref_c)
    print(f"diff: {diff}")
    assert diff < 1e-3


def main():
    test_gemm_fp8(1024, 1024, 1024, fp8_tl_dtype("e4m3"))
    test_gemm_fp8(1024, 1024, 1024, fp8_tl_dtype("e5m2"))


def run_regression_perf():
    M, N, K = 4096, 4096, 4096
    bm, bn, bk, stages = _gemm_tile_config()
    dtype = fp8_tl_dtype("e4m3")
    kernel_e4m3 = matmul(M, N, K, bm, bn, bk, dtype, num_stages=stages)
    profiler_e4m3 = kernel_e4m3.get_profiler(tilelang.TensorSupplyType.Integer)
    if torch.version.hip is None:
        latency_e4m3 = profiler_e4m3.do_bench(backend="cupti")
        dtype = determine_fp8_type("e5m2")
        kernel_e5m2 = matmul(M, N, K, 128, 128, 64, dtype)
        profiler_e5m2 = kernel_e5m2.get_profiler(tilelang.TensorSupplyType.Integer)
        latency_e5m2 = profiler_e5m2.do_bench(backend="cupti")
        return (latency_e4m3 + latency_e5m2) / 2
    latency_e4m3 = profiler_e4m3.do_bench()
    return latency_e4m3


if __name__ == "__main__":
    main()
