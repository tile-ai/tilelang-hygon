import argparse
import tilelang
import tilelang.language as T

tilelang.disable_cache()


@tilelang.jit
def matmul(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=512) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        A_local = T.alloc_fragment((block_M, block_K), dtype)
        B_local = T.alloc_fragment((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(
            T.ceildiv(K, block_K),
            num_stages=4,
            enable_register_pipeline=True,
            enable_warp_divergence=True,
        ):
            T.copy(A[by * block_M, k * block_K], A_shared, enable_async=True)
            T.copy(B[k * block_K, bx * block_N], B_shared, enable_async=True)
            T.copy(A_shared, A_local)
            T.copy(B_shared, B_local)
            T.gemm(A_local, B_local, C_local, transpose_B=False, annotations={"trans_c": True})

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def main(M, N, K):
    # Pick block sizes that adapt to the problem dimensions.
    # Prefer the reference values (256/256/16) but cap at M, N, K so we still
    # work for arbitrary inputs (e.g. very small matrices).
    # block_M = min(256, M)
    # block_N = min(256, N)
    # block_K = min(16, K)
    
    block_M = 256
    block_N = 256
    block_K = 16

    kernel = matmul.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
    )

    import torch

    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()

    c = kernel(a, b)

    ref_c = a @ b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # Get CUDA Source
    print("CUDA Source:")
    print(kernel.get_kernel_source())
    # print(kernel.get_kernel_source(kernel_only=False))

    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    # latency = profiler.do_bench()
    tflops = 2 * M * N * K / latency * 1e-9
    print(f"tilelang Latency: {latency}ms")
    print(f"tilelang TFlops: {tflops:.4f}")


def run_regression_perf(M=1024, N=1024, K=1024):
    block_M = min(256, M)
    block_N = min(256, N)
    block_K = min(16, K)

    kernel = matmul.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
    )
    profiler = kernel.get_profiler()
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TileLang GEMM batch example (mirrors perf/gemm/example_gemm.py)."
    )
    parser.add_argument(
        "--m",
        type=int,
        default=19200,
        help="Number of rows of matrix A / output matrix C (default: 19200)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5120,
        help="Number of columns of matrix B / output matrix C (default: 5120)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5120,
        help="Number of columns of matrix A / rows of matrix B (default: 5120)",
    )

    args = parser.parse_args()

    main(args.m, args.n, args.k)