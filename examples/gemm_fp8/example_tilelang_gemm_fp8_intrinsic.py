import torch
from tilelang import tvm as tvm
import tilelang.testing
from tvm import DataType
import tilelang.language as T
from tilelang.intrinsics import make_mmac_swizzle_layout, make_mfma_swizzle_layout
from tilelang.intrinsics.hcu_mmac_macro_generator import HCUMatrixCoreIntrinEmitter
from tilelang.intrinsics.mma_macro_generator import TensorCoreIntrinEmitter
from tilelang.intrinsics.mfma_macro_generator import MatrixCoreIntrinEmitter
from tilelang.utils import determine_fp8_type
from tilelang.utils.target import determine_target

from hcu_example_utils import fp8_tl_dtype, uses_mmac_intrinsic

tilelang.testing.set_random_seed(0)


def _emitter_dtype(dtype):
    if isinstance(dtype, str):
        return dtype
    name = getattr(dtype, "name", None)
    if name:
        return name
    return str(dtype)


def make_swizzle_layout(shared_buf, use_hcu_mmac: bool, use_mfma: bool):
    if use_hcu_mmac:
        return make_mmac_swizzle_layout(shared_buf)
    if use_mfma:
        return make_mfma_swizzle_layout(shared_buf)
    dtype = shared_buf.dtype
    shape = shared_buf.shape
    can_swizzle = shape[-1] * DataType(dtype).bits == 512
    if not can_swizzle:
        return T.Layout(shape, lambda *args: args)

    from tilelang.intrinsics import get_swizzle_layout

    def transform_func(i, j):
        new_warp_i, new_warp_j = get_swizzle_layout(i, j, shape[-1], dtype)
        return [new_warp_i, new_warp_j]

    return T.Layout(shape, transform_func)


@tilelang.jit(out_idx=[2])
def tl_matmul(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
):
    assert in_dtype in [
        T.float16,
        T.float8_e4m3fn,
        T.float8_e4m3fnuz,
        T.float8_e5m2,
        T.float8_e5m2fnuz,
        T.int8,
    ], "Currently only float16, float8, and int8 are supported"
    assert out_dtype in [
        T.float16,
        T.float32,
        T.int32,
    ], "Currently only float16, float32 and int32 are supported"

    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 32
    warp_col_tiles = 32
    chunk = 32 if in_dtype == T.float16 else 64

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, K)
    B_shape = (N, K)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)

    use_hcu_mmac = uses_mmac_intrinsic()
    use_mfma = torch.version.hip is not None and not use_hcu_mmac
    target = determine_target("auto", return_object=True) if torch.version.hip is not None else None
    emitter_dtype = _emitter_dtype(in_dtype)
    emitter_accum = _emitter_dtype(accum_dtype)

    if use_hcu_mmac:
        mma_emitter = HCUMatrixCoreIntrinEmitter(
            a_dtype=emitter_dtype,
            b_dtype=emitter_dtype,
            accum_dtype=emitter_accum,
            a_transposed=False,
            b_transposed=True,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            block_m=block_M,
            block_n=block_N,
            chunk=chunk,
        )
        shared_scope = "shared"
        stage = 0
    elif use_mfma:
        mma_emitter = MatrixCoreIntrinEmitter(
            a_dtype=in_dtype,
            b_dtype=in_dtype,
            accum_dtype=accum_dtype,
            a_transposed=False,
            b_transposed=True,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            target=target,
        )
        shared_scope = "shared"
        stage = 0
    else:
        mma_emitter = TensorCoreIntrinEmitter(
            a_dtype=in_dtype,
            b_dtype=in_dtype,
            accum_dtype=accum_dtype,
            a_transposed=False,
            b_transposed=True,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
        )
        shared_scope = "shared.dyn"
        stage = 2

    micro_size_x = mma_emitter.M_DIM
    micro_size_y = getattr(mma_emitter, "n_dim", getattr(mma_emitter, "N_DIM", micro_size_x))
    micro_size_k = mma_emitter.k_dim
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    threads = mma_emitter.threads
    local_size_a = mma_emitter.local_size_a
    local_size_b = mma_emitter.local_size_b
    local_size_c = mma_emitter.local_size_out
    warp_rows = mma_emitter.warp_rows
    warp_cols = mma_emitter.warp_cols
    k_pack = getattr(mma_emitter, "k_pack", 1)
    ki_extent = block_K // (k_pack * micro_size_k)

    @T.prim_func
    def gemm_fp8_intrinsic(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)

            T.annotate_layout(
                {
                    A_shared: make_swizzle_layout(A_shared, use_hcu_mmac, use_mfma),
                    B_shared: make_swizzle_layout(B_shared, use_hcu_mmac, use_mfma),
                }
            )

            T.use_swizzle(panel_size=10)
            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=stage):
                if use_hcu_mmac or use_mfma:
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[bx * block_N, ko * block_K], B_shared)
                else:
                    for i, k in T.Parallel(block_M, block_K):
                        A_shared[i, k] = A[by * block_M + i, ko * block_K + k]
                    for j, k in T.Parallel(block_N, block_K):
                        B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                for ki in T.serial(0, ki_extent):
                    mma_emitter.ldmatrix_a(A_local, A_shared, ki)
                    mma_emitter.ldmatrix_b(B_local, B_shared, ki)
                    if use_hcu_mmac:
                        mma_emitter.mmac(A_local, B_local, C_local)
                    elif use_mfma:
                        mma_emitter.mfma(A_local, B_local, C_local, ki)
                    else:
                        mma_emitter.mma(A_local, B_local, C_local)

            if use_hcu_mmac:
                mma_emitter.stmatrix(C_local, C, pid_m=by, pid_n=bx)
            else:
                mma_emitter.stmatrix(C_local, C_shared)
                for i, j in T.Parallel(block_M, block_N):
                    C[by * block_M + i, bx * block_N + j] = C_shared[
                        i // micro_size_x,
                        j // micro_size_y,
                        i % micro_size_x,
                        j % micro_size_y,
                    ]

    return gemm_fp8_intrinsic


def assert_tl_matmul_correctness(M, N, K, in_dtype, out_dtype, accum_dtype):
    kernel = tl_matmul(M, N, K, in_dtype, out_dtype, accum_dtype)
    src_code = kernel.get_kernel_source()
    assert src_code is not None

    in_dtype = in_dtype.as_torch()
    out_dtype = out_dtype.as_torch()
    accum_dtype = accum_dtype.as_torch()

    if in_dtype in {torch.int8, torch.int32}:
        A = torch.randint(-128, 128, (M, K), dtype=torch.int8).to(in_dtype).cuda()
        B = torch.randint(-128, 128, (N, K), dtype=torch.int8).to(in_dtype).cuda()
    elif in_dtype in {torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz}:
        A = torch.randn(M, K).to(in_dtype).cuda()
        B = torch.randn(N, K).to(in_dtype).cuda()
    else:
        A = torch.randn(M, K).to(in_dtype).cuda() - 0.5
        B = torch.randn(N, K).to(in_dtype).cuda() - 0.5

    profiler = kernel.get_profiler(tilelang.TensorSupplyType.Integer)
    C = profiler(A, B)
    latency = profiler.do_bench(warmup=25)
    assert latency is not None

    ref_c = torch.matmul(A.to(accum_dtype), B.T.to(accum_dtype)).to(out_dtype)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def main():
    e4m3_dtype = fp8_tl_dtype("e4m3")
    e5m2_dtype = fp8_tl_dtype("e5m2")
    assert_tl_matmul_correctness(128, 128, 128, e4m3_dtype, T.float32, T.float32)
    assert_tl_matmul_correctness(128, 128, 128, e5m2_dtype, T.float32, T.float32)


def run_regression_perf():
    M, N, K = 4096, 4096, 4096
    out_dtype, accum_dtype = T.float32, T.float32
    in_dtype = determine_fp8_type()
    kernel_e4m3 = tl_matmul(M, N, K, in_dtype, out_dtype, accum_dtype)
    profiler_e4m3 = kernel_e4m3.get_profiler(tilelang.TensorSupplyType.Integer)
    if torch.version.hip is None:
        latency_e4m3 = profiler_e4m3.do_bench(backend="cupti")
    else:
        latency_e4m3 = profiler_e4m3.do_bench()
    return latency_e4m3


if __name__ == "__main__":
    main()
