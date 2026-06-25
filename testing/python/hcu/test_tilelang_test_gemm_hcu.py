import pytest
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from hcu_test_utils import target_supports_fp8_mmac


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
    k_pack=1,
    policy=None,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    vec_size = (2 if in_dtype == "float32" else 4) * k_pack

    @T.prim_func
    def main(A: T.Tensor(A_shape, in_dtype), B: T.Tensor(B_shape, in_dtype), C: T.Tensor((M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared, coalesced_width=vec_size)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared, coalesced_width=vec_size)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared, coalesced_width=vec_size)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared, coalesced_width=vec_size)
                if policy is not None:
                    T.gemm(A_shared, B_shared, C_local, trans_A, trans_B, k_pack=k_pack, policy=policy)
                else:
                    T.gemm(A_shared, B_shared, C_local, trans_A, trans_B, k_pack=k_pack)
            # If using FullColK policy, need to reduce sum across k_warps
            if policy == T.GemmWarpPolicy.FullColK:
                T.reduce_sum_warp(C_local, C_local, clear=False)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=0,
    num_threads=128,
    k_pack=1,
    policy=None,
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
        k_pack=k_pack,
        policy=policy,
    )
    kernel = tl.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        return (A @ B).to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def _int8_ref_program(trans_A, trans_B):
    import torch

    def ref_program(A, B):
        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        return (A.to(torch.float32) @ B.to(torch.float32)).to(torch.int32)

    return ref_program


def _fp8_ref_program(trans_A, trans_B, out_dtype="float32"):
    import torch

    def ref_program(A, B):
        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        return (A.to(torch.float32) @ B.to(torch.float32)).to(getattr(torch, out_dtype))

    return ref_program


def run_gemm_int8(
    M,
    N,
    K,
    trans_A,
    trans_B,
    block_M,
    block_N,
    block_K,
    num_stages=0,
    num_threads=128,
    k_pack=1,
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        "int8",
        "int32",
        "int32",
        num_stages,
        num_threads,
        k_pack=k_pack,
    )
    kernel = tl.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    profiler.assert_allclose(_int8_ref_program(trans_A, trans_B), atol=0, rtol=0)


def run_gemm_fp8(
    M,
    N,
    K,
    trans_A,
    trans_B,
    block_M,
    block_N,
    block_K,
    in_dtype="float8_e4m3fn",
    out_dtype="float32",
    num_stages=0,
    num_threads=128,
    k_pack=1,
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        "float32",
        num_stages,
        num_threads,
        k_pack=k_pack,
    )
    kernel = tl.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    profiler.assert_allclose(_fp8_ref_program(trans_A, trans_B, out_dtype), atol=1e-2, rtol=1e-2)


def run_gemm_rs_int8(
    M,
    N,
    K,
    trans_A,
    trans_B,
    block_M,
    block_N,
    block_K,
    num_stages=0,
    num_threads=128,
    k_pack=1,
):
    program = matmul_rs(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        "int8",
        "int32",
        "int32",
        num_stages,
        num_threads,
        k_pack=k_pack,
    )
    kernel = tl.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    profiler.assert_allclose(_int8_ref_program(trans_A, trans_B), atol=0, rtol=0)


def run_gemm_rr_int8(
    M,
    N,
    K,
    trans_A,
    trans_B,
    block_M,
    block_N,
    block_K,
    num_stages=0,
    num_threads=128,
    k_pack=1,
):
    program = matmul_rr(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        "int8",
        "int32",
        "int32",
        num_stages,
        num_threads,
        k_pack=k_pack,
    )
    kernel = tl.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    profiler.assert_allclose(_int8_ref_program(trans_A, trans_B), atol=0, rtol=0)


def run_gemm_rs_fp8(
    M,
    N,
    K,
    trans_A,
    trans_B,
    block_M,
    block_N,
    block_K,
    in_dtype="float8_e4m3fn",
    out_dtype="float32",
    num_stages=0,
    num_threads=128,
    k_pack=1,
):
    program = matmul_rs(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        "float32",
        num_stages,
        num_threads,
        k_pack=k_pack,
    )
    kernel = tl.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    profiler.assert_allclose(_fp8_ref_program(trans_A, trans_B, out_dtype), atol=1e-2, rtol=1e-2)


def test_gemm_i8i32i32_nt():
    """gemm_ss: int8 A/B in shared -> T.gemm (Python lower path)."""
    run_gemm_int8(512, 512, 256, False, False, 128, 128, 32)
    run_gemm_int8(512, 512, 256, False, True, 128, 128, 32)
    run_gemm_int8(512, 512, 512, True, True, 128, 128, 32)
    run_gemm_int8(512, 512, 512, False, True, 128, 128, 64, k_pack=2)


def test_gemm_rs_i8i32i32_nt():
    """gemm_rs: A preloaded to fragment, B stays in shared."""
    run_gemm_rs_int8(512, 512, 256, False, False, 128, 128, 32)
    run_gemm_rs_int8(512, 512, 256, False, True, 128, 128, 32)
    run_gemm_rs_int8(512, 512, 512, True, False, 128, 128, 32)


def test_gemm_rr_i8i32i32_nt():
    """gemm_rr: A/B preloaded to fragment via shared."""
    run_gemm_rr_int8(512, 512, 256, False, False, 128, 128, 32)
    run_gemm_rr_int8(512, 512, 256, False, True, 128, 128, 32)
    run_gemm_rr_int8(512, 512, 512, True, True, 128, 128, 64, k_pack=2)


@pytest.mark.skipif(not target_supports_fp8_mmac(), reason="FP8 MMAC not supported on this target")
def test_gemm_fp8e4m3f32f32_nt():
    """gemm_ss: float8_e4m3fn A/B in shared -> T.gemm (Python lower path)."""
    run_gemm_fp8(512, 512, 256, False, False, 128, 128, 32)
    run_gemm_fp8(512, 512, 256, False, True, 128, 128, 32)
    run_gemm_fp8(512, 512, 512, True, True, 128, 128, 32)
    run_gemm_fp8(512, 512, 512, False, True, 128, 128, 64, k_pack=2)


@pytest.mark.skipif(not target_supports_fp8_mmac(), reason="FP8 MMAC not supported on this target")
def test_gemm_rs_fp8e4m3f32f32_nt():
    """gemm_rs: float8_e4m3fn A in fragment, B in shared -> T.gemm."""
    run_gemm_rs_fp8(512, 512, 256, False, False, 128, 128, 32)
    run_gemm_rs_fp8(512, 512, 256, False, True, 128, 128, 32)
    run_gemm_rs_fp8(512, 512, 512, True, False, 128, 128, 32)


@pytest.mark.skipif(not target_supports_fp8_mmac(), reason="FP8 MMAC not supported on this target")
def test_gemm_fp8e5m2f32f32_nt():
    """gemm_ss: float8_e5m2 (bf8) A/B in shared -> T.gemm."""
    run_gemm_fp8(
        512,
        512,
        256,
        False,
        True,
        128,
        128,
        32,
        in_dtype="float8_e5m2",
    )
    run_gemm_fp8(
        512,
        512,
        512,
        False,
        True,
        128,
        128,
        64,
        in_dtype="float8_e5m2",
        k_pack=2,
    )


def test_gemm_f16f32f32_nt():
    run_gemm(1024, 1024, 1024, False, False, "float16", "float32", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, False, True, "float16", "float32", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, True, True, "float16", "float32", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, True, False, "float16", "float32", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, False, True, "float16", "float32", "float32", 128, 128, 32, k_pack=2)


def test_gemm_bf16f32f32_nt():
    run_gemm(1024, 1024, 1024, False, False, "bfloat16", "float32", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, False, True, "bfloat16", "float32", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, True, True, "bfloat16", "float32", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, True, False, "bfloat16", "float32", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, False, True, "bfloat16", "float32", "float32", 128, 128, 32, k_pack=2)


def test_gemm_bf16bf16f32():
    run_gemm(1024, 1024, 1024, False, False, "bfloat16", "bfloat16", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, False, True, "bfloat16", "bfloat16", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, True, True, "bfloat16", "bfloat16", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, True, False, "bfloat16", "bfloat16", "float32", 128, 128, 32)
    run_gemm(1024, 1024, 1024, False, True, "bfloat16", "bfloat16", "float32", 128, 128, 32, k_pack=2)


def test_gemm_f16f32f32_nt_FullColK():
    """Test gemm_ss with FullColK policy (warp partitioning on K dimension)"""
    run_gemm(
        1024,
        1024,
        1024,
        False,
        False,
        "float16",
        "float32",
        "float32",
        128,
        64,
        32,
        k_pack=2,
        num_threads=256,
        policy=T.GemmWarpPolicy.FullColK,
    )
    run_gemm(1024, 1024, 1024, False, True, "float16", "float32", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK)
    run_gemm(1024, 1024, 1024, True, True, "float16", "float32", "float32", 128, 32, 64, num_threads=256, policy=T.GemmWarpPolicy.FullColK)
    run_gemm(
        1024,
        1024,
        1024,
        True,
        False,
        "float16",
        "float32",
        "float32",
        128,
        16,
        128,
        k_pack=2,
        num_threads=256,
        policy=T.GemmWarpPolicy.FullColK,
    )


def test_gemm_bf16f32f32_nt_FullColK():
    """Test gemm_ss with FullColK policy (warp partitioning on K dimension)"""
    run_gemm(
        1024, 1024, 1024, False, False, "bfloat16", "float32", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )
    run_gemm(
        1024, 1024, 1024, False, True, "bfloat16", "float32", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )
    run_gemm(1024, 1024, 1024, True, True, "bfloat16", "float32", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK)
    run_gemm(
        1024,
        1024,
        1024,
        True,
        False,
        "bfloat16",
        "float32",
        "float32",
        128,
        32,
        64,
        k_pack=2,
        num_threads=256,
        policy=T.GemmWarpPolicy.FullColK,
    )


def test_gemm_bf16bf16f32_FullColK():
    """Test gemm_ss with FullColK policy (warp partitioning on K dimension)"""
    run_gemm(
        1024, 1024, 1024, False, False, "bfloat16", "bfloat16", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )
    run_gemm(
        1024,
        1024,
        1024,
        False,
        True,
        "bfloat16",
        "bfloat16",
        "float32",
        128,
        32,
        64,
        k_pack=2,
        num_threads=256,
        policy=T.GemmWarpPolicy.FullColK,
    )
    run_gemm(
        1024, 1024, 1024, True, True, "bfloat16", "bfloat16", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )
    run_gemm(
        1024, 1024, 1024, True, False, "bfloat16", "bfloat16", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )


def test_gemm_f32f32f32_nt():
    """fp32 inputs; small shapes / blocks (HCU fp32 MMAC K-tile 8, keep block_K multiple of 8)."""
    run_gemm(512, 512, 512, False, False, "float32", "float32", "float32", 64, 64, 64, num_threads=128)
    run_gemm(512, 512, 512, False, True, "float32", "float32", "float32", 32, 32, 16, num_threads=128)
    run_gemm(512, 512, 512, True, True, "float32", "float32", "float32", 32, 32, 32, num_threads=128)
    run_gemm(512, 512, 512, True, False, "float32", "float32", "float32", 32, 32, 64, num_threads=128)


def matmul_rs(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
    k_pack=1,
    policy=None,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    vec_size = (2 if in_dtype == "float32" else 4) * k_pack

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            A_local = T.alloc_fragment(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared, coalesced_width=vec_size)
                    T.copy(A_shared, A_local)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared, coalesced_width=vec_size)
                    T.copy(A_shared, A_local)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared, coalesced_width=vec_size)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared, coalesced_width=vec_size)
                if policy is not None:
                    T.gemm(A_local, B_shared, C_local, trans_A, trans_B, k_pack=k_pack, policy=policy)
                else:
                    T.gemm(A_local, B_shared, C_local, trans_A, trans_B, k_pack=k_pack)
            # If using FullColK policy, need to reduce sum across k_warps
            if policy == T.GemmWarpPolicy.FullColK:
                T.reduce_sum_warp(C_local, C_local, clear=False)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def matmul_rr(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
    k_pack=1,
    policy=None,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    vec_size = (2 if in_dtype == "float32" else 4) * k_pack

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            A_local = T.alloc_fragment(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            B_local = T.alloc_fragment(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared, coalesced_width=vec_size)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared, coalesced_width=vec_size)
                T.copy(A_shared, A_local)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared, coalesced_width=vec_size)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared, coalesced_width=vec_size)
                T.copy(B_shared, B_local)
                if policy is not None:
                    T.gemm(A_local, B_local, C_local, trans_A, trans_B, k_pack=k_pack, policy=policy)
                else:
                    T.gemm(A_local, B_local, C_local, trans_A, trans_B, k_pack=k_pack)
            if policy == T.GemmWarpPolicy.FullColK:
                T.reduce_sum_warp(C_local, C_local, clear=False)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_rs(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=0,
    num_threads=128,
    k_pack=1,
    policy=None,
):
    program = matmul_rs(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
        k_pack=k_pack,
        policy=policy,
    )
    kernel = tl.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        return (A @ B).to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_gemm_rs_f16f32f32_nt():
    run_gemm_rs(1024, 1024, 1024, False, False, "float16", "float32", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, False, True, "float16", "float32", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, True, "float16", "float32", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, False, "float16", "float32", "float32", 128, 128, 32)


def test_gemm_rs_bf16f32f32_nt():
    run_gemm_rs(1024, 1024, 1024, False, False, "bfloat16", "float32", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, False, True, "bfloat16", "float32", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, True, "bfloat16", "float32", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, False, "bfloat16", "float32", "float32", 128, 128, 32)


def test_gemm_rs_bf16bf16f32_nt():
    run_gemm_rs(1024, 1024, 1024, False, False, "bfloat16", "bfloat16", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, False, True, "bfloat16", "bfloat16", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, True, "bfloat16", "bfloat16", "float32", 128, 128, 32)
    run_gemm_rs(1024, 1024, 1024, True, False, "bfloat16", "bfloat16", "float32", 128, 128, 32)


def test_gemm_rs_f16f32f32_nt_FullColK():
    """Test gemm_rs with FullColK policy (warp partitioning on K dimension)"""
    run_gemm_rs(
        1024, 1024, 1024, False, False, "float16", "float32", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )
    run_gemm_rs(
        1024, 1024, 1024, False, True, "float16", "float32", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )
    run_gemm_rs(
        1024, 1024, 1024, True, True, "float16", "float32", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )
    run_gemm_rs(
        1024,
        1024,
        1024,
        True,
        False,
        "float16",
        "float32",
        "float32",
        128,
        16,
        128,
        k_pack=2,
        num_threads=256,
        policy=T.GemmWarpPolicy.FullColK,
    )


def test_gemm_rs_bf16f32f32_nt_FullColK():
    """Test gemm_rs with FullColK policy (warp partitioning on K dimension)"""
    run_gemm_rs(
        1024, 1024, 1024, False, False, "bfloat16", "float32", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )
    run_gemm_rs(
        1024, 1024, 1024, False, True, "bfloat16", "float32", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )
    run_gemm_rs(
        1024, 1024, 1024, True, True, "bfloat16", "float32", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )
    run_gemm_rs(
        1024, 1024, 1024, True, False, "bfloat16", "float32", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )


def test_gemm_rs_bf16bf16f32_nt_FullColK():
    """Test gemm_rs with FullColK policy (warp partitioning on K dimension)"""
    run_gemm_rs(
        1024, 1024, 1024, False, False, "bfloat16", "bfloat16", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )
    run_gemm_rs(
        1024, 1024, 1024, False, True, "bfloat16", "bfloat16", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )
    run_gemm_rs(
        1024, 1024, 1024, True, True, "bfloat16", "bfloat16", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )
    run_gemm_rs(
        1024, 1024, 1024, True, False, "bfloat16", "bfloat16", "float32", 128, 32, 32, num_threads=256, policy=T.GemmWarpPolicy.FullColK
    )


def test_gemm_rs_f32f32f32_nt():
    """fp32 inputs with register-spill path (A_local); small shapes / blocks."""
    run_gemm_rs(512, 512, 256, False, False, "float32", "float32", "float32", 32, 32, 64, num_threads=128)
    run_gemm_rs(512, 512, 256, False, True, "float32", "float32", "float32", 32, 32, 16, num_threads=128)
    run_gemm_rs(256, 512, 256, True, True, "float32", "float32", "float32", 32, 32, 32, num_threads=128)
    run_gemm_rs(256, 256, 256, True, False, "float32", "float32", "float32", 32, 32, 64, num_threads=128)


if __name__ == "__main__":
    tilelang.testing.main()
