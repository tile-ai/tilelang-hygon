"""GEMM with MLS (Matrix Load Store) - both A and B from MLS LDS."""

import re

import pytest
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.utils.target import determine_target, target_is_hcu


def _is_hcu_target_available() -> bool:
    try:
        return target_is_hcu(tvm.target.Target(determine_target("auto")))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _is_hcu_target_available(),
    reason="matrix_load tests require an HCU target",
)


def _waitcnt_imms(source: str) -> list[int]:
    return [int(x) for x in re.findall(r"__builtin_amdgcn_s_waitcnt\((\d+)\)", source)]


def _vmcnt_keep(imm: int) -> int:
    return imm & 0xF


def _assert_each_waitcnt_followed_by_sync(source: str) -> None:
    waits = _waitcnt_imms(source)
    sync_after_waits = len(
        re.findall(
            r"__builtin_amdgcn_s_waitcnt\(\d+\);\s*\n\s*__syncthreads\(\);",
            source,
        )
    )
    assert sync_after_waits == len(waits), f"expected waitcnt+sync pairs for all {len(waits)} waits, got {sync_after_waits}"


def _assert_mls_direct_stage1_waitcnt(source: str) -> None:
    """Stage1 direct LDS: wait all outstanding MLS before each gemm_mls_mls consumer."""
    waits = _waitcnt_imms(source)
    assert waits == [16368, 16368], f"unexpected direct stage1 waitcnt sequence: {waits}"
    assert "gemm_mls_mls" in source
    assert "ds_read_format" not in source
    _assert_each_waitcnt_followed_by_sync(source)
    assert re.search(
        r"__builtin_amdgcn_s_waitcnt\(16368\);\s*\n\s*__syncthreads\(\);\s*\n\s*\{?\s*\n\s*"
        r"tl::gemm_mls_mls",
        source,
    )


def _assert_mls_ds_stage1_waitcnt(source: str) -> None:
    """Stage1 ds_read: A consumer keep=1; B consumer keep=1 after in-loop A reload."""
    waits = _waitcnt_imms(source)
    keeps = [_vmcnt_keep(w) for w in waits]
    assert keeps == [1, 1, 1, 0], f"unexpected ds stage1 vmcnt keep sequence: {keeps}"
    assert source.count("ds_read_format_tensor_a") >= 2
    assert source.count("ds_read_format_tensor_b") >= 2
    _assert_each_waitcnt_followed_by_sync(source)
    ds_waits = len(
        re.findall(
            r"__builtin_amdgcn_s_waitcnt\(\d+\);\s*\n\s*__syncthreads\(\);\s*\n\s*"
            r"tl::mls::ds_read_format",
            source,
        )
    )
    assert ds_waits == 4, f"expected wait+sync before 4 ds_read consumers, got {ds_waits}"


def _assert_mls_direct_stage2_waitcnt(source: str) -> None:
    """Stage2 direct LDS: merged wait before loop gemm (keep=2) and epilogue gemm (keep=0)."""
    waits = _waitcnt_imms(source)
    keeps = [_vmcnt_keep(w) for w in waits]
    assert keeps == [2, 0], f"unexpected direct stage2 vmcnt keep sequence: {keeps}"
    assert "gemm_mls_mls" in source
    assert "ds_read_format" not in source
    _assert_each_waitcnt_followed_by_sync(source)
    gemm_waits = len(
        re.findall(
            r"__builtin_amdgcn_s_waitcnt\(\d+\);\s*\n\s*__syncthreads\(\);\s*\n\s*\{?\s*\n?\s*"
            r"tl::gemm_mls_mls",
            source,
        )
    )
    assert gemm_waits == 2, f"expected wait+sync before 2 gemm consumers, got {gemm_waits}"


def _assert_mls_ds_stage2_waitcnt(source: str) -> None:
    """Stage2 ds_read: pipelined loop + epilogue drain keeps [3,3,3,2,1,0]."""
    waits = _waitcnt_imms(source)
    keeps = [_vmcnt_keep(w) for w in waits]
    assert keeps == [3, 3, 3, 2, 1, 0], f"unexpected ds stage2 vmcnt keep sequence: {keeps}"
    assert source.count("ds_read_format_tensor_a") >= 3
    assert source.count("ds_read_format_tensor_b") >= 3
    _assert_each_waitcnt_followed_by_sync(source)
    ds_waits = len(
        re.findall(
            r"__builtin_amdgcn_s_waitcnt\(\d+\);\s*\n\s*__syncthreads\(\);\s*\n\s*"
            r"tl::mls::ds_read_format",
            source,
        )
    )
    assert ds_waits == 6, f"expected wait+sync before 6 ds_read consumers, got {ds_waits}"


def _assert_mls_copy_a_mls_b_ds_stage1_waitcnt(source: str) -> None:
    """Mixed copy-A + MLS-B ds_read stage1: B-only MLS, no in-loop A reload -> keep=0."""
    waits = _waitcnt_imms(source)
    keeps = [_vmcnt_keep(w) for w in waits]
    assert keeps == [0, 0], f"unexpected mixed ds stage1 vmcnt keep sequence: {keeps}"
    assert "ds_read_format_tensor_b" in source
    assert "ds_read_format_tensor_a" not in source
    assert "gemm_rr" in source
    _assert_each_waitcnt_followed_by_sync(source)
    ds_waits = len(
        re.findall(
            r"__builtin_amdgcn_s_waitcnt\(\d+\);\s*\n\s*__syncthreads\(\);\s*\n\s*"
            r"tl::mls::ds_read_format_tensor_b",
            source,
        )
    )
    assert ds_waits == 2, f"expected wait+sync before 2 B ds_read consumers, got {ds_waits}"


def _assert_mls_copy_a_mls_b_ds_stage2_waitcnt(source: str) -> None:
    """Mixed copy-A + MLS-B ds_read stage2: pipelined B-only MLS keeps [1,1,0]."""
    waits = _waitcnt_imms(source)
    keeps = [_vmcnt_keep(w) for w in waits]
    assert keeps == [1, 1, 0], f"unexpected mixed ds stage2 vmcnt keep sequence: {keeps}"
    assert "ds_read_format_tensor_b" in source
    assert "ds_read_format_tensor_a" not in source
    assert "gemm_rr" in source
    _assert_each_waitcnt_followed_by_sync(source)
    ds_waits = len(
        re.findall(
            r"__builtin_amdgcn_s_waitcnt\(\d+\);\s*\n\s*__syncthreads\(\);\s*\n\s*"
            r"tl::mls::ds_read_format_tensor_b",
            source,
        )
    )
    assert ds_waits == 3, f"expected wait+sync before 3 B ds_read consumers, got {ds_waits}"


def matmul_mls(
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
    threads,
    k_pack=1,
    num_stages=0,
):
    """GEMM with both A and B loaded via matrix_load (MLS)."""
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.matrix_load(A[k * block_K, by * block_M], A_shared)
                else:
                    T.matrix_load(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.matrix_load(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.matrix_load(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B, k_pack=k_pack)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def matmul_mls_ds_read_format(
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
    threads,
    num_stages=0,
    k_pack=1,
):
    """GEMM with both A and B: matrix_load -> ds_read_format -> gemm."""
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            A_fragment = T.alloc_fragment(A_shared_shape, in_dtype)
            B_fragment = T.alloc_fragment(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.matrix_load(A[k * block_K, by * block_M], A_shared)
                else:
                    T.matrix_load(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.matrix_load(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.matrix_load(B[k * block_K, bx * block_N], B_shared)
                T.ds_read_format(A_shared, A_fragment)
                T.ds_read_format(B_shared, B_fragment)
                T.gemm(A_fragment, B_fragment, C_local, trans_A, trans_B, k_pack=k_pack)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def matmul_mls_ds_read_format_mul_scale(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    scale_a,
    scale_b,
    in_dtype,
    out_dtype,
    accum_dtype,
    threads,
    num_stages=0,
    k_pack=1,
):
    """GEMM with both A and B: matrix_load -> ds_read_format -> mul scale -> gemm."""
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            A_fragment = T.alloc_fragment(A_shared_shape, in_dtype)
            B_fragment = T.alloc_fragment(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                if trans_A:
                    T.matrix_load(A[k * block_K, by * block_M], A_shared)
                else:
                    T.matrix_load(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.matrix_load(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.matrix_load(B[k * block_K, bx * block_N], B_shared)
                T.ds_read_format(A_shared, A_fragment)
                T.ds_read_format(B_shared, B_fragment)
                for i, j in T.Parallel(A_shared_shape[0], A_shared_shape[1]):
                    A_fragment[i, j] = A_fragment[i, j] * scale_a
                for i, j in T.Parallel(B_shared_shape[0], B_shared_shape[1]):
                    B_fragment[i, j] = B_fragment[i, j] * scale_b
                T.gemm(A_fragment, B_fragment, C_local, trans_A, trans_B, k_pack=k_pack)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def matmul_mls_copy_a_mls_b_ds(
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
    threads,
    num_stages=0,
    k_pack=1,
):
    """GEMM with A via T.copy -> shared -> fragment, B via matrix_load -> ds_read_format -> gemm."""
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
            A_fragment = T.alloc_fragment(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            B_fragment = T.alloc_fragment(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared, coalesced_width=vec_size)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared, coalesced_width=vec_size)
                T.copy(A_shared, A_fragment)
                if trans_B:
                    T.matrix_load(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.matrix_load(B[k * block_K, bx * block_N], B_shared)
                T.ds_read_format(B_shared, B_fragment)
                T.gemm(A_fragment, B_fragment, C_local, trans_A, trans_B, k_pack=k_pack)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def matmul_mls_b_shared_a(
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
    threads,
    k_pack=1,
    num_stages=0,
):
    """GEMM with A shared and B loaded via matrix_load (MLS)."""
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    # T.matrix_load(A[k * block_K, by * block_M], A_shared)
                    T.copy(A[k * block_K, by * block_M], A_shared, coalesced_width=4)
                else:
                    # T.matrix_load(A[by * block_M, k * block_K], A_shared)
                    T.copy(A[by * block_M, k * block_K], A_shared, coalesced_width=4)
                if trans_B:
                    T.matrix_load(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.matrix_load(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B, k_pack=k_pack)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_mls_b_only(
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
    num_threads=128,
    num_stages=0,
    k_pack=1,
):
    program = matmul_mls_b_shared_a(
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
        num_threads,
        num_stages=num_stages,
        k_pack=k_pack,
    )
    kernel = tl.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if "float8" in in_dtype:
            A = A.to(torch.float32)
            B = B.to(torch.float32)
        return (A @ B).to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def run_gemm_mls(
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
    num_threads=128,
    k_pack=1,
    num_stages=0,
    verify_source=None,
):
    program = matmul_mls(
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
        num_threads,
        k_pack=k_pack,
        num_stages=num_stages,
    )
    kernel = tl.compile(program, out_idx=[2])
    if verify_source is not None:
        verify_source(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if "float8" in in_dtype:
            A = A.to(torch.float32)
            B = B.to(torch.float32)
        return (A @ B).to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def run_gemm_mls_ds_read_format(
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
    num_threads=128,
    num_stages=0,
    k_pack=1,
    verify_source=None,
):
    """Run GEMM with matrix_load + ds_read_format for both A and B."""
    program = matmul_mls_ds_read_format(
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
        num_threads,
        num_stages=num_stages,
        k_pack=k_pack,
    )
    kernel = tl.compile(program, out_idx=[2])
    if verify_source is not None:
        verify_source(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if "float8" in in_dtype:
            A = A.to(torch.float32)
            B = B.to(torch.float32)
        return (A @ B).to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def run_gemm_mls_copy_a_mls_b_ds(
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
    num_threads=128,
    num_stages=0,
    k_pack=1,
    verify_source=None,
):
    """Run GEMM with A via T.copy->fragment, B via matrix_load->ds_read_format->gemm."""
    program = matmul_mls_copy_a_mls_b_ds(
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
        num_threads,
        num_stages=num_stages,
        k_pack=k_pack,
    )
    kernel = tl.compile(program, out_idx=[2])
    if verify_source is not None:
        verify_source(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if "float8" in in_dtype:
            A = A.to(torch.float32)
            B = B.to(torch.float32)
        return (A @ B).to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def run_gemm_mls_ds_read_format_mul_scale(
    M,
    N,
    K,
    trans_A,
    trans_B,
    scale_a,
    scale_b,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_threads=128,
    k_pack=1,
):
    """Run GEMM with matrix_load + ds_read_format + mul scale for both A and B."""
    program = matmul_mls_ds_read_format_mul_scale(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        scale_a,
        scale_b,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_threads,
        k_pack=k_pack,
    )
    kernel = tl.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        return (scale_a * scale_b * (A @ B)).to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_gemm_shared_A_False_mls_B_True_1():
    """trans_A=False, trans_B=True. Single block: A(32,64), B(32,64), C(32,32)."""
    run_gemm_mls_b_only(
        M=32,
        N=32,
        K=64,
        trans_A=False,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
    )


def test_gemm_mls_A_False_B_True_1():
    """trans_A=False, trans_B=True. Single block: A(32,64), B(32,64), C(32,32)."""
    run_gemm_mls(
        M=32,
        N=32,
        K=64,
        trans_A=False,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
    )


def test_gemm_mls_A_False_B_True_2():
    """trans_A=False, trans_B=True. Multi-block: A(64,64), B(64,64), C(64,64), block 32x32x64."""
    run_gemm_mls(
        M=64,
        N=64,
        K=64,
        trans_A=False,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
    )


def test_gemm_mls_A_False_B_True_2_stage0():
    """trans_A=False, trans_B=True. Multi-iteration K loop with stage0 pipeline."""
    run_gemm_mls(
        M=64,
        N=64,
        K=256,
        trans_A=False,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=32,
        num_threads=128,
        num_stages=0,
    )


def test_gemm_mls_A_False_B_True_2_stage1():
    """trans_A=False, trans_B=True. Multi-iteration K loop with stage1 pipeline."""
    run_gemm_mls(
        M=64,
        N=64,
        K=256,
        trans_A=False,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=32,
        num_threads=128,
        num_stages=1,
        verify_source=_assert_mls_direct_stage1_waitcnt,
    )


def test_gemm_mls_A_False_B_True_2_stage2():
    """trans_A=False, trans_B=True. Multi-iteration K loop with stage2 pipeline."""
    run_gemm_mls(
        M=64,
        N=64,
        K=256,
        trans_A=False,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=32,
        num_threads=128,
        num_stages=2,
        verify_source=_assert_mls_direct_stage2_waitcnt,
    )
    """trans_A=False, trans_B=True. Non-power-of-2, multi-loop: M=96, N=96, K=192 (2x2x2 blocks)."""
    run_gemm_mls(
        M=96,
        N=96,
        K=198,
        trans_A=False,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=64,
        block_N=64,
        block_K=128,
        num_threads=128,
    )


def test_gemm_mls_A_True_B_False_1():
    """trans_A=True, trans_B=False. Single block: A(64,32), B(32,64), C(64,64)."""
    run_gemm_mls(
        M=64,
        N=64,
        K=32,
        trans_A=True,
        trans_B=False,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=64,
        block_N=64,
        block_K=32,
        num_threads=128,
    )


# AN BT
def test_gemm_mls_A_True_B_False_2():
    """trans_A=True, trans_B=False. Multi-block: A(64,64), B(64,64), C(64,64), block 32x32x64."""
    run_gemm_mls(
        M=64,
        N=64,
        K=64,
        trans_A=True,
        trans_B=False,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
    )


def test_gemm_mls_A_True_B_False_3():
    """trans_A=True, trans_B=False. Non-power-of-2, multi-loop: M=96, N=96, K=192 (2x2x2 blocks)."""
    run_gemm_mls(
        M=96,
        N=96,
        K=241,
        trans_A=True,
        trans_B=False,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=64,
        block_N=64,
        block_K=128,
        num_threads=128,
    )


def test_gemm_mls_A_False_B_False_1():
    """trans_A=False, trans_B=False. Single block: A(32,64), B(64,32), C(32,32)."""
    run_gemm_mls(
        M=32,
        N=32,
        K=64,
        trans_A=False,
        trans_B=False,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
    )


def test_gemm_mls_A_False_B_False_2():
    """trans_A=False, trans_B=False. Multi-block: A(64,64), B(64,64), C(64,64), block 32x32x64."""
    run_gemm_mls(
        M=64,
        N=64,
        K=64,
        trans_A=False,
        trans_B=False,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
    )


def test_gemm_mls_A_False_B_False_3():
    """trans_A=False, trans_B=False. Non-power-of-2, multi-loop: M=96, N=96, K=192 (2x2x2 blocks)."""
    run_gemm_mls(
        M=96,
        N=96,
        K=242,
        trans_A=False,
        trans_B=False,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=64,
        block_N=64,
        block_K=128,
        num_threads=128,
    )


def test_gemm_mls_A_True_B_True_1():
    """trans_A=True, trans_B=True. Single block: A(64,32), B(64,32), C(32,32)."""
    run_gemm_mls(
        M=32,
        N=32,
        K=64,
        trans_A=True,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
    )


def test_gemm_mls_A_True_B_True_2():
    """trans_A=True, trans_B=True. Multi-block: A(64,64), B(64,64), C(64,64), block 32x32x64."""
    run_gemm_mls(
        M=64,
        N=64,
        K=64,
        trans_A=True,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
    )


def test_gemm_mls_A_True_B_True_3():
    """trans_A=True, trans_B=True. Non-power-of-2, multi-loop: M=96, N=96, K=192 (2x2x2 blocks)."""
    run_gemm_mls(
        M=96,
        N=96,
        K=242,
        trans_A=True,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=64,
        block_N=64,
        block_K=128,
        num_threads=128,
    )


# -------------------------------------
# AN/BT matrix_load_64x16_b8 cases
# -------------------------------------


# NOTE: actually Compiler would choose matrix_load_64x32_b8
#       to test this, please comment out 64x32 in kMlsTileConfigsB8NonTrans table
def test_gemm_mls_A_True_mls_B_False_64x16_SingleBlock_fp8():
    """trans_A=True, trans_B=False. Single block float8 test: A(32,64), B(32,64), C(64,64)."""
    run_gemm_mls(
        M=64,
        N=64,
        K=32,
        trans_A=True,
        trans_B=False,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=64,
        block_N=64,
        block_K=32,
        num_threads=64,
    )


# -------------------------------------
# AN/BT matrix_load_64x32_b8 cases
# -------------------------------------


def test_gemm_mls_A_True_mls_B_False_64x32_SingleBlock_fp8():
    """trans_A=True, trans_B=False. Single block float8 test: A(32,64), B(32,64), C(64,64)."""
    run_gemm_mls(
        M=64,
        N=64,
        K=32,
        trans_A=True,
        trans_B=False,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=64,
        block_N=64,
        block_K=32,
        num_threads=64,
    )


def test_gemm_mls_A_True_mls_B_False_64x16_SingleBlock_KMask_fp8():
    """trans_A=True, trans_B=False. Single block float8 test: A(32,64), B(32,64), C(64,64)."""
    run_gemm_mls(
        M=64,
        N=64,
        K=41,
        trans_A=True,
        trans_B=False,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=64,
        block_N=64,
        block_K=32,
        num_threads=64,
    )


def test_gemm_mls_A_True_mls_B_False_64x32_MultiBlock_fp8():
    """trans_A=True, trans_B=False. Single block float8 test: A(32,64), B(32,64), C(64,64)."""
    run_gemm_mls(
        M=128,
        N=128,
        K=32,
        trans_A=True,
        trans_B=False,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=64,
        block_N=64,
        block_K=32,
        num_threads=64,
    )


# -------------------------------------
# AN/BT matrix_load_128x16_b8 cases
# -------------------------------------


def test_gemm_mls_A_True_mls_B_False_128x16_SingleBlock_fp8():
    """trans_A=True, trans_B=False. Single block float8 test: A(32,64), B(32,64), C(64,64)."""
    run_gemm_mls(
        M=128,
        N=128,
        K=32,
        trans_A=True,
        trans_B=False,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=128,
        block_N=128,
        block_K=32,
        num_threads=64,
    )


def test_gemm_mls_A_True_mls_B_False_128x16_MultiBlock_fp8():
    """trans_A=True, trans_B=False. Multi block float8 test: A(32,256), B(32,256), C(64,64)."""
    run_gemm_mls(
        M=256,
        N=256,
        K=32,
        trans_A=True,
        trans_B=False,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=128,
        block_N=128,
        block_K=32,
        num_threads=64,
    )


def test_gemm_mls_A_True_mls_B_False_128x16_MultiTile_fp8():
    """trans_A=True, trans_B=False. Multi block float8 test: A(32,256), B(32,256), C(64,64)."""
    run_gemm_mls(
        M=256,
        N=256,
        K=32,
        trans_A=True,
        trans_B=False,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=256,
        block_N=256,
        block_K=32,
        num_threads=128,
    )


# -----------------------------------------
# AT/BN  matrix_load_trans_64x16_b8 cases
# -----------------------------------------


def test_gemm_mls_A_False_mls_B_True_16x64_SingleBlock_fp8():
    """trans_A=False, trans_B=True. Single block float8 test: A(16,64), B(16,64), C(16,16)."""
    run_gemm_mls(
        M=16,
        N=16,
        K=64,
        trans_A=False,
        trans_B=True,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=16,
        block_N=16,
        block_K=64,
        num_threads=64,
    )


def test_gemm_mls_A_False_mls_B_True_16x64_MultiBlock_fp8():
    """trans_A=False, trans_B=True. Multi block_MN float8 test: A(32,64), B(32,64), C(32,32)."""
    run_gemm_mls(
        M=32,
        N=32,
        K=64,
        trans_A=False,
        trans_B=True,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=16,
        block_N=16,
        block_K=64,
        num_threads=64,
    )


# -----------------------------------------
# AT/BN matrix_load_trans_64x32_b8 cases
# -----------------------------------------


def test_gemm_mls_A_False_B_True_1_32x64_SingleBlock_fp8():
    """trans_A=True, trans_B=False. Both A and B from mls + ds_read_format -> gemm."""
    run_gemm_mls(
        M=32,
        N=32,
        K=64,
        trans_A=False,
        trans_B=True,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=64,
    )


def test_gemm_mls_A_False_B_True_1_32x64_MultiBlock():
    """trans_A=False, trans_B=True. Both A and B from mls + ds_read_format -> gemm."""
    run_gemm_mls(
        M=64,
        N=64,
        K=64,
        trans_A=False,
        trans_B=True,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=64,
        block_N=64,
        block_K=64,
        num_threads=128,
    )


# -----------------------------------------
# AT/BN matrix_load_trans_128x16_b8 cases
# -----------------------------------------


def test_gemm_mls_A_True_mls_B_False_16x128_SingleBlock_fp8():
    """trans_A=True, trans_B=False. Single block float8 test: A(32,64), B(32,64), C(64,64)."""
    run_gemm_mls(
        M=16,
        N=16,
        K=128,
        trans_A=False,
        trans_B=True,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=16,
        block_N=16,
        block_K=128,
        num_threads=64,
    )


def test_gemm_mls_A_True_mls_B_False_16x128_MultiBlock_fp8():
    """trans_A=True, trans_B=False. Single block float8 test: A(32,64), B(32,64), C(64,64)."""
    run_gemm_mls(
        M=32,
        N=32,
        K=256,
        trans_A=False,
        trans_B=True,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=16,
        block_N=16,
        block_K=128,
        num_threads=64,
    )


def test_gemm_mls_A_True_mls_B_False_16x128_MultiTile_fp8():
    """trans_A=True, trans_B=False. Single block float8 test: A(32,64), B(32,64), C(64,64)."""
    run_gemm_mls(
        M=32,
        N=32,
        K=256,
        trans_A=False,
        trans_B=True,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=128,
        num_threads=64,
    )


def test_gemm_mls_ds_A_True_B_True_1():
    """trans_A=True, trans_B=True. Both A and B from mls + ds_read_format -> gemm. Single block."""
    run_gemm_mls_ds_read_format(
        M=32,
        N=32,
        K=64,
        trans_A=True,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
    )


def test_gemm_mls_ds_A_True_B_True_2():
    """trans_A=True, trans_B=True. Both A and B from mls + ds_read_format -> gemm. Multi-block."""
    run_gemm_mls_ds_read_format(
        M=64,
        N=64,
        K=64,
        trans_A=True,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
    )


def test_gemm_mls_ds_A_True_B_False_1():
    """trans_A=True, trans_B=False. Both A and B from mls + ds_read_format -> gemm."""
    run_gemm_mls_ds_read_format(
        M=64,
        N=64,
        K=32,
        trans_A=True,
        trans_B=False,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=64,
        block_N=64,
        block_K=32,
        num_threads=128,
    )


def test_gemm_mls_ds_A_False_B_True_1():
    """trans_A=False, trans_B=True. Both A and B from mls + ds_read_format -> gemm."""
    run_gemm_mls_ds_read_format(
        M=32,
        N=32,
        K=64,
        trans_A=False,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
    )


def test_gemm_mls_ds_A_False_B_True_stage1():
    """trans_A=False, trans_B=True. Stage1 matrix_load -> ds_read_format -> gemm."""
    run_gemm_mls_ds_read_format(
        M=32,
        N=32,
        K=256,
        trans_A=False,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
        num_stages=1,
        verify_source=_assert_mls_ds_stage1_waitcnt,
    )


def test_gemm_mls_ds_A_False_B_True_stage2():
    """trans_A=False, trans_B=True. Pipelined matrix_load -> ds_read_format -> gemm."""
    run_gemm_mls_ds_read_format(
        M=32,
        N=32,
        K=256,
        trans_A=False,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
        num_stages=2,
        verify_source=_assert_mls_ds_stage2_waitcnt,
    )


def test_gemm_mls_copy_a_mls_b_ds_A_False_B_True_stage1():
    """A via T.copy->fragment, B via matrix_load->ds_read_format->gemm, stage1."""
    run_gemm_mls_copy_a_mls_b_ds(
        M=32,
        N=32,
        K=256,
        trans_A=False,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
        num_stages=1,
        verify_source=_assert_mls_copy_a_mls_b_ds_stage1_waitcnt,
    )


def test_gemm_mls_copy_a_mls_b_ds_A_False_B_True_stage2():
    """A via T.copy->fragment, B via matrix_load->ds_read_format->gemm, stage2."""
    run_gemm_mls_copy_a_mls_b_ds(
        M=32,
        N=32,
        K=256,
        trans_A=False,
        trans_B=True,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
        num_stages=2,
        verify_source=_assert_mls_copy_a_mls_b_ds_stage2_waitcnt,
    )


def test_gemm_mls_ds_A_True_B_False_64x32_SingleBlock_fp8():
    """trans_A=True, trans_B=False. FP8 ds_read_format coverage for 64x32 b8 shape."""
    run_gemm_mls_ds_read_format(
        M=64,
        N=64,
        K=32,
        trans_A=True,
        trans_B=False,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=64,
        block_N=64,
        block_K=32,
        num_threads=64,
    )


def test_gemm_mls_ds_A_True_B_False_128x16_SingleBlock_fp8():
    """trans_A=True, trans_B=False. FP8 ds_read_format coverage for 128x16 b8 shape."""
    run_gemm_mls_ds_read_format(
        M=128,
        N=128,
        K=32,
        trans_A=True,
        trans_B=False,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=128,
        block_N=128,
        block_K=32,
        num_threads=64,
    )


def test_gemm_mls_ds_A_False_B_True_16x64_SingleBlock_fp8():
    """trans_A=False, trans_B=True. FP8 ds_read_format coverage for 16x64 b8 shape (alt=1)."""
    run_gemm_mls_ds_read_format(
        M=16,
        N=16,
        K=64,
        trans_A=False,
        trans_B=True,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=16,
        block_N=16,
        block_K=64,
        num_threads=64,
    )


def test_gemm_mls_ds_A_False_B_True_32x64_SingleBlock_fp8():
    """trans_A=False, trans_B=True. FP8 ds_read_format coverage for 32x64 b8 shape."""
    run_gemm_mls_ds_read_format(
        M=32,
        N=32,
        K=64,
        trans_A=False,
        trans_B=True,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=64,
    )


def test_gemm_mls_ds_A_False_B_True_16x128_SingleBlock_fp8():
    """trans_A=False, trans_B=True. FP8 ds_read_format coverage for 16x128 b8 shape."""
    run_gemm_mls_ds_read_format(
        M=16,
        N=16,
        K=128,
        trans_A=False,
        trans_B=True,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=16,
        block_N=16,
        block_K=128,
        num_threads=64,
    )


def test_gemm_mls_ds_mul_scale_1():
    """Both A and B from mls + ds_read_format + mul scale -> gemm."""
    run_gemm_mls_ds_read_format_mul_scale(
        M=32,
        N=32,
        K=64,
        trans_A=False,
        trans_B=True,
        scale_a=0.5,
        scale_b=0.2,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=64,
        num_threads=128,
    )


def mls_ds_read_format_copy_to_global(
    M,
    K,
    block_M,
    block_K,
    in_dtype,
    out_dtype,
    threads,
):
    """matrix_load -> ds_read_format -> copy(fragment to global). No gemm, consumer is copy. No trans."""
    A_shape = (M, K)
    shared_shape = (block_M, block_K)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        C: T.Tensor(A_shape, out_dtype),
    ):
        with T.Kernel(T.ceildiv(K, block_K), T.ceildiv(M, block_M), threads=threads) as (bk, bm):
            A_shared = T.alloc_shared(shared_shape, in_dtype)
            A_fragment = T.alloc_fragment(shared_shape, in_dtype)
            T.matrix_load(A[bm * block_M, bk * block_K], A_shared)
            T.ds_read_format(A_shared, A_fragment)
            T.copy(A_fragment, C[bm * block_M, bk * block_K])

    return main


def run_mls_ds_read_format_copy_to_global(
    M,
    K,
    block_M,
    block_K,
    in_dtype,
    out_dtype,
    num_threads=128,
):
    """Run mls + ds_read_format + copy(global), no gemm."""
    program = mls_ds_read_format_copy_to_global(
        M,
        K,
        block_M,
        block_K,
        in_dtype,
        out_dtype,
        num_threads,
    )
    kernel = tl.compile(program, out_idx=[1])
    profiler = kernel.get_profiler()

    def ref_program(A):
        import torch

        return A.to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def matmul_mls_n_loop(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    threads,
    k_pack=1,
):
    """GEMM that serially walks N tiles inside one CTA, exercising MN-window MLS reuse."""
    A_shape = (M, K)
    B_shape = (K, N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as by:
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            for n in T.serial(T.ceildiv(N, block_N)):
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                    T.matrix_load(A[by * block_M, k * block_K], A_shared)
                    T.matrix_load(B[k * block_K, n * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local, False, False, k_pack=k_pack)
                T.copy(C_local, C[by * block_M, n * block_N])

    return main


def run_gemm_mls_n_loop(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_threads=128,
    k_pack=1,
):
    program = matmul_mls_n_loop(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_threads,
        k_pack=k_pack,
    )
    kernel = tl.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        return (A @ B).to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_mls_ds_read_format_copy_to_global_1():
    """matrix_load -> ds_read_format -> copy(fragment to global). No gemm."""
    run_mls_ds_read_format_copy_to_global(
        M=32,
        K=64,
        block_M=32,
        block_K=64,
        in_dtype="float16",
        out_dtype="float16",
        num_threads=128,
    )


def test_gemm_mls_n_loop_resource_hoist():
    """N loop outside K loop: MLS object should be reusable while the MN window changes."""
    run_gemm_mls_n_loop(
        M=32,
        N=64,
        K=64,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=32,
        num_threads=128,
    )


def test_gemm_mls_n_loop_resource_hoist_fixed_k():
    """N loop with a single K tile: MLS should update only the MN window in the loop."""
    run_gemm_mls_n_loop(
        M=32,
        N=64,
        K=32,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=32,
        block_N=32,
        block_K=32,
        num_threads=128,
    )


if __name__ == "__main__":
    tilelang.testing.main()
