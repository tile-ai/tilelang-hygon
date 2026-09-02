"""GEMM with MLS (Matrix Load Store) - both A and B from MLS LDS."""

import re

import pytest
import tilelang as tl
import tilelang.language as T
import tilelang.testing
import torch
from hcu_test_utils import (
    current_hcu_arch_string,
    target_supports_fp8_mmac,
    target_supports_mls,
    target_supports_mls_b4,
    target_supports_mls_b32,
    target_supports_mls_fp4_pad,
)


pytestmark = [
    pytest.mark.skipif(
        not target_supports_mls(),
        reason="matrix_load tests require an MLS-supported HCU target",
    ),
]

_WAITCNT_RE = r"__builtin_[A-Za-z0-9_]+_s_waitcnt"


def _assert_allclose_on_cpu(profiler, ref_program, atol=1e-2, rtol=1e-2):
    ins = profiler._get_inputs()
    ref_out = ref_program(*ins)
    torch.cuda.synchronize()
    lib_out = profiler.func(*ins)
    torch.cuda.synchronize()
    torch.testing.assert_close(lib_out.cpu(), ref_out.cpu(), atol=atol, rtol=rtol)


def _mls_dst_dtype(src_dtype):
    """gfx92a MLS expands packed FP4 into an explicitly unpacked LDS view."""
    if current_hcu_arch_string() == "gfx92a" and str(src_dtype) == "float4_e2m1fn":
        return T.float4_e2m1_unpacked
    return src_dtype


def _waitcnt_imms(source: str) -> list[int]:
    return [int(x) for x in re.findall(rf"{_WAITCNT_RE}\((\d+)\)", source)]


def _vmcnt_keep(imm: int) -> int:
    return imm & 0xF


def _assert_each_waitcnt_followed_by_sync(source: str) -> None:
    waits = _waitcnt_imms(source)
    sync_after_waits = len(
        re.findall(
            rf"{_WAITCNT_RE}\(\d+\);\s*\n\s*__syncthreads\(\);",
            source,
        )
    )
    assert sync_after_waits == len(waits), f"expected waitcnt+sync pairs for all {len(waits)} waits, got {sync_after_waits}"


def _assert_mls_direct_stage1_waitcnt(source: str) -> None:
    """Stage1 direct LDS: wait all outstanding MLS before each gemm consumer."""
    waits = _waitcnt_imms(source)
    assert waits == [16368, 16368], f"unexpected direct stage1 waitcnt sequence: {waits}"
    assert source.count("ds_read_format_tensor_a") >= 2
    assert source.count("ds_read_format_tensor_b") >= 2
    assert "__builtin_hcu_mmac" in source
    _assert_each_waitcnt_followed_by_sync(source)


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
            rf"{_WAITCNT_RE}\(\d+\);\s*\n\s*__syncthreads\(\);\s*\n\s*"
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
    assert source.count("ds_read_format_tensor_a") >= 2
    assert source.count("ds_read_format_tensor_b") >= 2
    assert "__builtin_hcu_mmac" in source
    _assert_each_waitcnt_followed_by_sync(source)


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
            rf"{_WAITCNT_RE}\(\d+\);\s*\n\s*__syncthreads\(\);\s*\n\s*"
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
    assert "__builtin_hcu_mmac" in source
    _assert_each_waitcnt_followed_by_sync(source)
    ds_waits = len(
        re.findall(
            rf"{_WAITCNT_RE}\(\d+\);\s*\n\s*__syncthreads\(\);\s*\n\s*"
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
    assert "__builtin_hcu_mmac" in source
    _assert_each_waitcnt_followed_by_sync(source)
    ds_waits = len(
        re.findall(
            rf"{_WAITCNT_RE}\(\d+\);\s*\n\s*__syncthreads\(\);\s*\n\s*"
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
    use_tf32=False,
):
    """GEMM with both A and B loaded via matrix_load (MLS)."""
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    mls_dtype = _mls_dst_dtype(in_dtype)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, mls_dtype)
            B_shared = T.alloc_shared(B_shared_shape, mls_dtype)
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
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B, k_pack=k_pack, use_tf32=use_tf32)
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
    mls_dtype=None,
    use_tf32=False,
):
    """GEMM with both A and B: matrix_load -> ds_read_format -> gemm."""
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    mls_dtype = _mls_dst_dtype(in_dtype) if mls_dtype is None else mls_dtype

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, mls_dtype)
            B_shared = T.alloc_shared(B_shared_shape, mls_dtype)
            A_fragment = T.alloc_fragment(A_shared_shape, mls_dtype)
            B_fragment = T.alloc_fragment(B_shared_shape, mls_dtype)
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
                T.gemm(A_fragment, B_fragment, C_local, trans_A, trans_B, k_pack=k_pack, use_tf32=use_tf32)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def matmul_mls_ds_read_format_mixed(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    a_dtype,
    b_dtype,
    out_dtype,
    accum_dtype,
    threads,
):
    """GEMM with mixed A/B dtypes: matrix_load -> T.gemm Python lowering."""
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    a_mls_dtype = _mls_dst_dtype(a_dtype)
    b_mls_dtype = _mls_dst_dtype(b_dtype)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, a_dtype),
        B: T.Tensor(B_shape, b_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, a_mls_dtype)
            B_shared = T.alloc_shared(B_shared_shape, b_mls_dtype)
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
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
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
        A = A.cpu()
        B = B.cpu()

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if "float8" in in_dtype:
            A = A.to(torch.float32)
            B = B.to(torch.float32)
        return (A @ B).to(torch.__getattribute__(out_dtype))

    _assert_allclose_on_cpu(profiler, ref_program, atol=1e-2, rtol=1e-2)


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
    use_tf32=False,
    atol=1e-2,
    rtol=1e-2,
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
        use_tf32=use_tf32,
    )
    kernel = tl.compile(program, out_idx=[2])
    if verify_source is not None:
        verify_source(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        A = A.cpu()
        B = B.cpu()

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if "float8" in in_dtype:
            A = A.to(torch.float32)
            B = B.to(torch.float32)
        return (A @ B).to(torch.__getattribute__(out_dtype))

    _assert_allclose_on_cpu(profiler, ref_program, atol=atol, rtol=rtol)


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
    use_tf32=False,
    atol=1e-2,
    rtol=1e-2,
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
        use_tf32=use_tf32,
    )
    kernel = tl.compile(program, out_idx=[2])
    if verify_source is not None:
        verify_source(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        A = A.cpu()
        B = B.cpu()

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if "float8" in in_dtype:
            A = A.to(torch.float32)
            B = B.to(torch.float32)
        return (A @ B).to(torch.__getattribute__(out_dtype))

    _assert_allclose_on_cpu(profiler, ref_program, atol=atol, rtol=rtol)


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
        A = A.cpu()
        B = B.cpu()

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if "float8" in in_dtype:
            A = A.to(torch.float32)
            B = B.to(torch.float32)
        return (A @ B).to(torch.__getattribute__(out_dtype))

    _assert_allclose_on_cpu(profiler, ref_program, atol=1e-2, rtol=1e-2)


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
        A = A.cpu()
        B = B.cpu()

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        return (scale_a * scale_b * (A @ B)).to(torch.__getattribute__(out_dtype))

    _assert_allclose_on_cpu(profiler, ref_program, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads",
    [
        pytest.param(32, 32, 64, False, True, 32, 32, 64, 128, id="at_bn_m32_n32_k64_bm32_bn32_bk64_t128_b16"),
    ],
)
def test_gemm_mls_b_only(M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads):
    """GEMM with A via shared copy and B via MLS."""
    run_gemm_mls_b_only(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_threads=num_threads,
    )


@pytest.mark.parametrize(
    "M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads",
    [
        pytest.param(16, 16, 32, False, True, 16, 16, 32, 64, id="at_bn_m16_n16_k32_bm16_bn16_bk32_t64_b16"),
        pytest.param(16, 16, 64, False, True, 16, 16, 64, 128, id="at_bn_m16_n16_k64_bm16_bn16_bk64_t128_b16"),
        pytest.param(16, 16, 64, False, True, 16, 16, 64, 64, id="at_bn_m16_n16_k64_bm16_bn16_bk64_t64_b16"),
        pytest.param(16, 16, 128, False, True, 16, 16, 128, 64, id="at_bn_m16_n16_k128_bm16_bn16_bk128_t64_b16"),
        pytest.param(32, 32, 64, False, True, 32, 32, 64, 128, id="at_bn_m32_n32_k64_bm32_bn32_bk64_t128_b16"),
        pytest.param(64, 64, 16, True, False, 64, 64, 16, 64, id="an_bt_m64_n64_k16_bm64_bn64_bk16_t64_b16"),
        pytest.param(64, 64, 64, True, False, 64, 64, 64, 128, id="an_bt_m64_n64_k64_bm64_bn64_bk64_t128_b16"),
    ],
)
def test_gemm_mls_b16(M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads):
    """B16 GEMM cases covering gfx946 MLS LDS layouts without duplicate legacy cases."""
    run_gemm_mls(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_threads=num_threads,
    )


@pytest.mark.parametrize(
    "M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads",
    [
        pytest.param(64, 64, 64, False, True, 32, 32, 64, 128, id="at_bn_multi_block"),
        pytest.param(96, 96, 198, False, True, 64, 64, 128, 128, id="at_bn_non_power_of_two"),
        pytest.param(64, 64, 32, True, False, 64, 64, 32, 128, id="an_bt_single_block"),
        pytest.param(64, 64, 64, True, False, 32, 32, 64, 128, id="an_bt_multi_block"),
        pytest.param(96, 96, 241, True, False, 64, 64, 128, 128, id="an_bt_non_power_of_two"),
        pytest.param(32, 32, 64, False, False, 32, 32, 64, 128, id="an_bn_single_block"),
        pytest.param(64, 64, 64, False, False, 32, 32, 64, 128, id="an_bn_multi_block"),
        pytest.param(96, 96, 242, False, False, 64, 64, 128, 128, id="an_bn_non_power_of_two"),
        pytest.param(32, 32, 64, True, True, 32, 32, 64, 128, id="at_bt_single_block"),
        pytest.param(64, 64, 64, True, True, 32, 32, 64, 128, id="at_bt_multi_block"),
        pytest.param(96, 96, 242, True, True, 64, 64, 128, 128, id="at_bt_non_power_of_two"),
    ],
)
def test_gemm_mls_f16(M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads):
    """Legacy float16 MLS GEMM coverage across transpose modes and grid shapes."""
    run_gemm_mls(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_threads=num_threads,
    )


@pytest.mark.parametrize(
    "num_stages, verify_source",
    [
        pytest.param(0, None, id="stage0"),
        pytest.param(1, _assert_mls_direct_stage1_waitcnt, id="stage1"),
        pytest.param(2, _assert_mls_direct_stage2_waitcnt, id="stage2"),
    ],
)
def test_gemm_mls_f16_pipeline(num_stages, verify_source):
    """Float16 direct MLS GEMM with explicit pipeline-stage waitcnt checks."""
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
        num_stages=num_stages,
        verify_source=verify_source,
    )


@pytest.mark.parametrize(
    "M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads",
    [
        pytest.param(64, 64, 32, True, False, 64, 64, 32, 64, id="an_bt_64x32_single"),
        pytest.param(64, 64, 41, True, False, 64, 64, 32, 64, id="an_bt_64x32_kmask"),
        pytest.param(128, 128, 32, True, False, 64, 64, 32, 64, id="an_bt_64x32_multi_block"),
        pytest.param(128, 128, 32, True, False, 128, 128, 32, 64, id="an_bt_128x16_single"),
        pytest.param(256, 256, 32, True, False, 128, 128, 32, 64, id="an_bt_128x16_multi_block"),
        pytest.param(256, 256, 32, True, False, 256, 256, 32, 128, id="an_bt_128x16_multi_tile"),
        pytest.param(16, 16, 64, False, True, 16, 16, 64, 64, id="at_bn_16x64_single"),
        pytest.param(32, 32, 64, False, True, 16, 16, 64, 64, id="at_bn_16x64_multi_block"),
        pytest.param(32, 32, 64, False, True, 32, 32, 64, 64, id="at_bn_32x64_single"),
        pytest.param(64, 64, 64, False, True, 64, 64, 64, 128, id="at_bn_32x64_multi_block"),
        pytest.param(16, 16, 128, False, True, 16, 16, 128, 64, id="at_bn_16x128_single"),
        pytest.param(32, 32, 256, False, True, 16, 16, 128, 64, id="at_bn_16x128_multi_block"),
        pytest.param(32, 32, 256, False, True, 32, 32, 128, 64, id="at_bn_16x128_multi_tile"),
    ],
)
def test_gemm_mls_fp8(M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads):
    """Float8 MLS GEMM coverage for AN/BT and AT/BN tile shapes."""
    run_gemm_mls(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_threads=num_threads,
    )


@pytest.mark.parametrize(
    "M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads",
    [
        pytest.param(32, 32, 64, True, True, 32, 32, 64, 128, id="at_bt_single_block"),
        pytest.param(64, 64, 64, True, True, 32, 32, 64, 128, id="at_bt_multi_block"),
        pytest.param(64, 64, 32, True, False, 64, 64, 32, 128, id="an_bt_single_block"),
        pytest.param(32, 32, 64, False, True, 32, 32, 64, 128, id="at_bn_single_block"),
    ],
)
def test_gemm_mls_ds_read_format_f16(M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads):
    """Float16 matrix_load -> ds_read_format -> GEMM coverage."""
    run_gemm_mls_ds_read_format(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_threads=num_threads,
    )


def _assert_mls_b32_tf32_source(source: str) -> None:
    assert "tilelang_mls_base<" in source
    assert "tl::sequence<16, 16>" in source
    assert "ds_read_format_tensor_a" in source
    assert "ds_read_format_tensor_b" in source
    assert "tl::hcu_target_enum::gfx" in source
    assert "int>" in source or "int," in source
    assert "_tf32" in source


@pytest.mark.parametrize(
    "M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads",
    [
        pytest.param(16, 16, 16, False, True, 16, 16, 16, 64, id="at_bn_m16_n16_k16_t64"),
        pytest.param(32, 32, 16, False, True, 32, 32, 16, 128, id="at_bn_m32_n32_k16_t128"),
        pytest.param(64, 32, 16, False, True, 32, 32, 16, 128, id="at_bn_m64_n32_k16_t128"),
        pytest.param(32, 32, 16, True, False, 32, 32, 16, 64, id="an_bt_m32_n32_k16_t64"),
        pytest.param(64, 64, 16, True, False, 64, 64, 16, 128, id="an_bt_m64_n64_k16_t128"),
        pytest.param(64, 64, 32, True, False, 64, 64, 16, 256, id="an_bt_m64_n64_k32_t256"),
    ],
)
@pytest.mark.skipif(
    not target_supports_mls_b32(),
    reason="b32 MLS matrix_load + ds_read_format is only supported on gfx92a/gfx946",
)
def test_gemm_mls_b32_tf32(M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads):
    """Float32 matrix_load_16x16_b32 -> TF32 ds_read_format -> TF32 GEMM coverage."""
    run_gemm_mls(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype="float32",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_threads=num_threads,
        verify_source=_assert_mls_b32_tf32_source,
        use_tf32=True,
        atol=1e-1,
        rtol=1e-1,
    )


@pytest.mark.parametrize(
    "num_stages, verify_source",
    [
        pytest.param(1, _assert_mls_ds_stage1_waitcnt, id="stage1"),
        pytest.param(2, _assert_mls_ds_stage2_waitcnt, id="stage2"),
    ],
)
def test_gemm_mls_ds_read_format_f16_pipeline(num_stages, verify_source):
    """Float16 matrix_load -> ds_read_format -> GEMM with pipeline waitcnt checks."""
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
        num_stages=num_stages,
        verify_source=verify_source,
    )


@pytest.mark.parametrize(
    "num_stages, verify_source",
    [
        pytest.param(1, _assert_mls_copy_a_mls_b_ds_stage1_waitcnt, id="stage1"),
        pytest.param(2, _assert_mls_copy_a_mls_b_ds_stage2_waitcnt, id="stage2"),
    ],
)
def test_gemm_mls_copy_a_mls_b_ds_f16_pipeline(num_stages, verify_source):
    """A via T.copy and B via matrix_load -> ds_read_format with pipeline waitcnt checks."""
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
        num_stages=num_stages,
        verify_source=verify_source,
    )


@pytest.mark.parametrize(
    "M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads",
    [
        pytest.param(64, 64, 32, True, False, 64, 64, 32, 64, id="an_bt_64x32_single"),
        pytest.param(128, 128, 32, True, False, 128, 128, 32, 64, id="an_bt_128x16_single"),
        pytest.param(16, 16, 64, False, True, 16, 16, 64, 64, id="at_bn_16x64_single"),
        pytest.param(32, 32, 64, False, True, 32, 32, 64, 64, id="at_bn_32x64_single"),
        pytest.param(16, 16, 128, False, True, 16, 16, 128, 64, id="at_bn_16x128_single"),
    ],
)
def test_gemm_mls_ds_read_format_fp8(M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads):
    """Float8 matrix_load -> ds_read_format -> GEMM coverage."""
    run_gemm_mls_ds_read_format(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        in_dtype="float8_e4m3fn",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_threads=num_threads,
    )


@pytest.mark.parametrize(
    "scale_a, scale_b",
    [
        pytest.param(0.5, 0.2, id="scale_a_0_5_b_0_2"),
    ],
)
def test_gemm_mls_ds_read_format_mul_scale(scale_a, scale_b):
    """matrix_load -> ds_read_format -> scaling -> GEMM."""
    run_gemm_mls_ds_read_format_mul_scale(
        M=32,
        N=32,
        K=64,
        trans_A=False,
        trans_B=True,
        scale_a=scale_a,
        scale_b=scale_b,
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
        A = A.cpu()
        return A.to(torch.__getattribute__(out_dtype))

    _assert_allclose_on_cpu(profiler, ref_program, atol=1e-2, rtol=1e-2)


def mls_ds_read_format_b4_copy_to_global(
    M,
    K,
    block_M,
    block_K,
    threads,
):
    """b4 matrix_load -> ds_read_format -> copy(fragment to global). No gemm."""
    in_dtype = "float4_e2m1fn"
    out_dtype = in_dtype
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


def _assert_b4_mls_ds_source(source: str, tile_mn: int, tile_k: int, trans: bool) -> None:
    expected_tile = f"tl::sequence<{tile_mn}, {tile_k}>"
    assert expected_tile in source, f"expected MLS tile {expected_tile} in generated source"
    assert "tl::pk_fp4_t" in source
    assert "tl::mls::ds_read_format" in source
    assert "padbyte" not in source
    assert "tl::hcu_target_enum::gfx946" in source
    if trans:
        assert "true, tl::hcu_target_enum::gfx946" in source
    else:
        assert "false, tl::hcu_target_enum::gfx946" in source


def run_mls_ds_read_format_b4_copy_to_global(
    M,
    K,
    block_M,
    block_K,
    num_threads=128,
):
    program = mls_ds_read_format_b4_copy_to_global(
        M,
        K,
        block_M,
        block_K,
        num_threads,
    )
    kernel = tl.compile(program, out_idx=[1])
    _assert_b4_mls_ds_source(
        kernel.get_kernel_source(),
        block_M,
        block_K,
        True,
    )
    profiler = kernel.get_profiler()

    rows = torch.arange(M, device="cuda", dtype=torch.uint8)[:, None]
    cols = torch.arange(K, device="cuda", dtype=torch.uint8)[None, :]
    logical = (rows * 3 + cols * 5 + rows // 7 + cols // 11) & 0x0F
    A_storage = (logical[:, 0::2] | (logical[:, 1::2] << 4)).to(torch.uint8)
    A = A_storage.view(torch.int8)
    torch.cuda.synchronize()
    lib_out = profiler.func(A)
    torch.cuda.synchronize()
    lib_storage = lib_out.view(torch.uint8).cpu()
    torch.testing.assert_close(lib_storage, A_storage.cpu(), rtol=0, atol=0)


def _fp4_e2m1fn_decode(logical: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=logical.device,
        dtype=torch.float32,
    )
    return table[(logical & 0x0F).long()]


def _pack_fp4_last_dim(logical: torch.Tensor) -> torch.Tensor:
    logical = logical.to(torch.uint8)
    return (logical[..., 0::2] | (logical[..., 1::2] << 4)).contiguous().view(torch.int8)


def _torch_dtype(dtype: str):
    if dtype == "float8_e4m3fn":
        return torch.float8_e4m3fn
    if dtype == "float8_e5m2":
        return torch.float8_e5m2
    return getattr(torch, dtype)


def _make_mixed_mls_input(shape, dtype: str, row_mul: int, col_mul: int):
    rows = torch.arange(shape[0], device="cuda", dtype=torch.float32)[:, None]
    cols = torch.arange(shape[1], device="cuda", dtype=torch.float32)[None, :]
    value = ((rows * row_mul + cols * col_mul) % 13 - 6) / 8.0
    if dtype == "float4_e2m1fn":
        logical = ((rows.to(torch.uint8) * row_mul + cols.to(torch.uint8) * col_mul) & 0x0F).to(torch.uint8)
        return _pack_fp4_last_dim(logical), _fp4_e2m1fn_decode(logical.cpu())
    tensor = value.to(_torch_dtype(dtype))
    return tensor, tensor.cpu().to(torch.float32)


def run_gemm_mls_ds_read_format_b4(
    M=32,
    N=32,
    K=128,
    trans_A=False,
    trans_B=True,
    block_M=32,
    block_N=32,
    block_K=128,
    num_threads=64,
    direct_shared=False,
):
    builder = matmul_mls if direct_shared else matmul_mls_ds_read_format
    program = builder(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        "float4_e2m1fn",
        "float32",
        "float32",
        num_threads,
    )
    kernel = tl.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    a_rows = torch.arange(A_shape[0], device="cuda", dtype=torch.uint8)[:, None]
    a_cols = torch.arange(A_shape[1], device="cuda", dtype=torch.uint8)[None, :]
    b_rows = torch.arange(B_shape[0], device="cuda", dtype=torch.uint8)[:, None]
    b_cols = torch.arange(B_shape[1], device="cuda", dtype=torch.uint8)[None, :]
    A_logical = (a_rows * 3 + a_cols * 5 + a_rows // 7 + a_cols // 11) & 0x0F
    B_logical = (b_rows * 7 + b_cols * 2 + b_rows // 5 + b_cols // 13) & 0x0F
    A = _pack_fp4_last_dim(A_logical)
    B = _pack_fp4_last_dim(B_logical)

    torch.cuda.synchronize()
    lib_out = profiler.func(A, B)
    torch.cuda.synchronize()

    A_ref = _fp4_e2m1fn_decode(A_logical.cpu())
    B_ref = _fp4_e2m1fn_decode(B_logical.cpu())
    if trans_A:
        A_ref = A_ref.T
    if trans_B:
        B_ref = B_ref.T
    ref = A_ref @ B_ref
    torch.testing.assert_close(lib_out.cpu(), ref, rtol=1e-2, atol=1e-1)


def run_gemm_mls_ds_read_format_b4_pad(
    M=16,
    N=16,
    K=128,
    trans_A=False,
    trans_B=True,
    block_M=16,
    block_N=16,
    block_K=128,
    num_threads=64,
):
    program = matmul_mls_ds_read_format(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        "float4_e2m1fn",
        "float32",
        "float32",
        num_threads,
        mls_dtype=T.float4_e2m1_unpacked,
    )
    kernel = tl.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    a_rows = torch.arange(A_shape[0], device="cuda", dtype=torch.uint8)[:, None]
    a_cols = torch.arange(A_shape[1], device="cuda", dtype=torch.uint8)[None, :]
    b_rows = torch.arange(B_shape[0], device="cuda", dtype=torch.uint8)[:, None]
    b_cols = torch.arange(B_shape[1], device="cuda", dtype=torch.uint8)[None, :]
    A_logical = (a_rows * 3 + a_cols * 5 + a_rows // 7 + a_cols // 11) & 0x0F
    B_logical = (b_rows * 7 + b_cols * 2 + b_rows // 5 + b_cols // 13) & 0x0F
    A = _pack_fp4_last_dim(A_logical)
    B = _pack_fp4_last_dim(B_logical)

    torch.cuda.synchronize()
    lib_out = profiler.func(A, B)
    torch.cuda.synchronize()

    A_ref = _fp4_e2m1fn_decode(A_logical.cpu())
    B_ref = _fp4_e2m1fn_decode(B_logical.cpu())
    if trans_A:
        A_ref = A_ref.T
    if trans_B:
        B_ref = B_ref.T
    ref = A_ref @ B_ref
    torch.testing.assert_close(lib_out.cpu(), ref, rtol=1e-2, atol=1e-1)


def run_gemm_mls_mix_f4f6f8(
    M=32,
    N=32,
    K=128,
    trans_A=False,
    trans_B=True,
    block_M=32,
    block_N=32,
    block_K=128,
    a_dtype="float8_e4m3fn",
    b_dtype="float4_e2m1fn",
    num_threads=64,
):
    program = matmul_mls_ds_read_format_mixed(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        a_dtype,
        b_dtype,
        "float32",
        "float32",
        num_threads,
    )
    kernel = tl.compile(program, out_idx=[2])
    source = kernel.get_kernel_source()
    arch = current_hcu_arch_string()
    assert "ds_read_format_tensor_a" in source
    assert "ds_read_format_tensor_b" in source
    assert "__builtin_hcu_mmac_f32_16x16x32_f8f6f4" in source
    assert "__builtin_hcu_mmac_f32_16x16x64_fp4" not in source
    if arch == "gfx946":
        assert "uint8_t, 4, 8" in source
        assert "__builtin_hcu_mmac_f32_16x16x32_f8f6f4_lit_lts" in source
    else:
        assert "uint8_t, 8, 8>" in source
        assert "__builtin_hcu_mmac_f32_16x16x32_f8f6f4(" in source

    profiler = kernel.get_profiler()
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A, A_ref = _make_mixed_mls_input(A_shape, a_dtype, 3, 5)
    B, B_ref = _make_mixed_mls_input(B_shape, b_dtype, 7, 2)

    torch.cuda.synchronize()
    lib_out = profiler.func(A, B)
    torch.cuda.synchronize()

    if trans_A:
        A_ref = A_ref.T
    if trans_B:
        B_ref = B_ref.T
    ref = A_ref @ B_ref
    torch.testing.assert_close(lib_out.cpu(), ref, rtol=2e-2, atol=2e-1)


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
    verify_source=None,
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
    if verify_source is not None:
        verify_source(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        A = A.cpu()
        B = B.cpu()
        return (A @ B).to(torch.__getattribute__(out_dtype))

    _assert_allclose_on_cpu(profiler, ref_program, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "M, K, block_M, block_K, num_threads",
    [
        pytest.param(16, 64, 16, 64, 128, id="16x32_B1_W2K_b16"),
        pytest.param(16, 64, 16, 64, 64, id="16x64_B1_W1_b16"),
        pytest.param(64, 64, 64, 64, 128, id="16x64_B1_W4M_b16"),
        pytest.param(16, 128, 16, 128, 64, id="16x64_B1_W1_4KB_b16"),
        pytest.param(16, 128, 16, 128, 128, id="16x64_B1_W2_4KB_b16"),
        pytest.param(32, 128, 32, 128, 128, id="16x64_B1_W2_8KB_b16"),
        pytest.param(32, 128, 32, 128, 256, id="16x64_B1_W4_8KB_b16"),
        pytest.param(32, 32, 32, 32, 64, id="32x32"),
    ],
)
def test_mls_ds_read_format_copy_to_global(M, K, block_M, block_K, num_threads):
    """matrix_load -> ds_read_format -> copy(fragment to global). No gemm."""
    run_mls_ds_read_format_copy_to_global(
        M=M,
        K=K,
        block_M=block_M,
        block_K=block_K,
        in_dtype="float16",
        out_dtype="float16",
        num_threads=num_threads,
    )


@pytest.mark.skipif(
    not target_supports_mls_b32(),
    reason="b32 MLS matrix_load + ds_read_format is only supported on gfx92a/gfx946",
)
def test_mls_ds_read_format_b32_copy_to_global():
    """b32 matrix_load -> ds_read_format -> copy(fragment to global). No gemm."""
    run_mls_ds_read_format_copy_to_global(
        M=16,
        K=16,
        block_M=16,
        block_K=16,
        in_dtype="float32",
        out_dtype="float32",
        num_threads=64,
    )


@pytest.mark.skipif(
    current_hcu_arch_string() != "gfx946",
    reason="packed b4 ds_read_format copy-to-global no-pad test is only supported on gfx946",
)
@pytest.mark.parametrize(
    "M, K, block_M, block_K, num_threads",
    [
        pytest.param(32, 128, 32, 128, 64, id="b4_format_32x128_t64"),
        pytest.param(32, 128, 32, 128, 128, id="b4_format_32x128_t128"),
        pytest.param(64, 256, 32, 128, 128, id="b4_format_64x256_mktiles_t128"),
        pytest.param(32, 256, 32, 256, 64, id="b4_format_32x256_trans_t64"),
    ],
)
def test_mls_ds_read_format_b4_copy_to_global(M, K, block_M, block_K, num_threads):
    """b4 matrix_load -> ds_read_format -> copy(fragment to global). No gemm."""
    run_mls_ds_read_format_b4_copy_to_global(
        M=M,
        K=K,
        block_M=block_M,
        block_K=block_K,
        num_threads=num_threads,
    )


@pytest.mark.skipif(
    not target_supports_mls_b4(),
    reason="packed FP4 matrix_load source is only supported on gfx92a/gfx946",
)
@pytest.mark.parametrize(
    "M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads",
    [
        pytest.param(32, 32, 128, False, True, 32, 32, 128, 64, id="at_bn_m32_n32_k128_t64"),
        pytest.param(32, 32, 256, False, True, 32, 32, 256, 64, id="at_bn_m32_n32_k256_t64"),
        pytest.param(32, 32, 256, False, True, 32, 32, 256, 128, id="at_bn_m32_n32_k256_t128"),
        pytest.param(64, 32, 256, False, True, 32, 32, 256, 64, id="at_bn_m64_n32_k256_t64"),
        pytest.param(128, 128, 64, True, False, 128, 128, 64, 64, id="an_bt_m128_n128_k64_t64"),
        pytest.param(128, 128, 64, True, False, 128, 128, 64, 128, id="an_bt_m128_n128_k64_t128"),
        pytest.param(256, 256, 128, True, False, 128, 128, 64, 128, id="an_bt_m256_n256_k128_t128"),
    ],
)
def test_gemm_mls_b4_nopad(M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads):
    """FP4-source GEMM; gfx92a explicitly expands the matrix_load destination to b8."""
    run_gemm_mls_ds_read_format_b4(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_threads=num_threads,
    )


@pytest.mark.skipif(
    not target_supports_mls_b4(),
    reason="packed FP4 matrix_load source is only supported on gfx92a/gfx946",
)
def test_gemm_mls_b4_shared_k32_auto_expand():
    """Packed FP4 LDS is expanded by the implicit ds_read when native K64 cannot fit."""
    run_gemm_mls_ds_read_format_b4(
        M=128,
        N=128,
        K=32,
        trans_A=True,
        trans_B=False,
        block_M=128,
        block_N=128,
        block_K=32,
        num_threads=64,
        direct_shared=True,
    )


@pytest.mark.skipif(
    not target_supports_mls_fp4_pad(),
    reason="fp4 b8-LDS MLS/MMAC path is only supported on gfx92a/gfx946",
)
@pytest.mark.parametrize(
    "M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads",
    [
        pytest.param(16, 16, 128, False, True, 16, 16, 128, 64, id="at_bn_m16_n16_k128_t64"),
        pytest.param(16, 16, 128, False, True, 16, 16, 128, 128, id="at_bn_m16_n16_k128_t128"),
        pytest.param(56, 64, 32, True, False, 64, 64, 32, 64, id="an_bt_m56_n64_k32_t64"),
        pytest.param(64, 64, 64, True, False, 64, 64, 64, 64, id="an_bt_m64_n64_k64_t64"),
    ],
)
def test_gemm_mls_b4_pad(M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads):
    """fp4 fallback path: matrix_load pads b4 to b8 LDS, ds_read outputs b8, MMAC uses f8f6f4."""
    run_gemm_mls_ds_read_format_b4_pad(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_threads=num_threads,
    )


@pytest.mark.skipif(
    not (target_supports_fp8_mmac() and target_supports_mls_fp4_pad()),
    reason="mixed f8/fp4 MLS f8f6f4 path requires fp8 MMAC and fp4 MLS support",
)
@pytest.mark.parametrize(
    "a_dtype, b_dtype",
    [
        pytest.param("float8_e4m3fn", "float4_e2m1fn", id="fp8_fp4"),
        pytest.param("float4_e2m1fn", "float8_e4m3fn", id="fp4_fp8"),
    ],
)
@pytest.mark.parametrize(
    "M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads",
    [
        pytest.param(32, 32, 128, False, True, 32, 32, 128, 64, id="at_bn_m32_n32_k128_t64"),
        pytest.param(64, 64, 256, False, True, 32, 32, 128, 64, id="at_bn_m64_n64_k256_t64"),
        pytest.param(128, 128, 64, True, False, 128, 128, 64, 64, id="an_bt_m128_n128_k64_t64"),
        pytest.param(256, 256, 128, True, False, 128, 128, 64, 128, id="an_bt_m256_n256_k128_t128"),
    ],
)
def test_gemm_mls_mix_f4f6f8(M, N, K, trans_A, trans_B, block_M, block_N, block_K, num_threads, a_dtype, b_dtype):
    """Mixed f8/fp4 MLS GEMM: fp4 operand is read as b8 for f8f6f4 MMAC."""
    run_gemm_mls_mix_f4f6f8(
        M=M,
        N=N,
        K=K,
        trans_A=trans_A,
        trans_B=trans_B,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        num_threads=num_threads,
    )


def _assert_mls_boundary_filter_live(source: str) -> None:
    has_k_filter = "async_mls_load_asm<half_t, true" in source
    has_mn_filter = (
        "update_mn_base<true" in source
        or "async_mls_load_asm<half_t, false, true" in source
        or "async_mls_load_asm<half_t, true, true" in source
    )
    assert has_k_filter or has_mn_filter, source


def _assert_mls_reverse_k_rebase(source: str) -> None:
    assert "update_k_base" in source
    _assert_mls_boundary_filter_live(source)


def _assert_mls_k_outer_update_mn(source: str) -> None:
    assert "update_mn_base" in source
    _assert_mls_boundary_filter_live(source)


def test_gemm_mls_n_loop_resource_hoist():
    """N-outer / K-inner GEMM with a partial tail; MLS reuses the MN window."""
    run_gemm_mls_n_loop(
        M=96,
        N=96,
        K=184,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=64,
        block_N=64,
        block_K=64,
        num_threads=128,
        verify_source=_assert_mls_boundary_filter_live,
    )


def test_gemm_mls_n_loop_resource_hoist_fixed_k():
    """N-outer GEMM with a single K tile; only the MN window updates in the loop."""
    run_gemm_mls_n_loop(
        M=96,
        N=96,
        K=64,
        in_dtype="float16",
        out_dtype="float32",
        dtypeAccum="float32",
        block_M=64,
        block_N=64,
        block_K=64,
        num_threads=128,
        verify_source=_assert_mls_boundary_filter_live,
    )


def matmul_mls_reverse_k(
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
    """GEMM that walks K tiles high-to-low so hoist cannot use forward_delta."""
    num_k = T.ceildiv(K, block_K)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.serial(num_k):
                k_tile = (num_k - 1 - k) * block_K
                T.matrix_load(A[by * block_M, k_tile], A_shared)
                T.matrix_load(B[k_tile, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, False, False, k_pack=k_pack)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def matmul_mls_k_outer_n_inner(
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
    """GEMM that walks N tiles inside a stable K tile."""
    n_tiles = T.ceildiv(N, block_N)
    k_tiles = T.ceildiv(K, block_K)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as by:
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            for k in T.serial(k_tiles):
                T.matrix_load(A[by * block_M, k * block_K], A_shared)
                for n in T.serial(n_tiles):
                    if k == 0:
                        T.clear(C_local)
                    else:
                        T.copy(C[by * block_M, n * block_N], C_local)
                    T.matrix_load(B[k * block_K, n * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local, False, False, k_pack=k_pack)
                    T.copy(C_local, C[by * block_M, n * block_N])

    return main


def _run_gemm_mls_program(program, verify_source=None):
    kernel = tl.compile(program, out_idx=[2])
    if verify_source is not None:
        verify_source(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        return (A.cpu() @ B.cpu()).to(torch.float32)

    _assert_allclose_on_cpu(profiler, ref_program, atol=1e-2, rtol=1e-2)


def test_gemm_mls_reverse_k_absolute_rebase():
    """Non-monotonic K uses update_k_base."""
    _run_gemm_mls_program(
        matmul_mls_reverse_k(
            M=96,
            N=96,
            K=184,
            block_M=64,
            block_N=64,
            block_K=64,
            in_dtype="float16",
            out_dtype="float32",
            accum_dtype="float32",
            threads=128,
        ),
        verify_source=_assert_mls_reverse_k_rebase,
    )


def matmul_mls_k_prefetch_then_loop(
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
    """Prefetch K=0 then Serial(k); hoist must still treat K as the inner axis."""
    num_k = T.ceildiv(K, block_K)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            T.matrix_load(A[by * block_M, 0], A_shared)
            T.matrix_load(B[0, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local, False, False, k_pack=k_pack)
            for k in T.serial(num_k - 1):
                T.matrix_load(A[by * block_M, (k + 1) * block_K], A_shared)
                T.matrix_load(B[(k + 1) * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, False, False, k_pack=k_pack)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _assert_mls_k_inner_despite_prefetch(source: str) -> None:
    # Prefetch K=0 must not flip the inner axis to MN.
    assert "update_k_base" in source, source
    assert "mls_resource_axis::k" not in source, source


def test_gemm_mls_k_prefetch_then_loop_k_inner():
    _run_gemm_mls_program(
        matmul_mls_k_prefetch_then_loop(
            M=96,
            N=96,
            K=184,
            block_M=64,
            block_N=64,
            block_K=64,
            in_dtype="float16",
            out_dtype="float32",
            accum_dtype="float32",
            threads=128,
        ),
        verify_source=_assert_mls_k_inner_despite_prefetch,
    )


def test_gemm_mls_k_outer_n_inner_update_mn():
    """K-outer / N-inner B loads use update_mn_base."""
    _run_gemm_mls_program(
        matmul_mls_k_outer_n_inner(
            M=96,
            N=96,
            K=184,
            block_M=64,
            block_N=64,
            block_K=64,
            in_dtype="float16",
            out_dtype="float32",
            accum_dtype="float32",
            threads=128,
        ),
        verify_source=_assert_mls_k_outer_update_mn,
    )


if __name__ == "__main__":
    tilelang.testing.main()
