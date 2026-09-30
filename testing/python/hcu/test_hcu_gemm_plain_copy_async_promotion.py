# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Tests for compiler-managed HCU GEMM async-copy promotion."""

import re

import pytest
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from hcu_test_utils import current_hcu_arch_string


pytestmark = pytest.mark.skipif(
    current_hcu_arch_string() not in {"gfx936", "gfx938", "gfx92a", "gfx946"},
    reason="plain GEMM copy promotion requires a supported HCU target",
)


def test_hcu_gemm_plain_copy_async_promotion():
    m = 256
    n = 256
    k = 128
    block_m = 256
    block_n = 256
    block_k = 16

    @T.prim_func
    def main(
        A: T.Tensor((m, k), T.float16),
        B: T.Tensor((k, n), T.float16),
        C: T.Tensor((m, n), T.float16),
    ):
        with T.Kernel(1, threads=512):
            A_shared = T.alloc_shared((block_m, block_k), T.float16)
            B_shared = T.alloc_shared((block_k, block_n), T.float16)
            C_local = T.alloc_fragment((block_m, block_n), T.float32)
            T.clear(C_local)
            for ko in T.Pipelined(k // block_k, num_stages=2):
                T.copy(A[0, ko * block_k], A_shared)
                T.copy(B[ko * block_k, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)

    kernel = tl.compile(main, out_idx=[2])
    source = kernel.get_kernel_source()

    assert source.count("tl::cp_async_commit();") == 2
    assert re.findall(r"tl::cp_async_wait<(\d+)>\(\);", source) == ["1", "0"]
    assert "load_async_lds" in source or "cp_async_gs" in source
    kernel.get_profiler().assert_allclose(lambda a, b: a @ b, atol=1e-2, rtol=1e-2)


def test_hcu_gemm_plain_copy_shared_at_bn_multiple_consumers():
    @T.prim_func
    def main(
        V: T.Tensor((2, 64, 32), T.float16),
        DO: T.Tensor((2, 64, 32), T.float16),
        DH: T.Tensor((2, 32, 32), T.float16),
        C_ds: T.Tensor((64, 64), T.float32),
        C_dk: T.Tensor((64, 32), T.float32),
    ):
        with T.Kernel(1, threads=128):
            V_shared = T.alloc_shared((64, 32), T.float16)
            DO_shared = T.alloc_shared((64, 32), T.float16)
            DH_shared = T.alloc_shared((32, 32), T.float16)
            C_ds_local = T.alloc_fragment((64, 64), T.float32)
            C_dk_local = T.alloc_fragment((64, 32), T.float32)
            T.clear(C_ds_local)
            T.clear(C_dk_local)

            for i in T.Pipelined(2, num_stages=2):
                T.copy(V[i, 0, 0], V_shared)
                T.copy(DO[i, 0, 0], DO_shared)
                T.copy(DH[i, 0, 0], DH_shared)
                # V_shared is an AT/BN B operand with n_warp=1.
                T.gemm(DO_shared, V_shared, C_ds_local, False, True)
                # The same LDS tile is an AT/BN A operand with m_warp=2.
                T.gemm(V_shared, DH_shared, C_dk_local, False, True)

            T.copy(C_ds_local, C_ds)
            T.copy(C_dk_local, C_dk)

    kernel = tl.compile(main, out_idx=[3, 4])
    kernel.get_profiler().assert_allclose(
        lambda v, do, dh: (
            sum(do[i].float() @ v[i].float().T for i in range(2)),
            sum(v[i].float() @ dh[i].float().T for i in range(2)),
        ),
        atol=1e-2,
        rtol=1e-2,
    )


if __name__ == "__main__":
    tilelang.testing.main()
