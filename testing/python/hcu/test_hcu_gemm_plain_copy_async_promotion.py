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


if __name__ == "__main__":
    tilelang.testing.main()
