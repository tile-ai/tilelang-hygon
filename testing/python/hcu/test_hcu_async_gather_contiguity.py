# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Paged AT/BN gathers must prove vector contiguity through LDS swizzles."""

import pytest
import torch
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey
from tilelang import tvm
from hcu_test_utils import current_hcu_arch_string

pytestmark = [
    *tilelang.testing.requires_rocm.marks(),
    pytest.mark.skipif(
        current_hcu_arch_string() not in {"gfx938", "gfx946"},
        reason="requires an HCU LDS target",
    ),
]


def gather_program(rows, source_stride=1):
    @T.prim_func
    def main(
        A: T.Tensor((8, 16, 128 * source_stride), T.bfloat16),
        P: T.Tensor((rows // 16,), T.int32),
        B: T.Tensor((16, 128), T.bfloat16),
        C: T.Tensor((rows, 16), T.float32),
    ):
        with T.Kernel(1, threads=256):
            s = T.alloc_shared((rows, 128), T.bfloat16)
            a = T.alloc_fragment((rows, 128), T.bfloat16)
            b = T.alloc_fragment((16, 128), T.bfloat16)
            c = T.alloc_fragment((rows, 16), T.float32)
            for i, j in T.Parallel(rows, 128, prefer_async=True):
                page = P[i // 16]
                s[i, j] = A[page, i % 16, j * source_stride]
            T.ptx_wait_group(0)
            T.sync_threads()
            T.copy(s, a)
            T.copy(B, b)
            T.gemm(a, b, c, transpose_B=True, clear_accum=True, k_pack=2, policy=T.GemmWarpPolicy.FullRow)
            T.copy(c, C)

    return main


@pytest.mark.parametrize("rows", [64, 128])
def test_paged_gather_swizzled_contiguity(rows):
    kernel = tl.compile(
        gather_program(rows),
        out_idx=[3],
        target=tvm.target.Target({"kind": "hcu", "mcpu": current_hcu_arch_string()}),
        pass_configs={PassConfigKey.TL_ENABLE_ASYNC_COPY: True},
    )
    assert "cp_async_gs" in kernel.get_kernel_source()
    gen = torch.Generator().manual_seed(27)
    a = torch.randn((8, 16, 128), generator=gen).bfloat16()
    b = torch.randn((16, 128), generator=gen).bfloat16()
    p = (torch.arange(rows // 16, dtype=torch.int32) * 3 + 1) % 8
    expected = a[p.long()].reshape(rows, 128).float() @ b.float().T
    actual = kernel(a.cuda(), p.cuda(), b.cuda()).cpu()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_reject_strided_gather():
    with pytest.raises(tvm.error.InternalError, match="eligible HCU global-to-shared copy"):
        tl.compile(
            gather_program(64, source_stride=2),
            out_idx=[3],
            target=tvm.target.Target({"kind": "hcu", "mcpu": current_hcu_arch_string()}),
            pass_configs={PassConfigKey.TL_ENABLE_ASYNC_COPY: True},
        )
