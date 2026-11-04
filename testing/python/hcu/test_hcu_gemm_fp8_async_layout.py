# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""FP8 async producers and layout-aware AN/BT / mixed GEMM consumers."""

import json
from itertools import product

import pytest
import torch
import tilelang as tl
import tilelang.language as T
from tilelang import tvm
from tilelang.hcu.language import gemm
from tilelang.transform import PassConfigKey
from test_hcu_plain_copy_lds_formula_b8 import make_layout
from test_hcu_plain_copy_lds_swizzle import lower_with_captures

import tilelang.testing
from hcu_test_utils import current_hcu_arch_string

pytestmark = [
    *tilelang.testing.requires_rocm.marks(),
    pytest.mark.skipif(current_hcu_arch_string() not in {"gfx938", "gfx946"}, reason="requires HCU LDS target"),
]


@pytest.fixture
def runtime_arch():
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    assert arch in {"gfx938", "gfx946"}
    return arch


def async_program(
    dtype="float8_e4m3fn", mixed=False, pack=1, width=64, stages=0, async_src=True, bad_layout=False, storage=None, threads=64
):
    @T.prim_func
    def main(
        A: T.Tensor((3, 32, 32), dtype),
        S: T.Tensor((3, 32, width), dtype),
        B: T.Tensor((3, 32, width), dtype),
        C: T.Tensor((3, 32, width), T.float32),
        D: T.Tensor((3, 32, 32), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            a = T.alloc_shared((32, 32), dtype)
            s = T.alloc_shared((32, width), dtype)
            b = T.alloc_shared((32, width), dtype)
            if storage is not None:
                T.annotate_layout({s: storage})
            if bad_layout:
                T.annotate_layout({s: tl.layout.Layout((32, width), lambda i, j: (i, j))})
            c = T.alloc_fragment((32, width), T.float32)
            d = T.alloc_fragment((32, 32), T.float32)
            for it in T.Pipelined(3, num_stages=stages):
                T.copy(A[it, :, :], a)
                if async_src:
                    T.async_copy(S[it, :, :], s)
                else:
                    T.copy(S[it, :, :], s)
                T.copy(B[it, :, :], b)
                T.ptx_wait_group(0)
                T.sync_threads()
                T.clear(c)
                T.clear(d)
                gemm(a, s, c, False, False)
                if mixed:
                    gemm(s, b, d, False, True, k_pack=pack)
                T.copy(c, C[it, :, :])
                T.copy(d, D[it, :, :])

    return main


@pytest.mark.parametrize("mixed", [False, True])
def test_async_storage_mismatch_diagnostic(mixed):
    with pytest.raises(tvm.error.InternalError, match="storage layout mismatch") as error:
        lower_with_captures(async_program(mixed=mixed, bad_layout=True), "gfx938")
    message = str(error.value).splitlines()[-1]
    expected = 16 if mixed else 32
    assert "logical coordinate (2, 0), dimension 1: actual=0" in message
    assert f"expected={expected}" in message
    print("ASYNC_GUARD_TRIGGERED", message)


@pytest.mark.parametrize("mixed", [False, True])
def test_async_equivalent_storage_is_accepted(mixed):
    storage = make_layout((32, 64), 32, (1,) if mixed else ())
    captures, artifact = lower_with_captures(async_program(mixed=mixed, storage=storage), "gfx938")
    assert "tl.hcu_gemm_an_bt_lds_strategy" in str(captures["tl.MaterializeHcuGemmLdsStrategy"])
    assert "tl::cp_async_gs<16>" in artifact.kernel_source
    assert "tl::ds_read_vector" in artifact.kernel_source


@pytest.mark.parametrize(
    "dtype,mixed,pack,stages",
    [
        (dtype, mixed, pack, stages)
        for dtype, stages in product(["float8_e4m3fn", "float8_e5m2"], [0, 2])
        for mixed, pack in [(False, 1), (True, 1), (True, 2)]
    ],
)
def test_fp8_async_numerical(runtime_arch, dtype, mixed, pack, stages):
    torch.manual_seed(20261028)
    s, b = (torch.randn((32, 64)).to(getattr(torch, dtype)) for _ in range(2))
    a = torch.randn((32, 32)).to(getattr(torch, dtype))
    inputs = [torch.stack([a, a, a]), torch.stack([s, (-s.float()).to(s.dtype), s]), torch.stack([b, b, b])]
    references = [
        torch.stack([a.float() @ x.float() for x in inputs[1]]),
        torch.stack([x.float() @ b.float().T for x in inputs[1]]) if mixed else torch.zeros((3, 32, 32)),
    ]
    kernel = tl.compile(
        async_program(dtype, mixed, pack, stages=stages),
        target={"kind": "hcu", "mcpu": runtime_arch},
        out_idx=[3, 4],
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_ENABLE_ASYNC_COPY: False, PassConfigKey.TIR_USE_ASYNC_COPY: False},
    )
    source = kernel.get_kernel_source()
    assert "tl::cp_async_gs<16>" in source
    assert ("tl::ds_read_vector" in source) == mixed
    actuals = [x.cpu() for x in kernel(*(x.cuda() for x in inputs))]
    torch.cuda.synchronize()
    errors = [(actual - expected).abs().max().item() for actual, expected in zip(actuals, references)]
    print("FP8_ASYNC_RESULT", json.dumps(dict(arch=runtime_arch, dtype=dtype, mixed=mixed, pack=pack, stages=stages, max_abs=errors)))
    for actual, expected in zip(actuals, references):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
