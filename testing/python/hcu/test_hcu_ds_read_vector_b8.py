# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Plain b8 matrix instruction: exact byte and native fragment ownership checks."""

import numpy as np
import pytest
import torch
import tilelang as tl
import tilelang.language as T
from tilelang import tvm
from tilelang.transform import PassConfigKey

import tilelang.testing
from hcu_test_utils import current_hcu_arch_string

pytestmark = [
    *tilelang.testing.requires_rocm.marks(),
    pytest.mark.skipif(
        current_hcu_arch_string() not in {"gfx938", "gfx946"},
        reason="requires a supported HCU LDS target",
    ),
]


def instruction_program(dtype):
    @T.prim_func
    def main(A: T.Tensor((32, 32), dtype), O: T.Tensor((64, 4), T.int32)):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            s = T.alloc_shared((32, 32), dtype)
            r = T.alloc_local((4,), T.float32)
            T.annotate_layout({s: tl.layout.Layout((32, 32), lambda i, j: (i, j))})
            T.copy(A, s)
            T.sync_threads()
            T.evaluate(T.call_intrin("handle", tvm.ir.Op.get("tl.ds_read_vector"), r.data, 0, s[tx // 2, (tx % 2) * 16]))
            for v in T.serial(4):
                O[tx, v] = T.reinterpret(T.int32, r[v])

    return main


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_b8_instruction_bytes(dtype):
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    assert arch in ("gfx938", "gfx946")
    program = instruction_program(dtype)
    kernel = tl.compile(
        program,
        target={"kind": "hcu", "mcpu": arch},
        out_idx=[1],
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_ENABLE_ASYNC_COPY: False, PassConfigKey.TIR_USE_ASYNC_COPY: False},
    )
    assert "tl::ds_read_vector" in kernel.get_kernel_source()
    # Two tag planes jointly distinguish all 1024 logical byte positions.
    # No floating arithmetic is allowed to normalize NaN or subnormal encodings.
    q = np.arange(1024, dtype=np.int32).reshape(32, 32)
    for tags in [q % 251, q // 251]:
        data = tags.astype(np.uint8)
        argument = torch.from_numpy(data.copy()).view(getattr(torch, dtype)).cuda()
        output = kernel(argument).cpu().numpy().view(np.uint8).reshape(64, 16)
        torch.cuda.synchronize()
        expected = np.empty((64, 16), dtype=np.uint8)
        for k in range(32):
            for n in range(32):
                expected[n % 16 + 16 * (k // 8), (n // 16) * 8 + k % 8] = data[k, n]
        np.testing.assert_array_equal(output, expected)
    print("B8_INSTRUCTION_BYTES_PASS", dtype, "64 lanes x 16 bytes; two independent tag planes")


def layout_program(dtype="float8_e4m3fn", pack=1, layout_mode="common", width=64):
    from tilelang.hcu.language import gemm
    from test_hcu_plain_copy_lds_formula_b8 import make_layout

    if layout_mode == "bad":
        # Exchange interior column bits 1/2. Both ends of every N16 segment
        # stay fixed, so checking only segment endpoints would miss this error.
        layout = tl.layout.Layout((32, width), lambda i, j: (i, j // 8 * 8 + j % 2 + (j // 2 % 2) * 4 + (j // 4 % 2) * 2))
    elif layout_mode == "atbn_kp1":
        # Valid for ordinary 8B reads, deliberately invalid for a matrix N16 read.
        from tilelang.layout.swizzle import make_hcu_swizzled_layout

        layout = make_hcu_swizzled_layout(tvm.tirx.decl_buffer((32, width), dtype, scope="shared"), 1)
    elif layout_mode == "common":
        layout = make_layout((32, width), 32, (pack,))
    else:
        layout = tl.layout.Layout((32, width), lambda i, j: (i, j))

    @T.prim_func
    def main(
        A: T.Tensor((32, 32), dtype),
        S: T.Tensor((32, width), dtype),
        B: T.Tensor((width, 32), dtype),
        C: T.Tensor((32, width), T.float32),
        D: T.Tensor((32, 32), T.float32),
    ):
        with T.Kernel(1, threads=64):
            a = T.alloc_shared((32, 32), dtype)
            s = T.alloc_shared((32, width), dtype)
            b = T.alloc_shared((width, 32), dtype)
            sf = T.alloc_fragment((32, width), dtype)
            c = T.alloc_fragment((32, width), T.float32)
            d = T.alloc_fragment((32, 32), T.float32)
            T.annotate_layout(
                {
                    a: tl.layout.Layout((32, 32), lambda i, j: (i, j)),
                    s: layout,
                    b: tl.layout.Layout((width, 32), lambda i, j: (i, j)),
                }
            )
            T.clear(c)
            T.clear(d)
            T.copy(A, a)
            T.copy(S, s)
            T.copy(B, b)
            T.copy(s, sf)
            gemm(a, sf, c)
            gemm(s, b, d, k_pack=pack)
            T.copy(c, C)
            T.copy(d, D)

    return main


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("layout_mode", ["bad", "atbn_kp1"])
def test_b8_layout_interior_guard(dtype, layout_mode):
    from test_hcu_plain_copy_lds_swizzle import lower_with_captures

    with pytest.raises(tvm.error.InternalError, match="16-byte segment to remain contiguous") as error:
        lower_with_captures(layout_program(dtype, layout_mode=layout_mode), "gfx938")
    print("B8_INTERIOR_GUARD_TRIGGERED", layout_mode, str(error.value).splitlines()[-1])


def paired_program(dtype="float8_e4m3fn", pack=1, mode="common"):
    """Step 2c's exact S@B.T/B@S.T inputs, with AN stored as [K,M]."""
    from tilelang.hcu.language import gemm
    from test_hcu_plain_copy_lds_formula_b8 import make_layout

    s_layout = make_layout((64, 32), 32, (1,)) if mode == "common" else tl.layout.Layout((64, 32), lambda i, j: (i, j))
    b_layout = make_layout((32, 64), 32, (1, pack)) if mode == "common" else tl.layout.Layout((32, 64), lambda i, j: (i, j))
    t_layout = make_layout((32, 64), 32, (pack,)) if mode == "common" else tl.layout.Layout((32, 64), lambda i, j: (i, j))

    @T.prim_func
    def main(
        ST: T.Tensor((64, 32), dtype),
        S: T.Tensor((32, 64), dtype),
        B: T.Tensor((32, 64), dtype),
        C: T.Tensor((32, 32), T.float32),
        D: T.Tensor((32, 32), T.float32),
    ):
        with T.Kernel(1, threads=64):
            s = T.alloc_shared((64, 32), dtype)
            t = T.alloc_shared((32, 64), dtype)
            b = T.alloc_shared((32, 64), dtype)
            bn = T.alloc_shared((32, 64), dtype)
            sf = T.alloc_fragment((64, 32), dtype)
            c = T.alloc_fragment((32, 32), T.float32)
            d = T.alloc_fragment((32, 32), T.float32)
            if mode != "auto":
                T.annotate_layout({s: s_layout, b: b_layout, bn: b_layout, t: t_layout})
            T.copy(ST, s)
            T.copy(S, t)
            T.copy(B, b)
            # Keep the two AT/BN pack contracts on separate shared buffers.
            T.copy(B, bn)
            T.clear(c)
            T.clear(d)
            if mode == "auto":
                gemm(s, b, c, True, True)
            else:
                T.copy(s, sf)
                gemm(sf, b, c, True, True)
            gemm(bn, t, d, False, True, k_pack=pack)
            T.copy(c, C)
            T.copy(d, D)

    return main


@pytest.mark.parametrize("dtype,pack", [(dtype, pack) for dtype in ["float8_e4m3fn", "float8_e5m2"] for pack in [1, 2]])
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="numerical fixture requires gfx938")
def test_b8_layout_numerical(dtype, pack):
    from test_hcu_plain_copy_lds_swizzle import lower_with_captures

    torch.manual_seed(20261028)
    s, b = (torch.randn((32, 64)).to(getattr(torch, dtype)) for _ in range(2))
    references = [s.float() @ b.float().T, b.float() @ s.float().T]
    results = []
    for mode in ["identity", "common", "auto"]:
        program = paired_program(dtype, pack, mode)
        captures, artifact = lower_with_captures(program, "gfx938")
        assert "tl::ds_read_vector" in artifact.kernel_source
        assert "tl.hcu_layout_ds_read" in str(captures["tl.MaterializeHcuGemmLdsStrategy"])
        kernel = tl.compile(
            program,
            target={"kind": "hcu", "mcpu": "gfx938"},
            out_idx=[3, 4],
            execution_backend="cython",
            pass_configs={PassConfigKey.TL_ENABLE_ASYNC_COPY: False, PassConfigKey.TIR_USE_ASYNC_COPY: False},
        )
        outputs = [x.cpu() for x in kernel(s.T.contiguous().cuda(), s.cuda(), b.cuda())]
        torch.cuda.synchronize()
        for actual, expected in zip(outputs, references):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        results.append(outputs)
        print("B8_LAYOUT_NUMERICAL_PASS", dtype, pack, mode, "max_abs=0 rms=0")
    for outputs in results[1:]:
        for actual, expected in zip(outputs, results[0]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
