# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Pure AT/BN fp8 retains its existing layout and ordinary LDS reads."""

import json
from itertools import product

import pytest
import torch
import tilelang as tl
import tilelang.language as T
from tilelang import tvm
from tilelang.hcu.language import gemm
from tilelang.hcu.transform import MaterializeHcuGemmLdsStrategy
from tilelang.transform import PassConfigKey
from test_hcu_plain_copy_lds_swizzle import (
    lower_with_captures,
    plans_and_maps,
    prepare_materialize,
    shared_program,
)
from test_hcu_plain_copy_lds_formula_b8 import make_layout

import tilelang.testing
from hcu_test_utils import current_hcu_arch_string

pytestmark = [
    *tilelang.testing.requires_rocm.marks(),
    pytest.mark.skipif(
        current_hcu_arch_string() not in {"gfx938", "gfx946"},
        reason="requires a supported HCU LDS target",
    ),
]


def pure_program(dtype="float8_e4m3fn", pack=1, identity=False, relay=False, fragment_writer=False, k=64, second_pack=None):
    second_pack = pack if second_pack is None else second_pack

    # s is consumed as A (AT) and B (BN) by separate GEMMs. Other operands
    # are also AT/BN, so no b8 matrix-read lowering is needed anywhere.
    @T.prim_func
    def main(S: T.Tensor((32, k), dtype), B: T.Tensor((32, k), dtype), C: T.Tensor((32, 32), T.float32), D: T.Tensor((32, 32), T.float32)):
        with T.Kernel(1, threads=64):
            s = T.alloc_shared((32, k), dtype)
            b = T.alloc_shared((32, k), dtype)
            sf = T.alloc_fragment((32, k), dtype)
            writer = T.alloc_fragment((32, k), dtype)
            c = T.alloc_fragment((32, 32), T.float32)
            d = T.alloc_fragment((32, 32), T.float32)
            if identity:
                T.annotate_layout({s: tl.layout.Layout((32, k), lambda i, j: (i, j)), b: tl.layout.Layout((32, k), lambda i, j: (i, j))})
            if fragment_writer:
                T.copy(S, writer)
                T.copy(writer, s)
            else:
                T.copy(S, s)
            T.copy(B, b)
            T.clear(c)
            T.clear(d)
            if relay:
                T.copy(s, sf)
                gemm(sf, b, c, False, True, k_pack=pack)
            else:
                gemm(s, b, c, False, True, k_pack=pack)
            gemm(b, s, d, False, True, k_pack=second_pack)
            T.copy(c, C)
            T.copy(d, D)

    return main


@pytest.mark.parametrize("banks,packs", list(product([32, 64], [(1,), (2,), (1, 2), (2, 1)])))
def test_pure_atbn_factory(banks, packs):
    factory = tvm.get_global_func("tl.hcu.MakePlainGemmLdsLayout")
    args = [(32, 128), banks, 4]
    actual = factory(*args, 8, 32, 32, 16, 16, [0] * len(packs), list(packs))
    assert actual is None
    assert factory(*args, 16, 16, 32, 8, 16, [0] * len(packs), list(packs)) is None


@pytest.mark.parametrize(
    "arch,dtype,pack,relay,fragment_writer",
    list(product(["gfx938", "gfx946"], ["float8_e4m3fn", "float8_e5m2"], [1, 2], [False, True], [False, True])),
)
def test_fp8_materialize(arch, dtype, pack, relay, fragment_writer):
    captures, artifact = lower_with_captures(pure_program(dtype, pack, relay=relay, fragment_writer=fragment_writer), arch)
    for mod in captures.values():
        plans, _ = plans_and_maps(mod)
        assert not plans
        assert "tl.hcu_a_respect_layout_map" not in str(mod)
        assert "tl.hcu_b_respect_layout_map" not in str(mod)
        assert "tl.hcu_layout_ds_read" not in str(mod)
    assert "ds_read_m32x32" not in artifact.kernel_source


@pytest.mark.parametrize("dtype,pack,relay", list(product(["float8_e4m3fn", "float8_e5m2"], [1, 2], [False, True])))
def test_fp8_anbt_materializes_entire_buffer(dtype, pack, relay):
    mod = prepare_materialize(shared_program(dtype=dtype, pack=pack, relay=relay), collect=False)
    result = MaterializeHcuGemmLdsStrategy()(mod)
    plans, _ = plans_and_maps(result)
    assert any(buffer.name == "s" for buffer, _ in plans)
    assert "tl.hcu_layout_ds_read" in str(result)


@pytest.mark.parametrize("dtype,identity", list(product(["float8_e4m3fn", "float8_e5m2"], [False, True])))
def test_fp8_anbt_materialize_guard(dtype, identity):
    mod = prepare_materialize(shared_program(dtype=dtype, stages=0), collect=False)

    def annotate(node):
        if isinstance(node, tvm.tirx.SBlock):
            for buffer in node.alloc_buffers:
                if buffer.name == "s":
                    annotations = dict(node.annotations)
                    annotations["layout_map"] = {
                        buffer.data: tl.layout.Layout((32, 64), lambda i, j: (i, j)) if identity else make_layout((32, 64), 64, (1,))
                    }
                    return tvm.tirx.SBlock(
                        node.iter_vars,
                        node.reads,
                        node.writes,
                        node.name_hint,
                        node.body,
                        node.init,
                        node.alloc_buffers,
                        node.match_buffers,
                        annotations,
                    )
        return node

    def transform(func, _, __):
        return func.with_body(tvm.tirx.stmt_functor.ir_transform(func.body, None, annotate, ["tirx.SBlock"]))

    mod = tvm.tirx.transform.prim_func_pass(transform, opt_level=0)(mod)
    if identity:
        MaterializeHcuGemmLdsStrategy()(mod)
    else:
        with pytest.raises(tvm.error.InternalError, match="fp8 AN/BT buffer s requires row-major storage") as error:
            MaterializeHcuGemmLdsStrategy()(mod)
        print("GUARD_TRIGGERED", str(error.value).splitlines()[-1])


@pytest.mark.parametrize("pack,k", [(1, 48), (2, 32)])
def test_fp8_rejects_incomplete_native_k(pack, k):
    result = MaterializeHcuGemmLdsStrategy()(prepare_materialize(pure_program(pack=pack, k=k), collect=False))
    assert not plans_and_maps(result)[0]


@pytest.mark.parametrize(
    "dtype,pack,relay,fragment_writer", list(product(["float8_e4m3fn", "float8_e5m2"], [1, 2], [False, True], [False, True]))
)
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="numerical fixture requires gfx938")
def test_fp8_numerical(dtype, pack, relay, fragment_writer):
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == "gfx938"
    assert props.multi_processor_count == 72
    torch.manual_seed(20261028)
    # Quantize on CPU; use the exact quantized inputs for both reference and LDS modes.
    s, b = (torch.randn((32, 64)).to(getattr(torch, dtype)) for _ in range(2))
    references = [s.float() @ b.float().T, b.float() @ s.float().T]
    results = []
    for identity in (True, False):
        program = pure_program(dtype, pack, identity=identity, relay=relay, fragment_writer=fragment_writer)
        captures, _ = lower_with_captures(program, "gfx938")
        plans, _ = plans_and_maps(captures["tl.MaterializeHcuGemmLdsStrategy"])
        assert not plans
        kernel = tl.compile(
            program,
            target={"kind": "hcu", "mcpu": "gfx938"},
            out_idx=[2, 3],
            execution_backend="cython",
            pass_configs={PassConfigKey.TL_ENABLE_ASYNC_COPY: False, PassConfigKey.TIR_USE_ASYNC_COPY: False},
        )
        output = [x.cpu() for x in kernel(s.cuda(), b.cuda())]
        torch.cuda.synchronize()
        errors = []
        for actual, expected in zip(output, references):
            errors.append({"max_abs": (actual - expected).abs().max().item(), "rms": (actual - expected).square().mean().sqrt().item()})
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        results.append(output)
        print(
            "FP8_RESULT",
            json.dumps(
                dict(
                    dtype=dtype,
                    pack=pack,
                    relay=relay,
                    fragment_writer=fragment_writer,
                    layout="row_major" if identity else "default_hcu",
                    errors=errors,
                )
            ),
        )
    for baseline, common in zip(*results):
        torch.testing.assert_close(baseline, common, atol=0, rtol=0)


@pytest.mark.parametrize("pack,second_pack", [(1, 2), (2, 1)])
def test_fp8_mixed_packs_materialize(pack, second_pack):
    mod = MaterializeHcuGemmLdsStrategy()(prepare_materialize(pure_program(pack=pack, second_pack=second_pack), collect=False))
    assert not plans_and_maps(mod)[0]
