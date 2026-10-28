# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Ordinary b16 LDS layout materialization and pipeline integration."""

import numpy as np
import pytest
import torch
import tilelang as tl
import tilelang.language as T
from tilelang import tvm
from tilelang.hcu.language import gemm
from tilelang.hcu.transform import MaterializeHcuGemmLdsStrategy
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


@pytest.fixture(params=["gfx938", "gfx946"])
def runtime_arch(request):
    """Both targets must run in separate PMD processes; never execute foreign ISA."""
    actual = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if request.param != actual:
        pytest.skip(f"requires {request.param}; runtime reports {actual}")
    return request.param


PLAIN = "tl.hcu_plain_gemm_lds_copy"
LAYOUT_MAP = "layout_map"


def shared_program(dtype="float16", width=64, threads=128, stages=2, pack=2, shared=True, shared_as_b=False, relay=False, identity=False):
    @T.prim_func
    def main(
        A: T.Tensor((2, 32, 32), dtype),
        S: T.Tensor((2, 32, width), dtype),
        B: T.Tensor((2, width, 32), dtype),
        C: T.Tensor((32, width), T.float32),
        D: T.Tensor((32, 32), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            a = T.alloc_shared((32, 32), dtype)
            s = T.alloc_shared((32, width), dtype)
            b = T.alloc_shared((width, 32), dtype)
            if identity:
                T.annotate_layout({s: tl.layout.Layout((32, width), lambda i, j: (i, j))})
            c = T.alloc_fragment((32, width), T.float32)
            d = T.alloc_fragment((32, 32), T.float32)
            s_fragment = T.alloc_fragment((32, width), dtype)
            T.clear(c)
            T.clear(d)
            for i in T.Pipelined(2, num_stages=stages):
                T.copy(A[i, 0, 0], a)
                T.copy(S[i, 0, 0], s)
                T.copy(B[i, 0, 0], b)
                if relay:
                    T.copy(s, s_fragment)
                    gemm(a, s_fragment, c)
                else:
                    gemm(a, s, c)
                if shared:
                    if shared_as_b:
                        gemm(b, s, d, True, True, k_pack=pack)
                    else:
                        gemm(s, b, d, k_pack=pack)
            T.copy(c, C)
            T.copy(d, D)

    return main


def lower_with_captures(program, arch="gfx946", enable_async=False):
    captures = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in {"tl.MaterializeHcuGemmLdsStrategy", "tl.InjectSoftwarePipeline"}:
                captures[info.name] = mod

    target = tvm.target.Target({"kind": "hcu", "mcpu": arch})
    with (
        target,
        tvm.transform.PassContext(
            config={PassConfigKey.TL_ENABLE_ASYNC_COPY: enable_async, PassConfigKey.TIR_USE_ASYNC_COPY: enable_async},
            instruments=[Capture()],
        ),
    ):
        result = tl.lower(program, target=target)
    return captures, result


def plans_and_maps(mod):
    plans = []
    maps = {}

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and PLAIN in node.annotations:
            dst = node.args[1].args[0].buffer
            plans.append((dst, node.annotations[PLAIN]))
        if isinstance(node, tvm.tirx.SBlock) and LAYOUT_MAP in node.annotations:
            maps.update(dict(node.annotations[LAYOUT_MAP]))

    for func in mod.functions.values():
        if isinstance(func, tvm.tirx.PrimFunc):
            tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return plans, maps


@pytest.mark.parametrize("arch", ["gfx938", "gfx946"])
@pytest.mark.parametrize("width", [32, 64, 128])
@pytest.mark.parametrize("pack", [1, 2])
@pytest.mark.parametrize("shared_as_b", [False, True])
def test_materialize_and_pipeline_layout(arch, width, pack, shared_as_b):
    from test_hcu_plain_copy_lds_formula import enumerate_layout, make_layout

    captures, _ = lower_with_captures(shared_program(width=width, pack=pack, shared_as_b=shared_as_b), arch)
    for phase, mod in captures.items():
        plans, maps = plans_and_maps(mod)
        assert plans
        assert any(buffer.name == "s" for buffer, _ in plans)
        for buffer, marker in plans:
            assert int(marker) == 1
            assert buffer.data in maps
            layout = maps[buffer.data]
            assert list(layout.get_input_shape()) == list(buffer.shape)
            if buffer.name == "s":
                shape = tuple(int(x) for x in buffer.shape)
                np.testing.assert_array_equal(
                    enumerate_layout(layout, shape), enumerate_layout(make_layout(shape, 32 if arch == "gfx938" else 64, pack), shape)
                )
            if phase == "tl.InjectSoftwarePipeline":
                assert len(buffer.shape) == 3
                assert int(buffer.shape[0]) == 2
        text = str(mod)
        assert "tl.hcu_layout_ds_read" in text
        assert ("tl.hcu_b_respect_layout_map" if shared_as_b else "tl.hcu_a_respect_layout_map") in text
    assert len(captures) == 2


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "width,threads,stages,pack,shared,shared_as_b",
    [
        (32, 64, 0, 1, False, False),
        (64, 128, 2, 1, True, False),
        (128, 64, 2, 2, True, False),
        (64, 128, 2, 2, True, True),
    ],
)
def test_plain_copy_swizzle_numerical(runtime_arch, dtype, width, threads, stages, pack, shared, shared_as_b):
    program = shared_program(dtype, width, threads, stages, pack, shared, shared_as_b)
    kernel = tl.compile(
        program,
        target={"kind": "hcu", "mcpu": runtime_arch},
        out_idx=[3, 4],
        pass_configs={PassConfigKey.TL_ENABLE_ASYNC_COPY: False, PassConfigKey.TIR_USE_ASYNC_COPY: False},
    )
    source = kernel.get_kernel_source()
    assert "ds_read" in source
    assert "cp_async_gs" not in source and "load_async_lds" not in source
    if shared:
        vector = "uint4" if pack == 2 else "uint2"
        assert any(f"*({vector}*)(" in line and "_local" in line and "*)s)" in line for line in source.splitlines())
    tdtype = getattr(torch, dtype)
    a = torch.randn((2, 32, 32), dtype=tdtype)
    s = torch.randn((2, 32, width), dtype=tdtype)
    b = torch.randn((2, width, 32), dtype=tdtype)
    c, d = (value.cpu() for value in kernel(a.cuda(), s.cuda(), b.cuda()))
    expected_c = sum(a[i].float() @ s[i].float() for i in range(2))
    expected_d = (
        sum(b[i].float().T @ s[i].float().T if shared_as_b else s[i].float() @ b[i].float() for i in range(2))
        if shared
        else torch.zeros_like(d)
    )
    torch.testing.assert_close(c, expected_c, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(d, expected_d, atol=2e-2, rtol=2e-2)


def prepare_materialize(program, enable_async=False, collect=True):
    target = tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"})
    mod = tvm.IRModule({"main": program})
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tl.transform.MaterializeKernelLaunch(lower_grid_binding=True, lower_thread_binding=True, default_threads=128)(mod)
    mod = tl.transform.Simplify()(mod)
    if collect:
        mod = tl.hcu.transform.AnnotateMlsGemmDep()(mod)
    with tvm.transform.PassContext(config={PassConfigKey.TIR_USE_ASYNC_COPY: enable_async}):
        mod = tl.transform.PipelinePlanning()(mod)
    return mod


def map_calls(mod, rewrite):
    def transform(func, _, __):
        return func.with_body(tvm.tirx.stmt_functor.ir_transform(func.body, None, rewrite, ["tirx.Call"]))

    return tvm.tirx.transform.prim_func_pass(transform, opt_level=0)(mod)


def test_no_pending_candidate_can_get_plain_layout():
    mod = prepare_materialize(shared_program(pack=2), collect=False)
    assert "pending_gemm" not in str(mod)
    result = MaterializeHcuGemmLdsStrategy()(mod)
    plans, _ = plans_and_maps(result)
    assert any(buffer.name == "s" for buffer, _ in plans)


@pytest.mark.parametrize("semantic", ["is_async_copy", "force_cp_async", "no_implicit_async_commit_wait"])
def test_async_semantic_copy_never_gets_plain_layout(semantic):
    mod = prepare_materialize(shared_program(pack=2), collect=False)

    def rewrite(node):
        if node.op.name == "tl.tileop.copy" and node.args[1].args[0].buffer.name == "s":
            annotations = dict(node.annotations)
            annotations[semantic] = tvm.tirx.IntImm("int32", 1)
            return tvm.tirx.Call(node.dtype, node.op, node.args, annotations=annotations)
        return node

    result = MaterializeHcuGemmLdsStrategy()(map_calls(mod, rewrite))
    plans, _ = plans_and_maps(result)
    assert all(buffer.name != "s" for buffer, _ in plans)


def test_planner_selected_copy_without_pending_is_excluded():
    mod = prepare_materialize(shared_program(pack=1), enable_async=True)
    assert "software_pipeline_async_producers" in str(mod)
    assert "tl.hcu_pending_gemm_lds_copy_strategy" in str(mod)

    def rewrite(node):
        annotations = {key: value for key, value in node.annotations.items() if not key.startswith("tl.hcu_pending_")}
        return tvm.tirx.Call(node.dtype, node.op, node.args, annotations=annotations)

    result = MaterializeHcuGemmLdsStrategy()(map_calls(mod, rewrite))
    plans, _ = plans_and_maps(result)
    assert all(buffer.name != "b" for buffer, _ in plans)
    assert any(buffer.name == "s" for buffer, _ in plans)


def mixed_writer_program(prefer_async=False):
    @T.prim_func
    def main(A: T.Tensor((32, 32), T.float16), B: T.Tensor((32, 64), T.float16), C: T.Tensor((32, 64), T.float32)):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((32, 32), T.float16)
            s = T.alloc_shared((32, 64), T.float16)
            c = T.alloc_fragment((32, 64), T.float32)
            T.copy(A, a)
            T.copy(B, s)
            if prefer_async:
                with T.Parallel(32, 64, prefer_async=True) as (i, j):
                    s[i, j] = B[i, j]
            else:
                T.async_copy(B, s)
            T.clear(c)
            gemm(a, s, c)
            T.copy(c, C)

    return main


@pytest.mark.parametrize("prefer_async", [False, True])
@pytest.mark.parametrize("collect", [False, True])
def test_mixed_buffer_writers_are_excluded(prefer_async, collect):
    mod = prepare_materialize(mixed_writer_program(prefer_async), collect=collect)
    result = MaterializeHcuGemmLdsStrategy()(mod)
    plans, _ = plans_and_maps(result)
    assert all(buffer.name != "s" for buffer, _ in plans)
    assert PLAIN not in str(result)


def single_program(dtype, trans_a, trans_b, explicit_async=False, pack=1):
    a_shape = (32, 64) if trans_a else (64, 32)

    @T.prim_func
    def main(A: T.Tensor(a_shape, dtype), B: T.Tensor((32, 32), dtype), C: T.Tensor((64, 32), T.float32)):
        with T.Kernel(1, threads=64):
            a = T.alloc_shared(a_shape, dtype)
            b = T.alloc_shared((32, 32), dtype)
            c = T.alloc_fragment((64, 32), T.float32)
            if explicit_async:
                T.async_copy(A, a)
                T.async_copy(B, b)
            else:
                T.copy(A, a)
                T.copy(B, b)
            T.clear(c)
            gemm(a, b, c, trans_a, trans_b, k_pack=pack)
            T.copy(c, C)

    return main


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "trans_a,trans_b,pack", [(True, True, 1), (True, False, 1), (False, True, 1), (False, False, 1), (False, False, 2), (True, False, 2)]
)
def test_single_consumer_numerical(runtime_arch, dtype, trans_a, trans_b, pack):
    program = single_program(dtype, trans_a, trans_b, pack=pack)
    kernel = tl.compile(
        program, target={"kind": "hcu", "mcpu": runtime_arch}, out_idx=[2], pass_configs={PassConfigKey.TIR_USE_ASYNC_COPY: False}
    )
    a = torch.randn((32, 64) if trans_a else (64, 32), dtype=getattr(torch, dtype))
    b = torch.randn((32, 32), dtype=getattr(torch, dtype))
    expected = (a.T if trans_a else a).float() @ (b.T if trans_b else b).float()
    torch.testing.assert_close(kernel(a.cuda(), b.cuda()).cpu(), expected, atol=2e-2, rtol=2e-2)


def test_explicit_async_path_numerical(runtime_arch):
    program = single_program("float16", False, False, explicit_async=True)
    captures, _ = lower_with_captures(program, arch=runtime_arch, enable_async=True)
    assert not plans_and_maps(captures["tl.MaterializeHcuGemmLdsStrategy"])[0]
    kernel = tl.compile(program, target={"kind": "hcu", "mcpu": runtime_arch}, out_idx=[2])
    source = kernel.get_kernel_source()
    assert "cp_async_gs" in source or "load_async_lds" in source
    a = torch.randn((64, 32), dtype=torch.float16)
    b = torch.randn((32, 32), dtype=torch.float16)
    torch.testing.assert_close(kernel(a.cuda(), b.cuda()).cpu(), a.float() @ b.float(), atol=2e-2, rtol=2e-2)


def test_unsupported_consumer_abandons_entire_buffer():
    mod = prepare_materialize(shared_program(pack=2), collect=False)

    # Make the BT consumer itself packed; the compatible AT consumer must not
    # cause a partial ordinary plan to survive on the same shared buffer.
    def rewrite(node):
        if node.op.name == "tl.tileop.gemm" and node.args[1].args[0].buffer.name == "s":
            annotations = dict(node.annotations)
            annotations["k_pack"] = tvm.tirx.IntImm("int32", 2)
            return tvm.tirx.Call(node.dtype, node.op, node.args, annotations=annotations)
        return node

    result = MaterializeHcuGemmLdsStrategy()(map_calls(mod, rewrite))
    plans, _ = plans_and_maps(result)
    assert all(buffer.name != "s" for buffer, _ in plans)


def test_ordinary_plan_rejects_late_async_semantics():
    mod = MaterializeHcuGemmLdsStrategy()(prepare_materialize(shared_program(stages=0), collect=False))

    def rewrite(node):
        if PLAIN in node.annotations:
            annotations = dict(node.annotations)
            annotations["force_cp_async"] = tvm.tirx.IntImm("int32", 1)
            return tvm.tirx.Call(node.dtype, node.op, node.args, annotations=annotations)
        return node

    with pytest.raises(tvm.error.InternalError, match="ordinary HCU LDS layout cannot acquire async"):
        tl.transform.LayoutInference()(map_calls(mod, rewrite))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_shared_fragment_copy_is_materialized(runtime_arch, dtype):
    program = shared_program(dtype=dtype, relay=True)
    captures, _ = lower_with_captures(program, arch=runtime_arch)
    assert any(buffer.name == "s" for buffer, _ in plans_and_maps(captures["tl.MaterializeHcuGemmLdsStrategy"])[0])
    kernel = tl.compile(
        program, target={"kind": "hcu", "mcpu": runtime_arch}, out_idx=[3, 4], pass_configs={PassConfigKey.TIR_USE_ASYNC_COPY: False}
    )
    a = torch.randn((2, 32, 32), dtype=getattr(torch, dtype))
    s = torch.randn((2, 32, 64), dtype=getattr(torch, dtype))
    b = torch.randn((2, 64, 32), dtype=getattr(torch, dtype))
    c, d = (value.cpu() for value in kernel(a.cuda(), s.cuda(), b.cuda()))
    torch.testing.assert_close(c, sum(a[i].float() @ s[i].float() for i in range(2)), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(d, sum(s[i].float() @ b[i].float() for i in range(2)), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("arch", ["gfx938", "gfx946"])
@pytest.mark.parametrize("pack", [1, 2])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_lowered_instruction_addresses(arch, pack, dtype, tmp_path):
    from plain_copy_lds_codegen import check_assembly, check_lowered_addresses

    _, artifact = lower_with_captures(shared_program(dtype=dtype, width=128, threads=64, stages=0, pack=pack), arch)
    check_lowered_addresses(artifact, arch, pack)
    check_assembly(artifact, arch, pack, tmp_path)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("pack", [1, 2])
def test_pure_atbn_retains_default_layout(dtype, pack):
    from test_hcu_plain_copy_lds_fp8 import pure_program

    captures, artifact = lower_with_captures(pure_program(dtype, pack, k=64), "gfx938")
    for mod in captures.values():
        assert not plans_and_maps(mod)[0]
        assert "tl.hcu_a_respect_layout_map" not in str(mod)
        assert "tl.hcu_b_respect_layout_map" not in str(mod)
        assert "tl.hcu_layout_ds_read" not in str(mod)
    assert "ds_read_m32x32" not in artifact.kernel_source
