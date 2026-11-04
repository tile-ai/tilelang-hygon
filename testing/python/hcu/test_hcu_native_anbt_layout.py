# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Matrix AN/BT strategy selection and generic HCU-layout fallback."""

from itertools import product

import pytest
import torch
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from hcu_test_utils import current_hcu_arch_string
from tilelang.hcu.language import gemm
from tilelang.transform import PassConfigKey
from test_hcu_plain_copy_lds_swizzle import lower_with_captures

pytestmark = [
    *tilelang.testing.requires_rocm.marks(),
    pytest.mark.skipif(current_hcu_arch_string() not in {"gfx938", "gfx946"}, reason="requires an HCU LDS target"),
]


def native_program(
    dtype="float16",
    width=32,
    role="B",
    async_src=False,
    stages=0,
    storage=None,
    matrix_read=False,
    threads=64,
    fragment_writer=False,
    k=32,
    k_pack=1,
):
    out_shape = (32, width) if role == "B" else (width, 32)
    other_shape = (32, k) if role == "B" else (k, 32)

    @T.prim_func
    def main(
        O: T.Tensor((3, *other_shape), dtype),
        S: T.Tensor((3, k, width), dtype),
        C: T.Tensor((3, *out_shape), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            s = T.alloc_shared((k, width), dtype)
            fragment = T.alloc_fragment((k, width), dtype)
            other = T.alloc_fragment(other_shape, dtype)
            c = T.alloc_fragment(out_shape, T.float32)
            if storage is not None:
                T.annotate_layout({s: storage})
            for it in T.Pipelined(3, num_stages=stages):
                if async_src:
                    T.async_copy(S[it, :, :], s)
                else:
                    if fragment_writer:
                        T.copy(S[it, :, :], fragment)
                        T.copy(fragment, s)
                    else:
                        T.copy(S[it, :, :], s)
                T.copy(O[it, :, :], other)
                T.ptx_wait_group(0)
                T.sync_threads()
                T.clear(c)
                if matrix_read:
                    T.copy(s, fragment)
                if role == "B":
                    if matrix_read:
                        gemm(other, fragment, c, False, False, k_pack=k_pack)
                    else:
                        gemm(other, s, c, False, False, k_pack=k_pack)
                else:
                    if matrix_read:
                        gemm(fragment, other, c, True, False, k_pack=k_pack)
                    else:
                        gemm(s, other, c, True, False, k_pack=k_pack)
                T.copy(c, C[it, :, :])

    return main


def has_matrix_read(mod):
    reads = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and node.op.same_as(tvm.ir.Op.get("tl.tileop.ds_read_format")):
            reads.append(node)

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return bool(reads)


def fp8_gather_program(
    dtype="float8_e4m3fn", width=64, role="B", threads=256, stages=0, source_padding=0, leading_unit=False, storage=None
):
    other_shape = (32, 32)
    out_shape = (32, width) if role == "B" else (width, 32)

    @T.prim_func
    def main(
        O: T.Tensor((3, *other_shape), dtype), S: T.Tensor((3, 64, width + source_padding), dtype), C: T.Tensor((3, *out_shape), T.float32)
    ):
        with T.Kernel(1, threads=threads):
            s = T.alloc_shared((1, 32, width) if leading_unit else (32, width), dtype)
            if storage is not None:
                T.annotate_layout({s: storage})
            other = T.alloc_fragment(other_shape, dtype)
            c = T.alloc_fragment(out_shape, T.float32)
            for it in T.Pipelined(3, num_stages=stages):
                if leading_unit:
                    for k, n in T.Parallel(32, width, annotations={"parallel_prefer_async": True}):
                        s[0, k, n] = S[it, k * 2, n + source_padding]
                else:
                    for k, n in T.Parallel(32, width, annotations={"parallel_prefer_async": True}):
                        s[k, n] = S[it, k * 2, n + source_padding]
                T.copy(O[it, :, :], other)
                T.ptx_wait_group(0)
                T.sync_threads()
                T.clear(c)
                if role == "B":
                    if leading_unit:
                        gemm(other, s[0, :, :], c, False, False)
                    else:
                        gemm(other, s, c, False, False)
                else:
                    if leading_unit:
                        gemm(s[0, :, :], other, c, True, False)
                    else:
                        gemm(s, other, c, True, False)
                T.copy(c, C[it, :, :])

    return main


def gather_anbt_plans(mod):
    plans = []

    def visit(node):
        if isinstance(node, tvm.tirx.For) and "tl.hcu_gemm_an_bt_lds_strategy" in node.annotations:
            plans.append(node.annotations["tl.hcu_gemm_an_bt_lds_strategy"])

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return plans


def run_native_case(dtype, width, role, async_src, stages, arch, storage=None, k=32):
    torch.manual_seed(20261028)
    s = torch.randn((k, width)).to(getattr(torch, dtype))
    torch.randn((k, width))  # Preserve the established input sequence.
    other_shape = (32, k) if role == "B" else (k, 32)
    other = torch.randn(other_shape).to(getattr(torch, dtype))
    inputs = [torch.stack([other] * 3), torch.stack([s, (-s.float()).to(s.dtype), s])]
    expected = torch.stack([other.float() @ x.float() if role == "B" else x.float().T @ other.float() for x in inputs[1]])
    kernel = tl.compile(
        native_program(dtype, width, role, async_src, stages, storage=storage, k=k),
        target={"kind": "hcu", "mcpu": arch},
        out_idx=[2],
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_ENABLE_ASYNC_COPY: False, PassConfigKey.TIR_USE_ASYNC_COPY: False},
    )
    source = kernel.get_kernel_source()
    assert ("tl::cp_async_gs<" in source) == async_src
    actual = kernel(*(x.cuda() for x in inputs)).cpu()
    torch.cuda.synchronize()
    tolerance = 0.02 if dtype == "float16" else 0.0
    error = actual - expected
    result = dict(
        dtype=dtype,
        width=width,
        role=role,
        async_src=async_src,
        stages=stages,
        arch=arch,
        matrix_read="ds_read_vector" in source,
        max_abs=error.abs().max().item(),
        rms=error.square().mean().sqrt().item(),
    )
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    return result


def strategy_keys(mod):
    keys = set()

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            keys.update(str(key) for key in node.annotations)

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return keys


@pytest.mark.parametrize("arch,dtype,async_src", list(product(["gfx938", "gfx946"], ["float16", "float8_e4m3fn"], [False, True])))
def test_matrix_anbt_read_plan(arch, dtype, async_src):
    k = 16 if dtype == "float16" else 32
    captures, artifact = lower_with_captures(native_program(dtype=dtype, width=32, k=k, async_src=async_src, stages=2), arch)
    mod = captures["tl.MaterializeHcuGemmLdsStrategy"]
    assert has_matrix_read(mod)
    assert "ds_read_vector" in artifact.kernel_source
    assert ("tl::cp_async_gs<" in artifact.kernel_source) == async_src
    keys = strategy_keys(mod)
    if async_src:
        assert "tl.hcu_gemm_an_bt_lds_strategy" in keys
        assert "tl.hcu_gemm_lds_copy_strategy" in keys
    else:
        assert "tl.hcu_plain_gemm_lds_copy" in keys


@pytest.mark.parametrize(
    "arch,dtype,role,async_src", list(product(["gfx938", "gfx946"], ["float16", "float8_e4m3fn"], ["A", "B"], [False, True]))
)
def test_non_matrix_anbt_uses_generic_layout(arch, dtype, role, async_src):
    k = 16 if dtype == "float16" else 32
    captures, artifact = lower_with_captures(native_program(dtype=dtype, width=16, k=k, role=role, async_src=async_src, stages=2), arch)
    mod = captures["tl.MaterializeHcuGemmLdsStrategy"]
    assert not has_matrix_read(mod)
    assert "ds_read_vector" not in artifact.kernel_source
    assert ("tl::cp_async_gs<" in artifact.kernel_source) == async_src
    keys = strategy_keys(mod)
    assert "tl.hcu_gemm_an_bt_lds_strategy" not in keys
    assert "tl.hcu_gemm_lds_copy_strategy" not in keys
    assert "tl.hcu_plain_gemm_lds_copy" not in keys


@pytest.mark.parametrize("dtype,k", [("float16", 16), ("float8_e4m3fn", 32)])
def test_generic_hcu_swizzle_map_is_unchanged(dtype, k):
    from tilelang.layout import make_hcu_swizzled_layout
    from tilelang.hcu.intrinsics.hcu_mmac_layout import make_mmac_swizzle_layout

    buffer = tvm.tirx.decl_buffer((k, 16), dtype, scope="shared")
    original = make_mmac_swizzle_layout(buffer, vecSize=64 // tvm.DataType(dtype).bits)
    selected = make_hcu_swizzled_layout(buffer, 1)
    tvm.ir.assert_structural_equal(original.get_input_shape(), selected.get_input_shape())
    tvm.ir.assert_structural_equal(original.index, selected.index)
    inverse = selected.inverse()
    analyzer = tvm.arith.Analyzer()
    for row, col in product(range(k), range(16)):
        physical = [int(analyzer.simplify(x)) for x in selected.map_forward_index([row, col])]
        logical = inverse.map_forward_index(physical)
        assert [int(analyzer.simplify(x)) for x in logical] == [row, col]


@pytest.mark.parametrize("arch,dtype", list(product(["gfx938", "gfx946"], ["float16", "float8_e4m3fn"])))
def test_anbt_pack2_uses_generic_layout(arch, dtype):
    k = 32 if dtype == "float16" else 64
    captures, artifact = lower_with_captures(native_program(dtype=dtype, width=32, k=k, k_pack=2, async_src=True, stages=2), arch)
    mod = captures["tl.MaterializeHcuGemmLdsStrategy"]
    assert not has_matrix_read(mod)
    assert "ds_read_vector" not in artifact.kernel_source
    assert "tl::cp_async_gs<" in artifact.kernel_source
    assert not strategy_keys(mod).intersection(
        {"tl.hcu_gemm_an_bt_lds_strategy", "tl.hcu_gemm_lds_copy_strategy", "tl.hcu_plain_gemm_lds_copy"}
    )


@pytest.mark.parametrize("arch,pack", list(product(["gfx938", "gfx946"], [1, 2])))
def test_fp16_mixed_consumers_share_matrix_strategy(arch, pack):
    from test_hcu_gemm_fp8_async_layout import async_program

    captures, artifact = lower_with_captures(async_program(dtype="float16", mixed=True, pack=pack, stages=2), arch)
    mod = captures["tl.MaterializeHcuGemmLdsStrategy"]
    assert has_matrix_read(mod)
    assert "ds_read_vector" in artifact.kernel_source
    assert "tl.hcu_gemm_an_bt_lds_strategy" in strategy_keys(mod)
    assert "tl.hcu_gemm_lds_copy_strategy" in strategy_keys(mod)


@pytest.mark.parametrize("arch,dtype,role", list(product(["gfx938", "gfx946"], ["float16", "bfloat16"], ["A", "B"])))
def test_fp16_async_anbt_wrap_phase_selection(arch, dtype, role):
    from test_hcu_plain_copy_lds_formula import make_layout

    width = 64 if arch == "gfx938" else 128
    captures, _ = lower_with_captures(
        native_program(dtype=dtype, width=width, role=role, k=16, threads=128, async_src=True, stages=2), arch
    )
    plans = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and "tl.hcu_gemm_an_bt_lds_strategy" in node.annotations:
            plans.append(node.annotations["tl.hcu_gemm_an_bt_lds_strategy"])

    mod = captures["tl.MaterializeHcuGemmLdsStrategy"]
    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert plans
    expected = make_layout((16, width), 32 if arch == "gfx938" else 64, None)
    for plan in plans:
        if arch == "gfx938":
            assert int(plan.wrap_offset) == 16
            assert int(plan.wrap_idx_mask) + 1 == 2
        else:
            assert int(plan.wrap_offset) == 0
            assert int(plan.wrap_idx_mask) == 0
            tvm.ir.assert_structural_equal(plan.storage_layout.index, expected.index)
    assert has_matrix_read(mod)


@pytest.mark.parametrize("arch", ["gfx938", "gfx946"])
def test_fp16_async_gather_wrap_phase_selection(arch):
    @T.prim_func
    def main(O: T.Tensor((32, 16), T.float16), S: T.Tensor((16, 128), T.float16), C: T.Tensor((32, 128), T.float32)):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((16, 128), T.float16)
            other = T.alloc_fragment((32, 16), T.float16)
            c = T.alloc_fragment((32, 128), T.float32)
            for k, n in T.Parallel(16, 128, annotations={"parallel_prefer_async": True}):
                s[k, n] = S[k, n]
            T.copy(O, other)
            T.ptx_wait_group(0)
            T.sync_threads()
            T.clear(c)
            gemm(other, s, c)
            T.copy(c, C)

    captures, _ = lower_with_captures(main, arch, enable_async=True)
    plans = []

    def visit(node):
        if isinstance(node, tvm.tirx.For) and "tl.hcu_gemm_an_bt_lds_strategy" in node.annotations:
            plans.append(node.annotations["tl.hcu_gemm_an_bt_lds_strategy"])

    tvm.tirx.stmt_functor.post_order_visit(captures["tl.MaterializeHcuGemmLdsStrategy"]["main"].body, visit)
    assert plans
    expected_count = 2 if arch == "gfx938" else 1
    assert all(int(plan.wrap_idx_mask) + 1 == expected_count for plan in plans)
    assert all(int(plan.wrap_offset) == (16 if arch == "gfx938" else 0) for plan in plans)
    assert has_matrix_read(captures["tl.MaterializeHcuGemmLdsStrategy"])


@pytest.mark.parametrize(
    "arch,dtype,geometry",
    list(
        product(
            ["gfx938", "gfx946"],
            ["float8_e4m3fn", "float8_e5m2"],
            [
                (32, 128, 0, 1, 2),
                (64, 256, 0, 2, 4),
                (64, 256, 4, 2, 4),
                (128, 256, 0, 4, 4),
                (128, 512, 0, 4, 8),
                (256, 256, 0, 4, 4),
                (256, 1024, 0, 4, 16),
            ],
        )
    ),
)
def test_fp8_async_gather_wrap_fallback(arch, dtype, geometry):
    width, threads, padding, phases32, phases64 = geometry
    captures, artifact = lower_with_captures(
        fp8_gather_program(dtype=dtype, width=width, threads=threads, source_padding=padding, leading_unit=True), arch, enable_async=True
    )
    mod = captures["tl.MaterializeHcuGemmLdsStrategy"]
    plans = gather_anbt_plans(mod)
    assert plans
    for plan in plans:
        phases = phases32 if arch == "gfx938" else phases64
        assert int(plan.wrap_idx_mask) == phases - 1
        expected_offset = 0 if phases == 1 else 8 if arch == "gfx938" else 4
        assert int(plan.wrap_offset) == expected_offset
        if padding:
            assert int(plan.copy_transaction_bytes) == 4
    assert has_matrix_read(mod)
    assert "ds_read_vector" in artifact.kernel_source
    assert "tl::cp_async_gs<" in artifact.kernel_source


@pytest.mark.parametrize("arch,threads", list(product(["gfx938", "gfx946"], [64, 128, 256, 512])))
def test_fp8_wrap_fallback_requires_legal_wave_partition(arch, threads):
    from tilelang.hcu.transform import AnnotateMlsGemmDep

    target = tvm.target.Target({"kind": "hcu", "mcpu": arch})
    program = fp8_gather_program(threads=threads, leading_unit=True).with_attr("target", target)
    with target:
        mod = tl.transform.MaterializeKernelLaunch(lower_grid_binding=True, lower_thread_binding=True)(tvm.IRModule({"main": program}))
        mod = AnnotateMlsGemmDep()(mod)
    assert bool(gather_anbt_plans(mod)) == (threads in (128, 256))


@pytest.mark.parametrize("arch,dtype", list(product(["gfx938", "gfx946"], ["float8_e4m3fn", "float8_e5m2"])))
def test_fp8_async_anbt_keeps_full_wrap(arch, dtype):
    captures, _ = lower_with_captures(native_program(dtype=dtype, width=64, k=32, threads=256, async_src=True, stages=2), arch)
    plans = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and "tl.hcu_gemm_an_bt_lds_strategy" in node.annotations:
            plans.append(node.annotations["tl.hcu_gemm_an_bt_lds_strategy"])

    tvm.tirx.stmt_functor.post_order_visit(captures["tl.MaterializeHcuGemmLdsStrategy"]["main"].body, visit)
    assert plans
    expected_count, expected_offset = (2, 8) if arch == "gfx938" else (4, 4)
    assert all(int(plan.wrap_idx_mask) + 1 == expected_count for plan in plans)
    assert all(int(plan.wrap_offset) == expected_offset for plan in plans)


@pytest.mark.parametrize("arch,dtype", list(product(["gfx938", "gfx946"], ["float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"])))
def test_copy_and_gather_use_same_phase_policy(arch, dtype):
    selected = []
    bits = tvm.DataType(dtype).bits
    expected_count = 2 if arch == "gfx938" else 4 if bits == 8 else 1
    expected_offset = 0 if expected_count == 1 else (8 if arch == "gfx938" else 4) if bits == 8 else 16
    for program in [
        native_program(dtype=dtype, width=64, k=32, threads=256, async_src=True, stages=2),
        fp8_gather_program(dtype=dtype, width=64, threads=256, stages=2),
    ]:
        captures, _ = lower_with_captures(program, arch, enable_async=True)
        mod = captures["tl.MaterializeHcuGemmLdsStrategy"]
        plans = []

        def visit(node, plans=plans):
            if isinstance(node, (tvm.tirx.For, tvm.tirx.Call)) and "tl.hcu_gemm_an_bt_lds_strategy" in node.annotations:
                plans.append(node.annotations["tl.hcu_gemm_an_bt_lds_strategy"])

        tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
        assert plans and has_matrix_read(mod)
        assert all(int(plan.wrap_idx_mask) + 1 == expected_count for plan in plans)
        assert all(int(plan.wrap_offset) == expected_offset for plan in plans)
        selected.append(plans[0])
    tvm.ir.assert_structural_equal(selected[0].storage_layout, selected[1].storage_layout)
    tvm.ir.assert_structural_equal(selected[0].copy_loop_layout, selected[1].copy_loop_layout)


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize(
    "arch,threads,leading_unit,expected_count",
    [("gfx946", 64, False, 1), ("gfx946", 256, False, 1), ("gfx946", 256, True, 4), ("gfx946", 512, False, 8), ("gfx938", 256, False, 4)],
)
def test_partial_wrap_prefers_legal_common(dtype, arch, threads, leading_unit, expected_count):
    from test_hcu_plain_copy_lds_formula_b8 import make_layout

    captures, artifact = lower_with_captures(
        fp8_gather_program(dtype=dtype, width=128, threads=threads, leading_unit=leading_unit), arch, enable_async=True
    )
    mod = captures["tl.MaterializeHcuGemmLdsStrategy"]
    plans = gather_anbt_plans(mod)
    assert plans and has_matrix_read(mod)
    for plan in plans:
        assert int(plan.wrap_idx_mask) + 1 == expected_count
        if expected_count == 1:
            assert int(plan.wrap_offset) == 0
            tvm.ir.assert_structural_equal(plan.storage_layout.index, make_layout((32, 128), 64, ()).index)
    assert "ds_read_vector" in artifact.kernel_source
    assert "cp_async_gs" in artifact.kernel_source


def test_partial_wrap_respects_explicit_storage():
    from test_hcu_plain_copy_lds_formula_b8 import make_layout

    captures, _ = lower_with_captures(fp8_gather_program(width=128, leading_unit=True), "gfx946", enable_async=True)
    partial = gather_anbt_plans(captures["tl.MaterializeHcuGemmLdsStrategy"])[0].storage_layout
    partial_2d = tl.layout.Layout((32, 128), lambda k, n: partial.map_forward_index([0, k, n])[1:])
    for storage, expected_count in [(partial_2d, 4), (make_layout((32, 128), 64, ()), 1)]:
        captures, _ = lower_with_captures(fp8_gather_program(width=128, storage=storage), "gfx946", enable_async=True)
        plans = gather_anbt_plans(captures["tl.MaterializeHcuGemmLdsStrategy"])
        assert plans and all(int(plan.wrap_idx_mask) + 1 == expected_count for plan in plans)
        for plan in plans:
            tvm.ir.assert_structural_equal(plan.storage_layout.index, storage.index)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("arch,width,threads", [("gfx938", 64, 128), ("gfx946", 64, 128), ("gfx946", 128, 128), ("gfx946", 128, 512)])
def test_pure_anbt_unified_wrap_phase_policy(dtype, arch, width, threads):
    bits = tvm.DataType(dtype).bits
    captures, artifact = lower_with_captures(native_program(dtype=dtype, width=width, async_src=True, threads=threads), arch)
    mod = captures["tl.MaterializeHcuGemmLdsStrategy"]
    plans = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and "tl.hcu_gemm_an_bt_lds_strategy" in node.annotations:
            plans.append(node.annotations["tl.hcu_gemm_an_bt_lds_strategy"])

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    expected_count = 2 if arch == "gfx938" else 8 if bits == 8 and threads == 512 else 1
    assert plans and has_matrix_read(mod)
    for plan in plans:
        assert int(plan.wrap_idx_mask) + 1 == expected_count
        if expected_count == 1:
            expected = tvm.get_global_func("tl.hcu.MakePlainGemmLdsLayout")(
                (32, width), 64, 4, bits, 256 // bits, 32, 128 // bits, 16, [1], [1]
            )
            assert int(plan.wrap_offset) == 0
            tvm.ir.assert_structural_equal(plan.storage_layout.index, expected.index)
    assert "ds_read_vector" in artifact.kernel_source and "cp_async_gs" in artifact.kernel_source


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_pure_anbt_phase_policy_preserves_explicit_linear(dtype):
    identity = tl.layout.Layout((32, 64), lambda k, n: (k, n))
    captures, _ = lower_with_captures(native_program(dtype=dtype, width=64, async_src=True, threads=128, storage=identity), "gfx946")
    plans = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and "tl.hcu_gemm_an_bt_lds_strategy" in node.annotations:
            plans.append(node.annotations["tl.hcu_gemm_an_bt_lds_strategy"])

    tvm.tirx.stmt_functor.post_order_visit(captures["tl.MaterializeHcuGemmLdsStrategy"]["main"].body, visit)
    assert plans
    for plan in plans:
        assert int(plan.wrap_offset) == 0 and int(plan.wrap_idx_mask) == 0
        coordinates = identity.get_forward_vars()
        actual = plan.storage_layout.map_forward_index(coordinates)
        analyzer = tvm.arith.Analyzer()
        assert all(analyzer.can_prove_equal(mapped, logical) for mapped, logical in zip(actual, coordinates))


def test_partial_wrap_invalid_explicit_storage_is_rejected():
    identity = tl.layout.Layout((32, 128), lambda k, n: (k, n))
    with pytest.raises(tvm.error.InternalError, match="storage layout mismatch"):
        lower_with_captures(fp8_gather_program(width=128, storage=identity), "gfx946", enable_async=True)


@pytest.mark.parametrize("arch,dtype,pack", list(product(["gfx938", "gfx946"], ["float16", "float8_e4m3fn", "float8_e5m2"], [1, 2])))
def test_async_mixed_prefers_common_storage(arch, dtype, pack):
    from test_hcu_gemm_fp8_async_layout import async_program

    captures, artifact = lower_with_captures(async_program(dtype=dtype, mixed=True, pack=pack, width=128, threads=256, stages=2), arch)
    mod = captures["tl.MaterializeHcuGemmLdsStrategy"]
    plans = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and "tl.hcu_gemm_an_bt_lds_strategy" in node.annotations:
            plans.append(node.annotations["tl.hcu_gemm_an_bt_lds_strategy"])

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert plans and has_matrix_read(mod)
    bits = 16 if dtype == "float16" else 8
    banks = 32 if arch == "gfx938" else 64
    expected = tvm.get_global_func("tl.hcu.MakePlainGemmLdsLayout")(
        [32, 128], banks, 4, bits, 256 // bits, 32, 128 // bits, 16, [1, 0], [1, pack]
    )
    for plan in plans:
        assert int(plan.wrap_offset) == int(plan.wrap_idx_mask) == 0
        tvm.ir.assert_structural_equal(plan.storage_layout.index, expected.index)
    assert "ds_read_vector" in artifact.kernel_source
    assert "cp_async_gs" in artifact.kernel_source


@pytest.mark.parametrize("arch,dtype", list(product(["gfx938", "gfx946"], ["float16", "float8_e4m3fn"])))
def test_async_mixed_respects_explicit_native_storage(arch, dtype):
    from test_hcu_gemm_fp8_async_layout import async_program

    captures, _ = lower_with_captures(fp8_gather_program(dtype=dtype, width=128, leading_unit=True), arch, enable_async=True)
    native = gather_anbt_plans(captures["tl.MaterializeHcuGemmLdsStrategy"])[0]
    storage = tl.layout.Layout((32, 128), lambda k, n: native.storage_layout.map_forward_index([0, k, n])[1:])
    captures, _ = lower_with_captures(async_program(dtype=dtype, mixed=True, width=128, threads=256, storage=storage), arch)
    plans = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and "tl.hcu_gemm_an_bt_lds_strategy" in node.annotations:
            plans.append(node.annotations["tl.hcu_gemm_an_bt_lds_strategy"])

    tvm.tirx.stmt_functor.post_order_visit(captures["tl.MaterializeHcuGemmLdsStrategy"]["main"].body, visit)
    assert plans
    for plan in plans:
        assert int(plan.wrap_offset) == int(native.wrap_offset) > 0
        assert int(plan.wrap_idx_mask) == int(native.wrap_idx_mask)
        tvm.ir.assert_structural_equal(plan.storage_layout.index, storage.index)


@pytest.mark.parametrize("arch,dtype", list(product(["gfx938", "gfx946"], ["float16", "float8_e4m3fn"])))
def test_async_mixed_invalid_storage_is_rejected(arch, dtype):
    from test_hcu_gemm_fp8_async_layout import async_program

    identity = tl.layout.Layout((32, 128), lambda k, n: (k, n))
    with pytest.raises(tvm.error.InternalError, match="storage layout mismatch"):
        lower_with_captures(async_program(dtype=dtype, mixed=True, width=128, threads=256, storage=identity), arch)
