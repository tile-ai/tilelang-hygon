"""Explicit ordinary-LDS Alt2 reads infer both operands and their accumulator."""

import pytest
import torch
import tilelang as tl
import tilelang.hcu.language as T
from tilelang import tvm
import tilelang.testing
from hcu_test_utils import current_hcu_arch_string
from tilelang.hcu.intrinsics.hcu_mmac_layout import make_gemm_fragment_hcu


@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_dual_alt2_accumulator_mapping():
    fragment = make_gemm_fragment_hcu(128, 128, 4, 1, 1, 32, 16, lit=True, a_interleave=2, b_interleave=2)
    analyzer = tvm.arith.Analyzer()
    # Natural coordinates -> independently derived physical state registers.
    for v in range(128):
        for k in (0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127):
            coords = [T.int32(v), T.int32(k)]
            thread = int(analyzer.simplify(fragment.map_forward_thread(coords)[0]))
            slot = int(analyzer.simplify(fragment.map_forward_index(coords)[0]))
            assert thread == (v // 32) * 64 + ((k % 32) // 8) * 16 + (v % 32) // 2
            assert slot == (v % 2) * 32 + (k // 32) * 8 + (k % 2) * 4 + (k % 8) // 2


@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_region_alt2_frontend_contract():
    s = tvm.tirx.decl_buffer((32, 128), "bfloat16", scope="shared")
    f = tvm.tirx.decl_buffer((32, 128), "bfloat16", scope="local.fragment")
    call = T.ds_read_vector(s, f, alt=2)
    assert int(call.args[2]) == 2
    assert int(call.annotations["tl.hcu_linear_ds_read"]) == 1
    assert "tl.hcu_layout_ds_read" not in call.annotations
    with pytest.raises(ValueError, match="alt must be"):
        T.ds_read_vector(s, f, alt=4)
    with pytest.raises(ValueError, match="alt must be"):
        T.ds_read_vector(s, f, alt=True)
    with pytest.raises(ValueError, match="fragment"):
        T.ds_read_vector(s, s, alt=2)
    half = tvm.tirx.decl_buffer((32, 128), "float16", scope="shared")
    with pytest.raises(ValueError, match="bfloat16"):
        T.ds_read_vector(half, f, alt=2)


@pytest.mark.parametrize("alt", [1, 2])
@pytest.mark.parametrize("rows,width", [(32, 64), (32, 128), (48, 128)])
@pytest.mark.parametrize("source_layout", ["explicit", "inferred", "mixed"])
@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_dual_alt2_gemm_consumer_thread_domain(rows, width, source_layout, alt):
    @T.prim_func
    def main(A: T.Tensor((rows, 128), "bfloat16"), B: T.Tensor((rows, width), "bfloat16"), C: T.Tensor((128, width), "float32")):
        with T.Kernel(1, threads=512):
            a = T.alloc_shared((rows, 128), "bfloat16")
            b = T.alloc_shared((rows, width), "bfloat16")
            if source_layout == "explicit":
                T.annotate_layout({a: T.Layout((rows, 128), lambda i, j: (i, j)), b: T.Layout((rows, width), lambda i, j: (i, j))})
            elif source_layout == "mixed":
                T.annotate_layout({a: T.Layout((rows, 128), lambda i, j: (i, (j // 8 ^ (i % 8)) * 8 + j % 8))})
            tx = T.get_thread_binding()
            if tx < 256:
                T.copy(A, a)
                T.copy(B, b)
            T.sync_threads()
            if tx >= 256:
                af = T.alloc_fragment((rows, 128), "bfloat16")
                bf = T.alloc_fragment((rows, width), "bfloat16")
                cf = T.alloc_fragment((128, width), "float32")
                T.ds_read_vector(a, af, alt=alt)
                T.ds_read_vector(b, bf, alt=alt)
                T.s_waitcnt(0, "lgkmcnt")
                T.gemm(af, bf, cf, transpose_A=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True, k_pack=1)
                T.copy(cf, C)

    kernel = tl.compile(main, out_idx=[2], target={"kind": "hcu", "mcpu": "gfx938"})
    torch.manual_seed(777)
    a = torch.randint(-2, 3, (rows, 128), device="cuda").to(torch.bfloat16)
    b = torch.randint(-2, 3, (rows, width), device="cuda").to(torch.bfloat16)
    torch.testing.assert_close(kernel(a, b), a.float().T @ b.float(), atol=0, rtol=0)
    assert ("__builtin_hcu_ds_read_m32x16_bf16_alt(" in kernel.get_kernel_source()) == (alt == 2)


@pytest.mark.parametrize("entry", ["copy", "region"])
@pytest.mark.parametrize("storage", ["explicit", "inferred", "row_major"])
@pytest.mark.parametrize("async_producer", [False, True])
@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_layout_flag_follows_materialized_storage(entry, storage, async_producer):
    # Width 96 cannot use the automatic storage planners, but is legal for
    # the original b16 B/WarpK=1 row-major read path.
    width = 96 if storage == "row_major" else 128
    # Match the existing async producer's storage contract when one is
    # selected; an arbitrary identity annotation is not valid for that plan.
    source_layout = tvm.get_global_func("tl.hcu.MakePlainGemmLdsLayout")(
        (16, 128),
        32,
        4,
        16,
        16,
        32,
        [8],
        [1],
    )

    @T.prim_func
    def main(A: T.Tensor((32, 16), "bfloat16"), B: T.Tensor((16, width), "bfloat16"), C: T.Tensor((32, width), "float32")):
        with T.Kernel(1, threads=64):
            b = T.alloc_shared((16, width), "bfloat16")
            af = T.alloc_fragment((32, 16), "bfloat16")
            bf = T.alloc_fragment((16, width), "bfloat16")
            c = T.alloc_fragment((32, width), "float32")
            if storage == "explicit":
                T.annotate_layout({b: source_layout})
            T.copy(A, af)
            if async_producer:
                T.async_copy(B, b)
                T.ptx_wait_group(0)
                T.s_waitcnt(0, "vmcnt")
                T.s_waitcnt(0, "lgkmcnt")
            else:
                T.copy(B, b)
            T.sync_threads()
            if entry == "copy":
                T.copy(b, bf, prefer_instruction="ds_read_vector")
            else:
                T.ds_read_vector(b, bf)
            T.s_waitcnt(0, "lgkmcnt")
            T.gemm(af, bf, c, clear_accum=True, k_pack=1)
            T.copy(c, C)

    captured = []

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.MaterializeHcuGemmLdsStrategy":
                captured.append(mod)

    target = tvm.target.Target({"kind": "hcu", "mcpu": "gfx938"})
    with target, tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tl.lower(main, target=target, enable_device_compile=False)
    assert len(captured) == 1
    layouts, reads = {}, []

    def visit(node):
        if isinstance(node, tvm.tirx.SBlock) and "layout_map" in node.annotations:
            layouts.update(dict(node.annotations["layout_map"]))
        if isinstance(node, tvm.tirx.Call) and node.op.name == "tl.tileop.ds_read_format":
            reads.append(node)

    for func in captured[0].functions.values():
        tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert len(reads) == 1
    read = reads[0]
    source = read.args[0].args[0].buffer
    has_layout = source.data in layouts
    assert has_layout == (storage != "row_major")
    assert bool(read.annotations.get("tl.hcu_layout_ds_read", 0)) == has_layout
    assert ("ds_read_format_tensor_b_linear<" in artifact.kernel_source) == (not has_layout)
