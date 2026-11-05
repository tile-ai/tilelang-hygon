# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""MN slice consumers share one layout on the complete LDS allocation."""

import importlib
from itertools import product

import pytest
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.hcu.language import gemm
from hcu_test_utils import current_hcu_arch_string

pytestmark = [
    *tilelang.testing.requires_rocm.marks(),
    pytest.mark.skipif(current_hcu_arch_string() not in {"gfx938", "gfx946"}, reason="requires HCU LDS strategies"),
]


def slice_program(kind="atbn", dtype="float16", producer="async", invalid=None):
    bits = tvm.DataType(dtype).bits
    k = 64 if bits == 8 else 32
    mn = 128
    pack = 2 if kind == "atbn" else 1
    full_mn = 256
    full_k = k
    if invalid == "k_slice":
        full_mn, full_k = mn, 2 * k
    shape = (full_mn, full_k) if kind == "atbn" else (full_k, full_mn)
    view = (mn, k) if kind == "atbn" else (k, mn)
    r0 = c0 = 0
    r1, c1 = (mn, 0) if kind == "atbn" else (0, mn)
    if invalid == "k_slice":
        r1, c1 = (0, k) if kind == "atbn" else (k, 0)
    elif invalid == "offset":
        r1, c1 = (64, 0) if kind == "atbn" else (0, 16)
    if invalid == "partial_writer":
        r1 = c1 = 0
    if invalid == "rewrite_offset":
        r0 = 64
    rewrite = invalid in {"rewrite", "rewrite_offset"}
    second_annotations = {"review_tag": 1} if invalid == "metadata" else {}
    other_pack = 1 if invalid == "read_mode" else pack

    @T.prim_func
    def main(S: T.Tensor(shape, dtype), B: T.Tensor((k, 64), dtype), C: T.Tensor((2 * mn, 64), "float32")):
        with T.Kernel(1, threads=512):
            shared = T.alloc_shared(shape, dtype)
            first = T.alloc_fragment(view, dtype)
            second = T.alloc_fragment(view, dtype)
            peer = T.alloc_fragment((k, 64), dtype)
            other_peer = T.alloc_fragment((k, 64), dtype)
            c_first = T.alloc_fragment((mn, 64), "float32")
            c_second = T.alloc_fragment((mn, 64), "float32")
            if producer == "async":
                if invalid == "partial_writer":
                    T.async_copy(S[0 : view[0], 0 : view[1]], shared[0 : view[0], 0 : view[1]])
                else:
                    T.async_copy(S, shared)
                T.ptx_wait_group(0)
                T.sync_threads()
            else:
                T.copy(S, shared)
            T.copy(B, peer)
            T.copy(shared[r0 : r0 + view[0], c0 : c0 + view[1]], first)
            if not rewrite:
                T.copy(shared[r1 : r1 + view[0], c1 : c1 + view[1]], second)
            T.clear(c_first)
            T.clear(c_second)
            gemm(first, peer, c_first, kind == "anbt", False, k_pack=pack)
            if rewrite:
                T.sync_threads()
                T.async_copy(S, shared)
                T.ptx_wait_group(0)
                T.sync_threads()
                T.copy(shared[r1 : r1 + view[0], c1 : c1 + view[1]], second)
            if invalid == "read_mode":
                T.copy(B, other_peer)
                gemm(second, other_peer, c_second, kind == "anbt", False, k_pack=other_pack)
            else:
                gemm(second, peer, c_second, kind == "anbt", False, k_pack=other_pack, annotations=second_annotations)
            T.copy(c_first, C[0:mn, :])
            T.copy(c_second, C[mn : 2 * mn, :])

    return main, shape, view, (r0, c0, r1, c1)


def compile_slice(kind="atbn", dtype="float16", producer="async", invalid=None):
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in {"tl.MaterializeHcuGemmLdsStrategy", "tl.LayoutInference"}:
                snapshots[info.name] = mod

    module = importlib.import_module("tilelang.jit.kernel")
    original = module.create_pass_instruments
    module.create_pass_instruments = lambda **kw: [*original(**kw), Capture()]
    try:
        fn, shape, view, origins = slice_program(kind, dtype, producer, invalid)
        configs = {} if producer == "async" else {tl.PassConfigKey.TL_ENABLE_ASYNC_COPY: False, tl.PassConfigKey.TIR_USE_ASYNC_COPY: False}
        kernel = tl.compile(
            fn, target={"kind": "hcu", "mcpu": current_hcu_arch_string()}, out_idx=[2], execution_backend="cython", pass_configs=configs
        )
    finally:
        module.create_pass_instruments = original
    return kernel, snapshots, shape, view, origins


def run_slice(kind="atbn", dtype="float16", producer="async", invalid=None):
    import torch

    kernel, snapshots, shape, view, origins = compile_slice(kind, dtype, producer, invalid)
    k = view[1] if kind == "atbn" else view[0]
    rng = torch.Generator().manual_seed(20261105)
    data = (torch.randint(-8, 9, shape, generator=rng).float() / 4).to(getattr(torch, dtype))
    peer = (torch.randint(-8, 9, (k, 64), generator=rng).float() / 4).to(getattr(torch, dtype))
    r0, c0, r1, c1 = origins
    pieces = [data[r : r + view[0], c : c + view[1]].float() for r, c in [(r0, c0), (r1, c1)]]
    if kind == "anbt":
        pieces = [x.T for x in pieces]
    expected = torch.cat([x @ peer.float() for x in pieces])
    exact = torch.cat([x.double() @ peer.double() for x in pieces])
    assert torch.equal(expected.double(), exact)
    actual = kernel(data.cuda(), peer.cuda()).cpu()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    return kernel, snapshots, shape


@pytest.mark.parametrize(
    "kind,dtype,producer",
    list(product(["atbn", "anbt"], ["float16", "float8_e4m3fn"], ["plain", "async"]))
    + list(product(["atbn", "anbt"], ["bfloat16", "float8_e5m2"], ["async"])),
)
def test_mn_slice_numerical(kind, dtype, producer):
    kernel, snapshots, shape = run_slice(kind, dtype, producer)
    materialized = str(snapshots["tl.MaterializeHcuGemmLdsStrategy"])
    key = "tl.hcu_gemm_at_bn_lds_strategy" if kind == "atbn" else "tl.hcu_gemm_an_bt_lds_strategy"
    # Pure ordinary plain copies retain their normal inference path.
    if producer == "async" or kind == "anbt":
        assert key in materialized or (kind == "anbt" and "tl.hcu_plain_gemm_lds_copy" in materialized)
    if kind == "anbt":
        assert "ds_read_vector" in kernel.get_kernel_source()
    layouts = []

    def visit(node):
        if isinstance(node, tvm.tirx.SBlock) and "layout_map" in node.annotations:
            for buffer, layout in node.annotations["layout_map"].items():
                if buffer.name == "shared":
                    layouts.append(tuple(int(x) for x in layout.get_input_shape()))

    for fn in snapshots["tl.LayoutInference"].functions.values():
        tvm.tirx.stmt_functor.post_order_visit(fn.body, visit)
    if producer == "async" or kind == "anbt":
        assert layouts
    assert all(x == shape for x in layouts)


@pytest.mark.parametrize(
    "kind,invalid",
    [("atbn", "k_slice"), ("anbt", "k_slice"), ("atbn", "offset"), ("anbt", "offset"), ("atbn", "read_mode"), ("atbn", "partial_writer")],
)
def test_unsupported_slice_keeps_fallback(kind, invalid):
    _, snapshots, _ = run_slice(kind, "float16", "async", invalid)
    text = str(snapshots["tl.MaterializeHcuGemmLdsStrategy"])
    assert "tl.hcu_gemm_at_bn_lds_strategy" not in text
    assert "tl.hcu_gemm_an_bt_lds_strategy" not in text


@pytest.mark.parametrize("kind,variant", [("atbn", "metadata"), ("anbt", "metadata"), ("atbn", "rewrite"), ("atbn", "rewrite_offset")])
def test_mn_slice_query_consistency(kind, variant):
    _, snapshots, _ = run_slice(kind, "float16", "async", variant)
    text = str(snapshots["tl.MaterializeHcuGemmLdsStrategy"])
    if variant == "rewrite_offset":
        assert "tl.hcu_gemm_at_bn_lds_strategy" not in text
        assert "tl.hcu_gemm_an_bt_lds_strategy" not in text
    else:
        key = "tl.hcu_gemm_at_bn_lds_strategy" if kind == "atbn" else "tl.hcu_gemm_an_bt_lds_strategy"
        assert key in text
