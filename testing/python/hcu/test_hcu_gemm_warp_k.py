# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Matrix reads preserve the GEMM K partition and native fragment."""

import importlib

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


@pytest.mark.parametrize("warp_k", [2, 4])
@pytest.mark.parametrize("kind", ["an", "bt"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"])
def test_matrix_read_warp_k(warp_k, kind, dtype, monkeypatch, n=32, manual_writer=False):
    import torch

    k = (32 if "float8" in dtype else 16) * warp_k * 2
    m = 32
    trans_a = kind == "an"
    trans_b = kind == "an"
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    @T.prim_func
    def main(A: T.Tensor(a_shape, dtype), B: T.Tensor(b_shape, dtype), C: T.Tensor((m, n), "float32")):
        with T.Kernel(1, threads=64 * (n // 32) * warp_k):
            a = T.alloc_shared(a_shape, dtype)
            b = T.alloc_shared(b_shape, dtype)
            af = T.alloc_fragment(a_shape, dtype)
            bf = T.alloc_fragment(b_shape, dtype)
            acc = T.alloc_fragment((m, n), "float32")
            T.copy(A, a)
            if manual_writer:
                for i, j in T.Parallel(k, n):
                    b[i, j] = B[i, j]
            else:
                T.copy(B, b)
            T.copy(a, af)
            T.copy(b, bf)
            T.clear(acc)
            gemm(
                af,
                bf,
                acc,
                transpose_A=trans_a,
                transpose_B=trans_b,
                k_pack=1,
                policy=T.GemmWarpPolicy.FullColK,
                annotations={"tl.hcu.scale_min_n_per_warp": 32},
            )
            T.reduce_sum_warp(acc, acc, clear=False)
            T.copy(acc, C)

    partitions = []

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name != "tl.MaterializeHcuGemmLdsStrategy":
                return

            def visit(node):
                if isinstance(node, tvm.tirx.Call) and "tl.mls_gemm_dep" in node.annotations:
                    meta = node.annotations["tl.mls_gemm_dep"]
                    if meta.plain_warp_m > 0:
                        partitions.append((int(meta.plain_warp_m), int(meta.plain_warp_n), int(meta.plain_warp_k)))

            for fn in mod.functions.values():
                tvm.tirx.stmt_functor.post_order_visit(fn.body, visit)

    module = importlib.import_module("tilelang.jit.kernel")
    original = module.create_pass_instruments
    monkeypatch.setattr(module, "create_pass_instruments", lambda **kw: [*original(**kw), Capture()])
    kernel = tl.compile(
        main,
        target={"kind": "hcu", "mcpu": current_hcu_arch_string()},
        out_idx=[2],
        execution_backend="cython",
        pass_configs={tl.PassConfigKey.TL_ENABLE_ASYNC_COPY: False, tl.PassConfigKey.TIR_USE_ASYNC_COPY: False},
    )
    if manual_writer:
        assert not partitions
        assert "ds_read_vector" not in kernel.get_kernel_source()
    else:
        assert partitions and all(p == (1, n // 32, warp_k) for p in partitions)
        assert "ds_read_vector" in kernel.get_kernel_source()
    rng = torch.Generator().manual_seed(0)
    a = torch.randint(-2, 3, a_shape, generator=rng).to(getattr(torch, dtype))
    b = torch.randint(-2, 3, b_shape, generator=rng).to(getattr(torch, dtype))
    expected = (a.float().T if trans_a else a.float()) @ (b.float().T if trans_b else b.float())
    actual = kernel(a.cuda(), b.cuda()).cpu()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["an", "bt"])
@pytest.mark.parametrize("dtype", ["float16", "float8_e4m3fn"])
def test_matrix_read_warp_k_with_mn_partition(kind, dtype, monkeypatch):
    test_matrix_read_warp_k(2, kind, dtype, monkeypatch, n=64)


@pytest.mark.parametrize("dtype", ["float16", "float8_e4m3fn"])
@pytest.mark.parametrize("ordinary_pack", [1, 2])
def test_common_mixed_warp_k(dtype, ordinary_pack, monkeypatch):
    import torch

    @T.prim_func
    def main(
        S: T.Tensor((64, 64), dtype), P: T.Tensor((32, 64), dtype), C: T.Tensor((32, 64), "float32"), D: T.Tensor((32, 64), "float32")
    ):
        with T.Kernel(1, threads=256):
            shared = T.alloc_shared((64, 64), dtype)
            peer = T.alloc_shared((32, 64), dtype)
            matrix_peer = T.alloc_shared((32, 64), dtype)
            ordinary = T.alloc_fragment((32, 64), "float32")
            matrix = T.alloc_fragment((32, 64), "float32")
            T.copy(S, shared)
            T.copy(P, peer)
            T.copy(P, matrix_peer)
            T.clear(ordinary)
            T.clear(matrix)
            gemm(peer, shared, ordinary, transpose_B=True, k_pack=ordinary_pack, policy=T.GemmWarpPolicy.FullCol)
            gemm(matrix_peer, shared, matrix, k_pack=1, policy=T.GemmWarpPolicy.FullColK, annotations={"tl.hcu.scale_min_n_per_warp": 32})
            T.reduce_sum_warp(matrix, matrix, clear=False)
            T.copy(ordinary, C)
            T.copy(matrix, D)

    module = importlib.import_module("tilelang.hcu.op.gemm.gemm_hcu_mmac")
    original = module._compute_hcu_warp_partition
    partitions = []

    def capture(gemm, threads, target, meta):
        value = original(gemm, threads, target, meta)
        partitions.append((bool(gemm.trans_B), tuple(value[:3])))
        return value

    monkeypatch.setattr(module, "_compute_hcu_warp_partition", capture)
    kernel = tl.compile(
        main,
        target={"kind": "hcu", "mcpu": current_hcu_arch_string()},
        out_idx=[2, 3],
        execution_backend="cython",
        pass_configs={tl.PassConfigKey.TL_ENABLE_ASYNC_COPY: False, tl.PassConfigKey.TIR_USE_ASYNC_COPY: False},
    )
    assert (True, (1, 4, 1)) in partitions
    assert (False, (1, 2, 2)) in partitions
    assert "ds_read_vector" in kernel.get_kernel_source()
    rng = torch.Generator().manual_seed(0)
    s = torch.randint(-2, 3, (64, 64), generator=rng).to(getattr(torch, dtype))
    p = torch.randint(-2, 3, (32, 64), generator=rng).to(getattr(torch, dtype))
    ordinary, matrix = kernel(s.cuda(), p.cuda())
    torch.testing.assert_close(ordinary.cpu(), p.float() @ s.float().T, atol=0, rtol=0)
    torch.testing.assert_close(matrix.cpu(), p.float() @ s.float(), atol=0, rtol=0)


def test_warp_k_without_storage_map_keeps_ordinary_read(monkeypatch):
    test_matrix_read_warp_k(2, "bt", "float16", monkeypatch, manual_writer=True)
