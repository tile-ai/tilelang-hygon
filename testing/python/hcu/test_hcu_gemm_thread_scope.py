# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Each GEMM uses its own kernel and consumer thread domain."""

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


@pytest.mark.parametrize("consumer_group", ["full", "lower", "upper"])
@pytest.mark.parametrize("gemm_threads,copy_threads", [(256, 128), (128, 256)])
@pytest.mark.parametrize("dtype", ["bfloat16", "float8_e4m3fn"])
def test_native_read_uses_own_kernel_threads(gemm_threads, copy_threads, dtype, consumer_group, monkeypatch):
    import torch

    k = 32
    block_threads = gemm_threads if consumer_group == "full" else 2 * gemm_threads
    consumer_start = gemm_threads if consumer_group == "upper" else 0

    @T.prim_func
    def main(A: T.Tensor((16, k), dtype), B: T.Tensor((k, 128), dtype), C: T.Tensor((16, 128), "float32")):
        with T.Kernel(1, threads=block_threads):
            a = T.alloc_shared((16, k), dtype)
            b = T.alloc_shared((k, 128), dtype)
            b_frag = T.alloc_fragment((k, 128), dtype)
            acc = T.alloc_fragment((16, 128), "float32")
            T.copy(A, a)
            T.copy(B, b)
            # The producer uses the full block; the GEMM has its own domain.
            tx = T.get_thread_binding()
            if tx >= consumer_start and tx < consumer_start + gemm_threads:
                T.copy(b, b_frag)
                T.clear(acc)
                gemm(a, b_frag, acc, k_pack=1)
                T.copy(acc, C)
        with T.Kernel(1, threads=copy_threads):
            for i, j in T.Parallel(16, 128):
                C[i, j] = C[i, j] + 1.0

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
    assert partitions
    assert all(wm * wn * wk == gemm_threads // 64 for wm, wn, wk in partitions)
    assert "ds_read_vector" in kernel.get_kernel_source()
    rng = torch.Generator().manual_seed(0)
    a = torch.randint(-2, 3, (16, k), generator=rng).to(getattr(torch, dtype))
    b = torch.randint(-2, 3, (k, 128), generator=rng).to(getattr(torch, dtype))
    expected = a.float() @ b.float() + 1
    actual = kernel(a.cuda(), b.cuda()).cpu()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
