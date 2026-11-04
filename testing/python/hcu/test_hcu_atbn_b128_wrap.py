# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""AT/BN b128 wrap against the complete 32/64-bank read phases."""

from collections import Counter
from itertools import product

import numpy as np
import pytest
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.hcu.language import gemm
from hcu_test_utils import current_hcu_arch_string
from plain_copy_lds_codegen import evaluate
from test_hcu_plain_copy_lds_formula_b8 import B128_PHASES_32, B128_READ_PHASES_64
from test_hcu_plain_copy_lds_swizzle import lower_with_captures

pytestmark = [
    *tilelang.testing.requires_rocm.marks(),
    pytest.mark.skipif(current_hcu_arch_string() not in {"gfx938", "gfx946"}, reason="requires HCU"),
]


def wrap_program(dtype="float16", width=128, threads=64, rows=32, pack=2):
    @T.prim_func
    def main(S: T.Tensor((rows, width), dtype), B: T.Tensor((32, width), dtype), C: T.Tensor((rows, 32), T.float32)):
        with T.Kernel(1, threads=threads):
            s = T.alloc_shared((rows, width), dtype)
            b = T.alloc_fragment((32, width), dtype)
            c = T.alloc_fragment((rows, 32), T.float32)
            T.async_copy(S, s)
            T.copy(B, b)
            T.ptx_wait_group(0)
            T.sync_threads()
            T.clear(c)
            gemm(s, b, c, False, True, k_pack=pack)
            T.copy(c, C)

    return main


def get_plan(mod):
    found = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and "tl.hcu_gemm_at_bn_lds_strategy" in node.annotations:
            found.append(node.annotations["tl.hcu_gemm_at_bn_lds_strategy"])

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert found and all(plan.same_as(found[0]) for plan in found)
    return found[0]


@pytest.mark.parametrize(
    "banks,dtype,row_bytes,threads", list(product([32, 64], ["float16", "float8_e4m3fn"], [64, 128, 256, 512], [64, 128]))
)
def test_wrap_full_b128_phase(banks, dtype, row_bytes, threads):
    element_bytes = 2 if dtype == "float16" else 1
    width = row_bytes // element_bytes
    captures, _ = lower_with_captures(wrap_program(dtype, width, threads), "gfx938" if banks == 32 else "gfx946")
    plan = get_plan(captures["tl.MaterializeHcuGemmLdsStrategy"])
    assert int(plan.read_bytes_per_lane) == 16
    period = banks // np.gcd(banks, row_bytes // 4)
    required = max(1, (banks * 4 // 16 // 2) // period)
    assert int(plan.wrap_idx_mask) + 1 == required
    assert int(plan.wrap_offset) == (8 if required > 1 else 0)
    coords = np.indices((32, width), dtype=np.int64)
    values = dict(zip(plan.storage_layout.get_forward_vars(), coords))
    physical = [np.broadcast_to(evaluate(expr, values), (32, width)) for expr in plan.storage_layout.index]
    addresses = np.ravel_multi_index(physical, (32, width)) * element_bytes
    assert np.unique(addresses).size == 32 * width
    vectors = addresses.reshape(-1, 16 // element_bytes)
    np.testing.assert_array_equal(vectors - vectors[:, :1], np.broadcast_to(np.arange(16 // element_bytes) * element_bytes, vectors.shape))
    phases = B128_PHASES_32 if banks == 32 else B128_READ_PHASES_64
    for rb, cb, phase in product(range(0, 32, 16), range(0, width, 64 // element_bytes), phases):
        words = Counter()
        for lane in phase:
            addr = int(addresses[rb + lane % 16, cb + lane // 16 * (16 // element_bytes)])
            assert addr % 16 == 0
            words.update((addr // 4 + word) % banks for word in range(4))
        assert max(words.values()) == 1


def test_wrap_insufficient_slots_keeps_legal_partial_plan():
    captures, _ = lower_with_captures(wrap_program(width=32, rows=16), "gfx946")
    plan = get_plan(captures["tl.MaterializeHcuGemmLdsStrategy"])
    assert int(plan.read_bytes_per_lane) == 16
    assert int(plan.wrap_idx_mask) == int(plan.wrap_offset) == 0
