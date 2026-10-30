# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""b8 producer bank checks against current NMZ PMD's actual b128 lane order.

Captured at LdsState::getBankConflict in CoreArch:HEAD_1773, gfx938/32bank.
The first phase is lanes 0..3 and 20..23, not eight consecutive lanes.
The captured groups are listed explicitly below.
Matrix-read phases remain the independent, documented table in formula_b8.
"""

from itertools import product

import pytest
from tilelang import tvm
from tilelang.layout.swizzle import make_hcu_swizzled_layout
from test_hcu_plain_copy_lds_formula import enumerate_layout
from test_hcu_plain_copy_lds_formula_b8 import B128_PHASES_32, MATRIX_PHASES, make_layout, ways

import tilelang.testing
from hcu_test_utils import current_hcu_arch_string

pytestmark = [
    *tilelang.testing.requires_rocm.marks(),
    pytest.mark.skipif(
        current_hcu_arch_string() not in {"gfx938", "gfx946"},
        reason="requires a supported HCU LDS target",
    ),
]


@pytest.mark.parametrize("width", [32, 64, 128, 256, 512])
@pytest.mark.parametrize("packs", [(), (1,), (2,), (1, 2)], ids=["anbt", "kp1", "kp2", "mixed"])
def test_b8_pmd_write_and_matrix_phases(width, packs):
    shape = (64, width)
    addresses = enumerate_layout(make_layout(shape, 32, packs), shape, require_n8=False)
    for base, phase in product(range(0, 64 * width, 64 * 16), B128_PHASES_32):
        starts = [((base + lane * 16) // width, (base + lane * 16) % width) for lane in phase]
        assert ways(addresses, starts, 16, 32) == 1, (width, packs, base, phase)
    for row, col, phase in product(range(0, 64, 32), range(0, width, 32), MATRIX_PHASES):
        starts = [(row + (lane >> 1), col + (lane & 1) * 16) for lane in phase]
        assert ways(addresses, starts, 16, 32) == 1, (width, packs, row, col, phase)
    if 2 in packs and width >= 64:
        for row, col, phase in product(range(0, 64, 16), range(0, width, 64), B128_PHASES_32):
            starts = [(row + lane % 16, col + (lane // 16) * 16) for lane in phase]
            assert ways(addresses, starts, 16, 32) == 1, (width, packs, "kp2", row, col, phase)


def test_b128_write_phase_partition():
    assert sorted(lane for phase in B128_PHASES_32 for lane in phase) == list(range(64))


@pytest.mark.parametrize("width", [32, 64, 128, 256, 512])
def test_b8_atbn_kp1_retains_default_layout(width):
    shape = (32, width)
    factory = tvm.get_global_func("tl.hcu.MakePlainGemmLdsLayout")
    assert factory(shape, 32, 4, 8, 32, 32, 16, 16, [0], [1]) is None
    layout = make_hcu_swizzled_layout(tvm.tirx.decl_buffer(shape, "float8_e4m3fn", scope="shared"), 1)
    addresses = enumerate_layout(layout, shape, require_n8=True)
    # Unlike b128, NMZ b64 ignores the lane-order table and consumes sixteen
    # consecutive addresses. Captured inside getBankConflict, not inferred
    # from its unused pointer argument.
    for row, col, first in product(range(0, 32, 16), range(0, width, 32), range(0, 64, 16)):
        starts = [(row + lane % 16, col + (lane // 16) * 8) for lane in range(first, first + 16)]
        assert ways(addresses, starts, 8, 32) == 1, (width, "kp1-read", row, col, first)
    for base, first in product(range(0, 32 * width, 64 * 8), range(0, 64, 16)):
        starts = [((base + lane * 8) // width, (base + lane * 8) % width) for lane in range(first, first + 16)]
        assert ways(addresses, starts, 8, 32) == 1, (width, "b64-write", base, first)


@pytest.mark.parametrize("kinds,packs", [([0, 0], [1, 2]), ([0, 1], [1, 1])])
def test_b8_scope_and_n16_consumers(kinds, packs):
    shape = (32, 128)
    factory = tvm.get_global_func("tl.hcu.MakePlainGemmLdsLayout")
    layout = factory(shape, 32, 4, 8, 32, 32, 16, 16, kinds, packs)
    if 1 not in kinds:
        assert layout is None
        return
    addresses = enumerate_layout(layout, shape, require_n8=False)
    for row, col in product(range(32), range(0, 128, 16)):
        assert list(addresses[row, col : col + 16] - addresses[row, col]) == list(range(16))
