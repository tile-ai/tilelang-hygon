# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""b8 address proofs, not hardware execution or arbitration verification.

Phase source: 0712-Triton-ds-read-m-support.html,
section 1.2.4, b8 thread/VGPR diagram. Do not reuse the b16 phase table.
All bank enumeration stays in tests; the compiler emits a closed-form Layout.
"""

from collections import Counter
from itertools import product

import numpy as np
import pytest
from tilelang import tvm
from tilelang.hcu.intrinsics.hcu_mmac_layout import (
    shared_16x32_to_local_64x8_layout_A,
    shared_16x64_to_local_64x16_layout_A,
)
from test_hcu_plain_copy_lds_formula import enumerate_layout

import tilelang.testing
from hcu_test_utils import current_hcu_arch_string

pytestmark = [
    *tilelang.testing.requires_rocm.marks(),
    pytest.mark.skipif(
        current_hcu_arch_string() not in {"gfx938", "gfx946"},
        reason="requires a supported HCU LDS target",
    ),
]

MATRIX_PHASES = (
    (0, 2, 4, 6, 17, 19, 21, 23),
    (32, 34, 36, 38, 49, 51, 53, 55),
    (16, 18, 20, 22, 1, 3, 5, 7),
    (48, 50, 52, 54, 33, 35, 37, 39),
    (8, 10, 12, 14, 25, 27, 29, 31),
    (40, 42, 44, 46, 57, 59, 61, 63),
    (24, 26, 28, 30, 9, 11, 13, 15),
    (56, 58, 60, 62, 41, 43, 45, 47),
)
# Actual NMZ b128 write AND ordinary b128 read phases, captured from
# CoreArch:HEAD_1773 getBankConflict on the 32-bank target.
B128_PHASES_32 = (
    (0, 1, 2, 3, 20, 21, 22, 23),
    (32, 33, 34, 35, 52, 53, 54, 55),
    (4, 5, 6, 7, 16, 17, 18, 19),
    (36, 37, 38, 39, 48, 49, 50, 51),
    (8, 9, 10, 11, 28, 29, 30, 31),
    (40, 41, 42, 43, 60, 61, 62, 63),
    (12, 13, 14, 15, 24, 25, 26, 27),
    (44, 45, 46, 47, 56, 57, 58, 59),
)
WIDTHS = [32, 64, 128, 256, 512]


def make_layout(shape, banks, packs):
    factory = tvm.get_global_func("tl.hcu.MakePlainGemmLdsLayout")
    raw = factory(shape, banks, 4, 8, 32, 32, 16, 16, [1] + [0] * len(packs), [1] + list(packs))
    assert raw is not None
    return raw


def ways(addresses, starts, elements, banks):
    """Count every 4B word in an actual 8B/16B lane transaction."""
    counts = Counter()
    for coord in starts:
        address = int(addresses[coord])  # fp8: one element is one byte
        assert address % elements == 0
        counts.update((address // 4 + word) % banks for word in range(elements // 4))
    return max(counts.values())


def expected_shift(row, banks, width, common):
    # Derived in byte-address bank bits: r0/r1 must survive the matrix phase;
    # r2 is phase-constant, while r3 flips together with the lane's c4.
    if banks == 32:
        if width == 32:
            return (row & 4) << 2 if common else 0
        if width == 64:
            return ((row & 6) << 3) ^ ((row & 8) * 6) if common else (row & 2) << 4
        if width == 128 or common:
            shift = ((row & 1) << 5) ^ ((row & 10) << 3)
            return shift ^ ((row & 4) << 4) if common else shift
        return (row & 3) << 5
    else:
        shift = 0 if width <= 64 else (row & 2) << 5 if width == 128 else (row & 3) << 6
    if common:
        shift ^= (row & 4) << 2
        if banks == 64 and width >= 64:
            shift ^= (row & 8) << 2
    return shift


def footprint_summary(addresses, banks, pack):
    rows, width = addresses.shape
    matrix = max(
        ways(addresses, [(r + (lane >> 1), c + (lane & 1) * 16) for lane in phase], 16, banks)
        for r, c, phase in product(range(0, rows, 32), range(0, width, 32), MATRIX_PHASES)
    )
    # Native fp8 AT/BN: kp1 = 8B/lane (16-lane group). For NMZ kp2,
    # use the captured full b128 phase, not a four-lane subset. The old
    # contiguous group below is retained only as a 64bank address invariant.
    elements, group = (8, 16) if pack == 1 else (16, 4)
    if width < 32 * pack:
        return matrix, None
    groups = B128_PHASES_32 if pack == 2 and banks == 32 else [range(first, first + group) for first in range(0, 64, group)]
    ordinary = max(
        ways(addresses, [(r + lane % 16, c + (lane // 16) * elements) for lane in phase], elements, banks)
        for r, c, phase in product(range(0, rows, 16), range(0, width, 32 * pack), groups)
    )
    return matrix, ordinary


@pytest.mark.parametrize("banks,width,stages,packs", list(product([32, 64], WIDTHS, [(), (2,), (2, 2)], [(), (1,), (2,), (1, 2)])))
def test_b8_production_layout(banks, width, stages, packs):
    shape = (*stages, 64, width)
    addresses = enumerate_layout(make_layout(shape, banks, packs), shape, require_n8=False)
    coords = np.indices(shape, dtype=np.int64)
    expected_col = ((coords[-1] // 16) ^ (expected_shift(coords[-2], banks, width, bool(packs)) // 16)) * 16 + coords[-1] % 16
    np.testing.assert_array_equal(addresses, np.ravel_multi_index((*coords[:-1], expected_col), shape))
    np.testing.assert_array_equal(addresses % 16, coords[-1] % 16)
    vectors = addresses.reshape(-1, 16)
    np.testing.assert_array_equal(vectors - vectors[:, :1], np.broadcast_to(np.arange(16), vectors.shape))
    memory = np.empty(np.prod(shape), dtype=np.int64)
    memory[addresses] = np.arange(np.prod(shape)).reshape(shape)
    for stage in np.ndindex(stages):
        tile = addresses[stage]
        assert footprint_summary(tile, banks, 1)[0] == 1
        for r, c, phase in product(range(0, 64, 32), range(0, width, 32), MATRIX_PHASES):
            for lane in phase:
                start = (*stage, r + (lane >> 1), c + (lane & 1) * 16)
                logical = np.ravel_multi_index(start, shape)
                physical = int(addresses[start])
                np.testing.assert_array_equal(memory[physical : physical + 16], logical + np.arange(16))
        if banks == 32:
            for base, phase in product(range(0, 64 * width, 64 * 16), B128_PHASES_32):
                starts = [((base + lane * 16) // width, (base + lane * 16) % width) for lane in phase]
                assert ways(tile, starts, 16, banks) == 1
        else:
            # A contiguous-footprint invariant only: the PMD b128 phase
            # capture above is for NMZ/32bank, not a 64bank phase claim.
            for base in range(0, 64 * width, 128):
                starts = [((base + lane * 16) // width, (base + lane * 16) % width) for lane in range(8)]
                assert ways(tile, starts, 16, banks) == 1
        for pack in packs:
            if width < 32 * pack:
                # L32 cannot hold a complete native kp2 K64 micro-tile.
                # Prove all in-bounds four-lane 16B groups, without inventing
                # a second panel outside the buffer or wrapping its addresses.
                for r, c in product(range(0, 64, 4), range(0, width, 16)):
                    assert ways(tile, [(r + lane, c) for lane in range(4)], 16, banks) == 1
                continue
            expected = (2 if banks == 32 or width == 32 else 1) if pack == 1 else 1
            assert footprint_summary(tile, banks, pack) == (1, expected)
            elements = 8 * pack
            fragment = shared_16x32_to_local_64x8_layout_A if pack == 1 else shared_16x64_to_local_64x16_layout_A
            for r, c, lane, local in product(range(0, 64, 16), range(0, width, 32 * pack), range(64), range(elements)):
                row, col = lane % 16, (lane // 16) * elements + local
                assert fragment(row, col) == (lane, local)
                coord = (*stage, r + row, c + col)
                assert memory[addresses[coord]] == np.ravel_multi_index(coord, shape)


def test_b8_phase_geometry():
    assert sorted(lane for phase in MATRIX_PHASES for lane in phase) == list(range(64))
    for phase in MATRIX_PHASES:
        assert len(phase) == 8
        assert len({((lane >> 1) & 4, (lane >> 1) & 16) for lane in phase}) == 1
        assert len({((lane >> 1) & 3, lane & 1) for lane in phase}) == 8
        assert len({((lane >> 4) & 1) ^ (lane & 1) for lane in phase}) == 1


@pytest.mark.parametrize("banks,width,pack", list(product([32, 64], WIDTHS, [1, 2])))
def test_b8_identity_single_common_comparison(banks, width, pack):
    shape = (64, width)
    identity = np.arange(np.prod(shape)).reshape(shape)
    single = enumerate_layout(make_layout(shape, banks, ()), shape, require_n8=False)
    common = enumerate_layout(make_layout(shape, banks, (pack,)), shape, require_n8=False)
    baseline = footprint_summary(identity, banks, pack)
    single_result = footprint_summary(single, banks, pack)
    common_result = footprint_summary(common, banks, pack)
    period = min(16, banks // (width // 4)) if width // 4 < banks else 1
    assert baseline[0] == max(1, 4 // period)
    assert single_result[0] == common_result[0] == 1
    if width < 32 * pack:
        assert baseline[1] is single_result[1] is common_result[1] is None
    else:
        assert baseline[1] == (16 if pack == 1 else 4) // min(period, 16 if pack == 1 else 4)
        expected = (2 if banks == 32 or width == 32 else 1) if pack == 1 else 1
        assert common_result[1] == expected


@pytest.mark.parametrize("field,value", [(3, 16), (4, 16), (5, 16), (6, 8), (7, 8)])
def test_b8_rejects_inconsistent_contract(field, value):
    args = [(32, 64), 32, 4, 8, 32, 32, 16, 16, [1], [1]]
    args[field] = value
    assert tvm.get_global_func("tl.hcu.MakePlainGemmLdsLayout")(*args) is None


def test_b8_rejects_anbt_pack2():
    assert tvm.get_global_func("tl.hcu.MakePlainGemmLdsLayout")((32, 64), 32, 4, 8, 32, 32, 16, 16, [1], [2]) is None


@pytest.mark.parametrize("shape", [(16, 64), (32, 16), (48, 64), (32, 48)])
def test_b8_rejects_incompatible_shape(shape):
    assert tvm.get_global_func("tl.hcu.MakePlainGemmLdsLayout")(shape, 32, 4, 8, 32, 32, 16, 16, [1], [1]) is None


def test_b8_combines_all_consumers():
    factory = tvm.get_global_func("tl.hcu.MakePlainGemmLdsLayout")
    args = [(32, 128), 64, 4, 8, 32, 32, 16, 16]
    expected = enumerate_layout(make_layout((32, 128), 64, (1, 2)), (32, 128), require_n8=False)
    for kinds, packs in [([0, 1, 0], [2, 1, 1]), ([0, 0, 1], [1, 2, 1])]:
        actual = enumerate_layout(factory(*args, kinds, packs), (32, 128), require_n8=False)
        np.testing.assert_array_equal(actual, expected)
    assert factory(*args, [0], [1]) is None
    assert factory(*args, [1, 0], [1, 3]) is None
    assert factory(*args, [0, 1], [1, 2]) is None
