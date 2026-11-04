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
# Actual SB/64-bank phases from CoreArch HEAD_1773 getBankConflict.
# Matrix: sb_trace_1793232249/.../bank_calls.jsonl, call 25 (size=16).
# b128: sb_trace_p2_1793232421/.../bank_calls.jsonl (read=16, write=8).
# Keep the documented MATRIX_PHASES unchanged. NMZ matches that table;
# SB is separately captured here.
MATRIX_PHASES_64 = tuple(tuple(range(first, first + 32, 2)) for first in (0, 32, 1, 33))
B128_WRITE_PHASES_64 = (
    (0, 1, 2, 3, 8, 9, 10, 11),
    (20, 21, 22, 23, 28, 29, 30, 31),
    (32, 33, 34, 35, 40, 41, 42, 43),
    (52, 53, 54, 55, 60, 61, 62, 63),
    (4, 5, 6, 7, 12, 13, 14, 15),
    (16, 17, 18, 19, 24, 25, 26, 27),
    (36, 37, 38, 39, 44, 45, 46, 47),
    (48, 49, 50, 51, 56, 57, 58, 59),
)
B128_READ_PHASES_64 = tuple(B128_WRITE_PHASES_64[i] + B128_WRITE_PHASES_64[i + 1] for i in range(0, 8, 2))
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
    if banks == 64:
        period = max(1, 256 // width)
        return ((row & 8) << 1) ^ ((row & (8 - period)) * (32 // period))
    if width == 32:
        return (row & 4) << 2 if common else 0
    if width == 64:
        return ((row & 6) << 3) ^ ((row & 8) * 6) if common else (row & 2) << 4
    if width == 128 or common:
        shift = ((row & 1) << 5) ^ ((row & 10) << 3)
        return shift ^ ((row & 4) << 4) if common else shift
    return (row & 3) << 5


def footprint_summary(addresses, banks, pack):
    rows, width = addresses.shape
    # SB 256B read phases follow documented order9 (matrix) and order3
    # (ordinary b128); b64 uses 32 adjacent lanes.
    matrix_phases = MATRIX_PHASES_64 if banks == 64 else MATRIX_PHASES
    matrix = max(
        ways(addresses, [(r + (lane >> 1), c + (lane & 1) * 16) for lane in phase], 16, banks)
        for r, c, phase in product(range(0, rows, 32), range(0, width, 32), matrix_phases)
    )
    # Native fp8 AT/BN: kp1 = 8B/lane, kp2 = 16B/lane. Use complete
    # instruction-specific groups, including every SB 256B read phase.
    elements, group = (8, 16) if pack == 1 else (16, 4)
    if width < 32 * pack:
        return matrix, None
    groups = B128_PHASES_32 if pack == 2 and banks == 32 else [range(first, first + group) for first in range(0, 64, group)]
    if banks == 64:
        groups = B128_READ_PHASES_64 if pack == 2 else [range(i, i + 32) for i in range(0, 64, 32)]
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
        expected_matrix = 1
        assert footprint_summary(tile, banks, 1)[0] == expected_matrix
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
            for base, phase in product(range(0, 64 * width, 64 * 16), B128_WRITE_PHASES_64):
                starts = [((base + lane * 16) // width, (base + lane * 16) % width) for lane in phase]
                assert ways(tile, starts, 16, banks) == 1
        for pack in packs:
            if width < 32 * pack:
                # L32 cannot hold a complete native kp2 K64 micro-tile.
                # Prove all in-bounds four-lane 16B groups, without inventing
                # a second panel outside the buffer or wrapping its addresses.
                for r, c in product(range(0, 64, 4), range(0, width, 16)):
                    assert ways(tile, [(r + lane, c) for lane in range(4)], 16, banks) == 1
                continue
            expected = 2 if banks == 32 and pack == 1 else 1
            assert footprint_summary(tile, banks, pack) == (expected_matrix, expected)
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
    expected_matrix = max(1, (16 if banks == 64 else 4) // period)
    assert baseline[0] == expected_matrix
    assert single_result[0] == 1
    assert common_result[0] == 1
    if width < 32 * pack:
        assert baseline[1] is single_result[1] is common_result[1] is None
    else:
        native_rows = 16 if pack == 1 else 8 if banks == 64 else 4
        assert baseline[1] == max(1, native_rows // period)
        expected = 2 if banks == 32 and pack == 1 else 1
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


@pytest.mark.parametrize("packs", [(), (1,), (2,), (1, 2)])
def test_sb_width64_measured_phases(packs):
    for phases in (MATRIX_PHASES_64, B128_READ_PHASES_64, B128_WRITE_PHASES_64):
        assert sorted(lane for phase in phases for lane in phase) == list(range(64))
    layout = make_layout((32, 64), 64, packs)
    tile = enumerate_layout(layout, (32, 64), require_n8=False)
    assert footprint_summary(tile, 64, 1)[0] == 1
    for pack in packs:
        assert footprint_summary(tile, 64, pack) == (1, 1)
    # Ordinary global copies also keep conflict-free b128 producer phases.
    for base, phase in product(range(0, 32 * 64, 64 * 16), B128_WRITE_PHASES_64):
        starts = [((base + lane * 16) // 64, (base + lane * 16) % 64) for lane in phase]
        assert ways(tile, starts, 16, 64) == 1


@pytest.mark.parametrize("width", WIDTHS)
def test_b8_pure_sb_actual_read_write_phases(width):
    tile = enumerate_layout(make_layout((32, width), 64, ()), (32, width), require_n8=False)
    for c, phase in product(range(0, width, 32), MATRIX_PHASES_64):
        assert ways(tile, [(lane >> 1, c + (lane & 1) * 16) for lane in phase], 16, 64) == 1
    for base, phase in product(range(0, 32 * width, 64 * 16), B128_WRITE_PHASES_64):
        starts = [((base + lane * 16) // width, (base + lane * 16) % width) for lane in phase]
        assert ways(tile, starts, 16, 64) == 1


@pytest.mark.parametrize("width,expected", [(32, 2), (128, 4), (256, 4)])
def test_b8_old_pure_sb_matrix_conflict(width, expected):
    rows, cols = np.indices((32, width))
    shift = 0 if width == 32 else (rows & 2) << 5 if width == 128 else (rows & 3) << 6
    old = rows * width + (cols ^ shift)
    assert ways(old, [(lane >> 1, (lane & 1) * 16) for lane in MATRIX_PHASES_64[0]], 16, 64) == expected
