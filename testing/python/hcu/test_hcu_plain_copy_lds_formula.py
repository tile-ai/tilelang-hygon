# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""b16 address proofs. Hardware arbitration models belong only in tests.

Spec: 0712-Triton-ds-read-m-support, ds_read_m32x16_b16 phase diagram;
The checks evaluate the production layout independently of arbitration.
No address enumeration participates in compiler strategy selection.
"""

from collections import Counter
from itertools import product

import numpy as np
import pytest
from tilelang import tvm
from tilelang.hcu.intrinsics.hcu_mmac_layout import (
    make_ds_read_format_fragment_hcu,
    shared_16x16_to_local_64x4_layout_A,
    shared_16x32_to_local_64x8_layout_A,
)
from tilelang.layout import Layout

import tilelang.testing
from hcu_test_utils import current_hcu_arch_string

pytestmark = [
    *tilelang.testing.requires_rocm.marks(),
    pytest.mark.skipif(
        current_hcu_arch_string() not in {"gfx938", "gfx946"},
        reason="requires a supported HCU LDS target",
    ),
]

# Full 128B phases, including both four-lane halves (not eight adjacent lanes).
MATRIX_PHASES = (
    (0, 4, 1, 5, 18, 22, 19, 23),
    (2, 6, 3, 7, 16, 20, 17, 21),
    (8, 12, 9, 13, 26, 30, 27, 31),
    (10, 14, 11, 15, 24, 28, 25, 29),
    (32, 36, 33, 37, 50, 54, 51, 55),
    (34, 38, 35, 39, 48, 52, 49, 53),
    (40, 44, 41, 45, 58, 62, 59, 63),
    (42, 46, 43, 47, 56, 60, 57, 61),
)
BANK_WIDTH_BYTES = 4
ELEMENT_BYTES = 2


def make_layout(shape, banks, common_pack):
    """Use the production closed-form factory, also used by materialization."""
    factory = tvm.get_global_func("tl.hcu.MakePlainGemmLdsLayout")
    kinds = [1] if common_pack is None else [1, 0]
    packs = [1] if common_pack is None else [1, common_pack]
    raw = factory(shape, banks, BANK_WIDTH_BYTES, 16, 16, 32, 8, 16, kinds, packs)
    assert raw is not None
    assert isinstance(raw, Layout)
    return raw


def enumerate_layout(layout, shape, require_n8=True):
    # Evaluate the actual Layout's expression over an array, avoiding a second
    # implementation of the swizzle. Enumeration is exclusively an assertion.
    variables = layout.get_forward_vars()
    coords = np.indices(shape, dtype=np.int64)

    def evaluate(expr):
        if isinstance(expr, tvm.tirx.IntImm):
            return int(expr)
        if isinstance(expr, tvm.tirx.Var):
            return coords[next(i for i, var in enumerate(variables) if var.same_as(expr))]
        if isinstance(expr, tvm.tirx.Call):
            ops = {
                "tirx.bitwise_xor": np.bitwise_xor,
                "tirx.bitwise_and": np.bitwise_and,
                "tirx.shift_left": np.left_shift,
                "tirx.shift_right": np.right_shift,
            }
            return ops[expr.op.name](*(evaluate(arg) for arg in expr.args))
        ops = {tvm.tirx.Add: np.add, tvm.tirx.Mul: np.multiply, tvm.tirx.FloorDiv: np.floor_divide, tvm.tirx.FloorMod: np.mod}
        return ops[type(expr)](evaluate(expr.a), evaluate(expr.b))

    physical = [np.broadcast_to(evaluate(expr), shape) for expr in layout.index]
    for axis, extent in zip(physical, shape):
        assert np.all((axis >= 0) & (axis < extent))
    addresses = np.ravel_multi_index(physical, shape)
    assert np.unique(addresses).size == np.prod(shape)
    for axis in range(len(shape) - 1):
        np.testing.assert_array_equal(physical[axis], coords[axis])
    if require_n8:
        np.testing.assert_array_equal(physical[-1] % 8, coords[-1] % 8)
        vectors = addresses.reshape(-1, 8)
        np.testing.assert_array_equal(vectors - vectors[:, :1], np.broadcast_to(np.arange(8), vectors.shape))
    return addresses


def ways(addresses, starts, elements, bank_count, require_alignment=True):
    # require_alignment=False is ONLY for explicitly invalid transaction
    # diagnostics in the default-swizzle baseline, never an executable proof.
    # One request per 4B word, not per half. Each lane contributes all words
    # of its real 8B/16B transaction to the complete arbitration group.
    banks = Counter()
    for start in starts:
        byte_address = int(addresses[start]) * ELEMENT_BYTES
        if require_alignment:
            assert byte_address % (elements * ELEMENT_BYTES) == 0
        banks.update((byte_address // BANK_WIDTH_BYTES + word) % bank_count for word in range(elements * ELEMENT_BYTES // BANK_WIDTH_BYTES))
    return max(banks.values())


@pytest.mark.parametrize(
    "banks,width,stages,common_pack", list(product([32, 64], [32, 64, 128, 256, 512], [(), (2,), (2, 2)], [None, 1, 2]))
)
def test_b16_production_layout(banks, width, stages, common_pack):
    shape = (*stages, 64, width)
    addresses = enumerate_layout(make_layout(shape, banks, common_pack), shape)
    # A logical tag written through the producer Layout must reach the same
    # lane/register via both consumers, including cross-panel permutations.
    memory = np.empty(np.prod(shape), dtype=np.int64)
    memory[addresses] = np.arange(np.prod(shape)).reshape(shape)
    for stage in np.ndindex(stages):
        # The emitted ordinary producer uses row-major N8 global vectors.
        # Prove disjoint store footprints over a complete bank ring; any
        # narrower hardware write group is a subset of this batch.
        for flat_base in range(0, 64 * width, banks * BANK_WIDTH_BYTES // ELEMENT_BYTES):
            starts = [(*stage, (flat_base + lane * 8) // width, (flat_base + lane * 8) % width) for lane in range(banks // 4)]
            assert ways(addresses, starts, 8, banks) == 1
        for row_base, panel in product(range(0, 64, 16), range(0, width, 32)):
            for phase in MATRIX_PHASES:
                starts = [(*stage, row_base + (lane >> 2), panel + (lane & 3) * 8) for lane in phase]
                assert ways(addresses, starts, 8, banks) == 1
                for start in starts:
                    for local in range(8):
                        coord = (*start[:-1], start[-1] + local)
                        assert memory[addresses[coord]] == np.ravel_multi_index(coord, shape)
            if common_pack is None:
                continue
            elements = 4 * common_pack
            fragment = shared_16x16_to_local_64x4_layout_A if common_pack == 1 else shared_16x32_to_local_64x8_layout_A
            # Actual AT/BN fragment mapping, both A and transposed B use this
            # physical [MN,K] micro-layout. Test every lane and register.
            for col_base in range(panel, panel + 32, 16 * common_pack):
                for lane, local in product(range(64), range(elements)):
                    r, c = lane % 16, (lane // 16) * elements + local
                    assert fragment(r, c) == (lane, local)
                    coord = (*stage, row_base + r, col_base + c)
                    assert memory[addresses[coord]] == np.ravel_multi_index(coord, shape)
                group_size = 16 if common_pack == 1 else 4
                for first_lane in range(0, 64, group_size):
                    starts = [
                        (*stage, row_base + lane % 16, col_base + (lane // 16) * elements)
                        for lane in range(first_lane, first_lane + group_size)
                    ]
                    expected = 2 if common_pack == 1 and banks == 32 else 1
                    assert ways(addresses, starts, elements, banks) == expected


def test_phase_table_covers_wave():
    assert sorted(lane for phase in MATRIX_PHASES for lane in phase) == list(range(64))
    assert all(len(phase) == 8 for phase in MATRIX_PHASES)


def test_native_matrix_fragment_ownership():
    # Hardware redistributes each 16B lane request into two N16 fragments.
    # Native fragment ownership is different from the request lane numbering.
    fragment = make_ds_read_format_fragment_hcu(32, 16, 1, 1, 16, 1, False)
    analyzer = tvm.arith.Analyzer()
    for k, n in product(range(16), range(32)):
        lane = analyzer.simplify(fragment.map_forward_thread([k, n])[0])
        register = analyzer.simplify(fragment.map_forward_index([k, n])[0])
        assert int(lane) == (n % 16) + 16 * (k // 4)
        assert int(register) == (n // 16) * 4 + k % 4


def footprint_summary(addresses, banks, pack):
    """Worst ways per instruction group; different consumers never arbitrate together."""
    rows, width = addresses.shape
    matrix = max(
        ways(addresses, [(r + (lane >> 2), c + (lane & 3) * 8) for lane in phase], 8, banks)
        for r, c, phase in product(range(0, rows, 16), range(0, width, 32), MATRIX_PHASES)
    )
    elements = 4 * pack
    group = 16 if pack == 1 else 4
    ordinary = max(
        ways(addresses, [(r + lane % 16, c + (lane // 16) * elements) for lane in range(first, first + group)], elements, banks)
        for r, c, first in product(range(0, rows, 16), range(0, width, 16 * pack), range(0, 64, group))
    )
    # ds_write_b128: eight consecutive lanes issue 16B each (128B).
    # This is deliberately independent from the four-lane ds_read_b128 model.
    store = max(
        ways(addresses, [((base + lane * 8) // width, (base + lane * 8) % width) for lane in range(8)], 8, banks)
        for base in range(0, rows * width, 64)
    )
    return matrix, ordinary, store


@pytest.mark.parametrize("banks,width,pack", list(product([32, 64], [32, 64, 128, 256, 512], [1, 2])))
def test_identity_single_common_comparison(banks, width, pack):
    shape = (64, width)
    identity = np.arange(np.prod(shape)).reshape(shape)
    single = enumerate_layout(make_layout(shape, banks, None), shape)
    common = enumerate_layout(make_layout(shape, banks, pack), shape)
    identity_result = footprint_summary(identity, banks, pack)
    single_result = footprint_summary(single, banks, pack)
    common_result = footprint_summary(common, banks, pack)
    row_period = max(1, banks // (width // 2))
    assert identity_result == (max(1, 2 // row_period), (16 if pack == 1 else 4) // min(row_period, 16 if pack == 1 else 4), 1)
    assert single_result[0] == 1
    assert single_result[1] == max(1, identity_result[1] // (2 if row_period == 1 else 1))
    assert single_result[2] == 1
    assert common_result == (1, 2 if banks == 32 and pack == 1 else 1, 1)
    # The common layout is one address array for both instructions. Separate
    # consumer groups are tested above; merging them would invent conflicts.


@pytest.mark.parametrize("width,pack", list(product([64, 128, 256, 512], [1, 2])))
def test_common_layout_scalar_anbt_bank_words(width, pack):
    """Check actual kp1 scalar lane addresses, separate from matrix phases.

    Opposite halfwords in one word are counted once. This is an address
    coverage invariant, not a prediction of DS service phases or PMD cycles.
    """
    addresses = enumerate_layout(make_layout((32, width), 32, pack), (32, width))
    for wave, ki, j, local in product(range(4), range(2), range(2), range(4)):
        words = set()
        for lane in range(64):
            row = ki * 16 + (lane // 16) * 4 + local
            col = (wave * 4 + j * 2 + lane % 16 // 8) * 8 + lane % 8
            if col < width:
                words.add(int(addresses[row, col]) // 2)
        counts = Counter(word % 32 for word in words)
        assert max(counts.values(), default=0) <= 1


def test_common_layout_separates_rows_eight_apart():
    addresses = enumerate_layout(make_layout((32, 128), 32, 2), (32, 128))
    words = [int(addresses[row, 0]) // 2 for row in (0, 4, 8, 12)]
    assert len({word % 32 for word in words}) == 4
