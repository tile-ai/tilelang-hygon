"""Alt2 bank phases from addr.md order8/order9; ordinary order4/order5."""

from itertools import product

import numpy as np
import pytest
from tilelang import tvm
from test_hcu_ds_read_vector_storage_layout import make_layout
from test_hcu_plain_copy_lds_formula import MATRIX_PHASES, MATRIX_PHASES_64, enumerate_layout, ways
from test_hcu_plain_copy_lds_formula_b8 import B128_PHASES_32, B128_WRITE_PHASES_64

# 32 banks service eight 16B lane segments per phase; 64 banks service sixteen.
ALT2_PHASES_32 = (
    (0, 2, 4, 6, 17, 19, 21, 23),
    (32, 34, 36, 38, 49, 51, 53, 55),
    (1, 3, 5, 7, 16, 18, 20, 22),
    (33, 35, 37, 39, 48, 50, 52, 54),
    (8, 10, 12, 14, 25, 27, 29, 31),
    (40, 42, 44, 46, 57, 59, 61, 63),
    (9, 11, 13, 15, 24, 26, 28, 30),
    (41, 43, 45, 47, 56, 58, 60, 62),
)
ALT2_PHASES_64 = tuple(tuple(range(first, first + 32, 2)) for first in (0, 32, 1, 33))


@pytest.mark.parametrize("banks,width,rows", list(product([32, 64], [32, 64, 128, 256, 512], [16, 48])))
def test_alt2_storage_phases_and_inverse(banks, width, rows):
    shape = (2, rows, width)
    layout = tvm.get_global_func("tl.hcu.MakeDsReadVectorLdsLayoutWithAlt")(shape, banks, 4, 64, 16, 2)
    addresses = enumerate_layout(layout, shape)
    inverse = enumerate_layout(layout.inverse(), shape, require_n8=False)
    np.testing.assert_array_equal(inverse.reshape(-1)[addresses], np.arange(np.prod(shape)).reshape(shape))
    # A shared allocation can contain both ordinary and Alt2 consumers.
    phases = (MATRIX_PHASES + ALT2_PHASES_32) if banks == 32 else (MATRIX_PHASES_64 + ALT2_PHASES_64)
    for stage, row, col, phase in product(range(2), range(0, rows, 16), range(0, width, 32), phases):
        coords = [(stage, row + lane // 4, col + lane % 4 * 8) for lane in phase]
        assert ways(addresses, coords, 8, banks) == 1
    # GDN stores V using its accumulator ownership, independently of matrix
    # lane order. Do not trade read conflicts for b128 write conflicts.
    write_phases = B128_PHASES_32 if banks == 32 else B128_WRITE_PHASES_64
    for stage, row, col, phase in product(range(2), range(0, rows, 16), range(0, width, 32), write_phases):
        coords = [(stage, row + lane % 16, col + lane // 16 * 8) for lane in phase]
        assert ways(addresses, coords, 8, banks) == 1


@pytest.mark.parametrize("width", [64, 128])
def test_old_layout_exposes_alt2_conflict(width):
    addresses = enumerate_layout(make_layout((64, width), 64), (64, width), preserve_rows=False)
    coords = [(lane // 4, lane % 4 * 8) for lane in ALT2_PHASES_64[0]]
    assert ways(addresses, coords, 8, 64) == 2
