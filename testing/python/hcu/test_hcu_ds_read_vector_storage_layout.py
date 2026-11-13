"""Native matrix-read storage: bank phases, inverse, and vector preservation."""

from itertools import product

import numpy as np
import pytest
from tilelang import tvm
from test_hcu_plain_copy_lds_formula import (
    MATRIX_PHASES,
    MATRIX_PHASES_64,
    enumerate_layout,
    ways,
)


def make_layout(shape, banks):
    return tvm.get_global_func("tl.hcu.MakeDsReadVectorLdsLayout")(shape, banks, 4, 64, 16)


@pytest.mark.parametrize(
    "banks,width,rows,stages",
    list(product([32, 64], [32, 64, 128, 256, 512], [16, 32, 64], [(), (2,)]))
    + list(product([32, 64], [128, 256, 512], [48, 80], [(), (3, 2)])),
)
@pytest.mark.skipif(tvm.get_global_func("tl.hcu.MakeDsReadVectorLdsLayout", allow_missing=True) is None, reason="requires HCU layout FFI")
def test_storage_contract(banks, width, rows, stages):
    shape = (*stages, rows, width)
    layout = make_layout(shape, banks)
    assert layout is not None
    addresses = enumerate_layout(layout, shape, preserve_rows=False)
    inverse = enumerate_layout(layout.inverse(), shape, require_n8=False, preserve_rows=False)
    np.testing.assert_array_equal(inverse.reshape(-1)[addresses], np.arange(np.prod(shape)).reshape(shape))
    phases = MATRIX_PHASES if banks == 32 else MATRIX_PHASES_64
    for stage in np.ndindex(stages):
        for row, col, phase in product(range(0, rows, 16), range(0, width, 32), phases):
            starts = [(*stage, row + lane // 4, col + lane % 4 * 8) for lane in phase]
            assert ways(addresses, starts, 8, banks) == 1


@pytest.mark.skipif(tvm.get_global_func("tl.hcu.MakeDsReadVectorLdsLayout", allow_missing=True) is None, reason="requires HCU layout FFI")
def test_existing_k_mapping_preserved():
    shape = (64, 128)
    actual = enumerate_layout(make_layout(shape, 32), shape, preserve_rows=False)
    t, k = np.indices(shape)
    # Frozen kernel contract, independent of the compiler's parameterization.
    expected = 8 * (256 * (t // 16) + 64 * (t % 4) + (16 * ((t % 16) // 4) + k // 8 + 4 * (t % 2)) % 64) + k % 8
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(), (128,), (0, 128), (16, 0), (15, 128), (16, 7), (16, 96), (0, 16, 128)])
@pytest.mark.skipif(tvm.get_global_func("tl.hcu.MakeDsReadVectorLdsLayout", allow_missing=True) is None, reason="requires HCU layout FFI")
def test_unsupported_shape_not_silently_linearized(shape):
    assert make_layout(shape, 32) is None


@pytest.mark.parametrize("leading", ["symbolic", "large"])
@pytest.mark.parametrize("banks", [32, 64])
@pytest.mark.skipif(tvm.get_global_func("tl.hcu.MakeDsReadVectorLdsLayout", allow_missing=True) is None, reason="requires HCU layout FFI")
def test_leading_axes_pass_through(leading, banks):
    stage = tvm.tirx.Var("stage", "int64") if leading == "symbolic" else tvm.tirx.IntImm("int64", 2**32)
    shape = (stage, 3, 48, 128)
    layout = make_layout(shape, banks)
    assert layout is not None
    inverse = layout.inverse()
    matrix = make_layout((48, 128), banks)
    coords = [tvm.tirx.Var(name, "int32") for name in ("s", "b", "i", "j")]
    analyzer = tvm.arith.Analyzer()
    for actual, expected in ((layout, matrix), (inverse, matrix.inverse())):
        mapped = actual.map_forward_index(coords)
        matrix_mapped = expected.map_forward_index(coords[-2:])
        for lhs, rhs in zip(mapped, [*coords[:2], *matrix_mapped]):
            assert analyzer.can_prove_equal(lhs, rhs)
        for lhs, rhs in zip(actual.get_input_shape(), shape):
            assert analyzer.can_prove_equal(lhs, rhs)
