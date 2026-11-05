# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Default HCU shared layout: ordinary AT/BN and non-matrix AN/BT."""

from collections import Counter, defaultdict
from itertools import product

import numpy as np
import pytest
import tilelang.testing
from tilelang import tvm
from test_hcu_plain_copy_lds_formula import enumerate_layout
from test_hcu_plain_copy_lds_formula_b8 import B128_READ_PHASES_64, B128_WRITE_PHASES_64
from hcu_test_utils import current_hcu_arch_string

pytestmark = [
    *tilelang.testing.requires_rocm.marks(),
    pytest.mark.skipif(current_hcu_arch_string() not in {"gfx938", "gfx946"}, reason="requires HCU layouts"),
]

DTYPES = {8: "float8_e4m3fn", 16: "float16", 32: "float32"}


def make_layout(shape, bits, arch="gfx946", pack=2):
    buffer = tvm.tirx.decl_buffer(shape, DTYPES[bits], scope="shared")
    factory = tvm.get_global_func("tl.make_hcu_swizzled_layout")
    if arch is None:
        return factory(buffer, pack)
    with tvm.target.Target({"kind": "hcu", "mcpu": arch}):
        return factory(buffer, pack)


def ways(addresses, coords, size, banks=64):
    counts = Counter()
    for row, col in coords:
        address = int(addresses[row, col])
        assert address % size == 0
        counts.update((address // 4 + word) % banks for word in range(size // 4))
    return max(counts.values())


def footprint(addresses, element_bytes, read_bytes):
    rows, width = addresses.shape
    row_bytes = width * element_bytes
    read_elements = read_bytes // element_bytes
    read_phases = B128_READ_PHASES_64 if read_bytes == 16 else tuple(tuple(range(i, i + 32)) for i in (0, 32))
    reads = max(
        ways(addresses, [(row + lane % 16, col + (lane // 16) * read_elements) for lane in phase], read_bytes)
        for row in range(0, rows, 16)
        for col in range(0, width, 4 * read_elements)
        for phase in read_phases
    )
    writes = max(
        ways(addresses, [((base + lane * 16) // row_bytes, (base + lane * 16) % row_bytes // element_bytes) for lane in phase], 16)
        for base in range(0, rows * row_bytes, 1024)
        for phase in B128_WRITE_PHASES_64
    )
    return reads, writes


@pytest.mark.parametrize("bits,row_bytes,pack,leading", list(product([8, 16, 32], [64, 128, 256, 512], [1, 2], [(), (2,)])))
def test_bank64_ordinary_read_and_write(bits, row_bytes, pack, leading):
    shape = (*leading, 64, row_bytes // (bits // 8))
    layout = make_layout(shape, bits, pack=pack)
    addresses = enumerate_layout(layout, shape, require_n8=False)
    inverse = enumerate_layout(layout.inverse(), shape, require_n8=False)
    np.testing.assert_array_equal(inverse.ravel()[addresses.ravel()], np.arange(np.prod(shape)))
    vectors = addresses.reshape(-1, 128 // bits)
    np.testing.assert_array_equal(vectors - vectors[:, :1], np.broadcast_to(np.arange(128 // bits), vectors.shape))
    for stage in np.ndindex(leading):
        assert footprint(addresses[stage] * (bits // 8), bits // 8, 8 * pack) == (1, 1)


@pytest.mark.parametrize("bits,row_bytes,pack", list(product([8, 16, 32], [64, 128, 256, 512], [1, 2])))
def test_nonmatrix_anbt_word_multiplicity(bits, row_bytes, pack):
    shape = (64, row_bytes // (bits // 8))
    actual = enumerate_layout(make_layout(shape, bits, pack=pack), shape, require_n8=False) * (bits // 8)
    legacy = enumerate_layout(make_layout(shape, bits, arch=None, pack=pack), shape, require_n8=False) * (bits // 8)
    # Ordinary transposed loads use the B index map, not a matrix-read map.
    # This is a whole-wave bank-word multiplicity bound, not an assumed
    # hardware arbitration phase for scalar/subword instructions.
    local_elements = {8: 8, 16: 4, 32: 1}[bits] * pack

    def bound(addresses):
        result = 0
        for rb, cb, item in product(range(0, 64, 4 * local_elements), range(0, shape[1], 16), range(local_elements)):
            words = defaultdict(set)
            for lane in range(64):
                address = int(addresses[rb + (lane // 16) * local_elements + item, cb + lane % 16])
                words[address // 4 % 64].add(address // 4)
            result = max(result, max(map(len, words.values())))
        return result

    assert bound(actual) <= bound(legacy)


@pytest.mark.parametrize("bits,pack,row_bytes", list(product([8, 16, 32], [1, 2], [64, 128, 192, 256, 512])))
def test_bank32_and_no_target_preserve_legacy(bits, pack, row_bytes):
    shape = (32, row_bytes // (bits // 8))
    legacy = make_layout(shape, bits, arch=None, pack=pack)
    bank32 = make_layout(shape, bits, arch="gfx938", pack=pack)
    tvm.ir.assert_structural_equal(bank32.index, legacy.index)


@pytest.mark.parametrize("bits,width", list(product([8, 16, 32], [24, 48, 96])))
def test_bank64_padding_remains_injective(bits, width):
    layout = make_layout((32, width), bits)
    variables = layout.get_forward_vars()
    from plain_copy_lds_codegen import evaluate

    coords = np.indices((32, width), dtype=np.int64)
    physical = [np.broadcast_to(evaluate(expr, dict(zip(variables, coords))), (32, width)) for expr in layout.index]
    output = tuple(int(x) for x in layout.get_output_shape())
    addresses = np.ravel_multi_index(physical, output)
    assert np.unique(addresses).size == 32 * width
    if width % (4 * (128 // bits)) == 0 and (32 * width * bits // 8) % 1024 == 0:
        read_way, write_way = footprint(addresses * (bits // 8), bits // 8, 16)
        assert read_way == 1
        baseline = make_layout((32, width), bits, arch=None)
        baseline_vars = dict(zip(baseline.get_forward_vars(), coords))
        baseline_physical = [np.broadcast_to(evaluate(expr, baseline_vars), (32, width)) for expr in baseline.index]
        baseline_output = tuple(int(x) for x in baseline.get_output_shape())
        baseline_addresses = np.ravel_multi_index(baseline_physical, baseline_output) * (bits // 8)
        # Partial rows can straddle a store phase; the old layout can also
        # have residual write conflicts. Preserve that baseline bound.
        assert write_way <= footprint(baseline_addresses, bits // 8, 16)[1]


def default_program(dtype, kind, width, pack=2, threads=128):
    import tilelang.language as T
    from tilelang.hcu.language import gemm

    shared_shape = (32, width) if kind == "atbn" else (64, width)
    peer_shape = (width, 32) if kind == "atbn" else (64, 32)
    out_shape = (32, 32) if kind == "atbn" else (width, 32)

    @T.prim_func
    def main(S: T.Tensor(shared_shape, dtype), B: T.Tensor(peer_shape, dtype), C: T.Tensor(out_shape, "float32")):
        with T.Kernel(1, threads=threads):
            shared = T.alloc_shared(shared_shape, dtype)
            peer = T.alloc_fragment(peer_shape, dtype)
            accum = T.alloc_fragment(out_shape, "float32")
            T.copy(S, shared)
            T.copy(B, peer)
            T.clear(accum)
            gemm(shared, peer, accum, kind == "anbt", False, k_pack=pack)
            T.copy(accum, C)

    return main


def run_default_case(dtype, kind, width, pack, threads, arch):
    import torch
    import tilelang as tl
    from tilelang.transform import PassConfigKey

    shared_shape = (32, width) if kind == "atbn" else (64, width)
    peer_shape = (width, 32) if kind == "atbn" else (64, 32)
    rng = torch.Generator().manual_seed(20261104)
    inputs = [(torch.randint(-8, 9, shape, generator=rng).float() / 4).to(getattr(torch, dtype)) for shape in (shared_shape, peer_shape)]
    left = inputs[0].float() if kind == "atbn" else inputs[0].float().T
    expected = left @ inputs[1].float()
    assert torch.equal(expected.double(), left.double() @ inputs[1].double())
    kernel = tl.compile(
        default_program(dtype, kind, width, pack, threads),
        target={"kind": "hcu", "mcpu": arch},
        out_idx=[2],
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_ENABLE_ASYNC_COPY: False, PassConfigKey.TIR_USE_ASYNC_COPY: False},
    )
    source = kernel.get_kernel_source()
    assert "ds_read_vector" not in source and "cp_async_gs" not in source
    actual = kernel(*(x.cuda() for x in inputs)).cpu()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    return kernel, float((actual - expected).abs().max())


@pytest.mark.parametrize("dtype,kind", list(product(["float16", "float8_e4m3fn"], ["atbn", "anbt"])))
def test_default_shared_numerical(dtype, kind):
    _, error = run_default_case(dtype, kind, 64, 2, 128, current_hcu_arch_string())
    assert error == 0
