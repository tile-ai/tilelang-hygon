# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Test-only checks against lowered lane addresses and assembled LDS opcodes."""

from itertools import product
from pathlib import Path
import subprocess

import numpy as np
from tilelang import tvm
from tilelang.contrib.hcu import tilelang_callback_hcu_compile

from test_hcu_plain_copy_lds_formula import MATRIX_PHASES, enumerate_layout, make_layout, ways


def evaluate(expr, values):
    if isinstance(expr, tvm.tirx.IntImm):
        return int(expr)
    if isinstance(expr, tvm.tirx.Var):
        return values[expr]
    if isinstance(expr, tvm.tirx.Call):
        ops = {
            "tirx.bitwise_xor": np.bitwise_xor,
            "tirx.bitwise_and": np.bitwise_and,
            "tirx.shift_left": np.left_shift,
            "tirx.shift_right": np.right_shift,
        }
        return ops[expr.op.name](*(evaluate(arg, values) for arg in expr.args))
    ops = {tvm.tirx.Add: np.add, tvm.tirx.Mul: np.multiply, tvm.tirx.FloorDiv: np.floor_divide, tvm.tirx.FloorMod: np.mod}
    return ops[type(expr)](evaluate(expr.a, values), evaluate(expr.b, values))


def check_lowered_addresses(artifact, arch, pack):
    """Enumerate the actual lowered s buffer, not a reconstructed swizzle.

    All consumers use the same s.data pointer. Reads in distinct instructions
    are distinct arbitration groups; no cross-consumer ways are added.
    """
    banks = 32 if arch == "gfx938" else 64
    layout = enumerate_layout(make_layout((32, 128), banks, pack), (32, 128))
    inverse = np.argsort(layout.ravel())
    extents = {}
    accesses = []
    lane_variables = []
    pointer = []

    def visit(node):
        if isinstance(node, tvm.tirx.For):
            assert int(node.min) == 0
            extents[node.loop_var] = int(node.extent)
        if isinstance(node, tvm.tirx.AttrStmt) and node.attr_key == "thread_extent":
            extents[node.node.var] = int(node.value)
            if node.node.thread_tag == "threadIdx.x":
                lane_variables.append(node.node.var)
        if isinstance(node, (tvm.tirx.BufferLoad, tvm.tirx.BufferStore)) and node.buffer.data.name == "s":
            assert len(node.indices) == 1
            index = node.indices[0]
            vector = isinstance(index, tvm.tirx.Ramp)
            if vector:
                assert int(index.stride) == 1
            kind = "store" if isinstance(node, tvm.tirx.BufferStore) else "ordinary" if vector else "matrix"
            elements = int(index.lanes) if vector else 8
            accesses.append((kind, elements, index.base if vector else index))
            pointer.append(node.buffer.data)

    # Collect extents first: post-order visits accesses before their loop.
    for func in artifact.device_mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert accesses and all(ptr.same_as(pointer[0]) for ptr in pointer)
    found = set()
    for kind, elements, expr in accesses:
        variables = []

        def collect(node, variables=variables):
            if isinstance(node, tvm.tirx.Var) and all(not node.same_as(var) for var in variables):
                variables.append(node)

        tvm.tirx.stmt_functor.post_order_visit(expr, collect)
        lane_var = next(var for var in variables if any(var.same_as(lane) for lane in lane_variables))
        variables = [var for var in variables if not var.same_as(lane_var)]
        variables.append(lane_var)
        shape = tuple(extents[var] for var in variables)
        coords = np.indices(shape, dtype=np.int64)
        values = dict(zip(variables, coords))
        addresses = np.broadcast_to(evaluate(expr, values), shape).reshape(-1, 64)
        assert np.all((addresses >= 0) & (addresses < 4096))
        logical = inverse[addresses]
        lanes = np.arange(64)
        if kind == "matrix":
            groups = MATRIX_PHASES
            r, c = lanes >> 2, (lanes & 3) * 8
            expected = 1
        elif kind == "ordinary":
            assert elements == 4 * pack
            group = 16 if pack == 1 else 4
            groups = [range(first, first + group) for first in range(0, 64, group)]
            r, c = lanes % 16, (lanes // 16) * elements
            expected = 2 if pack == 1 and banks == 32 else 1
        else:
            assert elements == 8
            groups = [range(first, first + 8) for first in range(0, 64, 8)]
            np.testing.assert_array_equal(logical - logical[:, :1], np.broadcast_to(lanes * 8, logical.shape))
            expected = 1
        if kind != "store":
            np.testing.assert_array_equal(logical // 128 - r, np.broadcast_to(logical[:, :1] // 128, logical.shape))
            np.testing.assert_array_equal(logical % 128 - c, np.broadcast_to(logical[:, :1] % 128, logical.shape))
        assert max(ways(row, list(group), elements, banks) for row, group in product(addresses, groups)) == expected
        found.add(kind)
    assert found == {"matrix", "ordinary", "store"}


def check_assembly(artifact, arch, pack, directory):
    """Assemble unchanged generated source, then check real instruction widths."""
    directory = Path(directory)
    binary = directory / "kernel.hsaco"
    binary.write_bytes(bytes(tilelang_callback_hcu_compile(artifact.kernel_source, tvm.target.Target({"kind": "hcu", "mcpu": arch}))))
    if binary.read_bytes().startswith(b"__CLANG_OFFLOAD_BUNDLE__"):
        bundler = "/opt/rocm-6.3.3/llvm/bin/clang-offload-bundler"
        targets = subprocess.check_output([bundler, "--list", "--type=o", f"--input={binary}"], text=True).splitlines()
        target = next(item for item in targets if arch in item)
        device_binary = directory / "device.hsaco"
        subprocess.run(
            [bundler, "--unbundle", "--type=o", f"--targets={target}", f"--input={binary}", f"--output={device_binary}"], check=True
        )
        binary = device_binary
    assembly = subprocess.check_output(["/opt/rocm-6.3.3/llvm/bin/llvm-objdump", "-d", f"--mcpu={arch}", str(binary)], text=True)
    (directory / "kernel.s").write_text(assembly)
    (directory / "device.cu").write_text(artifact.kernel_source)
    (directory / "device.tir").write_text(str(artifact.device_mod))
    assert "ds_read_m32x16_b16" in assembly
    assert "ds_write_b128" in assembly
    assert ("ds_read_b128" if pack == 2 else "ds_read_b64") in assembly
    # Link the checked shared buffer to generated vector operations as well:
    # the other operand also contains LDS reads and cannot prove s's width.
    vector = "uint4" if pack == 2 else "uint2"
    assert any(f"= *({vector}*)" in line and "*)s)" in line for line in artifact.kernel_source.splitlines())
