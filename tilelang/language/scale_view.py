# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Logical views over physically formatted HCU scale LDS buffers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import tvm
from tvm import tirx
from tilelang.language.utils import get_buffer_region_from_load


_FORMAT_IDS = {"identity": 0, "k2": 1, "k4": 2, "k2mn2": 3, "mn2": 4, "mn4": 5}


def _as_expr(value):
    return value if isinstance(value, tirx.PrimExpr) else tirx.IntImm("int32", int(value))


def _const_int(value, what: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, tirx.IntImm):
        return int(value.value)
    raise ValueError(f"{what} must be static, got {value}")


def _maybe_const_int(value):
    if isinstance(value, int):
        return value
    if isinstance(value, tirx.IntImm):
        return int(value.value)
    return None


@dataclass(frozen=True)
class ScaleFormat:
    name: str

    @property
    def format_id(self) -> int:
        return _FORMAT_IDS[self.name]

    def physical_shape(self, logical_shape: Sequence) -> tuple:
        k, mn = [_const_int(x, "logical shape") for x in logical_shape]
        if self.name == "identity":
            return (k, mn)
        if self.name == "k2":
            if k % 2:
                raise ValueError("K2 scale format requires ParentK divisible by 2")
            return (k // 2, mn, 2)
        if self.name == "k4":
            if k % 4:
                raise ValueError("K4 scale format requires ParentK divisible by 4")
            return (k // 4, mn, 4)
        if self.name == "k2mn2":
            if k % 2 or mn % 32:
                raise ValueError("K2MN2 scale format requires ParentK%2==0 and ParentMN%32==0")
            return (k // 2, mn // 32, 16, 2, 2)
        if self.name == "mn2":
            if mn % 2:
                raise ValueError("MN2 scale format requires ParentMN%2==0")
            return (k, mn // 2, 2)
        if self.name == "mn4":
            if mn % 4:
                raise ValueError("MN4 scale format requires ParentMN%4==0")
            return (k, mn // 4, 4)
        raise ValueError(f"unknown scale format {self.name}")


def scale_identity() -> ScaleFormat:
    return ScaleFormat("identity")


def scale_k2_interleaved() -> ScaleFormat:
    return ScaleFormat("k2")


def scale_k4_interleaved() -> ScaleFormat:
    return ScaleFormat("k4")


def scale_k2mn2_interleaved() -> ScaleFormat:
    return ScaleFormat("k2mn2")


def scale_mn2_interleaved() -> ScaleFormat:
    return ScaleFormat("mn2")


def scale_mn4_interleaved() -> ScaleFormat:
    return ScaleFormat("mn4")


@dataclass(frozen=True)
class ScaleView:
    buffer: tirx.Buffer | tirx.BufferLoad | tirx.BufferRegion
    logical_shape: tuple
    format: ScaleFormat
    origin: tuple
    extent: tuple

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) != 2:
            raise ValueError("ScaleView slicing expects [K, MN]")
        origin = []
        extent = []
        for dim, item in enumerate(indices):
            parent = _as_expr(self.logical_shape[dim])
            if isinstance(item, slice):
                if item.step not in (None, 1):
                    raise ValueError("ScaleView slice step must be 1")
                start = _as_expr(0 if item.start is None else item.start)
                stop = parent if item.stop is None else _as_expr(item.stop)
                origin.append(start)
                extent.append(stop - start)
            else:
                origin.append(_as_expr(item))
                extent.append(_as_expr(1))
            parent_c = _maybe_const_int(parent)
            origin_c = _maybe_const_int(origin[-1])
            extent_c = _maybe_const_int(extent[-1])
            if (
                parent_c is not None
                and origin_c is not None
                and extent_c is not None
                and (origin_c < 0 or extent_c < 0 or origin_c + extent_c > parent_c)
            ):
                raise ValueError(f"ScaleView slice dim {dim} [{origin_c}:{origin_c + extent_c}] is outside parent extent {parent_c}")
        return ScaleView(
            self.buffer,
            self.logical_shape,
            self.format,
            tuple(origin),
            tuple(extent),
        )


def scale_view(
    buffer: tirx.Buffer | tirx.BufferLoad | tirx.BufferRegion,
    *,
    logical_shape: Sequence,
    format: ScaleFormat,
) -> ScaleView:
    """Create a compile-time logical scale view over a physical LDS Buffer."""
    from tilelang.language.frame import get_let_value, has_let_value

    if isinstance(buffer, tirx.Var) and has_let_value(buffer):
        buffer = get_let_value(buffer)
    if not isinstance(buffer, (tirx.Buffer, tirx.BufferLoad, tirx.BufferRegion)):
        raise TypeError("scale_view buffer must be a TIR Buffer, BufferLoad, or BufferRegion")
    logical_shape = tuple(_as_expr(x) for x in logical_shape)
    if len(logical_shape) != 2:
        raise ValueError("scale_view logical_shape must be [K, MN]")
    expected = tuple(_as_expr(x) for x in format.physical_shape(logical_shape))
    source = buffer
    if isinstance(buffer, tirx.Buffer):
        physical_shape = tuple(buffer.shape)
        if len(physical_shape) != len(expected):
            raise ValueError(f"scale_view physical shape mismatch: format {format.name} expects {expected}, got {physical_shape}")
    else:
        if isinstance(buffer, tirx.BufferLoad):
            source = get_buffer_region_from_load(buffer)
            if source is None:
                raise ValueError("scale_view staged source must use explicit full slices for the non-stage dimensions")
        ranges = tuple(source.region)
        if len(ranges) < len(expected):
            raise ValueError(f"scale_view physical rank mismatch: format {format.name} needs {len(expected)} trailing dimensions")
        leading = ranges[: len(ranges) - len(expected)]
        if not leading:
            raise ValueError("scale_view BufferLoad/BufferRegion form is only for an explicit leading stage index")
        one = tirx.IntImm("int32", 1)
        if any(not tvm.ir.structural_equal(r.extent, one) for r in leading):
            raise ValueError("scale_view stage dimensions must be single indices, not ranges")
        tail = ranges[-len(expected) :]
        zero = tirx.IntImm("int32", 0)
        if any(not tvm.ir.structural_equal(r.min, zero) or not tvm.ir.structural_equal(r.extent, dim) for r, dim in zip(tail, expected)):
            got = tuple((r.min, r.extent) for r in tail)
            raise ValueError(f"scale_view must select the full physical scale plane {expected}, got {got}")
        physical_shape = tuple(r.extent for r in tail)
    if any(not tvm.ir.structural_equal(a, b) for a, b in zip(physical_shape, expected)):
        raise ValueError(f"scale_view physical shape mismatch: format {format.name} expects {expected}, got {physical_shape}")
    zeros = tuple(tirx.IntImm("int32", 0) for _ in range(2))
    return ScaleView(source, logical_shape, format, zeros, logical_shape)
