"""HCU-only matrix/scale copy language operations."""

from __future__ import annotations

from typing import Any, Literal

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language.copy_op import copy as _common_copy
from tilelang.language.common import get_let_value, has_let_value
from tilelang.language.utils import buffer_load_to_tile_region, buffer_region_to_tile_region, get_buffer_region_from_load
from tilelang.utils.language import is_fragment, is_global, is_scale_buffer, is_shared, legalize_pairwise_extents, to_buffer_region

from .scale_view import ScaleView


def copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    prefer_instruction: str | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tirx.PrimExpr | tirx.Stmt:
    """Copy data with HCU lowering hints.

    ``prefer_instruction="matrix_load"`` preserves the legacy HCU spelling
    while keeping the backend-neutral copy operation free of vendor-specific
    keyword arguments.  The hint is carried as tile-op metadata and consumed
    only by HCU lowering.

    ``disable_tma`` is accepted solely for source compatibility with kernels
    written against the former CUDA-shaped common facade.  HCU has no TMA
    lowering, so either value has identical semantics and no annotation is
    emitted into the target-neutral copy operation.
    """

    del disable_tma

    ann = dict(annotations) if annotations is not None else {}
    if "prefer_instruction" not in ann and prefer_instruction is not None:
        ann["prefer_instruction"] = tirx.StringImm(prefer_instruction)
    return _common_copy(
        src,
        dst,
        coalesced_width=coalesced_width,
        annotations=ann or None,
        loop_layout=loop_layout,
    )


def _resolve(value):
    return get_let_value(value) if isinstance(value, tirx.Var) and has_let_value(value) else value


def _buffer(value):
    value = _resolve(value)
    return value.buffer if isinstance(value, (tirx.BufferLoad, tirx.BufferRegion)) else value if isinstance(value, tirx.Buffer) else None


def _extent(value):
    value = _resolve(value)
    if isinstance(value, tirx.Buffer):
        return list(value.shape)
    if isinstance(value, tirx.BufferRegion):
        return [item.extent for item in value.region]
    if isinstance(value, tirx.BufferLoad):
        region = get_buffer_region_from_load(value)
        return None if region is None else [item.extent for item in region.region]
    return None


def _region(value, access_type, extents):
    value = _resolve(value)
    if isinstance(value, tirx.Buffer):
        return to_buffer_region(value, access_type=access_type, extents=extents)
    if isinstance(value, tirx.BufferRegion):
        return buffer_region_to_tile_region(value, access_type, extents)
    if isinstance(value, tirx.BufferLoad):
        region = get_buffer_region_from_load(value)
        if region is not None:
            return buffer_region_to_tile_region(region, access_type, extents)
        return buffer_load_to_tile_region(value, access_type, extents)
    return buffer_load_to_tile_region(value, access_type, extents)


def _boundary_modes(boundary):
    if boundary is None:
        boundary = (None, None)
    if len(boundary) != 2:
        raise ValueError("HCU matrix copy boundary must be (mn, k)")
    return tuple(-1 if value is None else int(bool(value)) for value in boundary)


def matrix_load(src, dst, boundary: tuple[bool | None, bool | None] | None = None, *, annotations: dict | None = None):
    """Load a logical matrix tile from global memory to HCU MLS shared memory."""

    dst_buffer = _buffer(dst)
    if dst_buffer is None or not is_shared(dst_buffer):
        raise ValueError("matrix_load destination must be a shared buffer")
    dst_extent = _extent(dst)
    if dst_extent is None:
        raise ValueError("matrix_load destination must have an extent")
    src_extent = _extent(src)
    dst_tile = list(dst_extent[-2:])
    src_tile = list(src_extent[-2:]) if src_extent is not None else list(dst_tile)
    if len(src_tile) < 2:
        src_tile = [tirx.IntImm("int32", 1)] * (2 - len(src_tile)) + src_tile
    src_tile, dst_tile = legalize_pairwise_extents(src_tile, dst_tile)
    mn_mode, k_mode = _boundary_modes(boundary)
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.matrix_load"),
        _region(src, "r", src_tile),
        _region(dst, "w", list(dst_extent)),
        tirx.IntImm("int32", mn_mode),
        tirx.IntImm("int32", k_mode),
        annotations=annotations,
    )


def matrix_store(src, dst, boundary: tuple[bool | None, bool | None] | None = None):
    """Store an HCU MMAC fragment to global memory through matrix-store lowering."""

    src_buffer, dst_buffer = _buffer(src), _buffer(dst)
    if src_buffer is None or not is_fragment(src_buffer):
        raise ValueError("matrix_store source must be a local.fragment buffer")
    if dst_buffer is None or not is_global(dst_buffer):
        raise ValueError("matrix_store destination must be a global buffer")
    src_extent = _extent(src)
    if src_extent is None:
        raise ValueError("matrix_store source must have an extent")
    dst_extent = _extent(dst) or list(src_extent[-2:])
    src_tile, dst_tile = legalize_pairwise_extents(list(src_extent[-2:]), list(dst_extent[-2:]))
    mn_mode, k_mode = _boundary_modes(boundary)
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.matrix_store"),
        _region(src, "r", list(src_extent)),
        _region(dst, "w", list(dst_extent)),
        tirx.IntImm("int32", mn_mode),
        tirx.IntImm("int32", k_mode),
    )


def ds_read_format(src, dst, alt: Literal[1, 2, 4] = 1):
    """Read MLS-formatted shared memory into an HCU register fragment."""

    if alt not in (1, 2, 4):
        raise ValueError(f"ds_read_format alt must be 1, 2, or 4, got {alt}")
    if _buffer(src) is None or not is_shared(_buffer(src)):
        raise ValueError("ds_read_format source must be shared memory")
    if isinstance(dst, tirx.BufferRegion):
        raise ValueError("ds_read_format destination must be Buffer or BufferLoad")
    src_extent, dst_extent = _extent(src), _extent(dst)
    if src_extent is None and dst_extent is None:
        raise ValueError("ds_read_format requires at least one known extent")
    src_extent = list(src_extent) if src_extent is not None else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent is not None else [1] * len(src_extent)
    src_extent, dst_extent = legalize_pairwise_extents(src_extent, dst_extent)
    extent = [tirx.max(lhs, rhs) for lhs, rhs in zip(src_extent, dst_extent)]
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.ds_read_format"),
        _region(src, "r", extent),
        _region(dst, "w", extent),
        tirx.IntImm("int32", alt),
    )


def copy_scale(src, dst, *, op_ctrl: int = 0):
    """Copy an LDS scale tile into HCU ``shared.scale`` storage."""

    scale_src = src if isinstance(src, ScaleView) else None
    if scale_src is not None:
        src = scale_src.buffer
        expected = {"identity": 0, "k2": 1, "k4": 2, "k2mn2": 2, "mn2": 1, "mn4": 2}[scale_src.format.name]
        if op_ctrl != expected:
            raise ValueError(f"copy_scale format {scale_src.format.name} requires op_ctrl={expected}, got {op_ctrl}")
    elif op_ctrl != 0:
        raise ValueError("copy_scale op_ctrl>0 requires an explicit ScaleView")
    if op_ctrl not in (0, 1, 2):
        raise ValueError(f"copy_scale op_ctrl must be 0/1/2, got {op_ctrl}")
    src_buffer, dst_buffer = _buffer(src), _buffer(dst)
    if src_buffer is None or not is_shared(src_buffer):
        raise ValueError("copy_scale source must be shared LDS")
    if dst_buffer is None or not is_scale_buffer(dst_buffer):
        raise ValueError("copy_scale destination must be shared.scale")
    src_extent, dst_extent = _extent(src), _extent(dst)
    if src_extent is None and dst_extent is None:
        raise ValueError("copy_scale requires at least one known extent")
    src_extent = list(src_extent or dst_extent)
    dst_extent = list(dst_extent or src_extent)
    if scale_src is None:
        src_extent, dst_extent = legalize_pairwise_extents(src_extent, dst_extent)
    args = [_region(src, "r", src_extent), _region(dst, "w", dst_extent), tirx.IntImm("int32", op_ctrl)]
    if scale_src is not None:
        args.extend(
            [
                tirx.IntImm("int32", scale_src.format.format_id),
                *scale_src.logical_shape,
                *scale_src.origin,
                *scale_src.extent,
            ]
        )
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.copy_scale"), *args)


__all__ = ("copy", "matrix_load", "matrix_store", "ds_read_format", "copy_scale")
