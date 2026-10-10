"""HCU-specific storage allocation helpers."""

from tilelang._typing import DType, ShapeType
from tvm.script import tirx as T
from tvm.tirx.buffer import Buffer


def alloc_scale_buffer(shape: ShapeType, dtype: DType = "uint8") -> Buffer:
    if len(shape) != 2:
        raise ValueError("scale_buffer shape must be 2D; use separate buffers for ping-pong stages")
    return T.sblock_alloc_buffer(shape, dtype, scope="shared.scale")


__all__ = ("alloc_scale_buffer",)
