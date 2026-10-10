"""HCU compatibility wrappers for warp-level reduction."""

from tvm import tirx

from tilelang.language.reduce_op import reduce


def reduce_warp(buffer: tirx.Buffer, out: tirx.Buffer, reduce_type: str, clear: bool = True):
    if list(buffer.shape) != list(out.shape):
        raise ValueError("warp reduce input and output must have the same logical shape")
    return reduce(buffer, out, reduce_type, 0, clear, annotations={"tl.warp_reduce": 1})


def reduce_sum_warp(buffer: tirx.Buffer, out: tirx.Buffer, clear: bool = True):
    return reduce_warp(buffer, out, "sum", clear)


__all__ = ("reduce_warp", "reduce_sum_warp")
