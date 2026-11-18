"""HCU cache warmup for contiguous global-memory regions."""

from tvm import DataType, arith, tirx
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script.builder import evaluate
from tvm.tirx.script.builder.frame import ForFrame, IfFrame, WhileFrame

from tilelang._typing import BufferLikeType
from tilelang.language.kernel import KernelLaunchFrame
from tilelang.utils.language import is_global, to_buffer_region

__all__ = ("cache_warmup",)


def cache_warmup(*srcs: BufferLikeType) -> None:
    """Issue cache warmup reads for contiguous global-memory regions.

    Regions must be contiguous in a compact two-dimensional row-major or
    column-major buffer. Calls must be at kernel scope, outside loops and
    conditional branches. Call this function at most once per kernel,
    passing all regions in that call.
    """
    if IRBuilder.is_in_scope():
        kernel = KernelLaunchFrame.Current()
        if kernel is None:
            raise ValueError("cache_warmup must be called inside T.Kernel")
        in_kernel = False
        for frame in IRBuilder.current().frames:
            if frame.same_as(kernel.frames[-1]):
                in_kernel = True
            elif in_kernel and isinstance(frame, (ForFrame, IfFrame, WhileFrame)):
                raise ValueError("cache_warmup must be outside loops and conditional branches")

    if not srcs:
        raise ValueError("cache_warmup requires at least one region")
    args = []
    for region in srcs:
        args.extend(_cache_warmup_args(region))
    evaluate(tirx.call_intrin("uint32", tirx.op.Op.get("tl.cache_warmup"), *args))


def _cache_warmup_args(src: BufferLikeType) -> tuple[tirx.PrimExpr, ...]:
    region = to_buffer_region(src)
    buffer = region.buffer
    if not is_global(buffer):
        raise ValueError("cache_warmup requires a global buffer")
    dtype = DataType(buffer.dtype)
    if dtype.bits % 8:
        raise ValueError("cache_warmup requires a byte-addressable dtype")
    if len(buffer.shape) != 2:
        raise ValueError("cache_warmup requires a two-dimensional buffer")
    if buffer.axis_separators:
        raise ValueError("cache_warmup does not support axis separators")

    analyzer = arith.Analyzer()
    shape = [tirx.Cast("int64", dim) for dim in buffer.shape]
    ranges = list(region.region)

    if buffer.strides:
        stride0, stride1 = buffer.strides
        row_major = analyzer.can_prove_equal(stride1, 1) and analyzer.can_prove_equal(stride0, shape[1])
        col_major = analyzer.can_prove_equal(stride0, 1) and analyzer.can_prove_equal(stride1, shape[0])
        if row_major:
            pass
        elif col_major:
            # Normalize column-major regions without transposing data.
            shape.reverse()
            ranges.reverse()
        else:
            raise ValueError("cache_warmup requires a compact row-major or column-major buffer")

    buffer_elements = analyzer.simplify(shape[0] * shape[1])

    # A contiguous region has single-element prefix dimensions and complete
    # suffix dimensions around one sliced axis in the normalized order.
    axis = 0
    for dim in range(1, len(shape)):
        full_dimension = analyzer.can_prove_equal(ranges[dim].min, 0) and analyzer.can_prove_equal(ranges[dim].extent, shape[dim])
        if not full_dimension:
            axis = dim
    if any(not analyzer.can_prove_equal(ranges[dim].extent, 1) for dim in range(axis)):
        raise ValueError("cache_warmup requires a contiguous region")

    prefix = tirx.IntImm("int64", 0)
    valid_prefix = tirx.const(True, "bool")
    for dim in range(axis):
        index = tirx.Cast("int64", ranges[dim].min)
        valid_prefix = valid_prefix & (index >= 0) & (index < shape[dim])
        prefix = analyzer.simplify(prefix * shape[dim] + index)
    row_elements = tirx.IntImm("int64", 1)
    for dim in shape[axis + 1 :]:
        row_elements = analyzer.simplify(row_elements * dim)
    offset = analyzer.simplify(prefix * shape[axis] * row_elements)
    # Invalid prefix indices suppress the request without forming an
    # out-of-bounds base pointer.
    zero = tirx.IntImm("int64", 0)
    offset = analyzer.simplify(tirx.Select(valid_prefix, offset, zero))
    rows = analyzer.simplify(tirx.Select(valid_prefix, tirx.Cast("int64", ranges[axis].extent), zero))
    return (
        buffer.access_ptr("r", offset=offset, extent=analyzer.simplify(buffer_elements - offset)),
        analyzer.simplify(tirx.Cast("int64", ranges[axis].min)),
        shape[axis],
        row_elements,
        rows,
    )
