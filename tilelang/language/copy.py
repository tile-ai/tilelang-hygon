"""The language interface for tl programs."""

from typing import Union, Optional, Literal
from tilelang import language as T
from tilelang.utils.language import get_buffer_region_from_load, is_shared
from tvm import ir, tir
from tilelang.language.utils import buffer_to_tile_region, buffer_region_to_tile_region, buffer_load_to_tile_region


def matrix_load(
    src: Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion],
    dst: Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion],
    last_k_load: Optional[bool] = None,
):
    """MLS (Matrix Load Store) load from global memory to shared memory.

    Args:
        src: Global source tensor. Buffer/BufferLoad/BufferRegion - offset (block_mn_base,
            block_k_base) is extracted from region in C++ (region[].min).
        dst: Destination shared tensor (must be shared memory, must have extent).
        last_k_load: If None (default), check_last_k_load=true (boundary k check). If set,
            check_last_k_load=false and use given last_k_load value.
            boundary mn check is always true.

    Returns:
        tir.Call: A handle to the matrix_load operation.
    """
    if last_k_load is None:
        check_last_k_load = True
        last_k_load_val = False  # unused when check_last_load
    else:
        check_last_k_load = False
        last_k_load_val = last_k_load

    def get_extent(data):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            return list(data.shape)
        if isinstance(data, tir.BufferRegion):
            return [x.extent for x in data.region]
        if isinstance(data, tir.BufferLoad):
            region = get_buffer_region_from_load(data)
            if region is None:
                return None
            return [x.extent for x in region.region]
        return None

    def get_buffer(data):
        if isinstance(data, tir.Buffer):
            return data
        if isinstance(data, (tir.BufferLoad, tir.BufferRegion)):
            return data.buffer
        return None

    dst_buf = get_buffer(dst)
    assert dst_buf is not None, "matrix_load dst must be Buffer, BufferLoad or BufferRegion"
    assert is_shared(dst_buf), f"matrix_load dst must be shared memory, got scope={dst_buf.scope()}"

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)
    assert dst_extent is not None, "matrix_load dst must have extent (use Buffer or BufferRegion)"
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent)
    extent = [max(a, b) for a, b in zip(src_extent, dst_extent)]

    def _to_region(data, access_type):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            return buffer_to_tile_region(data, access_type)
        if isinstance(data, tir.BufferRegion):
            return buffer_region_to_tile_region(data, access_type, extent)
        if isinstance(data, tir.BufferLoad):
            region = get_buffer_region_from_load(data)
            if region is None:
                return buffer_load_to_tile_region(data, access_type, extent)
            return buffer_region_to_tile_region(region, access_type, extent)
        return buffer_load_to_tile_region(data, access_type, extent)

    src_region = _to_region(src, "r")
    dst_region = _to_region(dst, "w")

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.matrix_load"),
        src_region,
        dst_region,
        tir.IntImm("int32", 1 if check_last_k_load else 0),
        tir.IntImm("int32", 1 if last_k_load_val else 0),
    )


def ds_read_format(
    src: Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion],
    dst: Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion],
):
    """Read MLS-formatted shared memory into register with ds_read_format layout.

    Args:
        src: Source shared tensor (must be shared memory, must have extent).
        dst: Destination register tensor (must have extent).
        Input and output must have at least one with extent; both passed as
        buffer regions to C++.

    Returns:
        tir.Call: A handle to the ds_read_format operation.
    """
    def get_extent(data):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            return list(data.shape)
        if isinstance(data, tir.BufferRegion):
            return [x.extent for x in data.region]
        if isinstance(data, tir.BufferLoad):
            region = get_buffer_region_from_load(data)
            if region is None:
                return None
            return [x.extent for x in region.region]
        return None

    def get_buffer(data):
        if isinstance(data, tir.Buffer):
            return data
        if isinstance(data, (tir.BufferLoad, tir.BufferRegion)):
            return data.buffer
        return None

    src_buf = get_buffer(src)
    dst_buf = get_buffer(dst)
    assert src_buf is not None, "ds_read_format src must be Buffer, BufferLoad or BufferRegion"
    assert dst_buf is not None, "ds_read_format dst must be Buffer, BufferLoad or BufferRegion"
    assert is_shared(src_buf), (
        f"ds_read_format src must be shared memory, got scope={src_buf.scope()}"
    )

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)
    assert src_extent is not None or dst_extent is not None, (
        "ds_read_format: src and dst must have at least one with extent"
    )
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)
    extent = [max(a, b) for a, b in zip(src_extent, dst_extent)]

    def _to_region(data, access_type):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            return buffer_to_tile_region(data, access_type)
        if isinstance(data, tir.BufferRegion):
            return buffer_region_to_tile_region(data, access_type, extent)
        if isinstance(data, tir.BufferLoad):
            region = get_buffer_region_from_load(data)
            if region is None:
                return buffer_load_to_tile_region(data, access_type, extent)
            return buffer_region_to_tile_region(region, access_type, extent)
        return buffer_load_to_tile_region(data, access_type, extent)

    src_region = _to_region(src, "r")
    dst_region = _to_region(dst, "w")

    return tir.call_intrin("handle", tir.op.Op.get("tl.ds_read_format"), src_region, dst_region)


def copy(src: Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion],
         dst: Union[tir.Buffer, tir.BufferLoad],
         coalesced_width: Optional[int] = None,
         disable_tma: bool = False,
         eviction_policy: Optional[Literal["evict_normal", "evict_first", "evict_last"]] = None):
    """Copy data between memory regions.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source memory region
        dst (Union[tir.Buffer, tir.BufferLoad]): Destination memory region
        coalesced_width (Optional[int], optional): Width for coalesced memory access. Defaults to None.

    Raises:
        TypeError: If copy extents cannot be deduced from arguments

    Returns:
        tir.Call: A handle to the copy operation
    """
    if isinstance(src, tir.Buffer) and isinstance(dst, tir.Buffer):
        ir.assert_structural_equal(src.shape, dst.shape)

    def get_extent(data):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            return data.shape
        elif isinstance(data, tir.BufferRegion):
            return [x.extent for x in data.region]
        elif isinstance(data, tir.BufferLoad):
            region = get_buffer_region_from_load(data)
            if region is None:
                return None
            return [x.extent for x in region.region]
        else:
            return None

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)
    # Combine the nested if statements into a single if statement as suggested by SIM102
    if (src_extent is None and dst_extent is None and isinstance(src, tir.BufferLoad) and
            isinstance(dst, tir.BufferLoad)):
        # check if the case is like this:
        # copy(buffer_a[i], buffer_b[i]) where both are BufferLoad nodes
        # In this case, lower it to a simple BufferStore: buffer_b[i] = buffer_a[i]
        return tir.BufferStore(dst.buffer, src, dst.indices)

    assert src_extent or dst_extent, "Can't deduce copy extents from args"
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)
    extent = max(src_extent, dst_extent)

    def _to_region(data, access_type):
        if isinstance(data, tir.Var) and T.has_let_value(data):
            data = T.get_let_value(data)
        if isinstance(data, tir.Buffer):
            return buffer_to_tile_region(data, access_type)
        elif isinstance(data, tir.BufferRegion):
            return buffer_region_to_tile_region(data, access_type, extent)
        elif isinstance(data, tir.BufferLoad):
            region = get_buffer_region_from_load(data)
            if region is None:
                return buffer_load_to_tile_region(data, access_type, extent)
            return buffer_region_to_tile_region(region, access_type, extent)
        else:
            return buffer_load_to_tile_region(data, access_type, extent)

    src = _to_region(src, "r")
    dst = _to_region(dst, "w")

    if coalesced_width is None:
        coalesced_width = -1  # PrimExpr can not be None
    if eviction_policy is None:
        eviction_policy = 0
    else:
        eviction_policy = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}[eviction_policy]
    return tir.call_intrin("handle", tir.op.Op.get("tl.copy"), src, dst, coalesced_width,
                           disable_tma, eviction_policy)


def c2d_im2col(img: tir.Buffer,
               col: tir.Buffer,
               nhw_step: tir.PrimExpr,
               c_step: tir.PrimExpr,
               kernel: int,
               stride: int,
               dilation: int,
               pad: int,
               eviction_policy: Optional[Literal["evict_normal", "evict_first",
                                                 "evict_last"]] = None):
    """Perform im2col transformation for 2D convolution.

    Args:
        img (tir.Buffer): Input image buffer
        col (tir.Buffer): Output column buffer
        nhw_step (tir.PrimExpr): Step size for batch and spatial dimensions
        c_step (tir.PrimExpr): Step size for channel dimension
        kernel (int): Kernel size
        stride (int): Stride of the convolution
        dilation (int): Dilation rate
        pad (int): Padding size

    Returns:
        tir.Call: A handle to the im2col operation
    """
    if eviction_policy is None:
        eviction_policy = 0
    else:
        eviction_policy = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}[eviction_policy]
    return tir.call_intrin("handle", tir.op.Op.get("tl.c2d_im2col"), img.access_ptr("r"),
                           col.access_ptr("w"), nhw_step, c_step, kernel, stride, dilation, pad,
                           eviction_policy)
