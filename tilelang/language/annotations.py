"""Annotation helpers exposed on the TileLang language surface."""

from typing import Callable

from tilelang.layout import Fragment, Layout
from tilelang.utils.language import is_fragment
from tvm.script.parser.tir import attr, block_attr
from tvm.tir import FloatImm, IntImm

__all__ = [
    "use_swizzle",
    "annotate_layout",
    "annotate_safe_value",
    "annotate_l2_hit_ratio",
    "annotate_direct_to_lds",
    "disable_buffer_ops",
    "annotate_padding",
    "annotate_restrict_buffers",
]


def use_swizzle(panel_size: int, order: str = "row", enable: bool = True):
    """Annotate a kernel to use a specific threadblock swizzle pattern."""
    device_func = "rasterization2DRow" if order == "row" else "rasterization2DColumn"
    if not enable:
        return None
    return attr(None, "threadblock_swizzle_pattern", f"tl::{device_func}<{panel_size}>")


def annotate_layout(layout_map: dict):
    """Annotate the layout of the buffer."""
    _layout_map = {}
    for buffer, layout in layout_map.items():
        if is_fragment(buffer):
            assert isinstance(layout, Fragment), f"for Fragment {buffer}, layout must be a Fragment, but got {type(layout)}"
        if isinstance(layout, Layout):
            _layout_map[buffer.data] = layout
        elif isinstance(layout, Callable):
            _layout_map[buffer.data] = Layout(buffer.shape, layout)
        else:
            raise ValueError(f"Invalid layout: {layout}")

    return block_attr({"layout_map": _layout_map})


def annotate_safe_value(safe_value_map: dict):
    """Annotate the safe value of the buffer."""
    _safe_value_map = {}
    for buffer, safe_value in safe_value_map.items():
        _safe_value_map[buffer.data] = safe_value
    return block_attr({"safe_value_map": _safe_value_map})


def annotate_l2_hit_ratio(l2_hit_ratio_map: dict):
    """Annotate the L2 hit ratio of the buffer."""
    _l2_hit_ratio_map = {}
    for buffer, hit_ratio in l2_hit_ratio_map.items():
        assert buffer.scope() == "global", "persistent L2 can only be applied to global buffers"
        _l2_hit_ratio_map[buffer.data] = FloatImm("float32", float(hit_ratio))
    return block_attr({"l2_hit_ratio_map": _l2_hit_ratio_map})


def annotate_direct_to_lds(buffers):
    """Annotate buffers to use direct-to-LDS loading

    This annotation enables direct global-to-LDS memory transfers for specified buffers,
    bypassing VGPR intermediate storage. This is an HCU-specific optimization that
    can improve memory bandwidth utilization for global-to-shared copies.

    Args:
        buffer_map: Either a single buffer, a list of buffers, or a dict mapping buffers to bool.
                    If a list/single buffer is provided, all buffers will be enabled for direct-to-LDS.
                    If a dict is provided, only buffers with True value will use direct-to-LDS.

    Returns:
        block_attr: an block attribute statement

    Example:
        @T.prim_func
        def gemm_kernel(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((K, N), dtype),
                C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(grid_m, grid_n, threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)

                # Enable direct-to-LDS for both shared buffers
                T.annotate_hcu_direct_to_lds([A_shared, B_shared])
                # Or use a dict for finer control:
                # T.annotate_hcu_direct_to_lds({A_shared: True, B_shared: False})

                # Copy from global to shared (will use direct-to-LDS)
                for i, j in T.Parallel(block_M, block_K):
                    A_shared[i, j] = A[bx * block_M + i, j]
                for i, j in T.Parallel(block_K, block_N):
                    B_shared[i, j] = B[i, by * block_N + j]
                # ... rest of kernel

        return gemm_kernel
    """
    _direct_to_lds_map = {}
    if isinstance(buffers, dict):
        for buffer, enabled in buffers.items():
            _direct_to_lds_map[buffer.data] = IntImm("int32", 1 if enabled else 0)
    elif isinstance(buffers, (list, tuple)):
        for buffer in buffers:
            _direct_to_lds_map[buffer.data] = IntImm("int32", 1)
    else:
        _direct_to_lds_map[buffers.data] = IntImm("int32", 1)
    return block_attr({"direct_to_lds": _direct_to_lds_map})


def disable_buffer_ops(*buffers):
    """Disable amd_buffer_load/amd_buffer_store for specified global buffers.

    Use when offset may exceed 2G (int32 limit). Applies to all accesses
    (T.copy, BufferLoad, BufferStore) in the entire kernel.

    Call inside a block (e.g. at the start of T.Kernel body) to annotate.

    Args:
        *buffers: One or more tir.Buffer to disable buffer ops for.

    Example:
        @T.prim_func
        def main(q: T.Tensor[...], k: T.Tensor[...], ...):
            with T.Kernel(...) as (...):
                T.disable_buffer_ops(q, k)
                # ... kernel body
    """
    _disable_map = {}
    for buf in buffers:
        if hasattr(buf, "data") and buf.data is not None:
            data_var = buf.data
            name = getattr(data_var, "name_hint", None) or getattr(data_var, "name", str(data_var))
        else:
            name = getattr(buf, "name_hint", None) or getattr(buf, "name", None)
            if name is None:
                raise TypeError(
                    f"disable_buffer_ops expects Buffer or Var, got {type(buf).__name__}"
                )
        _disable_map[name] = IntImm("int32", 1)
    return block_attr({"disable_buffer_ops_map": _disable_map})


def annotate_padding(padding_map: dict):
    """Annotate the padding of the buffer

    Args:
        padding_map (dict): a dictionary of buffer to padding value

    Returns:
        block_attr: a block attribute

    Example:
        @T.prim_func
        def main(
                A: T.Tensor((M, N), dtype),
                B: T.Tensor((M, N), dtype),
        ):
            # Initialize Kernel Context
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_N), dtype)

                T.annotate_padding({A_shared: pad_value})
                for i, j in T.Parallel(block_M, block_N):
                    A_shared[i, j] = A[by * block_M + i - 10, bx * block_N + j]

                for i, j in T.Parallel(block_M, block_N):
                    B[by * block_M + i, bx * block_N + j] = A_shared[i, j]

        return main
    """
    _padding_map = {}
    for buffer, padding_value in padding_map.items():
        assert buffer.scope() != "global", "padding can not be applied to global buffers"
        _padding_map[buffer.data] = padding_value
    return block_attr({"padding_map": _padding_map})


def annotate_restrict_buffers(*buffers):
    """Mark the given buffer parameters as non-restrict.

    This annotation tells codegen to omit the `__restrict__` qualifier for the
    specified kernel buffer parameters. Use this when two (or more) buffers may
    alias, for example overlapping slices from the same base tensor.

    Example
    -------
    >>> @T.prim_func
    ... def buggy_kernel(x: T.Tensor((N,), T.float32),
    ...                  y: T.Tensor((N,), T.float32)):
    ...     T.annotate_restrict_buffers(x, y)
    ...     with T.Kernel(N, threads=32) as pid:
    ...         y[pid] = x[pid] + 1
    """
    if not buffers:
        return None
    data_vars = []
    for buf in buffers:
        try:
            data_vars.append(buf.data)
        except Exception as e:
            raise TypeError(f"annotate_restrict_buffers expects Buffer arguments, got {type(buf)}") from e
    # Also return as block attribute (root block exists by default) for readability/tools.
    return block_attr({"tl.non_restrict_params": data_vars})
