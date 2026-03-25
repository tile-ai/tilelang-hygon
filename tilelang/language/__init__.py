"""The language interface for tl programs."""

from typing import Optional, Callable, Dict
# from .parser import *
# now is fully compatible with the upstream
# tir script
# TODO(lei): remove this import once the
# upstream tir script is fully compatible
from tvm.script.parser.tir import *
from . import overrides as _overrides  # noqa: F401
from .tir import (
    prim_func,  # noqa: F401
)
from .tir.ir import *  # noqa: F401
from tilelang.layout import Layout, Fragment  # noqa: F401
from .proxy import (
    ptr,  # noqa: F401
    make_tensor,  # noqa: F401
    Buffer,  # noqa: F401
    Tensor,  # noqa: F401
    StridedTensor,  # noqa: F401
    FragmentBuffer,  # noqa: F401
    SharedBuffer,  # noqa: F401
    LocalBuffer,  # noqa: F401
)
from .parallel import Parallel  # noqa: F401
from .pipeline import Pipelined  # noqa: F401
from .persistent import Persistent  # noqa: F401
from .frame import has_let_value, get_let_value  # noqa: F401
from .math_intrinsics import *  # noqa: F401
from .kernel import (
    Kernel,  # noqa: F401
    KernelLaunchFrame,  # noqa: F401
    get_thread_binding,  # noqa: F401
    get_thread_bindings,  # noqa: F401
    get_block_binding,  # noqa: F401
    get_block_bindings,  # noqa: F401
)
from .warpgroup import ws  # noqa: F401
from .allocate import (
    alloc_var,  # noqa: F401
    alloc_local,  # noqa: F401
    alloc_shared,  # noqa: F401
    alloc_fragment,  # noqa: F401
    alloc_barrier,  # noqa: F401
    alloc_tmem,  # noqa: F401
    alloc_reducer,  # noqa: F401
    alloc_descriptor,  # noqa: F401
)
from .copy import copy, c2d_im2col, matrix_load, ds_read_format  # noqa: F401
from .gemm import GemmWarpPolicy, gemm, gemm_v2  # noqa: F401
from .experimental.gemm_sp import gemm_sp  # noqa: F401
from .fill import fill, clear  # noqa: F401
from .reduce import (
    reduce,  # noqa: F401
    reduce_max,  # noqa: F401
    reduce_min,  # noqa: F401
    reduce_sum,  # noqa: F401
    reduce_abssum,  # noqa: F401
    reduce_absmax,  # noqa: F401
    reduce_warp,  # noqa: F401
    reduce_sum_warp,  # noqa: F401
    cumsum,  # noqa: F401
    finalize_reducer,  # noqa: F401
)
from .print import print  # noqa: F401
from .customize import (
    atomic_max,  # noqa: F401
    atomic_min,  # noqa: F401
    atomic_add,  # noqa: F401
    atomic_addx2,  # noqa: F401
    atomic_addx4,  # noqa: F401
    dp4a,  # noqa: F401
    clamp,  # noqa: F401
    reshape,  # noqa: F401
    view,  # noqa: F401
    atomic_load,  # noqa: F401
    atomic_store,  # noqa: F401
    loop_break,  # noqa: F401
)
from .logical import any_of, all_of  # noqa: F401
from .builtin import *  # noqa: F401

from .utils import index_to_coordinates  # noqa: F401


def symbolic(name: str, dtype: str = "int32"):
    """
    Create a TIR symbolic variable.

    Parameters:
        name (str): Identifier for the variable in generated TIR.
        dtype (str): Data type string for the variable (e.g., "int32"). Defaults to "int32".

    Returns:
        tir.Var: A TIR variable with the given name and dtype for use in TIR/TensorIR kernels.
    """
    return tir.Var(name, dtype)


def use_swizzle(panel_size: int, order: str = "row", enable: bool = True):
    # If order is row, use rasterization2DRow, otherwise use rasterization2DColumn
    # The panel size is the number of threads in a warp
    # Use to improve the L2 Cache Locality
    device_func = ("rasterization2DRow" if order == "row" else "rasterization2DColumn")
    return attr(None, "threadblock_swizzle_pattern",
                f"tl::{device_func}<{panel_size}>") if enable else None


def annotate_layout(layout_map: Dict):
    """Annotate the layout of the buffer

    Args:
        layout_map (Dict): a dictionary of buffer to layout

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

                T.annotate_layout({A_shared: layout})
                for i, j in T.Parallel(block_M, block_N):
                    A_shared[i, j] = A[by * block_M + i, bx * block_N + j]

                for i, j in T.Parallel(block_M, block_N):
                    B[by * block_M + i, bx * block_N + j] = A_shared[i, j]

        return main
    """
    # layout_map is a dictionary of buffer to layout
    _layout_map = {}
    for buffer, layout in layout_map.items():
        if isinstance(layout, Layout):
            _layout_map[buffer.data] = layout
        elif isinstance(layout, Callable):
            _layout_map[buffer.data] = Layout(buffer.shape, layout)
        else:
            raise ValueError(f"Invalid layout: {layout}")

    return block_attr({"layout_map": _layout_map})


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
    from tvm.tir import IntImm

    # Normalize input to dict format
    _direct_to_lds_map = {}
    if isinstance(buffers, dict):
        # Already a dict, use as-is
        for buffer, enabled in buffers.items():
            _direct_to_lds_map[buffer.data] = IntImm("int32", 1 if enabled else 0)
    elif isinstance(buffers, (list, tuple)):
        # List of buffers, enable all
        for buffer in buffers:
            _direct_to_lds_map[buffer.data] = IntImm("int32", 1)
    else:
        # Single buffer, enable it
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
    from tvm.tir import IntImm

    # Use param name for robust matching across passes (Var may be renamed)
    # Support both Buffer (buf.data) and Var (e.g. param in T.Kernel block)
    _disable_map = {}
    for buf in buffers:
        if hasattr(buf, "data") and buf.data is not None:
            # Buffer: use data Var's name
            data_var = buf.data
            name = getattr(data_var, "name_hint", None) or getattr(data_var, "name", str(data_var))
        else:
            # Var (param): use name directly
            name = getattr(buf, "name_hint", None) or getattr(buf, "name", None)
            if name is None:
                raise TypeError(
                    f"disable_buffer_ops expects Buffer or Var, got {type(buf).__name__}"
                )
        _disable_map[name] = IntImm("int32", 1)
    return block_attr({"disable_buffer_ops_map": _disable_map})


def annotate_padding(padding_map: Dict):
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
    # padding_map is a dictionary of buffer to padding value
    _padding_map = {}
    for buffer, padding_value in padding_map.items():
        # assert not global
        assert buffer.scope() != "global", "padding can not be applied to global buffers"
        _padding_map[buffer.data] = padding_value
    return block_attr({"padding_map": _padding_map})


def annotate_l2_hit_ratio(l2_hit_ratio_map: Dict):
    """Annotate the L2 hit ratio of the buffer, detailed explanation please refer to:
    https://docs.nvidia.com/cuda/cuda-c-programming-guide/#l2-policy-for-persisting-accesses

    Args:
        l2_hit_ratio_map (dict): a dictionary of buffer to L2 hit ratio value
    Example:
        # 0.5 is the hit ratio
        T.annotate_l2_hit_ratio({A: 0.5})
    """
    _l2_hit_ratio_map = {}
    for buffer, hit_ratio in l2_hit_ratio_map.items():
        assert buffer.scope() == "global", "persistent L2 can only be applied to global buffers"
        _l2_hit_ratio_map[buffer.data] = float(hit_ratio)
    return block_attr({"l2_hit_ratio_map": _l2_hit_ratio_map})


def import_source(source: Optional[str] = None):
    # source is the source code to be imported
    return block_attr({"pragma_import_c": source}) if source is not None else None
