"""Shared helpers for HCU MMAC / MLS emitter lowering."""

from __future__ import annotations

from tilelang import _ffi_api
from tilelang.hcu.target import get_hcu_arch_string
from tvm import DataType, tirx
from tvm.target import Target


def mls_block_mn_k_from_shape(shape, mls_trans: bool) -> tuple[int, int]:
    if mls_trans:
        return int(shape[-2]), int(shape[-1])
    return int(shape[-1]), int(shape[-2])


def mls_block_mn_k_from_region(region: tirx.BufferRegion, mls_trans: bool) -> tuple[int, int]:
    """ReadExtent (MN, K) from the last two region dims."""
    ext0 = region.region[-2].extent
    ext1 = region.region[-1].extent
    if isinstance(ext0, tirx.IntImm) and isinstance(ext1, tirx.IntImm):
        mn0, mn1 = int(ext0.value), int(ext1.value)
    else:
        raise ValueError(f"ldmatrix_mls requires static MN/K tile extents on the last two region dims, got {ext0}, {ext1}")
    if mls_trans:
        return mn0, mn1
    return mn1, mn0


def mls_logical_origin_from_region(region: tirx.BufferRegion, mls_trans: bool) -> tuple[int, int]:
    """Logical (origin_mn, origin_k) of a slice inside the full LDS buffer."""
    min0 = region.region[-2].min
    min1 = region.region[-1].min
    if isinstance(min0, tirx.IntImm) and isinstance(min1, tirx.IntImm):
        o0, o1 = int(min0.value), int(min1.value)
    else:
        raise ValueError(f"ldmatrix_mls requires static last-2 logical origins, got {min0}, {min1}")
    if mls_trans:
        return o0, o1
    return o1, o0


def mls_full_and_read_mn_k(
    src: tirx.Buffer | tirx.BufferLoad | tirx.BufferRegion,
    mls_trans: bool,
) -> tuple[int, int, int, int, int, int]:
    """Return (lds_mn, lds_k, read_mn, read_k, origin_mn, origin_k)."""
    from tilelang.language.utils import get_buffer_region_from_load
    from tvm.ir import Range

    if isinstance(src, tirx.Buffer):
        mins = [tirx.IntImm("int32", 0) for _ in src.shape]
        region = tirx.BufferRegion(src, [Range.from_min_extent(m, e) for m, e in zip(mins, src.shape)])
    elif isinstance(src, tirx.BufferRegion):
        region = src
    elif isinstance(src, tirx.BufferLoad):
        region = get_buffer_region_from_load(src)
        if region is None:
            raise ValueError(f"ldmatrix_mls cannot derive BufferRegion from {src}")
    else:
        raise TypeError(f"Unsupported mls src type: {type(src)}")

    lds_mn, lds_k = mls_block_mn_k_from_shape(region.buffer.shape, mls_trans)
    read_mn, read_k = mls_block_mn_k_from_region(region, mls_trans)
    origin_mn, origin_k = mls_logical_origin_from_region(region, mls_trans)
    return lds_mn, lds_k, read_mn, read_k, origin_mn, origin_k


def mls_packed_tile_elems(
    block_mn: int,
    block_k: int,
    tile_mn: int,
    tile_k: int,
    mls_trans: bool,
    dtype,
    lds_bits: int | None = None,
) -> int:
    """Per-stage element stride of a packed MLS dst tile; 0 when contiguous.

    Delegates to the C++ ``MlsPackedTileBytes`` so the ds_read consumer, the
    MLS producer and the shared-memory allocation cannot drift apart again --
    they did, and every ``num_stages > 1`` tile whose dst took a packed branch
    (fp8 trans tile (16,128), i.e. block_K >= 256 at bn16) silently read stage
    >= 1 from the wrong LDS address.
    """
    dt = DataType(dtype)
    return int(
        _ffi_api.MlsPackedTileElems(
            int(block_mn),
            int(block_k),
            int(tile_mn),
            int(tile_k),
            bool(mls_trans),
            int(dt.bits),
            int(dt.lanes),
            # Matches the C++ producer/allocator, which both pass
            # ``dst->dtype.bits()`` as lds_physical_bits.
            int(lds_bits) if lds_bits is not None else int(dt.bits),
        )
    )


def retrieve_mls_lds_base_ptr(
    src: tirx.Buffer | tirx.BufferLoad | tirx.BufferRegion,
    access_type: str = "r",
    *,
    tile_mn: int | None = None,
    tile_k: int | None = None,
    mls_trans: bool | None = None,
    lds_bits: int | None = None,
):
    """Pointer to full LDS base (+ leading-dim offset only). Last-2 slice mins are NOT applied.

    The leading dims are the pipeline stage (and any outer tile dims).  Their
    stride is the *packed* MLS tile size when the dst layout is packed, not the
    logical ``prod(shape[-2:])`` -- pass ``tile_mn``/``tile_k``/``mls_trans``
    so the stride matches what the producer and the allocator use.
    """
    from tilelang.language.utils import get_buffer_region_from_load
    from tilelang.utils.language import retrieve_ptr
    from tvm.ir import Range

    if isinstance(src, tirx.BufferRegion):
        region = src
    elif isinstance(src, tirx.Buffer):
        mins = [tirx.IntImm("int32", 0) for _ in src.shape]
        region = tirx.BufferRegion(src, [Range.from_min_extent(m, e) for m, e in zip(mins, src.shape)])
    elif isinstance(src, tirx.BufferLoad):
        region = get_buffer_region_from_load(src)
        if region is None:
            raise ValueError(f"cannot derive BufferRegion from {src}")
    else:
        raise TypeError(f"Unsupported mls src type: {type(src)}")

    n = len(region.region)
    buf = region.buffer
    if n > 2 and tile_mn is not None and tile_k is not None and mls_trans is not None:
        block_mn, block_k = mls_block_mn_k_from_shape(buf.shape, bool(mls_trans))
        packed_elems = mls_packed_tile_elems(
            block_mn, block_k, int(tile_mn), int(tile_k), bool(mls_trans), buf.dtype, lds_bits
        )
        if packed_elems > 0:
            # Flat leading index, scaled by the packed per-stage stride.
            idx = None
            stride = 1
            for i in range(n - 3, -1, -1):
                term = region.region[i].min * stride
                idx = term if idx is None else idx + term
                stride = stride * int(buf.shape[i])
            if idx is None:
                idx = 0
            return buf.access_ptr(access_type, offset=idx * packed_elems)

    ranges = []
    for i, r in enumerate(region.region):
        if i + 2 >= n:
            ranges.append(Range.from_min_extent(tirx.IntImm(r.min.dtype, 0), r.extent))
        else:
            ranges.append(r)
    return retrieve_ptr(tirx.BufferRegion(buf, ranges), access_type)


def check_mls_slice_aligned_to_tile(
    *,
    origin_mn: int,
    origin_k: int,
    read_mn: int,
    read_k: int,
    tile_mn: int,
    tile_k: int,
    what: str = "ldmatrix_mls",
) -> None:
    if origin_mn % tile_mn != 0:
        raise ValueError(f"{what}: origin_mn={origin_mn} must be divisible by mls_tile_mn={tile_mn}")
    if origin_k % tile_k != 0:
        raise ValueError(f"{what}: origin_k={origin_k} must be divisible by mls_tile_k={tile_k}")
    if read_mn % tile_mn != 0:
        raise ValueError(f"{what}: read_mn={read_mn} must be divisible by mls_tile_mn={tile_mn}")
    if read_k % tile_k != 0:
        raise ValueError(f"{what}: read_k={read_k} must be divisible by mls_tile_k={tile_k}")


def compute_mls_tiles(
    mls_trans: bool,
    block_mn: int,
    block_k: int,
    block_size: int,
    target: Target,
    dtype,
) -> tuple[int, int]:
    dtype = DataType(dtype)
    src_bits = 4 if dtype.is_float4() else int(dtype.bits)
    requested_lds_bits = int(dtype.bits)
    _warp_mn, _warp_k, tile_mn, tile_k = _ffi_api.ComputeMlsWarpPartition(
        bool(mls_trans),
        int(block_mn),
        int(block_k),
        int(block_size),
        target,
        src_bits,
        requested_lds_bits,
    )
    return int(tile_mn), int(tile_k)


def hcu_mls_ds_read_dtype_str(dtype) -> str:
    """C++ element type for ``tl::mls::ds_read_format_tensor_*`` templates."""
    s = str(dtype).lower()
    if "e4m3" in s:
        return "tl::fp8_t"
    if "e5m2" in s:
        return "tl::bf8_t"
    if DataType(dtype).is_float4():
        return "tl::pk_fp4_t"
    if "bfloat16" in s or s == "bf16":
        return "bfloat16_t"
    if ("float16" in s or s in ("fp16", "half")) and "float8" not in s:
        return "half_t"
    if "tfloat32" in s:
        return "int"
    if s == "int32":
        return "int"
    if s == "float32":
        return "float"
    raise ValueError(f"ldmatrix_mls unsupported dtype: {dtype}")


def build_ds_read_format_tensor_a_template(
    *,
    lds_block_mn: int,
    lds_block_k: int,
    ds_read_mn: int,
    ds_read_k: int,
    tile_mn: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    mls_trans: bool,
    dtype_str: str,
    target: Target,
    target_dtype_str: str | None = None,
    lds_bits: int | None = None,
    reg_bits: int | None = None,
) -> str:
    arch = get_hcu_arch_string(target)
    trans = "true" if mls_trans else "false"
    target_type = f", {target_dtype_str}" if target_dtype_str is not None else ""
    if lds_bits is not None:
        if target_dtype_str is None:
            raise ValueError("lds_bits requires target_dtype_str")
        if reg_bits is None:
            raise ValueError("lds_bits requires reg_bits")
        target_type += f", {int(lds_bits)}, {int(reg_bits)}"
    return (
        f"tl::mls::ds_read_format_tensor_a<tl::sequence<{lds_block_mn}, {lds_block_k}>, "
        f"tl::sequence<{ds_read_mn}, {ds_read_k}>, tl::sequence<{tile_mn}, {tile_k}>, "
        f"{warp_m}, {warp_n}, {warp_k}, {dtype_str}, 1, {trans}, tl::hcu_target_enum::{arch}{target_type}>"
    )


def build_ds_read_format_tensor_b_template(
    *,
    lds_block_mn: int,
    lds_block_k: int,
    ds_read_mn: int,
    ds_read_k: int,
    tile_mn: int,
    tile_k: int,
    total_warp: int,
    warp_n: int,
    warp_k: int,
    mls_trans: bool,
    dtype_str: str,
    target: Target,
    target_dtype_str: str | None = None,
    lds_bits: int | None = None,
    reg_bits: int | None = None,
    min_n_per_warp: int = 0,
) -> str:
    arch = get_hcu_arch_string(target)
    trans = "true" if mls_trans else "false"
    target_type = f", {target_dtype_str}" if target_dtype_str is not None else ""
    if lds_bits is not None:
        if target_dtype_str is None:
            raise ValueError("lds_bits requires target_dtype_str")
        if reg_bits is None:
            raise ValueError("lds_bits requires reg_bits")
        target_type += f", {int(lds_bits)}, {int(reg_bits)}"
    elif target_dtype_str is not None and min_n_per_warp:
        target_type += f", tl::mls::mls_elem_bits_v<{dtype_str}>, tl::mls::mls_elem_bits_v<{target_dtype_str}>"
    elif min_n_per_warp:
        # ExtraMinNPerWarp follows three defaulted template parameters, so spell
        # those defaults explicitly when no widened target type was requested.
        target_type = f", {dtype_str}, tl::mls::mls_elem_bits_v<{dtype_str}>, tl::mls::mls_elem_bits_v<{dtype_str}>"
    extra_floor = f", {int(min_n_per_warp)}" if min_n_per_warp else ""
    return (
        f"tl::mls::ds_read_format_tensor_b<tl::sequence<{lds_block_mn}, {lds_block_k}>, "
        f"tl::sequence<{ds_read_mn}, {ds_read_k}>, tl::sequence<{tile_mn}, {tile_k}>, "
        f"{total_warp}, {warp_n}, {warp_k}, {dtype_str}, 1, {trans}, "
        f"tl::hcu_target_enum::{arch}{target_type}{extra_floor}>"
    )


def build_ds_read_format_tensor_b_linear_template(
    *,
    block_n: int,
    block_k: int,
    total_warp: int,
    warp_n: int,
    dtype_str: str,
) -> str:
    return f"tl::mls::ds_read_format_tensor_b_linear<tl::sequence<{block_n}, {block_k}>, {total_warp}, {warp_n}, {dtype_str}>"


def build_b_vgpr_direct_load_template(
    *,
    block_n: int,
    block_k: int,
    total_warp: int,
    warp_n: int,
    warp_k: int,
    min_n_per_warp: int,
    warp_cols: int,
) -> str:
    """B-operand global->VGPR direct load (hipblas shape): same B_local layout as
    ds_read_format_tensor_b, but reads straight from global (8B dwordx2 per lane),
    no LDS round-trip. See tl::mls::b_vgpr_direct_load. ``warp_cols`` is the mmac
    warp_cols (n-panels of 16 rows per n-warp); the run order must match the
    mmac's ``B_local[(ki*WarpCols+j)*8]``."""
    return (
        f"tl::mls::b_vgpr_direct_load<{total_warp}, {warp_n}, {warp_k}, "
        f"{block_n}, {block_k}, {int(min_n_per_warp)}, {int(warp_cols)}>"
    )


def build_b_vgpr_direct_load_stage_template(
    *,
    block_n: int,
    total_warp: int,
    warp_n: int,
    warp_k: int,
    min_n_per_warp: int,
    warp_cols: int,
) -> str:
    """One 32-wide K-slice of the B-operand global->VGPR direct load, for the
    software-pipelined path (hipblas LoopBeginL_1 shape): each stage issues one
    ki-slice (ReadK=32) into a ping-pong VGPR buffer while the mmac chain for
    the previous slice creates the s_waitcnt (vmcnt) window. The per-slice dest
    layout matches one ki-slice of the mmac's B fragment
    (``warp_cols * local_size_b`` elems, consumed with stride 0). ``k_offset``
    is in elements along K (ki * 32 for fp8). See
    tl::mls::b_vgpr_direct_load_stage."""
    return (
        f"tl::mls::b_vgpr_direct_load_stage<{total_warp}, {warp_n}, {warp_k}, "
        f"{block_n}, 32, {int(min_n_per_warp)}, {int(warp_cols)}>"
    )


def block_col_warps_no_recompute(block_n: int, block_col_warps: int, min_n_per_warp: int) -> int:
    return min(block_col_warps, block_n // min_n_per_warp)


def scale_format_min_n_per_warp(scale_format_b: int | None) -> int:
    """Intrinsic ScaleB MN-atom floor (format ids match ``ScaleFormat``)."""
    if scale_format_b == 5:  # MN4
        return 64
    if scale_format_b in (3, 4):  # K2MN2 / MN2
        return 32
    return 16


def default_min_n_per_warp_for_b(*, b_from_mls: bool, b_mls_trans: bool, element_bits: int, scale_format_b: int | None = None) -> int:
    """Default final N floor for direct emitter construction."""
    mls_floor = 32 if b_from_mls and (element_bits == 4 or not b_mls_trans) else 16
    return max(mls_floor, scale_format_min_n_per_warp(scale_format_b))


def elem_bits(dtype) -> int:
    return int(DataType(dtype).bits)


def is_f8f6f4_operand_dtype(dtype) -> bool:
    dtype_str = str(dtype)
    return DataType(dtype).is_float4() or "float8_e4m3" in dtype_str or "float8_e5m2" in dtype_str


def hcu_mls_lds_bits(dtype, *, mls_trans: bool, tile_mn: int, tile_k: int, target: Target) -> int:
    """Physical LDS bits written by the selected MLS atom."""
    from tilelang.hcu.target import get_hcu_arch_string

    bits = elem_bits(dtype)
    if DataType(dtype).is_float4_e2m1_unpacked():
        return 8
    if bits != 4:
        return bits
    arch = get_hcu_arch_string(target)
    # The 64x16_fp4 atom pads fp4 to b8 while writing LDS. Other fp4 MLS
    # atoms keep LDS packed as b4; ds_read_matrix_padbyte may later expand
    # the register view to b8 when RegBits=8.
    if arch in ("gfx92a", "gfx946") and (
        (not mls_trans and tile_mn == 64 and tile_k == 16) or (mls_trans and tile_mn == 16 and tile_k == 64)
    ):
        return 8
    return bits


def hcu_mmac_k_dim(target: Target, element_bits: int, *, use_tf32: bool = False) -> int:
    """Per-instruction MMAC K along reduction axis (arch/dtype specific)."""
    from tilelang.hcu.target import get_hcu_arch_string, target_is_hcu

    bits = int(element_bits)
    if bits == 16:
        return 16
    if bits == 8:
        return 32
    if bits == 4:
        return 64
    if bits == 32:
        if use_tf32:
            return 8
        if target_is_hcu(target):
            mcpu = get_hcu_arch_string(target)
            if mcpu == "gfx938":
                return 8
            if mcpu in ("gfx92a", "gfx946"):
                return 4
        return 8
    raise ValueError(f"Unsupported element_bits for HCU MMAC: {element_bits}")


def hcu_mmac_k_dim_for_operand(target: Target, dtype, *, operand_mode: str = "native", use_tf32: bool = False) -> int:
    """Per-instruction MMAC K for a logical operand dtype under a selected operand mode."""
    dtype = DataType(dtype)
    if operand_mode == "f8f6f4" and is_f8f6f4_operand_dtype(dtype):
        return 32
    return hcu_mmac_k_dim(target, dtype.bits, use_tf32=use_tf32)
