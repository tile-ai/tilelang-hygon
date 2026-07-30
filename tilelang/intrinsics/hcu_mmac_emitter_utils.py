"""Shared helpers for HCU MMAC / MLS emitter lowering."""

from __future__ import annotations

from tilelang import _ffi_api
from tilelang.utils.target import get_hcu_arch_string
from tvm import DataType, tir
from tvm.target import Target


def mls_block_mn_k_from_shape(shape, mls_trans: bool) -> tuple[int, int]:
    if mls_trans:
        return int(shape[-2]), int(shape[-1])
    return int(shape[-1]), int(shape[-2])


def mls_block_mn_k_from_region(region: tir.BufferRegion, mls_trans: bool) -> tuple[int, int]:
    """ReadExtent (MN, K) from the last two region dims."""
    ext0 = region.region[-2].extent
    ext1 = region.region[-1].extent
    if isinstance(ext0, tir.IntImm) and isinstance(ext1, tir.IntImm):
        mn0, mn1 = int(ext0.value), int(ext1.value)
    else:
        raise ValueError(f"ldmatrix_mls requires static MN/K tile extents on the last two region dims, got {ext0}, {ext1}")
    if mls_trans:
        return mn0, mn1
    return mn1, mn0


def mls_logical_origin_from_region(region: tir.BufferRegion, mls_trans: bool) -> tuple[int, int]:
    """Logical (origin_mn, origin_k) of a slice inside the full LDS buffer."""
    min0 = region.region[-2].min
    min1 = region.region[-1].min
    if isinstance(min0, tir.IntImm) and isinstance(min1, tir.IntImm):
        o0, o1 = int(min0.value), int(min1.value)
    else:
        raise ValueError(f"ldmatrix_mls requires static last-2 logical origins, got {min0}, {min1}")
    if mls_trans:
        return o0, o1
    return o1, o0


def mls_full_and_read_mn_k(
    src: tir.Buffer | tir.BufferLoad | tir.BufferRegion,
    mls_trans: bool,
) -> tuple[int, int, int, int, int, int]:
    """Return (lds_mn, lds_k, read_mn, read_k, origin_mn, origin_k)."""
    from tilelang.language.utils import get_buffer_region_from_load
    from tvm.ir import Range

    if isinstance(src, tir.Buffer):
        mins = [tir.IntImm("int32", 0) for _ in src.shape]
        region = tir.BufferRegion(src, [Range.from_min_extent(m, e) for m, e in zip(mins, src.shape)])
    elif isinstance(src, tir.BufferRegion):
        region = src
    elif isinstance(src, tir.BufferLoad):
        region = get_buffer_region_from_load(src)
        if region is None:
            raise ValueError(f"ldmatrix_mls cannot derive BufferRegion from {src}")
    else:
        raise TypeError(f"Unsupported mls src type: {type(src)}")

    lds_mn, lds_k = mls_block_mn_k_from_shape(region.buffer.shape, mls_trans)
    read_mn, read_k = mls_block_mn_k_from_region(region, mls_trans)
    origin_mn, origin_k = mls_logical_origin_from_region(region, mls_trans)
    return lds_mn, lds_k, read_mn, read_k, origin_mn, origin_k


def retrieve_mls_lds_base_ptr(src: tir.Buffer | tir.BufferLoad | tir.BufferRegion, access_type: str = "r"):
    """Pointer to full LDS base (+ leading-dim offset only). Last-2 slice mins are NOT applied."""
    from tilelang.language.utils import get_buffer_region_from_load
    from tilelang.utils.language import retrieve_ptr
    from tvm.ir import Range

    if isinstance(src, tir.BufferRegion):
        region = src
    elif isinstance(src, tir.Buffer):
        mins = [tir.IntImm("int32", 0) for _ in src.shape]
        region = tir.BufferRegion(src, [Range.from_min_extent(m, e) for m, e in zip(mins, src.shape)])
    elif isinstance(src, tir.BufferLoad):
        region = get_buffer_region_from_load(src)
        if region is None:
            raise ValueError(f"cannot derive BufferRegion from {src}")
    else:
        raise TypeError(f"Unsupported mls src type: {type(src)}")

    ranges = []
    n = len(region.region)
    for i, r in enumerate(region.region):
        if i + 2 >= n:
            ranges.append(Range.from_min_extent(tir.IntImm(r.min.dtype, 0), r.extent))
        else:
            ranges.append(r)
    return retrieve_ptr(tir.BufferRegion(region.buffer, ranges), access_type)


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
    elem_bits: int,
) -> tuple[int, int]:
    _warp_mn, _warp_k, tile_mn, tile_k = _ffi_api.ComputeMlsWarpPartition(
        bool(mls_trans),
        int(block_mn),
        int(block_k),
        int(block_size),
        target,
        int(elem_bits),
    )
    return int(tile_mn), int(tile_k)


def hcu_mls_ds_read_dtype_str(dtype) -> str:
    """C++ element type for ``tl::mls::ds_read_format_tensor_*`` templates."""
    s = str(dtype).lower()
    if "e4m3" in s:
        return "tl::fp8_t"
    if "e5m2" in s:
        return "tl::bf8_t"
    if "float4_e2m1fn" in s:
        return "tl::pk_fp4_t"
    if "bfloat16" in s or s == "bf16":
        return "bfloat16_t"
    if ("float16" in s or s in ("fp16", "half")) and "float8" not in s:
        return "half_t"
    raise ValueError(f"ldmatrix_mls unsupported dtype: {dtype}")


def build_ds_read_format_tensor_a_template(
    *,
    lds_mn: int,
    lds_k: int,
    read_mn: int,
    read_k: int,
    tile_mn: int,
    tile_k: int,
    warp_m: int,
    warp_k: int,
    mls_trans: bool,
    dtype_str: str,
    target: Target,
) -> str:
    arch = get_hcu_arch_string(target)
    trans = "true" if mls_trans else "false"
    return (
        f"tl::mls::ds_read_format_tensor_a<tl::sequence<{lds_mn}, {lds_k}>, "
        f"tl::sequence<{read_mn}, {read_k}>, tl::sequence<{tile_mn}, {tile_k}>, "
        f"{warp_m}, {warp_k}, {dtype_str}, 1, {trans}, tl::hcu_target_enum::{arch}>"
    )


def build_ds_read_format_tensor_b_template(
    *,
    lds_mn: int,
    lds_k: int,
    read_mn: int,
    read_k: int,
    tile_mn: int,
    tile_k: int,
    total_warp: int,
    warp_n: int,
    warp_k: int,
    mls_trans: bool,
    dtype_str: str,
    target: Target,
) -> str:
    arch = get_hcu_arch_string(target)
    trans = "true" if mls_trans else "false"
    return (
        f"tl::mls::ds_read_format_tensor_b<tl::sequence<{lds_mn}, {lds_k}>, "
        f"tl::sequence<{read_mn}, {read_k}>, tl::sequence<{tile_mn}, {tile_k}>, "
        f"{total_warp}, {warp_n}, {warp_k}, {dtype_str}, 1, {trans}, "
        f"tl::hcu_target_enum::{arch}>"
    )


def min_n_per_warp_for_b(*, b_mls: bool, b_mls_trans: bool, element_bits: int | None = None) -> int:
    """Match ``gemm_mls`` / ``ds_read_format_tensor_b`` MinNPerWarp."""
    if b_mls and (element_bits == 4 or not b_mls_trans):
        return 32
    return 16


def block_col_warps_no_recompute(block_n: int, block_col_warps: int, min_n_per_warp: int) -> int:
    return min(block_col_warps, block_n // min_n_per_warp)


def elem_bits(dtype) -> int:
    return int(DataType(dtype).bits)
