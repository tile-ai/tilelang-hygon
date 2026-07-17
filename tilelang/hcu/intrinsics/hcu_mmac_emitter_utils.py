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
    ext0 = region.region[-2].extent
    ext1 = region.region[-1].extent
    if isinstance(ext0, tirx.IntImm) and isinstance(ext1, tirx.IntImm):
        mn0, mn1 = int(ext0.value), int(ext1.value)
    else:
        raise ValueError(f"ldmatrix_mls requires static MN/K tile extents on the last two region dims, got {ext0}, {ext1}")
    if mls_trans:
        return mn0, mn1
    return mn1, mn0


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
    if "bfloat16" in s or s == "bf16":
        return "bfloat16_t"
    if ("float16" in s or s in ("fp16", "half")) and "float8" not in s:
        return "half_t"
    raise ValueError(f"ldmatrix_mls unsupported dtype: {dtype}")


def build_ds_read_format_tensor_a_template(
    *,
    block_mn: int,
    block_k: int,
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
        f"tl::mls::ds_read_format_tensor_a<tl::sequence<{block_mn}, {block_k}>, "
        f"tl::sequence<{tile_mn}, {tile_k}>, {warp_m}, {warp_k}, {dtype_str}, 1, "
        f"{trans}, tl::hcu_target_enum::{arch}>"
    )


def build_ds_read_format_tensor_b_template(
    *,
    block_mn: int,
    block_k: int,
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
        f"tl::mls::ds_read_format_tensor_b<tl::sequence<{block_mn}, {block_k}>, "
        f"tl::sequence<{tile_mn}, {tile_k}>, {total_warp}, {warp_n}, {warp_k}, "
        f"{dtype_str}, 1, {trans}, tl::hcu_target_enum::{arch}>"
    )


def min_n_per_warp_for_b(*, b_mls: bool, b_mls_trans: bool) -> int:
    """Match ``gemm_mls`` / ``ds_read_format_tensor_b`` MinNPerWarp."""
    if b_mls and not b_mls_trans:
        return 32
    return 16


def block_col_warps_no_recompute(block_n: int, block_col_warps: int, min_n_per_warp: int) -> int:
    return min(block_col_warps, block_n // min_n_per_warp)


def elem_bits(dtype) -> int:
    return int(DataType(dtype).bits)


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
