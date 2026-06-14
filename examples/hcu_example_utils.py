"""Cross-platform helpers for TileLang examples (HCU-only overrides inside)."""

from __future__ import annotations

from typing import Literal

import torch
import tvm
from tilelang.utils import determine_fp8_type
from tilelang.utils.target import determine_target, target_get_warp_size, target_is_hcu


def _on_hcu() -> bool:
    if torch.version.hip is None:
        return False
    return target_is_hcu(determine_target("auto", return_object=True))


def warp_size() -> int:
    if _on_hcu():
        return target_get_warp_size(tvm.target.Target("hip"))
    return 32


def block_threads(num_warps: int = 1) -> int:
    """Full wavefront/warp thread count; widens to 64 only on HCU."""
    if _on_hcu():
        return target_get_warp_size(tvm.target.Target("hip")) * num_warps
    return 32 * num_warps


def uses_mmac_intrinsic() -> bool:
    """True when the manual GEMM intrinsic example should use HCU MMAC."""
    return _on_hcu()


def _ocp_fp8_tl_dtype(fp8_format: Literal["e4m3", "e5m2"] = "e4m3"):
    import tilelang.language as T

    if fp8_format == "e4m3":
        return T.float8_e4m3fn
    return T.float8_e5m2


def _ocp_fp8_torch_dtype(fp8_format: Literal["e4m3", "e5m2"] = "e4m3") -> torch.dtype:
    if fp8_format == "e4m3":
        return torch.float8_e4m3fn
    return torch.float8_e5m2


def fp8_tl_dtype(fp8_format: Literal["e4m3", "e5m2"] = "e4m3"):
    """TileLang FP8 dtype for examples (OCP on HCU, ``determine_fp8_type`` elsewhere)."""
    if _on_hcu():
        return _ocp_fp8_tl_dtype(fp8_format)
    return determine_fp8_type(fp8_format)


def fp8_torch_dtype(fp8_format: Literal["e4m3", "e5m2"] = "e4m3") -> torch.dtype:
    if _on_hcu():
        return _ocp_fp8_torch_dtype(fp8_format)
    return determine_fp8_type(fp8_format).as_torch()


def make_fp8_gemm_inputs(
    m: int,
    n_or_k: int,
    torch_dtype: torch.dtype,
    *,
    wide: bool = False,
) -> torch.Tensor:
    """Bounded FP8 inputs for GEMM checks (avoids fp8 saturation from large randn)."""
    if wide:
        t = torch.rand(m, n_or_k, device="cuda", dtype=torch.float16)
        return (100 * (2 * t - 1)).to(torch_dtype)
    t = torch.rand(m, n_or_k, device="cuda", dtype=torch.float16)
    return (t * 0.2 - 0.1).to(torch_dtype)


def gemm_tile_config(
    *,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    thread_num: int = 128,
    enable_rasteration: bool = True,
) -> dict:
    """GEMM tile config; applies HCU LDS-safe overrides only on HCU."""
    if _on_hcu():
        return {
            "block_M": 64,
            "block_N": 64,
            "block_K": 32,
            "num_stages": 2,
            "thread_num": 128,
            "enable_rasteration": False,
        }
    return {
        "block_M": block_M,
        "block_N": block_N,
        "block_K": block_K,
        "num_stages": num_stages,
        "thread_num": thread_num,
        "enable_rasteration": enable_rasteration,
    }


def gemm_autotune_seed_config() -> dict | None:
    """HCU seed config for GEMM autotune, or None to use platform defaults."""
    if _on_hcu():
        return gemm_tile_config()
    return None


def gemm_fp8_output_dtype(quant_dtype, accum_dtype):
    """Kernel output dtype for FP8 GEMM examples (fp32 accum buffer on HCU only)."""
    if _on_hcu():
        return accum_dtype
    return quant_dtype


def cast_gemm_fp8_result(c: torch.Tensor, quant_dtype) -> torch.Tensor:
    """Cast FP8 GEMM kernel result for compare (HCU fp32-accum path only)."""
    import tilelang.language as T

    torch_dtype = T.dtype(quant_dtype).as_torch()
    if _on_hcu():
        return c.to(torch_dtype)
    return c


def splitk_tile_config(
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
) -> tuple[int, int, int]:
    """Split-K GEMM tile sizes; shrinks tiles on HCU to fit LDS."""
    if _on_hcu():
        h = gemm_tile_config()
        return h["block_M"], h["block_N"], h["block_K"]
    return block_M, block_N, block_K


def flash_block_mn(
    d_head: int = 128,
    block_m: int = 128,
    block_n: int = 128,
) -> tuple[int, int]:
    """Flash-attention style (M, N) tile sizes within LDS budget."""
    if _on_hcu():
        return 64, 32
    return block_m, block_n


def gqa_decode_kernel_config(
    block_n: int = 128,
    block_h: int = 64,
    num_stages: int = 1,
    threads: int = 128,
) -> tuple[int, int, int, int]:
    """GQA decode kernel tile / pipeline settings (HCU overrides only)."""
    if _on_hcu():
        return 128, 64, 0, 256
    return block_n, block_h, num_stages, threads


def gemm_swizzle_panel_size() -> int | None:
    """``T.use_swizzle`` panel size, or None to skip swizzle (HCU only)."""
    if _on_hcu():
        return None
    return 10


def gdn_block_dk(block_dk: int) -> int:
    if _on_hcu():
        return min(block_dk, 32)
    return block_dk


def gdn_block_dv(block_dv: int) -> int:
    if _on_hcu():
        return min(block_dv, 16)
    return block_dv


def shared_swizzle_layout(shared_buf, major_pack: int = 1):
    """Swizzle for ``T.gemm`` shared operands (HCU layout only on HCU)."""
    import tilelang

    if _on_hcu():
        from tilelang.layout.swizzle import make_hcu_swizzled_layout

        return make_hcu_swizzled_layout(shared_buf, major_pack=major_pack)
    return tilelang.layout.make_swizzled_layout(shared_buf)


def mmac_intrinsic_swizzle_layout(shared_buf, vec_size: int = 8):
    """Swizzle for manual HCU matrix-core intrinsic emitters."""
    from tilelang.intrinsics import make_mmac_swizzle_layout

    return make_mmac_swizzle_layout(shared_buf, vecSize=vec_size)


def jit_pass_configs(extra: dict | None = None) -> dict:
    """JIT pass configs; aggressive smem merge on HCU only."""
    from tilelang import PassConfigKey

    out = dict(extra or {})
    if _on_hcu():
        out[PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE] = True
    return out
