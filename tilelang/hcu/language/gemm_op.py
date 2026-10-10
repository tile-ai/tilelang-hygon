"""HCU dialect wrappers for the common GEMM tile operations."""

from __future__ import annotations

from tvm import arith, tirx

from tilelang._typing import BufferLikeType
from tilelang.language.gemm_op import GemmWarpPolicy, _gemm_blockscaled_impl, _gemm_impl
from tilelang.language.utils import _normalize_annotations

__all__ = ["gemm", "gemm_blockscaled"]


def gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    mbar=None,
    use_tf32: bool = False,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """Common GEMM with the legacy HCU ``k_pack``/``use_tf32`` surface."""

    if mbar is not None:
        raise ValueError("HCU T.gemm does not support the CUDA mbar argument")
    if not isinstance(k_pack, int) or isinstance(k_pack, bool) or k_pack < 1:
        raise ValueError(f"HCU T.gemm k_pack must be a positive integer, got {k_pack!r}")
    ann = _normalize_annotations(annotations)
    if k_pack != 1:
        ann.setdefault("k_pack", k_pack)
    if use_tf32:
        ann.setdefault("tl.hcu.use_tf32", 1)
    return _gemm_impl(
        "tl.tileop.gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        None,
        annotations=ann,
    )


def _require_zero_k_start(k_start) -> None:
    value = int(k_start.value) if isinstance(k_start, tirx.IntImm) else k_start if isinstance(k_start, int) else None
    if value is None or value != 0:
        if value is None and isinstance(k_start, tirx.PrimExpr) and arith.Analyzer().can_prove_equal(k_start, 0):
            return
        raise ValueError("HCU gemm_blockscaled currently supports only k_start=0 with tile-local shared.scale buffers")


def gemm_blockscaled(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    SFA: BufferLikeType,
    SFB: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    *,
    k_start: int | tirx.PrimExpr = 0,
    sf_a_granularity_k: int,
    sf_b_granularity_k: int,
    sf_a_granularity_m: int = 1,
    sf_b_granularity_n: int = 1,
    a_scale_k_major: bool = False,
    b_scale_k_major: bool = False,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """Official block-scaled op with HCU scale-buffer compatibility metadata."""

    _require_zero_k_start(k_start)
    ann = _normalize_annotations(annotations)
    ann.update(
        {
            "tl.hcu.blockscaled": 1,
            # k_pack is part of the official GemmNode contract and is shared
            # by dense and block-scaled GEMM.  Keep the upstream annotation
            # spelling so InitFromDenseArgs populates GemmNode::kPack_.
            "k_pack": int(k_pack),
            "tl.hcu.sf_a_granularity_m": int(sf_a_granularity_m),
            "tl.hcu.sf_b_granularity_n": int(sf_b_granularity_n),
            "tl.hcu.a_scale_k_major": int(a_scale_k_major),
            "tl.hcu.b_scale_k_major": int(b_scale_k_major),
        }
    )
    return _gemm_blockscaled_impl(
        "tl.tileop.gemm_blockscaled",
        A,
        B,
        C,
        SFA,
        SFB,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        None,
        k_start=0,
        sf_a_granularity_k=sf_a_granularity_k,
        sf_b_granularity_k=sf_b_granularity_k,
        annotations=ann,
    )
