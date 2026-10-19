"""HCU dialect loop constructs and lowering hints."""

from __future__ import annotations

from typing import Any

from tvm import tirx
from tvm.tirx.script.builder import frame

from tilelang.language.loop import Parallel as _common_Parallel
from tilelang.language.loop import Pipelined as _common_Pipelined

__all__ = ["Parallel", "Pipelined"]


def Parallel(
    *extents: int | tirx.PrimExpr,
    coalesced_width: int | None = None,
    loop_layout: Any | None = None,
    prefer_async: bool | None = None,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
    """Construct a parallel loop with the HCU async-copy preference hint.

    ``prefer_async`` is recorded as ``parallel_prefer_async`` and consumed by
    the HCU MLS dependency analysis, pipeline planning, and async-copy
    injection passes.  Explicit entries in ``annotations`` take precedence.
    """

    ann: dict[str, Any] = dict(annotations) if annotations is not None else {}
    if prefer_async is not None:
        ann.setdefault("parallel_prefer_async", prefer_async)
    return _common_Parallel(
        *extents,
        coalesced_width=coalesced_width,
        loop_layout=loop_layout,
        annotations=ann or None,
    )


def _loop_requests_register_pipeline(ann: dict[str, Any]) -> bool:
    stages = ann.get("num_register_stages")
    if stages is not None:
        return int(stages) > 1
    return bool(ann.get("enable_register_pipeline", False))


def Pipelined(
    start: int | tirx.PrimExpr,
    stop: int | tirx.PrimExpr | None = None,
    num_stages: int = 0,
    order: list[int] | None = None,
    stage: list[int] | None = None,
    annotations: dict[str, Any] | None = None,
    *,
    register_stages: int | None = None,
    warp_divergence: bool | None = None,
) -> frame.ForFrame:
    """Construct a pipelined loop with typed HCU pipeline hints.

    ``register_stages`` requests a register-resident GEMM K loop and sets the
    rotation depth (at least two); it is recorded as ``num_register_stages``
    and read by the register-pipeline planner.  The legacy
    ``enable_register_pipeline`` annotation still asks for the default depth.

    ``warp_divergence`` splits a register-pipelined K loop across two warp
    groups and is recorded as ``enable_warp_divergence``.  It only applies to
    a loop that requests a register pipeline, so requesting it without one is
    rejected here instead of silently doing nothing.

    Explicit entries in ``annotations`` keep working; a typed argument that
    conflicts with the matching annotation raises ``ValueError``.
    """

    ann: dict[str, Any] = dict(annotations) if annotations is not None else {}
    if register_stages is not None:
        if type(register_stages) is not int or register_stages < 2:
            raise ValueError(f"register_stages must be an int >= 2, but got {register_stages!r}.")
        existing = ann.get("num_register_stages")
        if existing is not None and int(existing) != register_stages:
            raise ValueError(f"register_stages conflicts with annotations['num_register_stages'] ({register_stages} vs {existing}).")
        ann["num_register_stages"] = register_stages
    if warp_divergence is not None:
        if not isinstance(warp_divergence, bool):
            raise TypeError(f"warp_divergence must be a bool, but got {type(warp_divergence)}.")
        if warp_divergence and not _loop_requests_register_pipeline(ann):
            raise ValueError("warp_divergence requires a register pipeline; pass register_stages (>= 2) on the same loop.")
        existing = ann.get("enable_warp_divergence")
        if existing is not None and bool(existing) != warp_divergence:
            raise ValueError(f"warp_divergence conflicts with annotations['enable_warp_divergence'] ({warp_divergence} vs {existing}).")
        ann["enable_warp_divergence"] = 1 if warp_divergence else 0
    return _common_Pipelined(
        start,
        stop=stop,
        num_stages=num_stages,
        order=order,
        stage=stage,
        annotations=ann or None,
    )
