"""HCU dialect loop constructs and lowering hints."""

from __future__ import annotations

from typing import Any

from tvm import tirx
from tvm.tirx.script.builder import frame

from tilelang.language.loop import Parallel as _common_Parallel

__all__ = ["Parallel"]


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
