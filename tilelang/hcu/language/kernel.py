"""HCU dialect of ``T.Kernel``: the common launch plus HCU annotations."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.kernel import KernelLaunchFrame, kernel_launch_factory, launch_kernel

__all__ = ["Kernel"]


@kernel_launch_factory
def Kernel(
    *blocks: int | tirx.PrimExpr,
    threads: int | list[int] | tuple[int, ...] | None = None,
    prelude: str | None = None,
) -> KernelLaunchFrame:
    """Construct an HCU workgroup launch.

    ``threads`` and ``prelude`` are recorded as launch annotations and are
    materialized by the HCU backend pipeline.
    """
    return launch_kernel(blocks, threads=threads, prelude=prelude)
