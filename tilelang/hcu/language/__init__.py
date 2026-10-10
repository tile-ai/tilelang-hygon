"""HCU language dialect: common TileLang plus HCU-owned extensions."""

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from .allocate import *  # noqa: F401,F403
from .allocate import __all__ as _ALLOCATE_ALL
from .annotations import *  # noqa: F401,F403
from .annotations import __all__ as _ANNOTATION_ALL
from .copy_op import *  # noqa: F401,F403
from .copy_op import __all__ as _COPY_ALL
from .gemm_op import *  # noqa: F401,F403
from .gemm_op import __all__ as _GEMM_ALL
from .intrinsics import *  # noqa: F401,F403
from .intrinsics import __all__ as _INTRINSIC_ALL
from .kernel import *  # noqa: F401,F403
from .kernel import __all__ as _KERNEL_ALL
from .loop import *  # noqa: F401,F403
from .loop import __all__ as _LOOP_ALL
from .reduce_op import *  # noqa: F401,F403
from .reduce_op import __all__ as _REDUCE_ALL
from .scale_view import (
    ScaleFormat as ScaleFormat,
    ScaleView as ScaleView,
    scale_identity as scale_identity,
    scale_k2_interleaved as scale_k2_interleaved,
    scale_k4_interleaved as scale_k4_interleaved,
    scale_k2mn2_interleaved as scale_k2mn2_interleaved,
    scale_mn2_interleaved as scale_mn2_interleaved,
    scale_mn4_interleaved as scale_mn4_interleaved,
    scale_view as scale_view,
)

_SCALE_ALL = (
    "ScaleFormat",
    "ScaleView",
    "scale_identity",
    "scale_k2_interleaved",
    "scale_k4_interleaved",
    "scale_k2mn2_interleaved",
    "scale_mn2_interleaved",
    "scale_mn4_interleaved",
    "scale_view",
)
__tilelang_dialect__ = "hcu"
__all__ = tuple(
    dict.fromkeys(
        (
            *_COMMON_ALL,
            *_ALLOCATE_ALL,
            *_ANNOTATION_ALL,
            *_COPY_ALL,
            *_GEMM_ALL,
            *_INTRINSIC_ALL,
            *_KERNEL_ALL,
            *_LOOP_ALL,
            *_REDUCE_ALL,
            *_SCALE_ALL,
        )
    )
)

del _COMMON_ALL, _ALLOCATE_ALL, _ANNOTATION_ALL, _COPY_ALL, _GEMM_ALL, _INTRINSIC_ALL, _KERNEL_ALL, _LOOP_ALL, _REDUCE_ALL, _SCALE_ALL
