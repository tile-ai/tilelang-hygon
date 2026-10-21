from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl
from tilelang.hcu.target import target_is_hcu

from .gemm_hcu_mmac import GEMM_INST_HCU_MMAC, GemmHCUMMAC


def _match_hcu(target) -> bool:
    return target_is_hcu(target)


register_gemm_impl("hcu.mmac", GEMM_INST_HCU_MMAC, _match_hcu, GemmHCUMMAC)


def __getattr__(name):
    # ``gemm_fp8_channelwise`` is a user-facing kernel *builder*: it imports
    # torch and drives ``tilelang.compile``, so it must stay out of this module's
    # eager import path (which every GEMM lowering touches).  Lazy so that
    # ``from tilelang.hcu.op.gemm import decode_cfg`` still works.
    if name in ("DecodeCfg", "decode_cfg", "fp8_channelwise_gemm"):
        from . import gemm_fp8_channelwise as _m

        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
