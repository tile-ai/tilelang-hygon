from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl
from tilelang.hcu.target import target_is_hcu

from .gemm_hcu_mmac import GEMM_INST_HCU_MMAC, GemmHCUMMAC


def _match_hcu(target) -> bool:
    return target_is_hcu(target)


register_gemm_impl("hcu.mmac", GEMM_INST_HCU_MMAC, _match_hcu, GemmHCUMMAC)
