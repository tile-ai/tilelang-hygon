from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl
from tilelang.hcu.target import target_is_hcu

from .gemm_hcu_mmac import (
    GEMM_INST_HCU_MMAC,
    GEMM_INST_HCU_MMAC_BLOCKSCALED,
    GemmHCUMMAC,
    GemmHCUMMACBlockScaled,
)


def _match_hcu(target) -> bool:
    return target_is_hcu(target)


register_gemm_impl("hcu.mmac", GEMM_INST_HCU_MMAC, _match_hcu, GemmHCUMMAC)
register_gemm_impl(
    "hcu.mmac.blockscaled",
    GEMM_INST_HCU_MMAC_BLOCKSCALED,
    _match_hcu,
    GemmHCUMMACBlockScaled,
)
