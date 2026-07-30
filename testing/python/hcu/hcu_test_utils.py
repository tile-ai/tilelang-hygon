from tilelang import tvm as tvm
from tilelang.utils.target import determine_target, get_hcu_arch_string, target_is_hcu

# HCU gfx validated for FP8 ``__builtin_hcu_mmac_*`` (extend when verified on new arch).
_FP8_MMAC_SUPPORTED_HCU_ARCHES = frozenset({"gfx938", "gfx92a", "gfx946"})

# HCU gfx validated for MLS matrix_load tests (extend when verified on new arch).
MLS_SUPPORTED_HCU_ARCHES = frozenset({"gfx938", "gfx946", "gfx92a"})


def target_supports_fp8_mmac() -> bool:
    """Return whether the active target can lower FP8 operands via HCU MMAC."""
    try:
        target = tvm.target.Target(determine_target("auto"))
        return target_is_hcu(target) and get_hcu_arch_string(target) in _FP8_MMAC_SUPPORTED_HCU_ARCHES
    except Exception:
        return False


def target_supports_mls() -> bool:
    """Return whether the active target supports HCU MLS matrix_load tests."""
    try:
        target = tvm.target.Target(determine_target("auto"))
        return target_is_hcu(target) and get_hcu_arch_string(target) in MLS_SUPPORTED_HCU_ARCHES
    except Exception:
        return False


def target_supports_mls_b4() -> bool:
    """Return whether the active target supports b4 MLS matrix_load tests."""
    try:
        target = tvm.target.Target(determine_target("auto"))
        return target_is_hcu(target) and get_hcu_arch_string(target) == "gfx946"
    except Exception:
        return False
