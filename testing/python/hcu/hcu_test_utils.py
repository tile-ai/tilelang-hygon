from tilelang import tvm as tvm
from tilelang.backend.target import determine_target
from tilelang.hcu.target import get_hcu_arch_string, target_is_hcu

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
    """Return whether the target supports packed FP4 as an MLS source."""
    try:
        target = tvm.target.Target(determine_target("auto"))
        return target_is_hcu(target) and get_hcu_arch_string(target) in ("gfx92a", "gfx946")
    except Exception:
        return False


def target_supports_mls_b32() -> bool:
    """Return whether the target supports b32 as an MLS source."""
    try:
        target = tvm.target.Target(determine_target("auto"))
        return target_is_hcu(target) and get_hcu_arch_string(target) in ("gfx92a", "gfx946")
    except Exception:
        return False


def target_supports_mls_fp4_pad() -> bool:
    """Return whether the active target supports fp4 MLS b8-LDS fallback tests."""
    try:
        target = tvm.target.Target(determine_target("auto"))
        return target_is_hcu(target) and get_hcu_arch_string(target) in ("gfx92a", "gfx946")
    except Exception:
        return False


def current_hcu_arch_string() -> str:
    """Return the active HCU arch string, or an empty string when unavailable."""
    try:
        target = tvm.target.Target(determine_target("auto"))
        return get_hcu_arch_string(target) if target_is_hcu(target) else ""
    except Exception:
        return ""


# First-release gemm_blockscaled (FP4xFP4 + E8M0) requires lit/lts + scale builtins.
BLOCKSCALED_SUPPORTED_HCU_ARCHES = frozenset({"gfx946"})


def target_supports_blockscaled(target=None) -> bool:
    """Return whether ``target`` can compile HCU gemm_blockscaled (gfx946).

    Defaults to ``determine_target("auto")`` like other helpers. Pass an explicit
    target when offline-compiling for gfx946 on a non-matching auto detect.
    """
    try:
        tgt = tvm.target.Target(target if target is not None else determine_target("auto"))
        return target_is_hcu(tgt) and get_hcu_arch_string(tgt) in BLOCKSCALED_SUPPORTED_HCU_ARCHES
    except Exception:
        return False
