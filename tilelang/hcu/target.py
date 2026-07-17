from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer

# LLVM device triple passed to the offline compiler (DTK toolchain).
_HCU_DEVICE_TRIPLE = "amdgcn-amd-amdhsa-hcc"
HCU_MCPU_SET = frozenset({"gfx928", "gfx936", "gfx938", "gfx92a", "gfx946"})
HCU_WARP_SIZE = 64


def _target_ffi_api():
    from tilelang import _ffi_api

    return _ffi_api


def normalize_hcu_arch(arch: str | None) -> str | None:
    if arch is None:
        return None
    normalized = str(arch).strip().split(":", maxsplit=1)[0]
    return normalized if normalized.startswith("gfx") else None


def target_get_mcpu(target: str | Target | None) -> str | None:
    if target is None:
        return None
    if isinstance(target, str):
        target = Target(target)
    return normalize_hcu_arch(target.attrs.get("mcpu"))


def with_hcu_target_attrs(target: Target) -> Target:
    if target.kind.name != "hcu":
        return target
    arch = target_get_mcpu(target)
    if arch is None:
        return target

    target_dict = dict(target.export())
    # mtriple: LLVM target triple for device codegen / offline compile.
    target_dict.setdefault("mtriple", _HCU_DEVICE_TRIPLE)
    # Workgroup/block limits (not wave/SIMD). DCU supports 1024 threads/block.
    target_dict.setdefault("max_num_threads", 1024)
    target_dict.setdefault("max_threads_per_block", 1024)
    if arch in HCU_MCPU_SET:
        target_dict["thread_warp_size"] = HCU_WARP_SIZE
    else:
        target_dict.pop("thread_warp_size", None)
    return Target(target_dict)


def target_is_hcu(target: Target) -> bool:
    return _target_ffi_api().TargetIsHCU(target)


def get_hcu_arch_string(target: Target) -> str:
    return str(_target_ffi_api().GetHcuArchString(target))


def is_hcu_enable_auto_async_copy_target(target: Target) -> bool:
    return _target_ffi_api().IsHCUEnableAutoAsyncCopyTarget(target)


def default_enable_auto_async_copy(target: Target) -> bool:
    return _target_ffi_api().DefaultEnableAutoAsyncCopy(target)


def target_get_warp_size(target: Target) -> int:
    return _target_ffi_api().TargetHcuGetWarpSize(target)


def target_has_mmac_lit_lts(target: Target) -> bool:
    return _target_ffi_api().TargetHasMmacLitLts(target)


def _is_hcu_mcpu(arch: str | None) -> bool:
    return arch is not None and arch in HCU_MCPU_SET


def _target_from_hcu_arch(arch: str) -> Target:
    target_dict: dict[str, object] = {
        "kind": "hcu",
        "mcpu": arch,
        "mtriple": _HCU_DEVICE_TRIPLE,
        "thread_warp_size": HCU_WARP_SIZE,
    }
    return Target(target_dict)


def _detect_hcu_arch() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            arch = normalize_hcu_arch(getattr(torch.cuda.get_device_properties(0), "gcnArchName", None))
            if _is_hcu_mcpu(arch):
                return arch
    except Exception:
        pass
    return None


def check_hcu_availability() -> bool:
    """Return True when DTK device headers or a local HCU-capable GPU are present."""
    import os

    if os.path.isdir("/opt/dtk/include/hip"):
        return True
    return _detect_hcu_arch() is not None


def _detect_hcu_target() -> Target | str | None:
    arch = _detect_hcu_arch()
    if arch is None:
        return None
    return with_hcu_target_attrs(_target_from_hcu_arch(arch))


def normalize_hcu_target(target: TargetLike) -> Target | None:
    if isinstance(target, Target):
        parsed = target
    elif isinstance(target, dict):
        if target.get("kind") != "hcu":
            return None
        try:
            parsed = Target(target)
        except Exception:
            return None
    else:
        return None

    if parsed.kind.name != "hcu":
        return None
    return with_hcu_target_attrs(parsed)


register_target_detector("hcu", _detect_hcu_target, override=True)
register_target_normalizer("hcu", normalize_hcu_target, override=True)
