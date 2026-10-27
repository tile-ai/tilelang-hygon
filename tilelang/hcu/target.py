from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer

# LLVM device triple passed to the offline compiler (DTK toolchain).
_HCU_DEVICE_TRIPLE = "amdgcn-amd-amdhsa-hcc"
HCU_MCPU_SET = frozenset({"gfx928", "gfx936", "gfx938", "gfx92a", "gfx946", "gfx948"})
HCU_WARP_SIZE = 64
HCU_LDS_SIZE_BYTES = {
    "gfx946": 128 * 1024,
    "gfx948": 160 * 1024,
}
HCU_DEFAULT_LDS_SIZE_BYTES = 64 * 1024


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
    target_dict.setdefault("max_num_threads", 1024)
    target_dict.setdefault("max_threads_per_block", 1024)
    # The target kind materializes its generic 64 KiB default in export().
    # Upgrade that default for newer HCU architectures, while preserving a
    # caller-provided non-default limit.
    configured_lds_size = int(target_dict.get("max_shared_memory_per_block", HCU_DEFAULT_LDS_SIZE_BYTES))
    if configured_lds_size == HCU_DEFAULT_LDS_SIZE_BYTES:
        target_dict["max_shared_memory_per_block"] = HCU_LDS_SIZE_BYTES.get(arch, HCU_DEFAULT_LDS_SIZE_BYTES)
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


def target_get_lds_size_bytes(target: Target) -> int:
    return int(_target_ffi_api().TargetHcuGetLdsSizeBytes(target))


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
