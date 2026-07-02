"""Utility for HCU backend"""

from __future__ import annotations

import os
import re
import warnings

from tvm.target import Target

from tilelang.engine.callback import register_hip_postproc_callback
from tilelang.env import get_hip_compiler
from tilelang.transform.pass_config import PassConfigKey

_GLOBAL_KERNEL_RE = re.compile(
    r'extern\s+"C"\s+__global__\s+\w+(?:\s+__launch_bounds__\s*\([^)]*\))?\s+(\w+)\s*\(',
    re.MULTILINE,
)


def _extract_hip_kernel_symbols(code: str) -> list[str]:
    return list(dict.fromkeys(_GLOBAL_KERNEL_RE.findall(code)))


def _override_declares_symbols(override: str, symbols: list[str]) -> bool:
    for sym in symbols:
        if not re.search(
            rf'(?:extern\s+"C"\s+)?__global__\s+.+?\b{re.escape(sym)}\s*\(',
            override,
            re.MULTILINE | re.DOTALL,
        ):
            return False
    return True


@register_hip_postproc_callback
def hcu_recompute_from_source(code: str, _target: Target) -> str:
    """Optional HIP device source replacement before hipcc.

    ``TILELANG_OVERRIDE_DEVICE_SOURCE``: absolute or relative path to one ``.cu``
    file that should replace the full emitted HIP translation unit.

    ``TILELANG_OVERRIDE_DEVICE_SOURCE_DIR``: directory of ``{kernel_symbol}.cu``
    (only when codegen exposes exactly one ``__global__`` kernel).

    If unset, returns ``code`` unchanged. ``tilelang.engine.lower`` imports this module,
    so the callback is registered whenever lowering is used (no need to import ``hcu``
    manually in tests).

    With ``target="auto"``, ROCm PyTorch builds select HIP even when a CUDA toolkit is
    on PATH; override env vars only affect HIP device codegen, not ``tilelang_cuda``.

    A kernel cache hit skips lowering; set ``TILELANG_DISABLE_CACHE=1`` or clear the
    cache entry to force this hook to run.

    If both variables are set, the single-file override wins.
    """
    single = os.environ.get("TILELANG_OVERRIDE_DEVICE_SOURCE", "").strip()
    directory = os.environ.get("TILELANG_OVERRIDE_DEVICE_SOURCE_DIR", "").strip()
    if not single and not directory:
        return code

    symbols = _extract_hip_kernel_symbols(code)
    if not symbols:
        warnings.warn(
            'HIP device source override env is set but no `extern "C" __global__` kernels matched; skipping override.',
            stacklevel=2,
        )
        return code

    path: str | None = None
    if single:
        path = os.path.abspath(os.path.expanduser(single))
        if not path.endswith(".cu"):
            warnings.warn(
                "TILELANG_OVERRIDE_DEVICE_SOURCE should normally end with .cu.",
                stacklevel=2,
            )
    else:
        base = os.path.abspath(os.path.expanduser(directory))
        if len(symbols) != 1:
            warnings.warn(
                "TILELANG_OVERRIDE_DEVICE_SOURCE_DIR supports only a single "
                f"kernel in codegen; found {len(symbols)} symbols {symbols!r}; skipping.",
                stacklevel=2,
            )
            return code
        path = os.path.join(base, symbols[0] + ".cu")

    if path is None or not os.path.isfile(path):
        warnings.warn(
            f"HIP device source override path is missing or not a file: {path!r}; skipping.",
            stacklevel=2,
        )
        return code

    try:
        with open(path, encoding="utf-8") as f:
            override = f.read()
    except OSError as exc:
        warnings.warn(
            f"HIP device source override read failed ({path}): {exc}; skipping.",
            stacklevel=2,
        )
        return code

    if not _override_declares_symbols(override, symbols):
        warnings.warn(
            f"HIP device source override must declare every codegen kernel symbol {symbols!r}; skipping.",
            stacklevel=2,
        )
        return code

    print(f"override={path}")
    return override


def _pass_config_truthy(pass_configs: dict | None, key: PassConfigKey) -> bool:
    if not pass_configs:
        return False
    v = pass_configs.get(key)
    if v is None:
        v = pass_configs.get(key.value)
    return bool(v)


def get_hcu_compile_flags(arch: str, pass_configs: dict | None = None):
    # DTK toolchain (e.g. ROCM_PATH=/opt/dtk/...) uses its own defaults; do not inject LLVM hacks.
    # If get_hip_compiler() resolves to aicc (on PATH), still apply the LLVM tuning flags below.
    rocm_path = os.environ.get("ROCM_PATH", "")
    if ("dtk" in rocm_path.lower() or os.path.isdir("/opt/dtk")) and get_hip_compiler() != "aicc":
        return []
    if arch in ["gfx928", "gfx936", "gfx938", "gfx92a", "gfx946"]:
        flags = [
            "-mllvm=-support-768-vgprs=true",
            "-mllvm=-enable-latency-hack=true",
            "-mllvm=-mmac-latency=5",
            "-mllvm=-ds-load-store-latency=6",
            "-mllvm=-disable-machine-sink=True",
            "-mllvm=-check-valu-data-forward-hazards=0",
            "-mllvm=-disable-cluster-lds-memops=true",
            # "-mllvm=-amdgpu-disable-backoff-barrier=false",
        ]
        if _pass_config_truthy(pass_configs, PassConfigKey.TL_ENABLE_FAST_MATH):
            flags.append("-mllvm=-enable-hcu-approx-func-fp-math=true")
        if arch in ["gfx938", "gfx92a", "gfx946"]:
            flags.append("-mllvm=-hcu-update-wait-by-reverse-search=true")
            flags.append("-mllvm=-hcu-pre-emit-load-store-opt=false")
            # Pending upstream clang release; re-enable when -hcu-trust-special-waitcnt-for-lds-dma ships.
            # flags.append("-mllvm=-hcu-trust-special-waitcnt-for-lds-dma=true")
        return flags
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
