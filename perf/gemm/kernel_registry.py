# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Kernel dispatch and heuristic-config bindings for GEMM benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from perf.gemm.async_copy_gemm import gemm_async_copy_n_major
from perf.gemm.async_copy_gemm_pp import gemm_async_copy_k_major
from perf.gemm.persistent_gemm import (
    gemm_persistent_v1,
    gemm_persistent_v2,
    gemm_persistent_v3,
    gemm_persistent_v4,
    gemm_persistent_v4_split_m,
    gemm_persistent_v5,
)
from perf.gemm.splitk_gemm import gemm_splitk
from perf.gemm.streamk_gemm import gemm_streamk
from perf.gemm.utils import get_default_kernel_version, get_heuristic_config, get_persistent_split_m_group_size
from perf.gemm.vanilla_gemm import gemm_vanilla_v1, gemm_vanilla_v2


Kernel = Callable[..., Any]
ConfigGetter = Callable[[int, int, int], dict[str, Any]]


ASYNC_COPY_N_MAJOR_SWIZZLE_CONFIG: dict[int, dict[str, Any]] = {
    4096: {},
    5120: {},
    8192: {
        "swizzle_panel_size": 8,
        "swizzle_order": "row",
    },
    16384: {
        "swizzle_panel_size": 4,
        "swizzle_order": "row",
    },
}


ASYNC_COPY_K_MAJOR_SWIZZLE_CONFIG: dict[int, dict[str, Any]] = {
    4096: {
        "swizzle_enable": False,
    },
    5120: {
        "swizzle_panel_size": 10,
        "swizzle_order": "col",
        "swizzle_enable": True,
    },
    8192: {
        "swizzle_panel_size": 4,
        "swizzle_order": "row",
        "swizzle_enable": True,
    },
    16384: {
        "swizzle_panel_size": 4,
        "swizzle_order": "row",
        "swizzle_enable": True,
    },
}


@dataclass(frozen=True)
class KernelSpec:
    """A kernel together with its config policy and fixed invocation kwargs."""

    name: str
    kernel: Kernel
    get_config: ConfigGetter
    kernel_kwargs: dict[str, Any] = field(default_factory=dict)


def _heuristic_config_getter(impl: str, version: str | None) -> ConfigGetter:
    def get_config(M: int, N: int, K: int) -> dict[str, Any]:
        return get_heuristic_config(impl, version, M, N, K)

    return get_config


def _persistent_v4_split_m_config(M: int, N: int, K: int) -> dict[str, Any]:
    config = get_heuristic_config("persistent", "v4", M, N, K)
    config["group_size"] = get_persistent_split_m_group_size(
        M,
        N,
        config["block_M"],
        config["block_N"],
    )
    return config


def _async_copy_vanilla_config(M: int, N: int, K: int) -> dict[str, Any]:
    config = {
        "block_M": 256,
        "block_N": 256,
        "block_K": 16,
    }
    config.update(ASYNC_COPY_N_MAJOR_SWIZZLE_CONFIG.get(M, {}))
    return config


def _async_copy_pingpong_config(M: int, N: int, K: int) -> dict[str, Any]:
    config = {
        "block_M": 256,
        "block_N": 256,
        "block_K": 32,
    }
    config.update(ASYNC_COPY_K_MAJOR_SWIZZLE_CONFIG.get(M, {}))
    return config


_K_MAJOR = "k_major"
_N_MAJOR = "n_major"


KERNEL_REGISTRY: dict[tuple[str, str | None, str], KernelSpec] = {
    ("async_copy", "vanilla", _K_MAJOR): KernelSpec(
        "gemm_async_copy_k_major",
        gemm_async_copy_k_major,
        _async_copy_pingpong_config,
    ),
    ("async_copy", "vanilla", _N_MAJOR): KernelSpec(
        "gemm_async_copy_n_major",
        gemm_async_copy_n_major,
        _async_copy_vanilla_config,
    ),
    ("persistent", "v1", _K_MAJOR): KernelSpec("gemm_persistent_v1", gemm_persistent_v1, _heuristic_config_getter("persistent", "v1")),
    ("persistent", "v2", _K_MAJOR): KernelSpec("gemm_persistent_v2", gemm_persistent_v2, _heuristic_config_getter("persistent", "v2")),
    ("persistent", "v3", _K_MAJOR): KernelSpec("gemm_persistent_v3", gemm_persistent_v3, _heuristic_config_getter("persistent", "v3")),
    ("persistent", "v4", _K_MAJOR): KernelSpec(
        "gemm_persistent_v4",
        gemm_persistent_v4,
        _heuristic_config_getter("persistent", "v4"),
        {"transpose_B": True},
    ),
    ("persistent", "v4", _N_MAJOR): KernelSpec(
        "gemm_persistent_v4_split_m",
        gemm_persistent_v4_split_m,
        _persistent_v4_split_m_config,
        {"transpose_B": False},
    ),
    ("persistent", "v5", _K_MAJOR): KernelSpec("gemm_persistent_v5", gemm_persistent_v5, _heuristic_config_getter("persistent", "v5")),
    ("vanilla", "v1", _K_MAJOR): KernelSpec("gemm_vanilla_v1", gemm_vanilla_v1, _heuristic_config_getter("vanilla", "v1")),
    ("vanilla", "v2", _K_MAJOR): KernelSpec(
        "gemm_vanilla_v2",
        gemm_vanilla_v2,
        _heuristic_config_getter("vanilla", "v2"),
        {"transpose_B": True},
    ),
    ("vanilla", "v2", _N_MAJOR): KernelSpec(
        "gemm_vanilla_v2",
        gemm_vanilla_v2,
        _heuristic_config_getter("vanilla", "v2"),
        {"transpose_B": False},
    ),
    ("splitk", None, _K_MAJOR): KernelSpec("gemm_splitk", gemm_splitk, _heuristic_config_getter("splitk", None)),
    ("streamk", None, _K_MAJOR): KernelSpec("gemm_streamk", gemm_streamk, _heuristic_config_getter("streamk", None)),
}


def dispatch_kernel(impl: str, *, transpose_B: bool, version: str | None = None) -> KernelSpec:
    """Resolve a benchmark kernel, selecting the architecture default version when omitted."""
    if version is None:
        version = get_default_kernel_version(impl)
    layout = _K_MAJOR if transpose_B else _N_MAJOR
    key = (impl, version, layout)
    try:
        return KERNEL_REGISTRY[key]
    except KeyError as error:
        raise ValueError(f"Unsupported GEMM kernel: impl={impl}, version={version}, layout={layout}") from error


def get_kernel_config(spec: KernelSpec, M: int, N: int, K: int) -> dict[str, Any]:
    """Get a fresh shape-specific config using the policy bound to ``spec``."""
    return spec.get_config(M, N, K)
