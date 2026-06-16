"""
Common utility functions for GEMM implementations.
"""

import itertools
import torch
import tilelang as tl
from tilelang.autotuner import AutoTuner
from tilelang.carver.template import MatmulTemplate
from tilelang.carver.arch import CUDA
from tilelang.carver.arch import CDNA
from tilelang.carver.roller.rasterization import NoRasterization
from tilelang.contrib.rocm import find_rocm_path, get_rocm_arch


from aiter.ops.triton.gemm_unquantized import gemm_unquantized

def triton_gemm(A, B):
    """
    Compute the matrix product of A and the transpose of B.

    A and B are expected to be 2-D tensors where A has shape (M, K) and B has shape (N, K).
    The result is a tensor with shape (M, N) equal to A @ B.T, using the inputs' dtypes.
    """
    return  gemm_unquantized(A, B, torch.float16)


def ref_program(A, B):
    """
    Compute the matrix product of A and the transpose of B.

    A and B are expected to be 2-D tensors where A has shape (M, K) and B has shape (N, K).
    The result is a tensor with shape (M, N) equal to A @ B.T, using the inputs' dtypes.
    """
    return A @ B.T


def _generate_configs_from_product(param_dict):
    """
    Generate configuration dictionaries from Cartesian product of parameter lists.

    Args:
        param_dict: Dictionary mapping parameter names to lists of candidate values.
                    Keys are used in insertion order (Python 3.7+).

    Returns:
        List of configuration dictionaries
    """
    key_order = list(param_dict.keys())
    param_lists = [param_dict[key] for key in key_order]
    _configs = list(itertools.product(*param_lists))
    return [
        {key_order[i]: c[i] for i in range(len(key_order))}
        for c in _configs
    ]


def _get_roller_configs(M, N, K, topk=20):
    """
    Generate configurations using MatmulTemplate roller.

    Args:
        M, N, K: GEMM dimensions
        topk: Maximum number of roller hints to request

    Returns:
        List of configuration dictionaries

    Raises:
        ValueError: if roller returns no hints
    """
    arch = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
    carve_template = MatmulTemplate(
        M=M,
        N=N,
        K=K,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float",
    ).with_arch(arch)

    func = carve_template.equivalent_function()
    assert func is not None, "Function is None"
    roller_hints = carve_template.recommend_hints(topk=topk)
    if roller_hints is None:
        raise ValueError("No Roller Hints Found for TensorCore Scheduling")

    configs = []
    for hint in roller_hints:
        block_m, block_n = hint.block
        warp_m, warp_n = hint.warp
        # block_rows, block_cols represents warp partitioning
        block_rows, block_cols = block_m // warp_m, block_n // warp_n
        configs.append({
            "block_M": block_m,
            "block_N": block_n,
            "block_K": hint.rstep[0],
            "num_stages": hint.pipeline_stage if hint.pipeline_stage > 1 else 0,
            "thread_num": block_rows * block_cols * 32,
            "enable_rasteration": hint.rasterization_plan is not NoRasterization,
        })
    return configs


def get_configs(M, N, K, with_roller=False, topk=20):
    """
    Generate a list of kernel tuning configuration dictionaries for a tiled matrix-multiply.

    When with_roller is True this queries the MatmulTemplate roller to produce up to `topk` recommended
    configurations (device-specific TensorCore-friendly tilings). Each returned dict contains:
      - block_M, block_N, block_K: tile sizes
      - num_stages: pipeline staging (0 means no explicit staging)
      - thread_num: total threads used for the block
      - enable_rasteration: whether a rasterization/swizzle layout was recommended (note spelling)

    When with_roller is False this returns the Cartesian product of a fixed set of candidate
    parameters; the returned dicts use the backward-compatible key name "enable_rasteration" for that flag.

    Parameters:
        M, N, K (int): GEMM dimensions used to generate valid tile sizes.
        with_roller (bool): If True, use MatmulTemplate's roller to generate device-aware hints;
            otherwise use a predefined candidate grid.
        topk (int): Maximum number of roller hints to request when with_roller is True.

    Returns:
        List[dict]: A list of configuration dictionaries as described above.

    Raises:
        ValueError: if with_roller is True but the roller returns no hints.
    """
    if with_roller:
        return _get_roller_configs(M, N, K, topk)
    else:
        param_dict = {
            "block_M": [64, 128, 256],
            "block_N": [64, 128, 256],
            "block_K": [32, 64],
            "num_stages": [0, 1, 2, 3],
            "thread_num": [128, 256],
            "enable_rasteration": [True, False],
            "group_size": [1, 4, 8, 16],  # 1 disables group swizzling
        }
        return _generate_configs_from_product(param_dict)


def _create_autotuner(kernel, configs, pass_configs=None):
    """
    Create and configure an AutoTuner instance with common settings.

    Args:
        kernel: The kernel function to tune
        configs: List of configuration dictionaries to test

    Returns:
        Configured AutoTuner instance
    """
    return AutoTuner.from_kernel(
        kernel=kernel, configs=configs
    ).set_compile_args(
        out_idx=[-1],
        target="auto",
        verbose=True,
        pass_configs=pass_configs,
    ).set_profile_args(
        supply_type=tl.TensorSupplyType.Integer,
        ref_prog=ref_program,
        skip_check=False,
    )


def _run_autotuner(kernel, configs, warmup=3, rep=20, pass_configs=None):
    """
    Run autotuning with common settings.

    Args:
        kernel: The kernel function to tune
        configs: List of configuration dictionaries to test
        warmup: Number of warmup iterations
        rep: Number of benchmark repetitions

    Returns:
        Best configuration result from autotuner
    """
    autotuner = _create_autotuner(kernel, configs, pass_configs=pass_configs)
    return autotuner.run(warmup=warmup, rep=rep)


def get_heuristic_config(impl: str | None = None, version: str | None = None) -> dict:
    """Get a heuristic configuration for GEMM kernels.

    Args:
        impl: GEMM implementation family (vanilla, persistent, splitk, streamk).
        version: Kernel variant within the family (e.g. v1, v2, v3, v4, v5).
            When set, version-specific tile/thread overrides are applied on top
            of the base heuristic config.
    """
    config = {
        #"block_M": 128,
        #"block_N": 256,
        #"block_K": 32,
        "block_M": 128,
        "block_N": 256,
        "block_K": 64,
        "num_stages": 0,
        #"thread_num": 128,
        "thread_num": 256,
        # "thread_num": 512,
        "group_size": 1,  # Enable group swizzling optimization
        #"enable_rasteration": True,
        "wgs_per_cu": 2,
    }

    version_overrides = {
        ("persistent", "v4"): {
            "block_M": 256,
            "block_N": 256,
            "block_K": 32,
            "thread_num": 512,
        },
        ("vanilla", "v2"): {
            "block_M": 256,
            "block_N": 256,
            "block_K": 32,
            "thread_num": 512,
        },
    }
    if impl is not None and version is not None:
        config.update(version_overrides.get((impl, version), {}))

    if impl in ["vanilla"]:
        config["use_mls"] = True
    return config


def _get_compile_target_arch() -> str | None:
    try:
        return get_rocm_arch(find_rocm_path())
    except Exception:
        return None


def get_default_kernel_version(impl: str) -> str | None:
    """Default kernel variant used by perf/gemm/benchmark.py for each impl family."""
    arch = _get_compile_target_arch()
    if arch == "gfx938":
        defaults = {
            "persistent": "v4",
            "vanilla": "v2",
        }
    else:
        defaults = {
            "persistent": "v3",
            "vanilla": "v1",
        }
    return defaults.get(impl)
