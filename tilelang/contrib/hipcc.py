# pylint: disable=invalid-name
"""Utility to invoke hipcc compiler in the system"""
# File is copied from a modified version of hipcc.py to support
# compilation of HIP code with hipcc compiler
# Source Path:
# https://github1s.com/TileLang/tvm/blob/upstream/python/tvm/contrib/hipcc.py

from __future__ import absolute_import as _abs

import subprocess

import tvm_ffi

from tvm.contrib import utils
from tvm.base import py_str
from tvm.contrib.rocm import get_rocm_arch, find_rocm_path

from tilelang.env import COMPOSABLE_KERNEL_INCLUDE_DIR, TILELANG_TEMPLATE_PATH, get_hip_compiler


def compile_hip(
    code,
    target_format="hsaco",
    arch=None,
    options=None,
    path_target=None,
    verbose=False,
    pass_config=None,
):
    """Compile HIP code with the active HIP compiler (aicc if on PATH, else hipcc).

    Appends the same ``-mllvm=...`` tuning list as ``kernel_cache`` /
    ``libgen`` via ``get_hcu_compile_flags`` (DTK may yield an empty list).

    ``pass_config`` is the TVM ``PassContext`` config dict, forwarded from C++
    ``target.build.tilelang_hip`` into ``tilelang_callback_hip_compile`` (same
    pattern as CUDA). When ``tl.enable_fast_math`` is true, HCU targets append
    ``-mllvm=-enable-hcu-approx-func-fp-math=true``.
    If ``pass_config`` is omitted (e.g. direct ``compile_hip`` calls), it defaults
    to no optional LLVM tuning from config.

    Unsupported architectures raise ``ValueError`` from ``get_hcu_compile_flags``.

    Parameters
    ----------
    code : str
        The HIP code.

    target_format : str
        The target format of hipcc compiler.

    arch : str
        The AMD GPU architecture.

    options : str or list of str
        The additional options.

    path_target : str, optional
        Output file.

    pass_config : dict, optional
        Same object as the third argument of ``tilelang_callback_hip_compile``.

    Return
    ------
    hsaco : bytearray
        The bytearray of the hsaco
    """
    if arch is None:
        rocm_path = find_rocm_path()
        arch = get_rocm_arch(rocm_path)

    temp = utils.tempdir()
    if target_format not in ["hsaco"]:
        raise ValueError("target_format must be hsaco")
    temp_code = temp.relpath("my_kernel.cc")
    temp_target = temp.relpath(f"my_kernel.{target_format}")

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    file_target = path_target if path_target else temp_target
    cmd = [get_hip_compiler()]
    cmd += ["-O3", "-c"]
    if isinstance(arch, str):
        cmd += [f"--offload-arch={arch}"]
    if target_format == "hsaco":
        cmd += ["--genco"]
    if options:
        if isinstance(options, str):
            cmd += [options]
        elif isinstance(options, list):
            cmd += options
        else:
            raise ValueError("options must be str or list of str")

    cfg = pass_config or {}

    # Lazy import avoids circular import: contrib -> hipcc -> hcu -> engine -> utils.target
    from tilelang.contrib.hcu import get_hcu_compile_flags

    cmd.extend(get_hcu_compile_flags(arch, cfg))

    cmd += ["-o", file_target]
    cmd += [temp_code]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()
    if verbose:
        print(py_str(out))

    if proc.returncode != 0:
        msg = code
        msg += "\nCompilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data


@tvm_ffi.register_global_func("tilelang_callback_hip_compile", override=True)
def tilelang_callback_hip_compile(code, target, pass_config=None):
    """HIP hsaco compile callback; ``pass_config`` matches ``tilelang_callback_cuda_compile``."""
    cfg = pass_config or {}
    return compile_hip(
        code,
        target_format="hsaco",
        options=[
            "-std=c++17",
            "-I" + TILELANG_TEMPLATE_PATH,
            "-I" + COMPOSABLE_KERNEL_INCLUDE_DIR,
        ],
        pass_config=cfg,
        verbose=False,
    )


# Also registered (override) from ``tilelang.engine.lower`` when the compiler package loads.
