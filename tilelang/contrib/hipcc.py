# pylint: disable=invalid-name
"""Utility to invoke hipcc compiler in the system"""
# File is copied from a modified version of hipcc.py to support
# compilation of HIP code with hipcc compiler
# Source Path:
# https://github1s.com/TileLang/tvm/blob/upstream/python/tvm/contrib/hipcc.py

from __future__ import absolute_import as _abs

import os
import re
import subprocess

import tvm_ffi

from tvm.contrib import utils
from tvm.base import py_str
from tvm.contrib.rocm import get_rocm_arch, find_rocm_path

from tilelang.env import TILELANG_TEMPLATE_PATH, get_hip_compiler


def _debug_enabled():
    return os.environ.get("TILELANG_HIPCC_DEBUG", "").lower() in ("1", "true", "yes", "on")


def _debug_log(msg):
    if _debug_enabled():
        print(f"[TileLang][hipcc][debug] {msg}")


def _get_aillvm_tool(env_name, default_path):
    tool_path = os.environ.get(env_name, default_path)
    tool_path = os.path.abspath(os.path.expanduser(tool_path))
    if not os.path.isfile(tool_path):
        raise RuntimeError(f"{env_name} not found: {tool_path}")
    return tool_path


def _extract_device_bundle_asm(path, temp):
    """Extract hip-amdgcn offload bundle from clang .s if present."""
    start_marker = "# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--"
    end_marker = "# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--"

    with open(path, "r", encoding="utf-8") as in_file:
        asm_text = in_file.read()

    start_idx = asm_text.find(start_marker)
    if start_idx == -1:
        return path

    # Keep body after START line and before matching END line.
    body_start = asm_text.find("\n", start_idx)
    if body_start == -1:
        raise RuntimeError("Malformed asm bundle: START marker has no line break")
    body_start += 1

    end_idx = asm_text.find(end_marker, body_start)
    if end_idx == -1:
        raise RuntimeError("Malformed asm bundle: END marker not found for device bundle")

    extracted = asm_text[body_start:end_idx]
    extracted_path = temp.relpath("my_kernel_device_bundle.s")
    with open(extracted_path, "w", encoding="utf-8") as out_file:
        out_file.write(extracted)
    return extracted_path


def _extract_kernel_names_from_code(code):
    """Extract kernel function names from generated HIP source."""
    # Examples:
    #   extern "C" __global__ void _gemm_kernel(...)
    #   __global__ void __launch_bounds__(512) _gemm_kernel(...)
    #   __global__ void _gemm_kernel(...)
    pattern = re.compile(
        r'(?:extern\s+"C"\s+)?__global__\s+void(?:\s+__\w+__\s*\([^)]*\))*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('
    )
    return set(pattern.findall(code))


def _parse_asm_map(raw):
    """Parse env mapping: 'kernelA=/path/a.s;kernelB=/path/b.s'."""
    mapping = {}
    if not raw:
        return mapping
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError(
                "Invalid TILELANG_HIPCC_ASM_MAP item (missing '='): " + item
            )
        kernel_name, asm_path = item.split("=", 1)
        kernel_name = kernel_name.strip()
        asm_path = asm_path.strip()
        if not kernel_name or not asm_path:
            raise RuntimeError(
                "Invalid TILELANG_HIPCC_ASM_MAP item (empty key/value): " + item
            )
        mapping[kernel_name] = asm_path
    return mapping


def _normalize_asm_input_path(asm_input, temp):
    """Validate asm path and extract device bundle section when needed."""
    asm_input = os.path.abspath(os.path.expanduser(asm_input))
    if not os.path.exists(asm_input):
        raise RuntimeError(
            "TILELANG_HIPCC_ASM_MAP points to missing file: " + asm_input
        )
    if not os.path.isfile(asm_input):
        raise RuntimeError(
            "TILELANG_HIPCC_ASM_MAP points to non-file path: " + asm_input
        )
    return _extract_device_bundle_asm(asm_input, temp)


def _resolve_asm_input(code, temp, verbose=False):
    """Resolve asm override path by kernel name map only."""
    asm_map = _parse_asm_map(os.environ.get("TILELANG_HIPCC_ASM_MAP"))
    if asm_map:
        kernel_names = _extract_kernel_names_from_code(code)
        _debug_log(f"detected kernels: {sorted(kernel_names)}")
        for kernel_name in sorted(kernel_names):
            if kernel_name in asm_map:
                _debug_log(f"asm_map hit: kernel={kernel_name} asm={asm_map[kernel_name]}")
                if verbose:
                    print(
                        f"[TileLang][hipcc] asm override hit kernel '{kernel_name}' -> {asm_map[kernel_name]}"
                    )
                return _normalize_asm_input_path(asm_map[kernel_name], temp)
        _debug_log("asm_map configured but no kernel matched; fallback to source compile")
        if verbose:
            print(
                "[TileLang][hipcc] asm map configured but no kernel matched, fallback to source compile"
            )
    return None


def _compile_asm_with_aillvm(asm_input, arch, file_target, temp, verbose=False):
    """Compile AMDGPU assembly with aillvm clang++ and link to hsaco."""
    clangxx = _get_aillvm_tool("TILELANG_AILLVM_CLANGXX", "/opt/dtk/aillvm/bin/clang++")
    lld = _get_aillvm_tool("TILELANG_AILLVM_LLD", "/opt/dtk/aillvm/bin/ld.lld")
    temp_obj = temp.relpath("my_kernel_asm.o")

    compile_cmd = [
        clangxx,
        "-x",
        "assembler",
        "-target",
        "amdgcn-amd-amdhsa",
        f"-mcpu={arch}",
        "-c",
        asm_input,
        "-o",
        temp_obj,
    ]
    link_cmd = [lld, "-shared", temp_obj, "-o", file_target]
    _debug_log(f"asm compile cmd: {' '.join(compile_cmd)}")
    _debug_log(f"asm link cmd: {' '.join(link_cmd)}")

    compile_ret = subprocess.run(compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if verbose:
        print(py_str(compile_ret.stdout))
    if compile_ret.returncode != 0:
        raise RuntimeError("Assembly compile error:\n" + py_str(compile_ret.stdout))

    link_ret = subprocess.run(link_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if verbose:
        print(py_str(link_ret.stdout))
    if link_ret.returncode != 0:
        raise RuntimeError("Assembly link error:\n" + py_str(link_ret.stdout))


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

    If env ``TILELANG_HIPCC_ASM_MAP`` is set (format:
    ``kernelA=/path/a.s;kernelB=/path/b.s``), compile from mapped ``.s`` only
    when current HIP source contains a matching kernel name. Otherwise fallback
    to normal source compile.

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

    asm_input = _resolve_asm_input(code, temp, verbose=verbose)

    file_target = path_target if path_target else temp_target
    if asm_input:
        _debug_log("compilation mode: asm_map/aillvm")
        _compile_asm_with_aillvm(asm_input, arch, file_target, temp, verbose=verbose)
        with open(file_target, "rb") as f:
            data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        _debug_log(f"returning asm-built hsaco bytes={len(data)} target={file_target}")
        return data

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
        _debug_log(f"returning source-built hsaco bytes={len(data)} target={file_target}")
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
        ],
        pass_config=cfg,
        verbose=False,
    )


# Also registered (override) from ``tilelang.engine.lower`` when the compiler package loads.
