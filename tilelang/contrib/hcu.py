"""Utility for HCU backend"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings

import tvm_ffi
from tvm.base import py_str
from tvm.contrib import utils
from tvm.target import Target

from tilelang.engine.callback import register_hip_postproc_callback
from tilelang.env import TILELANG_TEMPLATE_PATH
from tilelang.transform.pass_config import PassConfigKey


def get_hcu_compiler() -> str:
    """Resolve the HIP/offload compiler for HCU toolchains."""
    return "aicc" if shutil.which("aicc") else "hipcc"


def _is_hcu_toolchain(toolchain_path: str) -> bool:
    """Return True when an HCU/DTK environment is detected."""
    if "dtk" in os.path.normpath(toolchain_path).lower():
        return True
    return shutil.which("hy-smi") is not None


def find_hcu_path() -> str:
    """Find the HCU/DTK toolchain root."""
    if "ROCM_PATH" in os.environ:
        return os.environ["ROCM_PATH"]
    if os.path.isdir("/opt/dtk"):
        return "/opt/dtk"
    hip_exe = shutil.which(get_hcu_compiler())
    if hip_exe:
        return os.path.realpath(os.path.join(hip_exe, "../.."))
    raise RuntimeError("Cannot find HCU toolchain path")


def get_hcu_arch(hcu_path: str | None = None) -> str:
    """Return gfx* architecture string from rocminfo for HCU devices."""
    gpu_arch = "gfx900"
    hcu_path = hcu_path or find_hcu_path()
    rocminfo = os.path.join(hcu_path, "bin", "rocminfo")
    if not os.path.isfile(rocminfo):
        print(f"HCU toolchain not detected at {hcu_path}, using default {gpu_arch}")
        return gpu_arch
    try:
        rocminfo_output = subprocess.check_output([rocminfo]).decode("utf-8")
        match = re.search(r"Name:\s+(gfx\d+[a-zA-Z]*)", rocminfo_output)
        if match:
            gpu_arch = match.group(1)
        return gpu_arch
    except subprocess.CalledProcessError:
        print(
            f"Unable to execute rocminfo command, "
            f"please ensure HCU toolchain is installed and you have an HCU device on your system. "
            f"using default {gpu_arch}."
        )
        return gpu_arch


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


def _sanitize_hcu_visible_text(text: str) -> str:
    if not text:
        return text
    text = text.replace("amdgcn-amd-amdhsa", "<hcu-target-triple>")
    text = text.replace("__builtin_amdgcn", "<hcu-builtin>")
    return text


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
    # If get_hcu_compiler() resolves to aicc (on PATH), still apply the LLVM tuning flags below.
    rocm_path = os.environ.get("ROCM_PATH", "")
    if ("dtk" in rocm_path.lower() or os.path.isdir("/opt/dtk")) and get_hcu_compiler() != "aicc":
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
            "-mllvm=-amdgpu-disable-backoff-barrier=false",
        ]
        if _pass_config_truthy(pass_configs, PassConfigKey.TL_ENABLE_FAST_MATH):
            flags.append("-mllvm=-enable-hcu-approx-func-fp-math=true")
        if _pass_config_truthy(pass_configs, PassConfigKey.TL_ENABLE_HCU_WDRA):
            flags.append("-mllvm=-vgpr-greedy-alloc-mode=local-wave")
            # flags.append("-mllvm=-run-on-model=true") # just for cmodel testing
        if arch in ["gfx938", "gfx92a", "gfx946"]:
            flags.append("-mllvm=-hcu-update-wait-by-reverse-search=true")
            flags.append("-mllvm=-hcu-pre-emit-load-store-opt=false")
            flags.append("-mllvm=-hcu-trust-special-waitcnt-for-lds-dma=true")
        return flags
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


def _debug_enabled():
    return os.environ.get("TILELANG_HIPCC_DEBUG", "").lower() in ("1", "true", "yes", "on")


def _debug_log(msg):
    if _debug_enabled():
        print(f"[TileLang][hcu][debug] {_sanitize_hcu_visible_text(msg)}")


def _get_aillvm_tool(env_name, default_path):
    tool_path = os.environ.get(env_name, default_path)
    tool_path = os.path.abspath(os.path.expanduser(tool_path))
    if not os.path.isfile(tool_path):
        raise RuntimeError(f"{env_name} not found: {tool_path}")
    return tool_path


def _extract_device_bundle_asm(path, temp):
    start_marker = "# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--"
    end_marker = "# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--"

    with open(path, encoding="utf-8") as in_file:
        asm_text = in_file.read()

    start_idx = asm_text.find(start_marker)
    if start_idx == -1:
        return path

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
    pattern = re.compile(r'(?:extern\s+"C"\s+)?__global__\s+void(?:\s+__\w+__\s*\([^)]*\))*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')
    return set(pattern.findall(code))


def _parse_asm_map(raw):
    mapping = {}
    if not raw:
        return mapping
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError("Invalid TILELANG_HIPCC_ASM_MAP item (missing '='): " + item)
        kernel_name, asm_path = item.split("=", 1)
        kernel_name = kernel_name.strip()
        asm_path = asm_path.strip()
        if not kernel_name or not asm_path:
            raise RuntimeError("Invalid TILELANG_HIPCC_ASM_MAP item (empty key/value): " + item)
        mapping[kernel_name] = asm_path
    return mapping


def _normalize_asm_input_path(asm_input, temp):
    asm_input = os.path.abspath(os.path.expanduser(asm_input))
    if not os.path.exists(asm_input):
        raise RuntimeError("TILELANG_HIPCC_ASM_MAP points to missing file: " + asm_input)
    if not os.path.isfile(asm_input):
        raise RuntimeError("TILELANG_HIPCC_ASM_MAP points to non-file path: " + asm_input)
    return _extract_device_bundle_asm(asm_input, temp)


def _resolve_asm_input(code, temp, verbose=False):
    asm_map = _parse_asm_map(os.environ.get("TILELANG_HIPCC_ASM_MAP"))
    if asm_map:
        kernel_names = _extract_kernel_names_from_code(code)
        _debug_log(f"detected kernels: {sorted(kernel_names)}")
        for kernel_name in sorted(kernel_names):
            if kernel_name in asm_map:
                _debug_log(f"asm_map hit: kernel={kernel_name} asm={asm_map[kernel_name]}")
                if verbose:
                    print(f"[TileLang][hcu] asm override hit kernel '{kernel_name}' -> {asm_map[kernel_name]}")
                return _normalize_asm_input_path(asm_map[kernel_name], temp)
        _debug_log("asm_map configured but no kernel matched; fallback to source compile")
        if verbose:
            print("[TileLang][hcu] asm map configured but no kernel matched, fallback to source compile")
    return None


def _compile_asm_with_aillvm(asm_input, arch, file_target, temp, verbose=False):
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
        print(_sanitize_hcu_visible_text(py_str(compile_ret.stdout)))
    if compile_ret.returncode != 0:
        raise RuntimeError(
            "Assembly compile error:\n"
            + _sanitize_hcu_visible_text(py_str(compile_ret.stdout))
        )

    link_ret = subprocess.run(link_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if verbose:
        print(_sanitize_hcu_visible_text(py_str(link_ret.stdout)))
    if link_ret.returncode != 0:
        raise RuntimeError(
            "Assembly link error:\n"
            + _sanitize_hcu_visible_text(py_str(link_ret.stdout))
        )


def compile_hcu(
    code,
    target_format="hsaco",
    arch=None,
    options=None,
    path_target=None,
    verbose=False,
    pass_config=None,
):
    """Compile HIP device code for HCU with DTK/aicc tuning flags."""
    if arch is None:
        arch = get_hcu_arch(find_hcu_path())

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

    cmd = [get_hcu_compiler()]
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
    cmd.extend(get_hcu_compile_flags(arch, cfg))

    cmd += ["-o", file_target]
    cmd += [temp_code]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = proc.communicate()
    if verbose:
        print(py_str(out))

    if proc.returncode != 0:
        msg = "Compilation failed for HCU source build.\n"
        msg += f"Source path: {temp_code}\n"
        msg += f"Command: {_sanitize_hcu_visible_text(' '.join(cmd))}\n"
        msg += "Compilation diagnostics:\n"
        msg += _sanitize_hcu_visible_text(py_str(out))
        raise RuntimeError(msg)

    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        _debug_log(f"returning source-built hsaco bytes={len(data)} target={file_target}")
        return data


@tvm_ffi.register_global_func("tilelang_callback_hcu_compile", override=True)
def tilelang_callback_hcu_compile(code, target, pass_config=None):
    """HCU hsaco compile callback used by ``target.build.tilelang_hcu``."""
    from tilelang.rocm.target import target_get_mcpu

    cfg = pass_config or {}
    arch = target_get_mcpu(target)
    return compile_hcu(
        code,
        target_format="hsaco",
        arch=arch,
        options=[
            "-std=c++17",
            "-I" + TILELANG_TEMPLATE_PATH,
        ],
        pass_config=cfg,
        verbose=False,
    )


@tvm_ffi.register_global_func("tilelang_callback_hcu_postproc", override=True)
def tilelang_callback_hcu_postproc(code, target):
    return hcu_recompute_from_source(code, target)
