"""Compile-only coverage for HCU's native FP32 -> BF16 pair conversion."""

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

import tilelang as tl
from tilelang import tvm
from tvm import tirx


def cast_source(lanes=4, masked=True, arch="gfx946", dst="bfloat16", annotations=None):
    a = tirx.decl_buffer((lanes,), "float32", name="A")
    mask = tirx.decl_buffer((lanes,), "int32", name="Mask")
    b = tirx.decl_buffer((lanes,), dst, name="B")
    i = tirx.Var("i", "int32")
    value = tirx.Select(mask[i] > 0, a[i], tirx.const(0, "float32")) if masked else a[i]
    body = tirx.BufferStore(b, tirx.Cast(dst, value, annotations=annotations), [i])
    if lanes == 1:
        body = tirx.stmt_functor.substitute(body, {i: tirx.const(0, "int32")})
    else:
        body = tirx.For(i, 0, lanes, tirx.ForKind.VECTORIZED, body)
    buffers = (a, mask, b)
    func = tirx.PrimFunc([buf.data for buf in buffers], body, buffer_map={buf.data: buf for buf in buffers})
    func = func.with_attr("global_symbol", "convert").with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tl.transform.VectorizeLoop()(tvm.IRModule({"convert": func}))
    target = tvm.target.Target({"kind": "hcu", "mcpu": arch})
    mod = tirx.transform.BindTarget(target)(mod)
    build = tvm.get_global_func("target.build.tilelang_hcu_without_compile")
    return build(mod, target).inspect_source()


@pytest.mark.parametrize("lanes", [2, 4, 8])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("arch", ["gfx936", "gfx946"])
def test_packed_bf16_cast_codegen(lanes, masked, arch):
    source = cast_source(lanes, masked, arch=arch)
    assert source.count("__builtin_hcu_cvt_pk_bf16_f32(") == lanes // 2
    assert source.count(", false));") == lanes // 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lanes": 1},
        {"dst": "float16"},
        {"annotations": {"round": tirx.StringImm("rn")}},
    ],
)
def test_other_cast_paths_unchanged(kwargs):
    assert "__builtin_hcu_cvt_pk_bf16_f32(" not in cast_source(**kwargs)


@pytest.mark.parametrize("lanes", [2, 4, 8])
@pytest.mark.parametrize("masked", [False, True])
def test_packed_bf16_cast_assembly(tmp_path, lanes, masked):
    compiler = shutil.which("hipcc")
    if compiler is None:
        pytest.skip("requires the HCU hipcc toolchain")
    cpp = tmp_path / "convert.cpp"
    asm = tmp_path / "convert.s"
    cpp.write_text(cast_source(lanes, masked))
    root = Path(__file__).resolve().parents[3]
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    subprocess.run(
        [
            compiler,
            "--offload-arch=gfx946",
            "--cuda-device-only",
            "-O3",
            "-std=c++17",
            f"-I{rocm / 'include'}",
            f"-I{root / 'src'}",
            "-S",
            str(cpp),
            "-o",
            str(asm),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assembly = asm.read_text()
    assert len(re.findall(r"\bv_cvt_pk_bf16_f32\b", assembly)) == lanes // 2
    assert not re.search(r"\bv_cvt_bf16_f32\b|\bv_perm_b32\b", assembly)
