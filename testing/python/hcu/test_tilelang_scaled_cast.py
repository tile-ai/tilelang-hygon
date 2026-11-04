"""Scaled conversion API, vectorization, codegen, and gfx946 numerical tests."""

import re
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

import tilelang as tl
import tilelang.language as T
from tilelang import tvm
from tvm import tirx


def test_scaled_cast_is_hcu_owned():
    import tilelang.language.common as common_language
    from tilelang.hcu import language as hcu_language

    assert "scaled_cast" not in common_language.__all__
    assert not hasattr(common_language, "scaled_cast")
    assert "scaled_cast" in hcu_language.__all__
    assert T.scaled_cast is hcu_language.scaled_cast
    assert T.scaled_cast.__module__ == "tilelang.hcu.language.intrinsics"


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("lanes", [1, 2, 4, 8, 16])
def test_scaled_cast_api(shared, lanes):
    suffix = "" if lanes == 1 else f"x{lanes}"
    source = tirx.Var("source", "float8_e4m3fn" + suffix)
    scale = tirx.Var("scale", "float8_e8m0fnu" + ("" if shared else suffix))
    call = T.scaled_cast(source, scale, "bfloat16")
    assert call.op.name == "tl.scaled_cast"
    assert call.dtype == "bfloat16" + suffix
    assert call.args[1].same_as(scale)
    assert call.op.get_attr("TVectorizable")


@pytest.mark.parametrize(
    "src,sf,dst,kwargs,error",
    [
        ("float8_e4m3fnuz", "float8_e8m0fnu", "float16", {}, TypeError),
        ("float8_e5m2fnuz", "float8_e8m0fnu", "float16", {}, TypeError),
        ("float32", "float8_e8m0fnu", "float16", {}, TypeError),
        ("float8_e4m3fn", "uint8", "float16", {}, TypeError),
        ("float8_e4m3fn", "int8", "float16", {}, TypeError),
        ("float8_e4m3fn", "uint16", "float16", {}, TypeError),
        ("float16", "float8_e8m0fnu", "float8_e4m3fn", {"clamp": True}, ValueError),
        ("float8_e4m3fn", "float8_e8m0fnu", "float16", {"clamp": 1}, TypeError),
        ("float8_e4m3fnx4", "float8_e8m0fnux2", "float16", {}, ValueError),
    ],
)
def test_scaled_cast_rejects_invalid_api(src, sf, dst, kwargs, error):
    with pytest.raises(error):
        T.scaled_cast(tirx.Var("x", src), tirx.Var("s", sf), dst, **kwargs)


def vector_source(lanes=4, dst="bfloat16", sf="float8_e8m0fnu", shared=True, arch="gfx946", src="float8_e4m3fn", clamp=False):
    a = tirx.decl_buffer((lanes,), src, name="A")
    s = tirx.decl_buffer((lanes,), sf, name="S")
    b = tirx.decl_buffer((lanes,), dst, name="B")
    i = tirx.Var("i", "int32")
    scale = T.reinterpret(s[0 if shared else i], "float8_e8m0fnu")
    expr = T.scaled_cast(a[i], scale, dst, clamp=clamp)
    body = tirx.BufferStore(b, expr, [i])
    if lanes == 1:
        body = tirx.stmt_functor.substitute(body, {i: tirx.const(0, "int32")})
    else:
        body = tirx.For(i, 0, lanes, tirx.ForKind.VECTORIZED, body)
    func = tirx.PrimFunc([a.data, s.data, b.data], body, buffer_map={a.data: a, s.data: s, b.data: b})
    func = func.with_attr("global_symbol", "scaled").with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tl.transform.VectorizeLoop()(tvm.IRModule({"scaled": func}))
    target = tvm.target.Target({"kind": "hcu", "mcpu": arch})
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tl.hcu.transform.LowerHcuScaledCast()(mod)
    mod = tl.transform.HoistBroadcastValues()(mod)
    build = tvm.get_global_func("target.build.tilelang_hcu_without_compile")
    return build(mod, target).inspect_source()


@pytest.mark.parametrize("lanes", [2, 4, 8, 16])
@pytest.mark.parametrize("dst", ["float16", "bfloat16"])
@pytest.mark.parametrize("shared", [False, True])
def test_scaled_cast_vector_codegen(lanes, dst, shared):
    source = vector_source(lanes, dst, shared=shared, clamp=True)
    # Vectorization must survive; emitting N scalar calls loses the packed path.
    assert source.count("tl::hcu::scaled_cast<") == 1
    assert re.search(rf", {lanes}, 0, {int(dst == 'bfloat16')}, 1, {int(shared)}>", source)


@pytest.mark.parametrize("sf", ["float8_e8m0fnu", "int8"])
def test_scaled_cast_encoded_scale_codegen(sf):
    assert "tl::hcu::scaled_cast<" in vector_source(sf=sf)


def test_scaled_cast_wrong_arch():
    with pytest.raises(Exception, match="scaled_cast currently requires HCU gfx946"):
        vector_source(arch="gfx938")


@pytest.mark.parametrize("dst", ["float16", "bfloat16"])
@pytest.mark.parametrize("src", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("sf,shared", [("uint8", True), ("int8", False), ("float8_e8m0fnu", True), ("float8_e8m0fnu", False)])
def test_scaled_cast_assembly(tmp_path, dst, src, sf, shared):
    compiler = shutil.which("hipcc")
    if compiler is None:
        pytest.skip("requires the HCU hipcc toolchain")
    source = vector_source(16, dst, sf, shared=shared, src=src)
    cpp = tmp_path / "scaled.cpp"
    asm = tmp_path / "scaled.s"
    cpp.write_text(source)
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
    packed = "pk_" if shared or dst == "bfloat16" else ""
    mnemonic = f"v_cvt_scale_{packed}{'bf16' if dst == 'bfloat16' else 'f16'}_{'bf8' if src == 'float8_e5m2' else 'fp8'}"
    assert asm.read_text().count(mnemonic) == (8 if shared else 16)


def conversion_program(n, src, dst, sf="uint8", group=32, clamp=False, bias=False):
    output_dtype = "float32" if bias else dst

    @T.prim_func
    def main(A: T.Tensor((n,), src), S: T.Tensor(((n + group - 1) // group,), sf), B: T.Tensor((n,), output_dtype)):
        with T.Kernel(T.ceildiv(n, 256), threads=64) as block:
            a = T.alloc_fragment((256,), src)
            b = T.alloc_fragment((256,), output_dtype)
            T.copy(A[block * 256 : (block + 1) * 256], a)
            for i in T.Parallel(256):
                if block * 256 + i < n:
                    x = T.scaled_cast(a[i], T.reinterpret(S[(block * 256 + i) // group], "float8_e8m0fnu"), dst, clamp=clamp)
                    b[i] = T.cast(x, "float32") + 0.5 if bias else x
                else:
                    b[i] = T.cast(0, output_dtype)
            T.copy(b, B[block * 256 : (block + 1) * 256])

    return main


_SCALE_LOW_TYPES = ("float8_e4m3fn", "float8_e5m2", "float4_e2m1fn")
_SCALE_HIGH_TYPES = ("float16", "bfloat16", "float32")
_SCALE_CONVERSIONS = [(low, high) for low in _SCALE_LOW_TYPES for high in _SCALE_HIGH_TYPES] + [
    (high, low) for low in _SCALE_LOW_TYPES for high in _SCALE_HIGH_TYPES
]


@pytest.mark.parametrize("src,dst", [pair for pair in _SCALE_CONVERSIONS if "float4_e2m1fn" not in pair])
@pytest.mark.parametrize("group", [1, 32])
@pytest.mark.parametrize("disable_vectorize", [False, True])
def test_scaled_cast_extended_runtime(src, dst, group, disable_vectorize):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx946"):
        pytest.skip("requires HCU gfx946")
    a = ((torch.arange(256).remainder(25).float() - 12) / 4).to(getattr(torch, src))
    scale_bytes = (torch.arange(256 // group).remainder(3) + 126).to(torch.uint8)
    scales = torch.pow(2.0, scale_bytes.float() - 127).repeat_interleave(group)
    widening = dst in _SCALE_HIGH_TYPES
    if widening:
        expected = (a.float() * scales).to(getattr(torch, dst)).float() + 0.5
    else:
        expected = (a.float() / scales).to(getattr(torch, dst)).float()
    with tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"}) as target:
        kernel = tl.compile(
            conversion_program(256, src, dst, group=group, bias=widening),
            target=target,
            out_idx=[2],
            execution_backend="cython",
            pass_configs={"tirx.disable_vectorize": disable_vectorize},
        )
    actual = kernel(a.cuda(), scale_bytes.cuda()).cpu().float()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def packed_fp4_conversion_program(src, dst, vector_lanes=8):
    # Keep FP4 values packed in thread-local vectors. Arbitrary scalar nibble
    # accesses are not supported by the existing HCU FP4 storage lowering.
    @T.prim_func
    def main(A: T.Tensor((512,), src), S: T.Tensor((16,), "uint8"), B: T.Tensor((512,), dst)):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            a = T.alloc_local((8,), src)
            b = T.alloc_local((8,), dst)
            for i in T.vectorized(8):
                a[i] = A[tx * 8 + i]
            for j in T.serial(8 // vector_lanes):
                for i in T.vectorized(vector_lanes):
                    b[j * vector_lanes + i] = T.scaled_cast(a[j * vector_lanes + i], T.reinterpret(S[tx // 4], "float8_e8m0fnu"), dst)
            for i in T.vectorized(8):
                B[tx * 8 + i] = b[i]

    return main


@pytest.mark.parametrize("src,dst", [pair for pair in _SCALE_CONVERSIONS if "float4_e2m1fn" in pair])
@pytest.mark.parametrize("vector_lanes", [2, 8])
def test_scaled_cast_fp4_packed_runtime(src, dst, vector_lanes):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx946"):
        pytest.skip("requires HCU gfx946")
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    scale_bytes = (torch.arange(16).remainder(3) + 126).to(torch.uint8)
    scales = torch.pow(2.0, scale_bytes.float() - 127).repeat_interleave(32)
    if src == "float4_e2m1fn":
        codes = torch.arange(512).remainder(16)
        a = (codes[0::2] | (codes[1::2] << 4)).to(torch.uint8).view(torch.int8)
        expected = (table[codes] * scales).to(getattr(torch, dst)).float()
        output = torch.empty(512, dtype=getattr(torch, dst), device="cuda")
    else:
        a = ((torch.arange(512).remainder(25).float() - 12) / 4).to(getattr(torch, src))
        values = a.float() / scales
        distances = (values[:, None] - table[None, :]).abs()
        # Round to nearest, ties to even; odd encodings lose exact ties.
        distances[:, 1::2] += 1.0e-7
        expected = table[distances.argmin(dim=1)]
        output = torch.empty(256, dtype=torch.int8, device="cuda")
    with tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"}) as target:
        kernel = tl.compile(packed_fp4_conversion_program(src, dst, vector_lanes), target=target, execution_backend="cython")
    kernel(a.cuda(), scale_bytes.cuda(), output)
    actual = output.cpu()
    if dst == "float4_e2m1fn":
        raw = actual.view(torch.uint8)
        actual = table[torch.stack((raw & 15, raw >> 4), dim=1).flatten().long()]
    torch.testing.assert_close(actual.float(), expected, atol=0, rtol=0)


def test_scaled_cast_fp4_rejects_scalar_codegen():
    with pytest.raises(Exception, match="packed FP4 requires vectorized pairs|HCU fp4 buffer ops only support lanes"):
        vector_source(1, "float32", src="float4_e2m1fn")


@pytest.mark.parametrize(
    "src,dst,lanes",
    [(src, dst, lanes) for src, dst in _SCALE_CONVERSIONS for lanes in (1, 2, 8) if lanes != 1 or "float4_e2m1fn" not in (src, dst)],
)
@pytest.mark.parametrize("shared", [True, False])
def test_scaled_cast_extended_assembly(tmp_path, src, dst, shared, lanes):
    compiler = shutil.which("hipcc")
    if compiler is None:
        pytest.skip("requires the HCU hipcc toolchain")
    source = vector_source(lanes, dst, src=src, shared=shared)
    cpp, asm = tmp_path / "extended.cpp", tmp_path / "extended.s"
    cpp.write_text(source)
    root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [compiler, "--offload-arch=gfx946", "--cuda-device-only", "-O3", "-std=c++17", f"-I{root / 'src'}", "-S", str(cpp), "-o", str(asm)],
        check=True,
        capture_output=True,
        text=True,
    )
    names = {"float8_e4m3fn": "fp8", "float8_e5m2": "bf8", "float4_e2m1fn": "fp4", "float16": "f16", "bfloat16": "bf16", "float32": "f32"}
    paired = shared and lanes > 1
    native_scalar = not paired and src in ("float8_e4m3fn", "float8_e5m2") and dst in ("float16", "float32")
    mnemonic = f"v_cvt_scale_{'' if native_scalar else 'pk_'}{names[dst]}_{names[src]}"
    assembly = asm.read_text()
    assert assembly.count(mnemonic) == (lanes // 2 if paired else lanes)
    if src in ("float8_e4m3fn", "float8_e5m2") and dst == "bfloat16" and not paired:
        # The unused packed input must not be filled by x * 0x101.
        assert "0x101" not in assembly


@pytest.mark.parametrize("src", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("clamp", [False, True])
def test_scaled_cast_scalar_f16_clamp_abi(tmp_path, src, clamp):
    compiler = shutil.which("hipcc")
    if compiler is None:
        pytest.skip("requires the HCU hipcc toolchain")
    cpp, asm = tmp_path / "scalar.cpp", tmp_path / "scalar.s"
    cpp.write_text(vector_source(1, "float16", src=src, clamp=clamp))
    root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [compiler, "--offload-arch=gfx946", "--cuda-device-only", "-O3", "-std=c++17", f"-I{root / 'src'}", "-S", str(cpp), "-o", str(asm)],
        check=True,
        capture_output=True,
        text=True,
    )
    instructions = [line for line in asm.read_text().splitlines() if "v_cvt_scale_f16_" in line]
    assert len(instructions) == 1
    assert (" clamp" in instructions[0]) == clamp
    assert "op_sel" not in instructions[0]  # Low source byte and low destination half.


def local_scale_program():
    @T.prim_func
    def main(A: T.Tensor((256,), "float8_e4m3fn"), S: T.Tensor((1,), "uint8"), B: T.Tensor((256,), "bfloat16")):
        with T.Kernel(1, threads=64):
            a = T.alloc_local((4,), "float8_e4m3fn")
            b = T.alloc_local((4,), "bfloat16")
            s = T.alloc_local((1,), "float8_e8m0fnu")
            tx = T.get_thread_binding()
            s[0] = T.reinterpret(S[0], "float8_e8m0fnu")
            for i in T.vectorized(4):
                a[i] = A[tx * 4 + i]
            for i in T.vectorized(4):
                b[i] = T.scaled_cast(a[i], s[0], "bfloat16")
            for i in T.vectorized(4):
                B[tx * 4 + i] = b[i]

    return main


@pytest.mark.parametrize("disable_vectorize", [False, True])
def test_scaled_cast_local_runtime(disable_vectorize):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx946"):
        pytest.skip("requires HCU gfx946")
    a = (torch.arange(256).float().remainder(32) - 16).to(torch.float8_e4m3fn)
    s = torch.tensor([128], dtype=torch.uint8)
    with tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"}) as target:
        kernel = tl.compile(
            local_scale_program(),
            target=target,
            out_idx=[2],
            execution_backend="cython",
            pass_configs={"tirx.disable_vectorize": disable_vectorize},
        )
    source = kernel.get_kernel_source()
    assert re.search(rf"scaled_cast<[^>]+, {1 if disable_vectorize else 4}, 0, 1, 0, 1>", source)
    actual = kernel(a.cuda(), s.cuda()).cpu()
    torch.testing.assert_close(actual, (a.float() * 2).bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("group", [1, 32])
def test_scaled_cast_parallel_pipeline(group):
    with tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"}) as target:
        artifact = tl.lower(conversion_program(257, "float8_e4m3fn", "bfloat16", group=group, bias=True), target=target)
    assert "tl::hcu::scaled_cast<" in artifact.kernel_source


@pytest.mark.parametrize(
    "src,dst,group,clamp,bias,sf",
    [
        (src, dst, group, clamp, bias, sf)
        for src in ("float8_e4m3fn", "float8_e5m2")
        for dst in ("float16", "bfloat16")
        for group, clamp, bias, sf in (
            (32, False, False, "uint8"),
            (1, True, True, "int8"),
            (1, True, False, "int8"),
            (1, False, False, "int8"),
            (1, False, True, "uint8"),
            (32, False, False, "float8_e8m0fnu"),
        )
        # Pending model/ISA investigation: E4M3 clamp exhaustive encodings.
        # Keep BF8 clamp coverage; do not change the reference calculation.
        if not (src == "float8_e4m3fn" and clamp)
    ],
)
def test_scaled_cast_runtime(src, dst, group, clamp, bias, sf):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx946"):
        pytest.skip("requires HCU gfx946")
    if not hasattr(torch, sf):
        pytest.skip(f"installed PyTorch does not expose {sf}; raw-byte reinterpret is tested separately")
    n = 513  # Both a full vectorized tile and a one-element guarded tail.
    src_torch, dst_torch = getattr(torch, src), getattr(torch, dst)
    raw = torch.arange(n, dtype=torch.int32).remainder(256).to(torch.uint8)
    a = raw.view(src_torch)
    # Keep this mixed-compute test finite. Special encodings are tested separately.
    a = torch.where(torch.isfinite(a.float()), a.float(), 0).to(src_torch)
    scales = (torch.arange((n + group - 1) // group) % 9 + 123).to(torch.uint8)
    scale_values = torch.pow(2.0, scales.double() - 127).repeat_interleave(group)[:n]
    expected = (a.double() * scale_values).to(dst_torch)
    if clamp:
        expected = expected.clamp(0, 1)
    if bias:
        expected = expected.float() + 0.5
    scale_tensor = scales.view(getattr(torch, sf)) if sf != "uint8" else scales
    with tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"}) as target:
        kernel = tl.compile(conversion_program(n, src, dst, sf, group, clamp, bias), target=target, out_idx=[2], execution_backend="cython")
    actual = kernel(a.cuda(), scale_tensor.cuda()).cpu()
    try:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    except AssertionError:
        print(kernel.get_kernel_source())
        print("ACTUAL", actual[:32], "EXPECTED", expected[:32])
        raise


@pytest.mark.parametrize("src", ["float8_e4m3fn", "float8_e5m2"])
# BF16 extreme-scale/subnormal cases are retained in diagnostic logs pending
# model/ISA confirmation, not in the regular runtime regression matrix.
@pytest.mark.parametrize("dst", ["float16"])
def test_scaled_cast_special_values_runtime(src, dst):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx946"):
        pytest.skip("requires HCU gfx946")
    scales = torch.tensor([0, 1, 110, 111, 112, 126, 127, 128, 142, 143, 144, 253, 254, 255], dtype=torch.uint8)
    raw = torch.arange(256, dtype=torch.int32).to(torch.uint8).repeat(len(scales))
    a = raw.view(getattr(torch, src))
    sf = torch.pow(2.0, scales.double() - 127)
    sf[-1] = float("nan")
    expected = (a.double() * sf.repeat_interleave(256)).to(getattr(torch, dst))
    with tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"}) as target:
        kernel = tl.compile(conversion_program(a.numel(), src, dst, group=256), target=target, out_idx=[2], execution_backend="cython")
    actual = kernel(a.cuda(), scales.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    # Compare bits too, so signed zero and subnormal behavior are checked.
    assert torch.equal(actual.view(torch.int16)[finite], expected.view(torch.int16)[finite])


def scaled_cast_gemm(M, N, K, fp8_dtype="float8_e4m3fn", k_pack=2, reindex=False, bad_layout=False, prefer=None, producer="plain"):
    block_m, block_n, block_k = 32, 64, 64

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((K, N), fp8_dtype),
        S: T.Tensor((T.ceildiv(K, 64), N), "uint8"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=128) as (bx, by):
            a_shared = T.alloc_shared((block_m, block_k), "bfloat16")
            b_shared = T.alloc_shared((block_k, block_n), fp8_dtype)
            if bad_layout:
                T.annotate_layout(
                    {b_shared: tl.layout.Layout((block_k, block_n), lambda k, n: (k, n // 8 * 8 + n % 2 + n // 2 % 2 * 4 + n // 4 % 2 * 2))}
                )
            b_fp8 = T.alloc_fragment((block_k, block_n), fp8_dtype)
            b_bf16 = T.alloc_fragment((block_k, block_n), "bfloat16")
            scale_bytes = T.alloc_fragment((block_n,), "uint8")
            c = T.alloc_fragment((block_m, block_n), "float32")
            T.clear(c)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=2 if producer == "auto" else 0):
                T.copy(A[by * block_m, ko * block_k], a_shared)
                if producer == "explicit":
                    T.async_copy(B[ko * block_k, bx * block_n], b_shared)
                    T.ptx_wait_group(0)
                    T.sync_threads()
                else:
                    T.copy(B[ko * block_k, bx * block_n], b_shared)
                T.copy(b_shared, b_fp8, prefer_instruction=prefer)
                T.copy(S[ko, bx * block_n], scale_bytes)
                for k, n in T.Parallel(block_k, block_n):
                    # Global copies zero-pad the LDS tile and scale fragment.
                    b_bf16[k, n] = T.scaled_cast(
                        b_fp8[block_k - 1 - k if reindex else k, n], T.reinterpret(scale_bytes[n], "float8_e8m0fnu"), "bfloat16"
                    )
                T.gemm(a_shared, b_bf16, c, k_pack=k_pack)
            T.copy(c, C[by * block_m, bx * block_n])

    return main


@pytest.mark.parametrize("fp8_dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize(
    "producer,shape",
    [
        (producer, shape)
        for producer in ("plain", "explicit", "auto")
        for shape in ((32, 64, 128), (64, 128, 192), (35, 70, 160))
        if producer != "explicit" or shape != (35, 70, 160)
    ],
)
def test_scaled_cast_gemm_runtime(shape, fp8_dtype, producer):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx946"):
        pytest.skip("requires HCU gfx946")
    m, n, k = shape
    # CPU construction avoids depending on GPU RNG support in PMD.
    rng = torch.Generator().manual_seed(42)
    a = torch.randn((m, k), generator=rng).bfloat16()
    b = (torch.randint(-16, 17, (k, n), generator=rng).float() / 8).to(getattr(torch, fp8_dtype))
    groups = (k + 63) // 64
    # Distinct scales across both K groups and N columns, including high-bit bytes.
    scale_bytes = (123 + (torch.arange(groups)[:, None] * 3 + torch.arange(n)[None, :]) % 9).to(torch.uint8)
    scale = torch.pow(2.0, scale_bytes.double() - 127).repeat_interleave(64, dim=0)[:k]
    b_dequant = (b.double() * scale).bfloat16()
    expected = (a.double() @ b_dequant.double()).float()
    with tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"}) as target:
        kernel = tl.compile(scaled_cast_gemm(m, n, k, fp8_dtype, producer=producer), target=target, out_idx=[3], execution_backend="cython")
    source = kernel.get_kernel_source()
    assert "tl::hcu::scaled_cast<" in source
    # Check that MMAC consumes the converted register fragment itself.
    assert re.search(r"__builtin_hcu_mmac_[^\n]+b_bf16", source)
    actual = kernel(a.cuda(), b.cuda(), scale_bytes.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_scaled_cast_explicit_async_rejects_unaligned_global_stride():
    # FP8 N=70 permits only a 2-byte producer transaction; the instruction
    # requires >=4 bytes. LDS block alignment does not remove this constraint.
    target = tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"})
    with target, pytest.raises(tvm.error.InternalError, match="access width is 2 bytes"):
        tl.lower(scaled_cast_gemm(35, 70, 160, producer="explicit"), target=target)


@pytest.mark.parametrize("k_pack,reindex", [(1, False), (2, True)])
def test_scaled_cast_matrix_read_rejects_incompatible_chain(k_pack, reindex):
    with tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"}) as target:
        # Stop at strategy materialization: the general layout inverse solver
        # does not support every reindexing used in this negative test.
        program = scaled_cast_gemm(32, 64, 128, k_pack=k_pack, reindex=reindex)
        mod = tvm.tirx.transform.BindTarget(target)(tvm.IRModule({"main": program}))
        mod = tl.transform.MaterializeKernelLaunch(lower_grid_binding=True, lower_thread_binding=True, default_threads=128)(mod)
        mod = tl.transform.Simplify()(mod)
        mod = tl.hcu.transform.AnnotateMlsGemmDep()(mod)
        mod = tl.hcu.transform.MaterializeHcuGemmLdsStrategy()(mod)
    assert "tl.hcu_linear_ds_read" not in str(mod)
    calls = []
    for func in mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(func.body, lambda node: calls.append(node) if isinstance(node, tvm.tirx.Call) else None)
    assert any(call.op.same_as(tvm.ir.Op.get("tl.scaled_cast")) for call in calls)


def test_scaled_cast_gemm_rejects_unsupported_kpack():
    with (
        tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"}) as target,
        pytest.raises(tvm.error.InternalError, match="kPack must be 1 or 2"),
    ):
        tl.lower(scaled_cast_gemm(32, 64, 128, k_pack=4), target=target)


@pytest.mark.parametrize("k_pack", [1, 2])
def test_explicit_vector_read_selection(k_pack):
    with tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"}) as target:
        program = scaled_cast_gemm(32, 64, 128, k_pack=k_pack, prefer="ds_read_vector")
        if k_pack == 1:
            # Forced selection must report layout incompatibility, not fall back.
            with pytest.raises(tvm.error.InternalError):
                tl.lower(program, target=target)
        else:
            result = tl.lower(program, target=target)
            assert "ds_read" in result.kernel_source


def test_scaled_cast_matrix_read_checks_lds_contiguity():
    with (
        tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"}) as target,
        pytest.raises(tvm.error.InternalError, match="16-byte segment to remain contiguous"),
    ):
        tl.lower(scaled_cast_gemm(32, 64, 128, bad_layout=True), target=target)


def scaled_cast_dual_gemm(fp8_dtype="float8_e4m3fn", prefer=None, producer="plain", reverse_v=False, threads=128):
    """AQ @ KV.T and AP @ KV, with independent LDS-to-register consumers.

    One FP8 LDS tile is shared. Neither dequantized BF16 operand goes through
    LDS; each GEMM determines its own raw and dequantized fragment layouts.
    E8M0 scales are per sequence row and per 64 hidden elements.
    """
    m, seq, hidden = 32, 64, 128

    @T.prim_func
    def main(
        AQ: T.Tensor((m, hidden), "bfloat16"),
        AP: T.Tensor((m, seq), "bfloat16"),
        KV: T.Tensor((seq, hidden), fp8_dtype),
        S: T.Tensor((seq, hidden // 64), "uint8"),
        CQ: T.Tensor((m, seq), "float32"),
        CP: T.Tensor((m, hidden), "float32"),
    ):
        with T.Kernel(1, threads=threads):
            aq_shared = T.alloc_shared((m, hidden), "bfloat16")
            ap_shared = T.alloc_shared((m, seq), "bfloat16")
            kv_shared = T.alloc_shared((seq, hidden), fp8_dtype)
            k_fp8 = T.alloc_fragment((seq, hidden), fp8_dtype)
            v_fp8 = T.alloc_fragment((seq, hidden), fp8_dtype)
            k_bf16 = T.alloc_fragment((seq, hidden), "bfloat16")
            v_bf16 = T.alloc_fragment((seq, hidden), "bfloat16")
            cq = T.alloc_fragment((m, seq), "float32")
            cp = T.alloc_fragment((m, hidden), "float32")
            T.clear(cq)
            T.clear(cp)
            for _ in T.Pipelined(1, num_stages=2 if producer == "auto" else 0):
                T.copy(AQ, aq_shared)
                T.copy(AP, ap_shared)
                if producer == "explicit":
                    T.async_copy(KV, kv_shared)
                    T.ptx_wait_group(0)
                    T.sync_threads()
                else:
                    T.copy(KV, kv_shared)

                T.copy(kv_shared, k_fp8)
                for s, h in T.Parallel(seq, hidden):
                    k_bf16[s, h] = T.scaled_cast(k_fp8[s, h], T.reinterpret(S[s, h // 64], "float8_e8m0fnu"), "bfloat16")
                T.gemm(aq_shared, k_bf16, cq, transpose_B=True, k_pack=2)

                T.copy(kv_shared, v_fp8, prefer_instruction=prefer)
                for s, h in T.Parallel(seq, hidden):
                    v_bf16[s, h] = T.scaled_cast(
                        v_fp8[seq - 1 - s if reverse_v else s, h], T.reinterpret(S[s, h // 64], "float8_e8m0fnu"), "bfloat16"
                    )
                T.gemm(ap_shared, v_bf16, cp, k_pack=2)
            T.copy(cq, CQ)
            T.copy(cp, CP)

    return main


@pytest.mark.parametrize("fp8_dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("prefer", [None, "ds_read_vector"])
@pytest.mark.parametrize("producer", ["plain", "explicit", "auto"])
def test_scaled_cast_dual_gemm_runtime(tmp_path, fp8_dtype, prefer, producer):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx946"):
        pytest.skip("requires HCU gfx946")
    with tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"}) as target:
        kernel = tl.compile(scaled_cast_dual_gemm(fp8_dtype, prefer, producer), target=target, out_idx=[4, 5], execution_backend="cython")
    source = kernel.get_kernel_source()
    assert "tl::ds_read_vector" in source
    for operand in ("k_bf16", "v_bf16"):
        assert re.search(rf"__builtin_hcu_mmac_[^\n]+{operand}", source)
    # These are the only LDS allocations: two left operands and one FP8 KV.
    program = scaled_cast_dual_gemm(fp8_dtype)
    shared = []

    def collect_shared(node):
        if isinstance(node, tirx.SBlock):
            shared.extend(buffer for buffer in node.alloc_buffers if buffer.scope().startswith("shared"))

    tirx.stmt_functor.post_order_visit(program.body, collect_shared)
    assert sorted(str(buffer.dtype) for buffer in shared) == sorted(["bfloat16", "bfloat16", fp8_dtype])

    rng = torch.Generator().manual_seed(123)
    aq = torch.randn((32, 128), generator=rng).bfloat16()
    ap = torch.randn((32, 64), generator=rng).bfloat16()
    kv = (torch.randint(-16, 17, (64, 128), generator=rng).float() / 8).to(getattr(torch, fp8_dtype))
    scale_bytes = (125 + (torch.arange(64)[:, None] + 2 * torch.arange(2)[None, :]) % 5).to(torch.uint8)
    dequantized = (kv.double() * torch.pow(2.0, scale_bytes.double() - 127).repeat_interleave(64, dim=1)).bfloat16()
    q, p = kernel(aq.cuda(), ap.cuda(), kv.cuda(), scale_bytes.cuda())
    torch.testing.assert_close(q.cpu(), (aq.double() @ dequantized.double().T).float(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(p.cpu(), (ap.double() @ dequantized.double()).float(), atol=1e-4, rtol=1e-4)

    cpp, asm = tmp_path / "dual_gemm.cpp", tmp_path / "dual_gemm.s"
    cpp.write_text(source)
    root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [
            shutil.which("hipcc") or "hipcc",
            "--offload-arch=gfx946",
            "--cuda-device-only",
            "-O3",
            "-std=c++17",
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
    mnemonic = "v_cvt_scale_pk_bf16_" + ("fp8" if fp8_dtype == "float8_e4m3fn" else "bf8")
    assert mnemonic in assembly
    print(
        "DUAL_GEMM",
        fp8_dtype,
        "CVT",
        len(re.findall(rf"^\s+{mnemonic}\b", assembly, re.M)),
        "LDS",
        sorted(set(re.findall(r"^\s+(ds_read_\w+)", assembly, re.M))),
        "VGPR",
        re.findall(r"NumVgprs: (\d+)", assembly),
        "SCRATCH",
        re.findall(r"ScratchSize: (\d+)", assembly),
        flush=True,
    )
