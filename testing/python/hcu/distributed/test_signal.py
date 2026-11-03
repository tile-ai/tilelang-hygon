"""Signal API validation and source-only HCU lowering; no device launch."""

import pytest

import tilelang as tl
from tilelang import tvm
from tilelang.hcu import language as T
from tilelang.distributed.metadata import requirements_for_backend
from tilelang.hcu.language.distributed import signal_op, signal_wait_until
from tvm import tirx


@pytest.mark.skipif("hcu" not in tvm.target.Target.list_kinds(), reason="Requires the HCU code generator")
def test_signal_codegen():
    @T.prim_func
    def main(signal: T.Tensor((1,), "uint64"), observed: T.Tensor((6,), "uint64")):
        with T.Kernel(1, threads=64):
            if T.get_thread_binding() == 0:
                T.signal_op(signal[0], (1 << 63) + 7, "set", T.get_rank())
                T.signal_op(T.address_of(signal[0]), 1, "add", T.get_rank())
                observed[0] = T.signal_wait_until(signal[0], "eq", 1)
                observed[1] = T.signal_wait_until(signal[0], "ne", 1)
                observed[2] = T.signal_wait_until(signal[0], "gt", 1)
                observed[3] = T.signal_wait_until(signal[0], "le", 1)
                observed[4] = T.signal_wait_until(signal[0], "lt", 1)
                observed[5] = T.signal_wait_until(T.address_of(signal[0]), "ge", 1)

    target = tvm.target.Target({"kind": "hcu", "mcpu": "gfx938", "dist_backend": "ipc"})
    with target:
        result = tl.lower(main, target=target, enable_host_codegen=False, enable_device_compile=False)
    source = result.kernel_source
    assert "tl::distributed_signal_op<0>" in source
    assert "tl::distributed_signal_op<1>" in source
    for comparison in range(6):
        assert f"tl::distributed_signal_wait_until<{comparison}>" in source
    assert "9223372036854775815" in source
    assert "// tilelang_distributed_metadata: 1" in source
    assert "// tilelang_distributed_abi: 1:1" in source
    assert "signal" in requirements_for_backend("ipc")["capabilities"]


@pytest.mark.parametrize("operation", ["set", "add"])
def test_signal_ir(operation):
    flag = tirx.decl_buffer((1,), "uint64")
    call = signal_op(flag[0], 1, operation, 0)
    assert call.op.same_as(tirx.op.Op.get("tl.signal_op"))
    value = tirx.Var("epoch", "uint64")
    dynamic = signal_op(flag[0], value, operation, 0)
    assert str(dynamic.args[1].dtype) == "uint64"
    wait = signal_wait_until(tirx.address_of(flag[0]), "ge", 1)
    assert str(wait.dtype) == "uint64"
    assert wait.op.same_as(tirx.op.Op.get("tl.signal_wait_until"))


@pytest.mark.skipif("hcu" not in tvm.target.Target.list_kinds(), reason="Requires the HCU code generator")
def test_signal_requires_distributed_target():
    @T.prim_func
    def main(signal: T.Tensor((1,), "uint64")):
        with T.Kernel(1, threads=64):
            if T.get_thread_binding() == 0:
                T.signal_op(signal[0], 1, "set", 0)

    target = tvm.target.Target({"kind": "hcu", "mcpu": "gfx938"})
    with target, pytest.raises(ValueError, match="dist_backend"):
        tl.lower(main, target=target, enable_host_codegen=False, enable_device_compile=False)


def test_signal_rejects_invalid_arguments():
    flag = tirx.decl_buffer((1,), "uint64")
    with pytest.raises(ValueError, match="op must"):
        signal_op(flag[0], 1, "exchange", 0)
    with pytest.raises(ValueError, match="cmp must"):
        signal_wait_until(flag[0], "invalid", 1)
    with pytest.raises(ValueError, match="explicit PE"):
        signal_op(flag[0], 1, "set", None)
    for value in (-1, 1 << 64):
        with pytest.raises(ValueError, match="fit uint64"):
            signal_op(flag[0], value, "set", 0)
    with pytest.raises(ValueError, match="scalar integer"):
        signal_wait_until(flag[0], "eq", tirx.FloatImm("float32", 1.5))
    for dtype, scope in (("int32", "global"), ("uint64", "shared")):
        invalid = tirx.decl_buffer((1,), dtype, scope=scope)
        for address in (invalid[0], tirx.address_of(invalid[0])):
            with pytest.raises(ValueError, match="uint64 global"):
                signal_op(address, 1, "set", 0)
