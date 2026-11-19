"""HCU thread-index alias scopes must restore nested and outer bindings."""

import pytest
import tilelang
from tilelang import tvm
import tilelang.hcu.language as T


def source_for_scope(alias_dtype="int32", unbound=False):
    tir = tvm.tirx
    tx = tir.Var("tx", "int32")
    iv = tir.IterVar(tvm.ir.Range(0, 64), tx, tir.IterVar.ThreadIndex, "threadIdx.x")
    a = tir.Var("local_a", alias_dtype)
    b = tir.Var("local_b", "int32")

    def emit(name):
        return tir.Evaluate(tir.call_extern("void", name, tx))

    body = tir.SeqStmt(
        [
            emit("before"),
            tir.AttrStmt(
                tir.Var("unknown", "int32") if unbound else tx,
                "tl.hcu.thread_binding_alias",
                a,
                tir.SeqStmt([emit("outer_before"), tir.AttrStmt(tx, "tl.hcu.thread_binding_alias", b, emit("inner")), emit("outer_after")]),
            ),
            emit("after"),
        ]
    )
    body = tir.AttrStmt(iv, "thread_extent", 64, body)
    func = tir.PrimFunc([a, b], body).with_attr({"global_symbol": "alias_scope", "calling_conv": 2})
    build = tvm.get_global_func("target.build.tilelang_hcu_without_compile")
    return build(tvm.IRModule({"main": func}), tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"})).inspect_source()


def test_nested_scope_restores_binding():
    source = source_for_scope()
    assert "before(((int)threadIdx.x))" in source
    assert "outer_before(local_a)" in source
    assert "inner(local_b)" in source
    assert "outer_after(local_a)" in source
    assert "after(((int)threadIdx.x))" in source


@pytest.mark.parametrize("dtype", ["float32", "int32x2"])
def test_reject_non_scalar_integer_alias(dtype):
    with pytest.raises(Exception, match="scalar integer alias"):
        source_for_scope(alias_dtype=dtype)


def test_reject_unbound_variable():
    with pytest.raises(Exception, match="bound thread variable"):
        source_for_scope(unbound=True)


def test_facade_rejects_non_variable():
    with pytest.raises(TypeError, match="bound thread variable"):
        T.thread_binding_alias(0, tvm.tirx.IntImm("int32", 0))


def test_storage_rewrite_tracks_local_alias_buffer():
    tir = tvm.tirx
    tx = tir.Var("tx", "int32")
    iv = tir.IterVar(tvm.ir.Range(0, 64), tx, tir.IterVar.ThreadIndex, "threadIdx.x")
    lane = tir.decl_buffer((1,), "int32", name="task_lane", scope="local")
    body = tir.SeqStmt(
        [
            tir.AllocBuffer(lane),
            tir.BufferStore(lane, tx, [0]),
            tir.AttrStmt(tx, "tl.hcu.thread_binding_alias", lane[0], tir.Evaluate(tir.call_extern("void", "consume_lane", tx))),
        ]
    )
    body = tir.AttrStmt(iv, "thread_extent", 64, body)
    func = tir.PrimFunc([], body).with_attr({"global_symbol": "buffer_alias", "calling_conv": 2})
    mod = tilelang.transform.StorageRewrite()(tvm.IRModule({"main": func}))
    build = tvm.get_global_func("target.build.tilelang_hcu_without_compile")
    source = build(mod, tvm.target.Target({"kind": "hcu", "mcpu": "gfx946"})).inspect_source()
    assert "consume_lane(task_lane[0])" in source
