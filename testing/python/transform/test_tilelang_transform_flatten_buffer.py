import tilelang
import tilelang.testing
from tilelang import tvm
import tilelang.language as T


def _collect_tvm_access_ptr_offsets(func: tvm.tirx.PrimFunc):
    offsets = []

    def _visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op) and str(node.op.name) == "tirx.tvm_access_ptr":
            offsets.append(node.args[2])

    tvm.tirx.stmt_functor.post_order_visit(func.body, _visit)
    return offsets


def test_flatten_buffer_promotes_tvm_access_ptr_offset_to_int64():
    @T.prim_func
    def before(A: T.Buffer((1,), "float16")):
        for i in T.serial(1 << 30):
            T.evaluate(
                T.tvm_access_ptr(
                    T.type_annotation(T.float16),
                    A.data,
                    i * 4,
                    1,
                    1,
                )
            )

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    before_offsets = _collect_tvm_access_ptr_offsets(mod["main"])
    assert len(before_offsets) == 1
    assert str(before_offsets[0].dtype) == "int32"

    mod = tilelang.transform.FlattenBuffer()(mod)
    after_offsets = _collect_tvm_access_ptr_offsets(mod["main"])
    assert len(after_offsets) == 1
    assert str(after_offsets[0].dtype) == "int64"


def test_flatten_buffer_keeps_safe_tvm_access_ptr_offset_int32():
    @T.prim_func
    def before(A: T.Buffer((1,), "float16")):
        for i in T.serial(1 << 20):
            T.evaluate(
                T.tvm_access_ptr(
                    T.type_annotation(T.float16),
                    A.data,
                    i * 2,
                    1,
                    1,
                )
            )

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    before_offsets = _collect_tvm_access_ptr_offsets(mod["main"])
    assert len(before_offsets) == 1
    assert str(before_offsets[0].dtype) == "int32"

    mod = tilelang.transform.FlattenBuffer()(mod)
    after_offsets = _collect_tvm_access_ptr_offsets(mod["main"])
    assert len(after_offsets) == 1
    assert str(after_offsets[0].dtype) == "int32"


def test_flatten_buffer_preserves_nested_div_mod_offset():
    @T.prim_func
    def before(ready: T.Buffer((24, 24, 32), "uint64"), output: T.Buffer((80, 32), "uint64")):
        for block_id in T.serial(80):
            for turn in T.serial(32):
                if block_id >= 64:
                    if block_id - 64 + turn * 16 < 504:
                        output[block_id, turn] = ready[
                            ((block_id - 64 + turn * 16) // 24 + 9) % 24,
                            (block_id - 64 + turn * 16) % 24,
                            0,
                        ]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tilelang.transform.FlattenBuffer()(mod)
    loads = []

    def _visit(node):
        if isinstance(node, tvm.tirx.BufferLoad) and node.buffer.name.startswith("ready"):
            loads.append(node)

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, _visit)
    assert len(loads) == 1
    index = loads[0].indices[0]
    variables = {}

    def _collect_var(node):
        if isinstance(node, tvm.tirx.Var):
            variables[node.name] = node

    tvm.tirx.stmt_functor.post_order_visit(index, _collect_var)
    analyzer = tvm.arith.Analyzer()
    for turn, block_id in [(0, 64), (0, 72), (31, 71)]:
        task = block_id - 64 + turn * 16
        row, panel = divmod(task, 24)
        expected = (((row + 9) % 24) * 24 + panel) * 32
        actual = tvm.tirx.stmt_functor.substitute(
            index,
            {
                variables["turn"]: tvm.tirx.const(turn, "int32"),
                variables["block_id"]: tvm.tirx.const(block_id, "int32"),
            },
        )
        assert analyzer.simplify(actual).value == expected


if __name__ == "__main__":
    tilelang.testing.main()
