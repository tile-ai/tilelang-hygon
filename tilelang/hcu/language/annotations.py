"""HCU-owned annotation helpers."""

from tvm.tirx import IntImm, PrimExpr, Var
from tvm.tirx.script.builder.ir import attr
from tvm.tirx.script.builder.ir import sblock_attr


def _buffer_flag_map(buffers):
    if isinstance(buffers, dict):
        return {buffer.data: IntImm("int32", int(bool(enabled))) for buffer, enabled in buffers.items()}
    if isinstance(buffers, (list, tuple)):
        return {buffer.data: IntImm("int32", 1) for buffer in buffers}
    return {buffers.data: IntImm("int32", 1)}


def annotate_direct_to_lds(buffers):
    return sblock_attr({"tl.hcu.direct_to_lds": _buffer_flag_map(buffers)})


def annotate_buffer_ops_rebase(buffers):
    return sblock_attr({"tl.hcu.buffer_ops_rebase_map": _buffer_flag_map(buffers)})


def thread_binding_alias(thread_var, local_index):
    """Use an equal task-local thread index inside a lexical HCU codegen scope.

    The caller must keep ``local_index`` equal to ``thread_var`` throughout
    the scope. This changes register lifetime, not logical thread mapping.
    Layout inference still sees the original thread binding; generated loads
    and stores use the alias. Nested scopes restore the enclosing alias.
    """
    if not isinstance(thread_var, Var):
        raise TypeError("thread_var must be a bound thread variable")
    if not isinstance(local_index, PrimExpr):
        raise TypeError("local_index must be an integer PrimExpr")
    return attr(thread_var, "tl.hcu.thread_binding_alias", local_index)


__all__ = ("annotate_direct_to_lds", "annotate_buffer_ops_rebase", "thread_binding_alias")
