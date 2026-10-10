"""HCU-owned annotation helpers."""

from tvm.tirx import IntImm
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


__all__ = ("annotate_direct_to_lds", "annotate_buffer_ops_rebase")
