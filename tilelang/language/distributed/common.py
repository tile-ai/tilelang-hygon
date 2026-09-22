"""Backend-neutral distributed scalar intrinsics."""

from tvm import tirx
from tvm.tirx import IntImm, PrimExpr


def get_rank():
    """Return the rank associated with the running distributed kernel."""
    return tirx.call_intrin("int32", tirx.op.Op.get("tl.get_rank"))


def get_num_ranks():
    """Return the number of ranks associated with the running kernel."""
    return tirx.call_intrin("int32", tirx.op.Op.get("tl.get_num_ranks"))


def get_block(src: PrimExpr, dst: PrimExpr, size: PrimExpr, src_pe: PrimExpr | IntImm):
    """Fetch a block-scoped range from an IPC peer into local memory."""
    if src_pe is None:
        raise ValueError("get_block requires an explicit source PE")
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.get_block"), src, dst, size, src_pe)
