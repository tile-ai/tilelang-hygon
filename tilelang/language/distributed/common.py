"""Backend-neutral distributed scalar intrinsics."""

from tvm import tirx


def get_rank():
    """Return the rank associated with the running distributed kernel."""
    return tirx.call_intrin("int32", tirx.op.Op.get("tl.get_rank"))


def get_num_ranks():
    """Return the number of ranks associated with the running kernel."""
    return tirx.call_intrin("int32", tirx.op.Op.get("tl.get_num_ranks"))
