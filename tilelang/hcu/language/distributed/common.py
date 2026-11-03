"""HCU distributed scalar intrinsics."""

from tvm import DataType, tirx
from tvm.runtime import const
from tvm.tirx import IntImm, PrimExpr


def get_rank():
    """Return the rank associated with the running distributed kernel."""
    return tirx.call_intrin("int32", tirx.op.Op.get("tl.get_rank"))


def get_num_ranks():
    """Return the number of ranks associated with the running kernel."""
    return tirx.call_intrin("int32", tirx.op.Op.get("tl.get_num_ranks"))


def get_block(src: PrimExpr, dst: PrimExpr, size: PrimExpr, src_pe: PrimExpr | IntImm):
    """Fetch a block-scoped range from an IPC peer into local memory.

    The caller guarantees 0 <= src_pe < get_num_ranks(), matching peer arena
    layouts, a source range inside the initialized allocator arena, and valid
    destination storage. Launch does not check source arena ownership.
    Invalid peers are not checked or reported.
    """
    if src_pe is None:
        raise ValueError("get_block requires an explicit source PE")
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.get_block"), src, dst, size, src_pe)


_SIGNAL_OPERATIONS = {"set": 0, "add": 1}
_SIGNAL_COMPARISONS = {"eq": 0, "ne": 1, "gt": 2, "le": 3, "lt": 4, "ge": 5}


def _signal_address(signal):
    if isinstance(signal, tirx.BufferLoad):
        if str(signal.dtype) != "uint64" or signal.buffer.scope() != "global":
            raise ValueError("signal requires a uint64 global-memory element")
        return tirx.address_of(signal)
    if not isinstance(signal, PrimExpr) or str(signal.dtype) != "handle":
        raise ValueError("signal requires a uint64 element or its address")
    if isinstance(signal, tirx.Call) and signal.op.same_as(tirx.op.Op.get("tirx.address_of")):
        element = signal.args[0]
        if isinstance(element, tirx.BufferLoad) and (str(element.dtype) != "uint64" or element.buffer.scope() != "global"):
            raise ValueError("signal requires a uint64 global-memory element")
    return signal


def _signal_value(value):
    if isinstance(value, int):
        if not 0 <= value < (1 << 64):
            raise ValueError("signal value must fit uint64")
        return const(value, "uint64")
    if isinstance(value, IntImm) and int(value.value) < 0:
        raise ValueError("signal value must fit uint64")
    if not isinstance(value, PrimExpr):
        raise ValueError("signal value must be a scalar integer")
    dtype = DataType(value.dtype)
    if dtype.type_code not in (0, 1) or dtype.lanes != 1:
        raise ValueError("signal value must be a scalar integer")
    return tirx.Cast("uint64", value)


def signal_op(signal, value, op, pe):
    """Atomically SET or ADD a uint64 signal on PE ``pe`` (release, system scope).

    ``op`` is "set" or "add". Each calling thread performs one update; this is
    not a block collective. ``signal`` is a uint64 global element or its address
    in the local symmetric IPC arena. The caller guarantees 8-byte alignment,
    a valid PE, and matching arena layouts. Use ``pe=get_rank()`` to publish a
    local signal. Other threads' data writes must be completed and synchronized
    before the publishing thread calls this operation; it is not a block fence.
    """
    if not isinstance(op, str) or op not in _SIGNAL_OPERATIONS:
        raise ValueError('signal_op op must be "set" or "add"')
    if pe is None:
        raise ValueError("signal_op requires an explicit PE")
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.signal_op"),
        _signal_address(signal),
        _signal_value(value),
        _SIGNAL_OPERATIONS[op],
        pe,
    )


def signal_wait_until(signal, cmp, value):
    """Wait for a uint64 signal comparison and return its observed value.

    ``cmp`` is "eq", "ne", "gt", "ge", "lt" or "le". Loads are acquire at
    system scope. Each calling thread waits independently; no block barrier is
    implied. The IPC backend accepts local or directly peer-mapped addresses.
    Other backends may require a local signal. Alignment/lifetime are the
    caller's responsibility. Like SHMEM, this primitive has no timeout; the
    kernel author must ensure progress and avoid cyclic scheduling waits.
    """
    if not isinstance(cmp, str) or cmp not in _SIGNAL_COMPARISONS:
        raise ValueError('signal_wait_until cmp must be "eq", "ne", "gt", "ge", "lt" or "le"')
    return tirx.call_intrin(
        "uint64",
        tirx.op.Op.get("tl.signal_wait_until"),
        _signal_address(signal),
        _SIGNAL_COMPARISONS[cmp],
        _signal_value(value),
    )
