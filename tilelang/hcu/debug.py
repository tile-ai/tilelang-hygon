"""HCU device-side debugging helpers."""

from __future__ import annotations

import warnings

from tvm import tirx

import tilelang.language as T
from tilelang.language.eager.builder import Builder, macro


def get_stack_str(msg: str, stacklevel: int = 1) -> str:
    stack = Builder.current().get_fileline_stack(stacklevel)
    msg += "\n"
    for fileline, lineno, macro_name in stack:
        msg += f"  at {fileline}:{lineno} in {macro_name}\n"
    return msg


@macro
def device_assert(condition: tirx.PrimExpr, msg: str = "", no_stack_info: bool = False):
    """Emit an HCU device assertion, optionally with DSL source context."""
    if no_stack_info and msg == "":
        T.call_intrin("void", tirx.op.Op.get("tl.device_assert"), condition)
    else:
        if no_stack_info:
            warnings.warn("Non-empty msg may slightly slow down the kernel", stacklevel=2)
            assert_msg = msg
        else:
            assert_msg = get_stack_str(msg, stacklevel=2)
        T.call_intrin("void", tirx.op.Op.get("tl.device_assert_with_msg"), condition, assert_msg)


__all__ = ["device_assert"]
