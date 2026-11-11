"""Explicit layouts preserve computation in a consumer-only thread interval."""

import pytest
import torch
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from hcu_test_utils import current_hcu_arch_string


@pytest.mark.parametrize("annotate_buffer", [False, True])
@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_explicit_consumer_fragment_thread_range(annotate_buffer):
    layout = T.Fragment((64, 4), forward_thread_fn=lambda i, j: i, forward_index_fn=lambda i, j: j)

    @T.prim_func
    def main(A: T.Tensor((64, 4), "float32"), B: T.Tensor((64, 4), "float32")):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            if tx >= 64:
                x = T.alloc_fragment((64, 4), "float32")
                if annotate_buffer:
                    T.annotate_layout({x: layout})
                T.copy(A, x, loop_layout=layout)
                for i, j in T.Parallel(64, 4, loop_layout=layout):
                    x[i, j] = x[i, j] * 3 + 1
                T.copy(x, B, loop_layout=layout)

    kernel = tl.compile(main, out_idx=[1], target={"kind": "hcu", "mcpu": "gfx938"})
    a = torch.arange(256, device="cuda", dtype=torch.float32).reshape(64, 4)
    torch.testing.assert_close(kernel(a), a * 3 + 1, atol=0, rtol=0)
