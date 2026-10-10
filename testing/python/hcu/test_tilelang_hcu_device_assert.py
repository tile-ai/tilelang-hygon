import torch

import tilelang
import tilelang.language as T


def test_hcu_device_assert_facade_codegen_and_execution():
    @T.prim_func
    def kernel(A: T.Tensor((64,), T.float32), B: T.Tensor((64,), T.float32)):
        with T.Kernel(1, threads=64):
            i = T.get_thread_binding()
            T.device_assert(i < 64, no_stack_info=True)
            B[i] = A[i]

    compiled = tilelang.compile(kernel, out_idx=-1)
    assert "device_assert(" in compiled.get_kernel_source()
    A = torch.randn(64, dtype=torch.float32, device="cuda")
    B = compiled(A)
    torch.testing.assert_close(B, A, atol=0, rtol=0)
