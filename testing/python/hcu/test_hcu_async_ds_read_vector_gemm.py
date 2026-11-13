"""Automatic LDS layouts with an async producer and separate GEMM consumers."""

import pytest
import torch
import tilelang as tl
import tilelang.hcu.language as T
import tilelang.testing
from hcu_test_utils import current_hcu_arch_string


@pytest.mark.parametrize("entry,alt", [("copy", 1), ("region", 1), ("region", 2)])
@pytest.mark.parametrize("rows,width", [(32, 64), (32, 128)])
@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_async_copy_ds_read_vector_gemm(rows, width, entry, alt):
    @T.prim_func
    def main(A: T.Tensor((rows, 128), "bfloat16"), B: T.Tensor((rows, width), "bfloat16"), C: T.Tensor((128, width), "float32")):
        with T.Kernel(1, threads=512):
            a = T.alloc_shared((rows, 128), "bfloat16")
            b = T.alloc_shared((rows, width), "bfloat16")
            tx = T.get_thread_binding()
            if tx < 256:
                T.async_copy(A, a, coalesced_width=8)
                T.async_copy(B, b, coalesced_width=8)
                T.ptx_wait_group(0)
                T.s_waitcnt(0, "vmcnt")
                T.s_waitcnt(0, "lgkmcnt")
            T.sync_threads()
            if tx >= 256:
                af = T.alloc_fragment((rows, 128), "bfloat16")
                bf = T.alloc_fragment((rows, width), "bfloat16")
                cf = T.alloc_fragment((128, width), "float32")
                if entry == "copy":
                    T.copy(a, af, prefer_instruction="ds_read_vector")
                    T.copy(b, bf, prefer_instruction="ds_read_vector")
                else:
                    T.ds_read_vector(a, af, alt=alt)
                    T.ds_read_vector(b, bf, alt=alt)
                T.s_waitcnt(0, "lgkmcnt")
                T.gemm(af, bf, cf, transpose_A=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True, k_pack=1)
                T.copy(cf, C)

    kernel = tl.compile(main, out_idx=[2], target={"kind": "hcu", "mcpu": "gfx938"})
    torch.manual_seed(777)
    a = torch.randint(-2, 3, (rows, 128), device="cuda").to(torch.bfloat16)
    b = torch.randint(-2, 3, (rows, width), device="cuda").to(torch.bfloat16)
    for trial in range(20):
        torch.testing.assert_close(kernel(a, b), a.float().T @ b.float(), atol=0, rtol=0, msg=f"trial={trial}")
