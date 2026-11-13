"""A native matrix read and other consumers agree on shared storage."""

import pytest
import torch
import tilelang as tl
import tilelang.hcu.language as T
import tilelang.testing
from hcu_test_utils import current_hcu_arch_string


@pytest.mark.parametrize("direct_gemm", [False, True])
@pytest.mark.parametrize("read_first", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_shared_source_with_direct_gemm_and_copy(read_first, explicit, direct_gemm):
    @T.prim_func
    def main(
        A: T.Tensor((32, 128), "bfloat16"),
        B: T.Tensor((32, 128), "bfloat16"),
        C: T.Tensor((128, 128), "float32"),
        D: T.Tensor((128, 128), "float32"),
        E: T.Tensor((32, 128), "bfloat16"),
    ):
        with T.Kernel(1, threads=512):
            a = T.alloc_shared((32, 128), "bfloat16")
            b = T.alloc_shared((32, 128), "bfloat16")
            if explicit:
                T.annotate_layout(
                    {
                        a: T.Layout((32, 128), lambda i, j: (i, j)),
                        b: T.Layout((32, 128), lambda i, j: (i, j)),
                    }
                )
            tx = T.get_thread_binding()
            if tx < 256:
                T.copy(A, a)
                T.copy(B, b)
            T.sync_threads()
            if tx >= 256:
                af = T.alloc_fragment((32, 128), "bfloat16")
                bf = T.alloc_fragment((32, 128), "bfloat16")
                cf = T.alloc_fragment((128, 128), "float32")
                df = T.alloc_fragment((128, 128), "float32")
                if direct_gemm and not read_first:
                    T.gemm(a, b, df, transpose_A=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True, k_pack=1)
                T.ds_read_vector(a, af, alt=2)
                T.ds_read_vector(b, bf, alt=2)
                T.s_waitcnt(0, "lgkmcnt")
                T.gemm(af, bf, cf, transpose_A=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True, k_pack=1)
                if direct_gemm and read_first:
                    T.gemm(a, b, df, transpose_A=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True, k_pack=1)
                T.copy(cf, C)
                if direct_gemm:
                    T.copy(df, D)
                else:
                    T.copy(cf, D)
                T.copy(b, E)

    # Source storage is now attached before GEMM lowering chooses its access
    # mode. Both consumer orders must agree, including inferred storage.
    kernel = tl.compile(main, out_idx=[2, 3, 4], target={"kind": "hcu", "mcpu": "gfx938"})
    torch.manual_seed(777)
    a = torch.randint(-2, 3, (32, 128), device="cuda").to(torch.bfloat16)
    b = torch.randint(-2, 3, (32, 128), device="cuda").to(torch.bfloat16)
    c, d, e = kernel(a, b)
    ref = a.float().T @ b.float()
    torch.testing.assert_close(c, ref, atol=0, rtol=0)
    torch.testing.assert_close(d, ref, atol=0, rtol=0)
    torch.testing.assert_close(e, b, atol=0, rtol=0)
