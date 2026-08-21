import tilelang as tl
import tilelang.language as T


@T.prim_func
def main(
    A: T.Tensor((256, 16), T.float16),
    B: T.Tensor((16, 32), T.float16),
    C: T.Tensor((256, 32), T.float32),
):
    with T.Kernel(1, threads=512):
        A_shared = T.alloc_shared((256, 16), T.float16)
        B_shared = T.alloc_shared((16, 32), T.float16)
        A_local = T.alloc_fragment((256, 16), T.float16)
        B_local = T.alloc_fragment((16, 32), T.float16)
        C_local = T.alloc_fragment((256, 32), T.float32)
        T.clear(C_local)

        T.async_copy(A, A_shared)
        T.async_copy(B, B_shared)
        T.ptx_wait_group(0)
        T.sync_threads()
        T.copy(A_shared, A_local)
        T.copy(B_shared, B_local)
        T.s_waitcnt(0, "lgkmcnt")
        T.gemm(A_local, B_local, C_local, False, False)
        T.copy(C_local, C)


@T.prim_func
def main_an_identity(
    A: T.Tensor((16, 32), T.float16),
    B: T.Tensor((16, 32), T.float16),
    C: T.Tensor((32, 32), T.float32),
):
    with T.Kernel(1, threads=64):
        A_shared = T.alloc_shared((16, 32), T.float16)
        B_shared = T.alloc_shared((16, 32), T.float16)
        A_local = T.alloc_fragment((16, 32), T.float16)
        B_local = T.alloc_fragment((16, 32), T.float16)
        C_local = T.alloc_fragment((32, 32), T.float32)
        T.clear(C_local)

        T.async_copy(A, A_shared)
        T.async_copy(B, B_shared)
        T.ptx_wait_group(0)
        T.sync_threads()
        T.copy(A_shared, A_local)
        T.copy(B_shared, B_local)
        T.s_waitcnt(0, "lgkmcnt")
        T.gemm(A_local, B_local, C_local, True, False)
        T.copy(C_local, C)


def test_hcu_gemm_bt_identity_layout():
    kernel = tl.compile(main, out_idx=[2])
    profiler = kernel.get_profiler()
    profiler.assert_allclose(
        lambda a, b: a.float() @ b.float(), atol=1e-2, rtol=1e-2
    )


def test_hcu_gemm_an_identity_layout():
    kernel = tl.compile(main_an_identity, out_idx=[2])
    profiler = kernel.get_profiler()
    profiler.assert_allclose(
        lambda a, b: a.T.float() @ b.float(),
        atol=1e-2,
        rtol=1e-2,
    )


if __name__ == "__main__":
    test_hcu_gemm_bt_identity_layout()
    test_hcu_gemm_an_identity_layout()
