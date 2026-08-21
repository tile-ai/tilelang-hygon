import tilelang as tl
import tilelang.language as T


@T.prim_func
def main(
    X: T.Tensor((64, 64), T.float16),
    B_at: T.Tensor((64, 64), T.float16),
    B_an: T.Tensor((64, 64), T.float16),
    C_at: T.Tensor((64, 64), T.float32),
    C_an: T.Tensor((64, 64), T.float32),
):
    with T.Kernel(1, threads=256):
        X_shared = T.alloc_shared((64, 64), T.float16)
        B_at_shared = T.alloc_shared((64, 64), T.float16)
        B_an_shared = T.alloc_shared((64, 64), T.float16)
        X_at_local = T.alloc_fragment((64, 64), T.float16)
        X_an_local = T.alloc_fragment((64, 64), T.float16)
        B_at_local = T.alloc_fragment((64, 64), T.float16)
        B_an_local = T.alloc_fragment((64, 64), T.float16)
        C_at_local = T.alloc_fragment((64, 64), T.float32)
        C_an_local = T.alloc_fragment((64, 64), T.float32)
        T.clear(C_at_local)
        T.clear(C_an_local)

        T.async_copy(X, X_shared)
        T.async_copy(B_at, B_at_shared)
        T.async_copy(B_an, B_an_shared)
        T.ptx_wait_group(0)
        T.sync_threads()

        T.copy(X_shared, X_at_local)
        T.copy(B_at_shared, B_at_local)
        T.s_waitcnt(0, "lgkmcnt")
        T.gemm(X_at_local, B_at_local, C_at_local, False, False)

        T.copy(X_shared, X_an_local)
        T.copy(B_an_shared, B_an_local)
        T.s_waitcnt(0, "lgkmcnt")
        T.gemm(X_an_local, B_an_local, C_an_local, True, False)

        T.copy(C_at_local, C_at)
        T.copy(C_an_local, C_an)


def test_hcu_gemm_mixed_lds_access():
    kernel = tl.compile(main, out_idx=[3, 4])
    source = kernel.get_kernel_source()
    assert "tl::ds_read_vector" in source

    profiler = kernel.get_profiler()
    profiler.assert_allclose(
        lambda x, b_at, b_an: (
            x.float() @ b_at.float(),
            x.T.float() @ b_an.float(),
        ),
        atol=1e-2,
        rtol=1e-2,
    )


if __name__ == "__main__":
    test_hcu_gemm_mixed_lds_access()
