import tilelang.testing
import example_gemm_intrinsics
import example_gemm
import example_gemm_autotune
import example_gemm_persistent


def test_example_gemm_intrinsics():
    example_gemm_intrinsics.main(M=1024, N=1024, K=1024)


def test_example_gemm():
    example_gemm.main()


def test_example_gemm_autotune():
    example_gemm_autotune.main(M=1024, N=1024, K=1024)


def test_example_gemm_persistent():
    example_gemm_persistent.main(M=1024, N=1024, K=1024)


if __name__ == "__main__":
    tilelang.testing.main()
