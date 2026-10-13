import tilelang.testing
import example_tilelang_gemm_fp8_2xAcc
import example_tilelang_gemm_fp8
import pytest


def target_supports_fp8_mmac() -> bool:
    """Return whether the active HCU target supports FP8 MMAC lowering."""
    try:
        from tilelang import tvm
        from tilelang.backend.target import determine_target
        from tilelang.hcu.target import get_hcu_arch_string, target_is_hcu

        target = tvm.target.Target(determine_target("auto"))
        return target_is_hcu(target) and get_hcu_arch_string(target) in {"gfx938", "gfx92a", "gfx946"}
    except Exception:
        return False


@tilelang.testing.requires_rocm
@pytest.mark.skipif(not target_supports_fp8_mmac(), reason="FP8 MMAC not supported on this target")
def test_example_tilelang_gemm_fp8_2xAcc():
    example_tilelang_gemm_fp8_2xAcc.main()


@tilelang.testing.requires_rocm
@pytest.mark.skipif(not target_supports_fp8_mmac(), reason="FP8 MMAC not supported on this target")
def test_example_tilelang_gemm_fp8():
    example_tilelang_gemm_fp8.main()


if __name__ == "__main__":
    tilelang.testing.main()
