"""End-to-end WASP regressions, separate from basic MLS operator tests."""

import pytest
import torch
import tilelang.language as T
from tilelang.contrib.rocm import get_rocm_arch
from perf.gemm.gemm_wasp import gemm_wasp_mls_4p4c4c_persistent


def current_hcu_arch_string():
    return get_rocm_arch()


@pytest.mark.skipif(current_hcu_arch_string() != "gfx946", reason="gfx946 WASP regression")
@pytest.mark.parametrize("matrix_store", [False, True])
@pytest.mark.parametrize("alignments", [(1, 1, 1), (8, 16, 32), (1024, 1024, 512)])
def test_gemm_wasp_dynamic_mls_let(matrix_store, alignments):
    """Dynamic floor division/modulo must compile without extra assumptions."""
    kernel = gemm_wasp_mls_4p4c4c_persistent(
        T.dynamic("M"),
        T.dynamic("N"),
        T.dynamic("K"),
        use_matrix_store=matrix_store,
        max_m_align=alignments[0],
        max_n_align=alignments[1],
        max_k_align=alignments[2],
    )
    generator = torch.Generator().manual_seed(321)
    a = torch.randint(-3, 4, (1024, 512), generator=generator).half()
    b = torch.randint(-3, 4, (1024, 512), generator=generator).half()
    expected = (a.float() @ b.float().T).half()
    actual = kernel(a.cuda(), b.cuda()).cpu()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("size,expected", [(1032, 8), (1040, 16), (96, 32), (1024, 1024), (257, 1)])
def test_wasp_infer_dimension_alignment(size, expected):
    from perf.gemm.gemm_wasp import _max_power_of_two_alignment

    assert _max_power_of_two_alignment(size) == expected


@pytest.mark.skipif(current_hcu_arch_string() != "gfx946", reason="gfx946 WASP regression")
@pytest.mark.parametrize("persistent,matrix_store", [(False, False), (True, False), (True, True)])
def test_gemm_wasp_dynamic_pong_k_tail(persistent, matrix_store):
    """An even ceildiv(K, 64) does not imply that the final pong is full."""
    from perf.gemm.gemm_wasp import gemm_wasp_mls_4p4c4c

    factory = gemm_wasp_mls_4p4c4c_persistent if persistent else gemm_wasp_mls_4p4c4c
    options = {"use_matrix_store": matrix_store} if persistent else {}
    kernel = factory(T.dynamic("M"), T.dynamic("N"), T.dynamic("K"), **options)
    generator = torch.Generator().manual_seed(321)
    # Isolate K filtering from MatrixStore M/N boundary behavior.
    mn = 256 if matrix_store else 258
    for k in (96, 128, 160, 224, 256):
        a = torch.randint(-3, 4, (mn, k), generator=generator).half()
        b = torch.randint(-3, 4, (mn, k), generator=generator).half()
        expected = (a.float() @ b.float().T).half()
        actual = kernel(a.cuda(), b.cuda()).cpu()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, msg=f"K={k}, matrix_store={matrix_store}")


@pytest.mark.parametrize("size", [0, -8])
def test_wasp_reject_invalid_dimension_alignment(size):
    from perf.gemm.gemm_wasp import _max_power_of_two_alignment

    with pytest.raises(ValueError):
        _max_power_of_two_alignment(size)


@pytest.mark.skipif(current_hcu_arch_string() != "gfx946", reason="gfx946 B16 store")
@pytest.mark.parametrize("m,n,k", [(258, 259, 96), (257, 225, 128), (32, 1, 64)])
def test_wasp_matrix_store_b16_odd_output(m, n, k):
    kernel = gemm_wasp_mls_4p4c4c_persistent(T.dynamic("M"), T.dynamic("N"), T.dynamic("K"), use_matrix_store=True)
    generator = torch.Generator().manual_seed(321)
    a = torch.randint(-3, 4, (m, k), generator=generator).half()
    b = torch.randint(-3, 4, (n, k), generator=generator).half()
    expected = (a.float() @ b.float().T).half()
    actual = kernel(a.cuda(), b.cuda()).cpu()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert "half_t, 4, true" in kernel.get_kernel_source()


@pytest.mark.parametrize("variant", ["mls_4p4c4c", "mls_4p4c4c_persistent"])
@pytest.mark.parametrize("axis,alignment", [("M", 8), ("N", 16), ("K", 32)])
def test_wasp_independent_dimension_assumptions(variant, axis, alignment):
    from perf.gemm.gemm_wasp import KERNELS

    func = KERNELS[variant].get_tir(
        T.dynamic("M"),
        T.dynamic("N"),
        T.dynamic("K"),
        **{f"max_{axis.lower()}_align": alignment},
    )
    assumptions = [line.strip() for line in func.script().splitlines() if "T.assume(" in line]
    assert len(assumptions) == 2, assumptions
    assert any(f"% {alignment}" in line for line in assumptions), assumptions
    for line in assumptions:
        assert axis in line, assumptions
        assert all(other not in line for other in {"M", "N", "K"} - {axis}), assumptions
