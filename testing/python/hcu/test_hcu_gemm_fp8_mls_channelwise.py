# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Correctness test for the fp8 channelwise GEMM on the MLS (matrix-LDS-read)
producer path.

Kernel: ``perf/gemm/fp8_mls_channelwise.py``. Semantics:
``C[m, n] = scale_a[m] * scale_b[n] * sum_k A[m, k] * B[n, k]`` with A (M, K)
fp8, B (N, K) fp8 (N-major), fp32 per-token / per-channel scales. Validated
configuration on HCU (gfx938): block 256x256x64, num_stages=1, 512 threads.
"""

import os
import sys

import pytest
import torch
from hcu_test_utils import current_hcu_arch_string, target_supports_fp8_mmac, target_supports_mls

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from perf.gemm.fp8_mls_channelwise import fp8_mls_channelwise  # noqa: E402


pytestmark = [
    pytest.mark.skipif(
        current_hcu_arch_string() != "gfx938",
        reason="fp8 MLS channelwise GEMM is validated on gfx938 only",
    ),
    pytest.mark.skipif(
        not target_supports_fp8_mmac(),
        reason="FP8 MMAC not supported on this target",
    ),
    pytest.mark.skipif(
        not target_supports_mls(),
        reason="MLS (matrix_load) not supported on this target",
    ),
]


def _ref_program(A, B, scale_a, scale_b):
    return (A.float() @ B.float().T) * scale_a[:, None] * scale_b[None, :]


def _make_inputs(M, N, K, seed=0):
    torch.manual_seed(seed)
    A = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.rand(M, device="cuda", dtype=torch.float32) + 0.5
    scale_b = torch.rand(N, device="cuda", dtype=torch.float32) + 0.5
    return A, B, scale_a, scale_b


def test_hcu_gemm_fp8_mls_channelwise_validated_config():
    M, N, K = 8192, 8192, 8192
    block_M, block_N, block_K, num_stages, threads = 256, 256, 64, 1, 512
    A, B, scale_a, scale_b = _make_inputs(M, N, K)

    kernel = fp8_mls_channelwise(
        M, N, K, block_M, block_N, block_K, num_stages=num_stages, threads=threads
    )

    # The MLS producer path must actually lower to the hardware matrix LDS read.
    src = kernel.get_kernel_source()
    assert "ds_read_format_tensor_a" in src or "ds_read_format_tensor_b" in src, (
        "fp8 MLS channelwise GEMM did not use the ds_read_format matrix LDS read"
    )

    C = kernel(A, B, scale_a, scale_b)
    ref = _ref_program(A, B, scale_a, scale_b)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 0.01, f"fp8 MLS channelwise GEMM rel error {rel:.6f} >= 0.01"

    kernel.get_profiler().assert_allclose(_ref_program, atol=1e-2, rtol=1e-2)


def test_hcu_gemm_fp8_mls_channelwise_small():
    # Smaller shape, ns=0 (explicit wait path), to exercise the non-pipelined branch.
    M, N, K = 1024, 1024, 1024
    A, B, scale_a, scale_b = _make_inputs(M, N, K)
    kernel = fp8_mls_channelwise(M, N, K, 128, 128, 64, num_stages=0, threads=256)
    C = kernel(A, B, scale_a, scale_b)
    ref = _ref_program(A, B, scale_a, scale_b)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 0.01, f"small fp8 MLS channelwise GEMM rel error {rel:.6f} >= 0.01"


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
