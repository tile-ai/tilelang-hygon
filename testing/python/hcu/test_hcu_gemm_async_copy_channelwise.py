# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Correctness test for the hand-written 4-stage async fp8 channelwise GEMM.

Kernel: ``perf/gemm/async_copy_gemm_channelwise.py``. Semantics:
``C[m, n] = scale_a[m] * scale_b[n] * sum_k A[m, k] * B[n, k]`` with A (M, K)
fp8, B (N, K) fp8 (N-major), scale_a (M,) fp32 (per-token), scale_b (N,) fp32
(per-channel). Validated configuration: block 256x128x32 on gfx938.
"""

import os
import sys

import pytest
import torch
from hcu_test_utils import current_hcu_arch_string, target_supports_fp8_mmac

# The kernel lives in perf/gemm/ (imported as ``perf.gemm.*`` from the repo root).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from perf.gemm.async_copy_gemm_channelwise import gemm_async_copy_channelwise  # noqa: E402


pytestmark = [
    pytest.mark.skipif(
        current_hcu_arch_string() != "gfx938",
        reason="async-copy channelwise GEMM is validated on gfx938 only",
    ),
    pytest.mark.skipif(
        not target_supports_fp8_mmac(),
        reason="FP8 MMAC not supported on this target",
    ),
]


def _ref_program(A, B, scale_a, scale_b):
    return (A.float() @ B.float().T) * scale_a[:, None] * scale_b[None, :]


def test_hcu_gemm_async_copy_channelwise_fp8():
    M, N, K = 4096, 4096, 4096
    block_M, block_N, block_K = 256, 128, 32

    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    B = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.rand(M, device="cuda", dtype=torch.float32) + 0.5
    scale_b = torch.rand(N, device="cuda", dtype=torch.float32) + 0.5

    kernel = gemm_async_copy_channelwise(M, N, K, block_M, block_N, block_K, dtype="float8_e4m3fn")

    C = kernel(A, B, scale_a, scale_b)
    ref = _ref_program(A, B, scale_a, scale_b)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 0.01, f"channelwise GEMM rel error {rel:.6f} >= 0.01"

    kernel.get_profiler().assert_allclose(_ref_program, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
