# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Sweep small threadblock swizzle configs for async_copy GEMM."""

import argparse
from dataclasses import dataclass
from typing import Any

import torch
import tilelang as tl
from tilelang.profiler import do_bench_cudagraph

from perf.gemm.kernel_registry import dispatch_kernel, get_kernel_config
from perf.utils.device import get_free_devices


@dataclass(frozen=True)
class SwizzleCandidate:
    enable: bool
    panel_size: int
    order: str


def _get_device(device: int) -> int:
    if device != -1:
        return device
    free_hcus = get_free_devices()
    if len(free_hcus) == 0:
        raise RuntimeError("No free HCU devices found")
    return free_hcus[0]


def _candidate_kwargs(candidate: SwizzleCandidate, transpose_b: bool) -> dict[str, Any]:
    if transpose_b:
        return {
            "swizzle_enable": candidate.enable,
            "swizzle_panel_size": candidate.panel_size,
            "swizzle_order": candidate.order,
        }
    if not candidate.enable:
        return {"swizzle_panel_size": 0, "swizzle_order": candidate.order}
    return {
        "swizzle_panel_size": candidate.panel_size,
        "swizzle_order": candidate.order,
    }


def _bench_candidate(
    *,
    M: int,
    N: int,
    K: int,
    dtype: str,
    device_id: int,
    transpose_b: bool,
    candidate: SwizzleCandidate,
) -> tuple[float, float]:
    spec = dispatch_kernel("async_copy", transpose_B=transpose_b, version="vanilla")
    config = get_kernel_config(spec, M, N, K)
    config.update(_candidate_kwargs(candidate, transpose_b))
    kernel = spec.kernel(M, N, K, dtype=dtype, **spec.kernel_kwargs, **config)
    torch.cuda.set_device(device_id)
    profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Auto)
    inputs = profiler._get_inputs()

    def tilelang_run():
        profiler.func(*inputs)

    if transpose_b:

        def ref_program(a, b):
            return a @ b.T
    else:

        def ref_program(a, b):
            return a @ b

    latency = do_bench_cudagraph(tilelang_run)
    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)
    tflops = 2 * M * N * K / latency * 1e-9
    return latency, tflops


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune async_copy GEMM swizzle configs")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("-d", "--device", type=int, default=-1)
    parser.add_argument("--n-major", action="store_true", help="Use N-major B layout [K, N]")
    args = parser.parse_args()

    transpose_b = not args.n_major
    device_id = _get_device(args.device)
    torch.cuda.set_device(device_id)

    candidates = [
        SwizzleCandidate(False, 4, "row"),
        SwizzleCandidate(True, 4, "row"),
        SwizzleCandidate(True, 8, "row"),
        SwizzleCandidate(True, 10, "row"),
        SwizzleCandidate(True, 4, "col"),
        SwizzleCandidate(True, 8, "col"),
        SwizzleCandidate(True, 10, "col"),
    ]

    print(f"Using HCU device: {device_id}")
    print(f"GEMM shape: M={args.m}, N={args.n}, K={args.k}")
    print(f"Data type: {args.dtype}")
    print(f"B layout: {'K-major [N, K]' if transpose_b else 'N-major [K, N]'}")

    best_candidate = None  # type: Optional[SwizzleCandidate]
    best_tflops = -1.0

    for candidate in candidates:
        latency, tflops = _bench_candidate(
            M=args.m,
            N=args.n,
            K=args.k,
            dtype=args.dtype,
            device_id=device_id,
            transpose_b=transpose_b,
            candidate=candidate,
        )
        print(
            "candidate:",
            {
                "enable": candidate.enable,
                "panel_size": candidate.panel_size,
                "order": candidate.order,
                "latency_ms": round(latency, 6),
                "tflops": round(tflops, 4),
            },
        )
        if tflops > best_tflops:
            best_tflops = tflops
            best_candidate = candidate

    assert best_candidate is not None
    print(
        "best:",
        {
            "enable": best_candidate.enable,
            "panel_size": best_candidate.panel_size,
            "order": best_candidate.order,
            "tflops": round(best_tflops, 4),
        },
    )


if __name__ == "__main__":
    main()
