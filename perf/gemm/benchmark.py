"""
Benchmark script for all GEMM implementations.
"""

import argparse
import torch
import tilelang as tl
from perf.gemm.utils import ref_program, get_heuristic_config
from perf.gemm.vanilla_gemm import gemm_vanilla, get_best_vanilla_config
from perf.gemm.persistent_gemm import gemm_persistent, get_best_persistent_config
from perf.gemm.splitk_gemm import gemm_splitk
from perf.gemm.streamk_gemm import gemm_streamk
from perf.utils.device import get_free_devices


def normalize_dtype(dtype_str: str) -> str:
    """Convert dtype string to torch dtype string."""
    dtype_map = {
        "fp16": "float16",
        "float16": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
    }
    return dtype_map.get(dtype_str.lower(), "float16")


def main(M: int = 4096,
         N: int = 4096,
         K: int = 4096,
         dtype: str = "fp16",
         autotune: bool = False,
         impl: str = "persistent",
         with_roller: bool = False,
         device: int = -1,
         ):
    """
    Main benchmark function for GEMM implementations.

    Args:
        M: Matrix dimension M (default: 4096)
        N: Matrix dimension N (default: 4096)
        K: Matrix dimension K (default: 4096)
        dtype: Data type (fp16, bf16, fp32, default: fp16)
        autotune: Whether to use autotune (default: False)
        impl: GEMM implementation (vanilla, persistent, splitk, streamk, default: persistent)
        with_roller: Whether to enable BitBLAS roller for search space (default: False)
        device: Device ID (default: -1, auto find free device)
    """
    # Convert dtype string to torch dtype
    dtype = normalize_dtype(dtype)

    if autotune:
        if impl == "persistent":
            result = get_best_persistent_config(M, N, K)
            print(f"Best config: {result.config}")
            kernel = result.kernel
        elif impl == "vanilla":
            result = get_best_vanilla_config(M, N, K, with_roller)
            print(f"Best config: {result.config}")
            kernel = result.kernel
        else:
            raise ValueError(f"Autotune not supported for {impl} implementation. "
                           f"Supported: persistent, vanilla")
    else:
        config = get_heuristic_config()

        if impl == "persistent":
            kernel = gemm_persistent(M, N, K, dtype=dtype, **config)
        elif impl == "vanilla":
            kernel = gemm_vanilla(M, N, K, dtype=dtype, **config)
        elif impl == "splitk":
            kernel = gemm_splitk(M, N, K, dtype=dtype, **config)
        elif impl == "streamk":
            # streamk doesn't need additional config parameters
            kernel = gemm_streamk(M, N, K, dtype=dtype, **config)
        else:
            raise ValueError(f"Unknown implementation: {impl}. "
                           f"Supported: vanilla, persistent, splitk, streamk")

    free_hcus = get_free_devices()
    if len(free_hcus) == 0:
        raise RuntimeError("No free HCU devices found")
    if device == -1:
        device_id = free_hcus[0]
    else:
        device_id = device
    torch.cuda.set_device(device_id)
    print(f"Using HCU device: {device_id}")
    print(f"GEMM shape: M={M}, N={N}, K={K}")
    print(f"Data type: {dtype}")
    print(f"GEMM implementation: {impl}")
    print(f"Autotune: {autotune}")
    print(f"With roller: {with_roller}")

    # benchmark
    profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Auto)
    tilelang_latency = profiler.do_bench()
    ref_latency = profiler.do_bench(ref_program)
    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)
    print("\n=== Benchmark Results ===")
    print(f"TileLang latency: {tilelang_latency:.6f} ms")
    print(f"Ref latency: {ref_latency:.6f} ms")
    print(f"TileLang TFlops: {2 * M * N * K / tilelang_latency * 1e-9:.4f}")
    print(f"Ref TFlops: {2 * M * N * K / ref_latency * 1e-9:.4f}")
    print(f"Speedup: {ref_latency / tilelang_latency:.4f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEMM Benchmark for all implementations")
    parser.add_argument("--m", type=int, default=4096, help="Matrix dimension M (default: 4096)")
    parser.add_argument("--n", type=int, default=4096, help="Matrix dimension N (default: 4096)")
    parser.add_argument("--k", type=int, default=4096, help="Matrix dimension K (default: 4096)")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16", "fp32", "float16", "bfloat16", "float32"],
        default="fp16",
        help="Data type (default: fp16)")
    parser.add_argument(
        "--autotune",
        action="store_true",
        default=False,
        help="Whether to use autotune for GEMM configs")
    parser.add_argument(
        "-d", "--device",
        type=int,
        default=-1,
        help="Device ID (default: auto find free device)")
    parser.add_argument(
        "-i", "--impl",
        type=str,
        choices=["vanilla", "persistent", "splitk", "streamk"],
        default="persistent",
        help="GEMM implementation (default: persistent)")
    parser.add_argument(
        "--with_roller",
        action="store_true",
        default=False,
        help="Whether to enable BitBLAS roller for search space")
    args = parser.parse_args()
    main(args.m, args.n, args.k, args.dtype, args.autotune, args.impl,
         args.with_roller, args.device)
