"""
Batch GEMM benchmark sweep: square matrices m=n=k from MIN to MAX, step 256.

All non-shape parameters use the defaults set in perf/gemm/benchmark.py:
  - dtype: fp16, autotune: False, impl: persistent, with_roller: False
  - device: 7 (benchmark.py CLI default)

Results are written incrementally to a CSV so partial progress survives a crash.

Usage:
    python perf/gemm/run_sweep.py [--min 256] [--max 16384] [--csv <path>] [--append]
  --append: append rows to an existing CSV (header already present); default is
            to create a new CSV with a header.
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from perf.gemm.benchmark import main as bench_main

STEP = 256

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CSV_PATH = os.path.join(RESULTS_DIR, "gemm_sweep_square_256_to_16384.csv")

CSV_FIELDS = [
    "M", "N", "K", "dtype", "impl",
    "tilelang_latency_ms", "ref_latency_ms",
    "tilelang_tflops", "ref_tflops", "speedup",
]


def main():
    parser = argparse.ArgumentParser(description="Square GEMM sweep (m=n=k)")
    parser.add_argument("--min", type=int, default=256)
    parser.add_argument("--max", type=int, default=16384)
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    parser.add_argument("--append", action="store_true", default=False,
                        help="append to existing CSV without rewriting the header")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sizes = list(range(args.min, args.max + 1, STEP))
    total = len(sizes)
    print(f"Sweeping {total} square sizes: {args.min} -> {args.max} step {STEP} "
          f"(append={args.append}) -> {args.csv}", flush=True)

    mode = "a" if args.append else "w"
    with open(args.csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not args.append:
            writer.writeheader()
            f.flush()

        for i, size in enumerate(sizes, start=1):
            t0 = time.time()
            print(f"\n[{i}/{total}] M=N=K={size}  start={time.strftime('%H:%M:%S')}", flush=True)
            try:
                # device pinned to benchmark.py's CLI default (7); auto-picking
                # would skip HCUs whose VRAM still shows allocated from earlier
                # sizes and eventually fail with "No free HCU devices found"
                result = bench_main(M=size, N=size, K=size, device=7)
                writer.writerow({k: result[k] for k in CSV_FIELDS})
                f.flush()
                elapsed = time.time() - t0
                print(
                    f"[{i}/{total}] {size}^3 done: tilelang={result['tilelang_latency_ms']:.3f} ms, "
                    f"tflops={result['tilelang_tflops']:.1f}, speedup={result['speedup']:.2f}x "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )
            except Exception as e:
                # record the failure row so the sweep continues with the next size
                print(f"[{i}/{total}] {size}^3 FAILED: {e}", flush=True)
                writer.writerow(
                    {
                        "M": size, "N": size, "K": size,
                        "dtype": "fp16", "impl": "persistent",
                        "tilelang_latency_ms": "", "ref_latency_ms": "",
                        "tilelang_tflops": "", "ref_tflops": "", "speedup": "",
                    }
                )
                f.flush()

    print(f"\nAll done. Results saved to {args.csv}", flush=True)


if __name__ == "__main__":
    main()
