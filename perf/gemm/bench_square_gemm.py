"""
Square FP16 GEMM sweep: M=N=K from 256 to 16384, step 256.

Compares:
  - PyTorch  (A @ B)
  - example_gemm.py  (T.Pipelined, BM=BN=256, BK=16)
  - async_copy_gemm.py  (hand pipeline, same tile)

CSV is written incrementally. Plot style follows correct/images/draw.py (MFU).

Usage (from tilelang-hygon root, after `source ~/venv-tl-hygon/bin/activate`):
    python perf/gemm/bench_square_gemm.py
    python perf/gemm/bench_square_gemm.py --min 256 --max 1024 --device 0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from perf.utils.device import get_free_devices  # noqa: E402

STEP = 256
BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 16
GPU_THEORETICAL_TFLOPS = 480
KEY_SIZES = [1024, 2048, 4096, 8192, 16384]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CSV_PATH = os.path.join(RESULTS_DIR, "square_gemm_256_to_16384.csv")
PNG_PATH = os.path.join(RESULTS_DIR, "square_gemm_fp16_mfu.png")

CSV_FIELDS = [
    "m",
    "pytorch_latency_ms",
    "pytorch_TFlops",
    "example_gemm_latency_ms",
    "example_gemm_TFlops",
    "async_copy_latency_ms",
    "async_copy_TFlops",
]


def tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    return 2.0 * m * n * k / latency_ms * 1e-9


def pick_device(explicit: int) -> int:
    if explicit >= 0:
        return explicit
    free = get_free_devices()
    if free:
        return free[0]
    if torch.cuda.is_available():
        return 0
    raise RuntimeError("No CUDA/HCU device available")


def bench_pytorch(m: int, n: int, k: int) -> float:
    from tilelang.profiler.bench import do_bench

    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)

    def run():
        return a @ b

    return float(do_bench(run, backend="event"))


def bench_example_gemm(m: int, n: int, k: int) -> float:
    from perf.gemm.example_gemm import matmul

    kernel = matmul.compile(
        M=m, N=n, K=k, block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K
    )
    profiler = kernel.get_profiler()
    return float(profiler.do_bench(backend="event"))


def bench_async_copy(m: int, n: int, k: int) -> float:
    from perf.gemm.async_copy_gemm import gemm_async_copy_n_major

    kernel = gemm_async_copy_n_major(
        m,
        n,
        k,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        dtype="float16",
        accum_dtype="float32",
    )
    profiler = kernel.get_profiler()
    return float(profiler.do_bench(backend="event"))


def _empty_row(size: int) -> dict:
    return {k: (size if k == "m" else "") for k in CSV_FIELDS}


def plot_csv(csv_path: str, png_path: str, start: int, end: int, step: int) -> None:
    df = pd.read_csv(csv_path)
    for col in CSV_FIELDS:
        if col not in df.columns:
            raise ValueError(f"CSV missing column {col}: {df.columns.tolist()}")

    df = df[df["m"].between(start, end)]
    df = df[df["m"] % step == 0].sort_values("m")
    sizes = df["m"].values

    def series(tflops_col: str):
        raw = pd.to_numeric(df[tflops_col], errors="coerce").to_numpy(dtype=float)
        mfu = raw / GPU_THEORETICAL_TFLOPS
        return raw, mfu

    raw_pt, mfu_pt = series("pytorch_TFlops")
    raw_ex, mfu_ex = series("example_gemm_TFlops")
    raw_ac, mfu_ac = series("async_copy_TFlops")

    plt.figure(figsize=(14, 7))
    (line_pt,) = plt.plot(
        sizes, mfu_pt, "o-", markersize=5, linewidth=1.5, label="PyTorch FP16 GEMM (A @ B)"
    )
    (line_ex,) = plt.plot(
        sizes,
        mfu_ex,
        "o-",
        markersize=5,
        linewidth=1.5,
        label="TileLang T.Pipelined (example_gemm)",
    )
    (line_ac,) = plt.plot(
        sizes,
        mfu_ac,
        "o-",
        markersize=5,
        linewidth=1.5,
        label="TileLang async_copy (hand pipeline)",
    )

    def annotate_peak(raw, mfu, line, y_off):
        if not np.any(np.isfinite(mfu)):
            return
        peak_idx = int(np.nanargmax(mfu))
        color = line.get_color()
        plt.annotate(
            f"Peak MFU: {mfu[peak_idx]:.1%}\n({raw[peak_idx]:.1f} TFLOPS)",
            xy=(sizes[peak_idx], mfu[peak_idx]),
            xytext=(20, y_off),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            fontsize=10,
            color=color,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
        )

    annotate_peak(raw_pt, mfu_pt, line_pt, 10)
    annotate_peak(raw_ex, mfu_ex, line_ex, 40)
    annotate_peak(raw_ac, mfu_ac, line_ac, 70)

    plt.axhline(
        y=1.0,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Theoretical Peak (100% MFU / {GPU_THEORETICAL_TFLOPS} TFLOPS)",
    )
    for size in KEY_SIZES:
        if start <= size <= end:
            plt.axvline(x=size, color="gray", linestyle=":", alpha=0.7)
            ymax = plt.ylim()[1]
            plt.text(
                size,
                ymax * 0.95 if ymax > 0 else 0.95,
                f"{size}",
                rotation=0,
                verticalalignment="top",
                horizontalalignment="center",
                fontsize=9,
                color="gray",
            )

    plt.title(
        f"FP16 GEMM MFU vs Square Size (Peak: {GPU_THEORETICAL_TFLOPS} TFLOPS)",
        fontsize=16,
        pad=20,
    )
    plt.xlabel("Matrix Dimension (M = N = K)", fontsize=14)
    plt.ylabel("MFU Ratio (Achieved / Theoretical Peak)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(start - step, end + step)
    xticks = np.arange(start, end + step, 2048)
    if len(xticks) == 0:
        xticks = sizes
    plt.xticks(xticks, rotation=45)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"plot saved: {png_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Square GEMM sweep + MFU plot")
    parser.add_argument("--min", type=int, default=256)
    parser.add_argument("--max", type=int, default=16384)
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("-d", "--device", type=int, default=-1)
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    parser.add_argument("--png", type=str, default=PNG_PATH)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="only redraw from existing CSV")
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--skip-example", action="store_true")
    parser.add_argument("--skip-async", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot_csv(args.csv, args.png, args.min, args.max, args.step)
        return

    device_id = pick_device(args.device)
    torch.cuda.set_device(device_id)
    print(f"device={device_id}  BM={BLOCK_M} BN={BLOCK_N} BK={BLOCK_K}", flush=True)

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    sizes = list(range(args.min, args.max + 1, args.step))
    total = len(sizes)
    print(
        f"sweep {total} squares: {args.min} -> {args.max} step {args.step} -> {args.csv}",
        flush=True,
    )

    mode = "a" if args.append else "w"
    with open(args.csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not args.append:
            writer.writeheader()
            f.flush()

        for i, size in enumerate(sizes, start=1):
            row = _empty_row(size)
            t0 = time.time()
            print(
                f"\n[{i}/{total}] {size}^3  start={time.strftime('%H:%M:%S')}",
                flush=True,
            )
            try:
                if not args.skip_pytorch:
                    lat = bench_pytorch(size, size, size)
                    row["pytorch_latency_ms"] = f"{lat:.6f}"
                    row["pytorch_TFlops"] = f"{tflops(size, size, size, lat):.4f}"
                    print(
                        f"  pytorch    {lat:.3f} ms  {row['pytorch_TFlops']} TFLOPS",
                        flush=True,
                    )
                if not args.skip_example:
                    lat = bench_example_gemm(size, size, size)
                    row["example_gemm_latency_ms"] = f"{lat:.6f}"
                    row["example_gemm_TFlops"] = f"{tflops(size, size, size, lat):.4f}"
                    print(
                        f"  example    {lat:.3f} ms  {row['example_gemm_TFlops']} TFLOPS",
                        flush=True,
                    )
                if not args.skip_async:
                    lat = bench_async_copy(size, size, size)
                    row["async_copy_latency_ms"] = f"{lat:.6f}"
                    row["async_copy_TFlops"] = f"{tflops(size, size, size, lat):.4f}"
                    print(
                        f"  async_copy {lat:.3f} ms  {row['async_copy_TFlops']} TFLOPS",
                        flush=True,
                    )
            except Exception as exc:
                print(f"  FAILED: {exc}", flush=True)
                traceback.print_exc()
            writer.writerow(row)
            f.flush()
            torch.cuda.empty_cache()
            print(f"[{i}/{total}] done in {time.time() - t0:.0f}s", flush=True)

    plot_csv(args.csv, args.png, args.min, args.max, args.step)


if __name__ == "__main__":
    main()
