"""
Plot FP16 GEMM MFU curves for three implementations (square M=N=K):
  1. Ours            - tilelang_gemm_performance_transpose_level9_warpKLoop.csv
                       max(tilelang_TFlops, tilelang_warpK_TFlops)
  2. rocBLAS         - same CSV, rocBLAS_TFlops column
  3. TileLang-Hygon  - gemm_sweep_square_256_to_16384.csv (official repo default
                       persistent kernel, measured on this machine)
Y axis is MFU = achieved TFLOPS / GPU theoretical peak.
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless-friendly

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------
# 配置参数
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_OURS = os.path.join(BASE_DIR, "tilelang_gemm_performance_transpose_level9_warpKLoop.csv")
CSV_HYGON = os.path.join(BASE_DIR, "gemm_sweep_square_256_to_20480.csv")
OUTPUT_PNG = os.path.join(BASE_DIR, "tilelang_gemm_fp16_performance_transpose_ours_rocblas_tilelanghygon_MFU.png")

# GPU 理论峰值 (TFLOPS) - 用于计算 MFU
GPU_THEORETICAL_TFLOPS = 480
GPU_NAME = "DCU BW200"

# 横轴范围配置
START = 256
END = 20480
STEP = 256

# 关键尺寸标记（用于画垂直虚线）
key_sizes = [1024, 2048, 4096, 8192, 16384]

# ----------------------------
# 读取与处理数据
# ----------------------------
try:
    df = pd.read_csv(CSV_OURS)
except FileNotFoundError:
    print(f"错误: 找不到文件 {CSV_OURS}")
    exit(1)

required_cols = ["m", "tilelang_TFlops", "tilelang_warpK_TFlops", "rocBLAS_TFlops"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV 缺少必要的列: '{col}'。当前列名: {df.columns.tolist()}")

# 过滤数据：确保在设定范围内且是 STEP 的倍数
df = df[df["m"].between(START, END)]
df = df[df["m"] % STEP == 0]
df = df.sort_values("m")

try:
    hygon = pd.read_csv(CSV_HYGON)
except FileNotFoundError:
    print(f"错误: 找不到文件 {CSV_HYGON}")
    exit(1)
hygon = hygon[hygon["M"].between(START, END)]
hygon = hygon[hygon["M"] % STEP == 0]
hygon = hygon.sort_values("M")

sizes = df["m"].values

# Ours: 两个 tilelang 变体取最大值; rocBLAS: 最后一列; TileLang-Hygon: 官方仓库实测
raw_tflops_ours = np.maximum(df["tilelang_TFlops"].values, df["tilelang_warpK_TFlops"].values)
raw_tflops_rocblas = df["rocBLAS_TFlops"].values
hygon_sizes = hygon["M"].values
raw_tflops_hygon = hygon["tilelang_tflops"].values

# 【核心修改】计算 MFU 比例 (Current TFLOPS / Peak TFLOPS)
mfu_ours = raw_tflops_ours / GPU_THEORETICAL_TFLOPS
mfu_rocblas = raw_tflops_rocblas / GPU_THEORETICAL_TFLOPS
mfu_hygon = raw_tflops_hygon / GPU_THEORETICAL_TFLOPS

# ----------------------------
# 绘图
# ----------------------------
plt.figure(figsize=(14, 7))

# 绘制三条性能曲线 (使用 MFU 数据)
(line_ours,) = plt.plot(sizes, mfu_ours, "o-", markersize=5, linewidth=1.5, color="red",
                        label="Ours (TileLang-Transpose L9, max of both variants)")
(line_rocblas,) = plt.plot(sizes, mfu_rocblas, "s-", markersize=5, linewidth=1.5, color="blue",
                           label="rocBLAS FP16")
(line_hygon,) = plt.plot(hygon_sizes, mfu_hygon, "^-", markersize=5, linewidth=1.5, color="green",
                         label="TileLang-Hygon (official repo)")


def annotate_peak(line, mfu, raw_tflops, sizes_arr, offset):
    """标注一条曲线的峰值点 (MFU% + 原始 TFLOPS)。"""
    if len(mfu) == 0:
        return
    peak_idx = np.argmax(mfu)
    color = line.get_color()
    text = f"Peak MFU: {mfu[peak_idx]:.1%}\n({raw_tflops[peak_idx]:.1f} TFLOPS)"
    plt.annotate(
        text,
        xy=(sizes_arr[peak_idx], mfu[peak_idx]),
        xytext=offset,
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        fontsize=10,
        color=color,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
    )


annotate_peak(line_ours, mfu_ours, raw_tflops_ours, sizes, (20, 40))
annotate_peak(line_rocblas, mfu_rocblas, raw_tflops_rocblas, sizes, (-70, 25))
annotate_peak(line_hygon, mfu_hygon, raw_tflops_hygon, hygon_sizes, (20, -45))

# 添加理论峰值线 (在 MFU 坐标系中，理论峰值即为 1.0)
plt.axhline(
    y=1.0,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label=f"Theoretical Peak (100% MFU / {GPU_THEORETICAL_TFLOPS} TFLOPS)",
)

# 标记关键尺寸（垂直虚线）
for size in key_sizes:
    if START <= size <= END:
        plt.axvline(x=size, color="gray", linestyle=":", alpha=0.7)
        plt.text(size, plt.ylim()[1] * 0.95, f"{size}", rotation=0,
                 verticalalignment="top", horizontalalignment="center",
                 fontsize=9, color="gray")

# 图表美化
plt.title(f"FP16 GEMM MFU vs Matrix Size (Peak: {GPU_THEORETICAL_TFLOPS} TFLOPS)", fontsize=16, pad=20)
plt.xlabel("Matrix Dimension (M = N = K)", fontsize=14)
plt.ylabel("MFU Ratio (Achieved / Theoretical Peak)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(START - STEP, END + STEP)

# X轴刻度设置
xticks = np.arange(START, END + STEP, 2048)
plt.xticks(xticks, rotation=45)

plt.legend(fontsize=12, loc="lower right")
plt.tight_layout()

# 保存与显示
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
print(f"✅ 图表已保存为: {OUTPUT_PNG}")
