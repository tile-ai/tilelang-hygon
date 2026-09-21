import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 配置参数
# ----------------------------
#CSV_FILE = "csv/rocblas_gemm_performance_half_Radeon(TM)ProVII.csv"
CSV_FILE = "csv/gemm_fp16.csv"
OUTPUT_PNG = "imgs/gemm_fp16.png"

GPU_THEORETICAL_TFLOPS = 480  # 示例：BW200 FP16 峰值

GPU_NAME = "DCU BW200"

COL_SUFFIX = "_TFlops"

# 横轴范围
START = 256
END = 20480
STEP = 256
expected_sizes = set(range(START, END + 1, STEP))

# 关键尺寸标记（2 的幂或硬件对齐点）
key_sizes = [1024, 2048, 4096, 8192, 16384]

# ----------------------------
# 读取 CSV
# ----------------------------
df = pd.read_csv(CSV_FILE)

# 假设列名为：
# 我们只关心 (m,n,k) , tilelang_TFlops , torch_TFlops
if 'm' not in df.columns or 'tilelang_TFlops' not in df.columns or 'torch_TFlops' not in df.columns:
    raise ValueError("CSV must contain columns: 'm' and 'tilelang_TFlops' and 'torch_TFlops' ")

# 过滤出在 [256, 20480] 且是 256 倍数的行（可选）
df = df[df['m'].between(START, END)]
df = df[df['m'] % STEP == 0]

# 排序
df = df.sort_values('m')

sizes = df['m'].values
tilelang_tflops = df['tilelang_TFlops'].values   # 转为 TFLOPS
rocblas_tflops = df['torch_TFlops'].values   # 转为 TFLOPS

mfu_tilelang = tilelang_tflops / GPU_THEORETICAL_TFLOPS
mfu_rocblas = rocblas_tflops / GPU_THEORETICAL_TFLOPS

# ----------------------------
# 绘图
# ----------------------------
markers = ["o", "s", "D", "^", "v", "P"]

plt.figure(figsize=(14, 7))
(line_tilelang,) = plt.plot(sizes, tilelang_tflops, 'o-', markersize=5, linewidth=1.5, color="green", label='TileLang-Hygon FP16 GEMM Performance')
(line_rocblas,) = plt.plot(sizes, rocblas_tflops, 's-', markersize=5, linewidth=1.5, color="blue", label='rocBLAS FP16 GEMM Performance')


def annotate_peak(line, mfu, raw_tflops, sizes_arr, offset):
    """标注一条曲线的峰值点 (MFU% + 原始 TFLOPS)。"""
    if len(mfu) == 0:
        return
    peak_idx = np.argmax(mfu)
    color = line.get_color()
    text = f"Peak MFU: {mfu[peak_idx]:.1%}\n({raw_tflops[peak_idx]:.1f} TFLOPS)"
    plt.annotate(
        text,
        xy=(sizes_arr[peak_idx], raw_tflops[peak_idx]),
        xytext=offset,
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        fontsize=10,
        color=color,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
    )

annotate_peak(line_tilelang, mfu_tilelang, tilelang_tflops, sizes, (-70, 25))
annotate_peak(line_rocblas, mfu_rocblas, rocblas_tflops, sizes, (20, -45))

# 添加理论峰值线
plt.axhline(
    y=GPU_THEORETICAL_TFLOPS,
    color='r',
    linestyle='--',
    linewidth=1.5,
    label=f'Theoretical Peak ({GPU_NAME}: {GPU_THEORETICAL_TFLOPS} TFLOPS)'
)

# 标记关键尺寸（垂直虚线）
for size in key_sizes:
    if START <= size <= END:
        plt.axvline(x=size, color='gray', linestyle=':', alpha=0.7)
        plt.text(size, plt.ylim()[1] * 0.95, f'{size}', rotation=0,
                 verticalalignment='top', horizontalalignment='center',
                 fontsize=9, color='gray')

# 图表美化
plt.title('FP16 GEMM MFU vs Matrix Size (M = N = K)', fontsize=16, pad=20)
plt.xlabel('Matrix Dimension (M = N = K)', fontsize=14)
plt.ylabel('MFU Ratio (Achieved / Theoretical Peak)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(START - STEP, END + STEP)
plt.xticks(np.arange(START, END + STEP, 2048), rotation=45)  # 每 2048 标一个 tick
plt.tight_layout()

# 图例
plt.legend(fontsize=12)

# 保存与显示
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
print(f"✅ 图表已保存为: {OUTPUT_PNG}")
plt.show()