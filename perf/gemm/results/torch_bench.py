import torch
import torch.utils.benchmark as benchmark
import csv
import os
import time
from datetime import datetime

# ================= 配置区域 =================
START_SIZE = 256
END_SIZE = 20480
STEP_SIZE = 256
DEVICE_ID = 0  # 对应你 shell 脚本中的 dev_id
OUTPUT_FILE = "csv/pytorch_gemm_performance_fp16.csv"
NUM_RUNS = 100 # 每个规模测试的次数，类似 shell 中的循环测试
# ===========================================

def run_benchmark(size):
    """
    使用 torch.utils.benchmark 测试指定规模的矩阵乘法性能
    """
    try:
        # 1. 准备数据 (FP16)
        # 这里的 setup 代码只会在计时开始前运行一次
        setup_code = f"""
import torch
torch.cuda.set_device({DEVICE_ID})
# 创建随机矩阵并转为半精度 (FP16)
a = torch.randn({size}, {size}, device='cuda', dtype=torch.float16)
b = torch.randn({size}, {size}, device='cuda', dtype=torch.float16)
"""
        # 2. 定义测试语句
        stmt_code = "a @ b"

        # 3. 创建 Timer 对象
        # label 和 sub_label 用于在结果中区分不同的测试
        t = benchmark.Timer(
            stmt=stmt_code,
            setup=setup_code,
            label="GEMM Performance",
            sub_label=f"{size}x{size}"
        )

        # 4. 执行测试
        # timeit 会自动处理预热(warmup)和多次运行取平均
        measurement = t.timeit(NUM_RUNS)
        
        # 5. 计算性能指标
        # GEMM 浮点运算次数 = 2 * M * N * K
        flops = 2 * size * size * size
        # measurement.mean 是秒，转换为 TFLOPS
        tflops = (flops / measurement.mean) / 1e12
        # 转换为毫秒
        time_ms = measurement.mean * 1000

        return time_ms, tflops, measurement.median, measurement.iqr

    except RuntimeError as e:
        print(f"❌ 错误 (尺寸 {size}): {e}")
        return None, None, None, None

def main():
    print(f"🚀 开始 PyTorch GEMM 性能测试 (FP16)")
    print(f"📅 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 结果将保存至: {OUTPUT_FILE}")
    print("-" * 70)
    print(f"{'Size':<10} | {'Time (ms)':<15} | {'TFLOPS':<15} | {'Status'}")
    print("-" * 70)

    # 打开 CSV 文件准备写入
    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["Matrix_Size", "Time_Mean_ms", "Time_Median_ms", "Time_IQR_ms", "TFLOPS", "FLOPS"])

        # 循环遍历尺寸
        current_size = START_SIZE
        while current_size <= END_SIZE:
            time_ms, tflops, median, iqr = run_benchmark(current_size)

            if time_ms is not None:
                # 打印到终端
                status = "OK"
                print(f"{current_size:<10} | {time_ms:<15.4f} | {tflops:<15.2f} | {status}")
                
                # 写入 CSV
                writer.writerow([current_size, f"{time_ms:.4f}", f"{median*1000:.4f}", f"{iqr*1000:.4f}", f"{tflops:.2f}", f"{tflops*1e12:.2e}"])
            else:
                print(f"{current_size:<10} | {'Failed':<15} | {'Failed':<15} | ❌")
                writer.writerow([current_size, "Error", "Error", "Error", "Error", "Error"])

            current_size += STEP_SIZE

    print("-" * 70)
    print(f"✅ 测试完成！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("❌ 错误: 未检测到 CUDA 设备，请检查显卡驱动或 PyTorch 安装。")
    else:
        # 确保在指定的 GPU 上运行
        torch.cuda.set_device(DEVICE_ID)
        main()