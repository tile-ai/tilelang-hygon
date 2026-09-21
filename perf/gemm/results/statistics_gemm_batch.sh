#!/bin/bash

# 定义参数范围：256的整数倍
start=256
#start=4096
end=20480
step=256

# TileLang 测试脚本路径（请根据实际路径修改）
PYTHON_SCRIPT="example_gemm_batch.py"

# 输出结果文件
result_file="csv/gemm_fp16_aicc_nightly_09.csv"

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python 脚本 $PYTHON_SCRIPT 不存在！"
    exit 1
fi

# 创建输出目录（如果不存在）
mkdir -p "$(dirname "$result_file")"

if [ ! -f "$result_file" ]; then
    echo "m,n,k,tilelang_latency_ms,tilelang_TFlops" >> "$result_file"
    echo "📝 检测到是新文件，已创建并写入表头。"
else
    echo "📂 检测到文件已存在，将以追加模式继续写入数据。"
fi
# # 写入CSV表头（保持五列）
# echo "m,n,k,tilelang_latency_ms,tilelang_TFlops" >> "$result_file"

# 遍历 m = n = k 的方阵情况
for size in $(seq $start $step $end); do
    m=$size
    n=$size
    k=$size

    echo "========================================"
    echo "Testing shape: m=$m, n=$n, k=$k (GEMM via TileLang)"
    echo "========================================"
    
    # 执行 TileLang 测试脚本，捕获全部输出（包括stderr）
    output=$(python3 "$PYTHON_SCRIPT" --m=$m --n=$n --k=$k 2>&1)
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "⚠️  程序执行失败（退出码: $exit_code），跳过此组数据。"
        echo "$output"
        # 写入空值行以保持 CSV 结构
        echo "$m,$n,$k,," >> "$result_file"
        continue
    fi

    # 提取 tilelang Latency (单位: ms)
    # 筛选包含 "tilelang Latency:" 的行，并提取第3个字段
    tilelang_latency=$(echo "$output" | grep "tilelang Latency:" | awk '{print $3}' | sed 's/ms.*//' | awk '{printf "%.3f", $1}')

    # 如果提取失败，输出警告并跳过计算
    if [ -z "$tilelang_latency" ]; then
        echo "⚠️  未能从输出中提取到 Latency，请检查脚本输出格式。"
        echo "$output"
        echo "$m,$n,$k,," >> "$result_file"
        continue
    fi

    # 计算 TFLOPS
    # 公式：2 * M * N * K / latency_ms * 1e-9
    # 使用 awk 进行浮点数运算，并保留6位小数
    tilelang_tflops=$(echo "$m $n $k $tilelang_latency" | awk '{printf "%.6f", (2 * $1 * $2 * $3) / $4 * 0.000000001}')
    # tilelang_tflops=$(echo "$m $n $k $tilelang_latency" | awk '{printf "%.6f", (2 * $1 * $2 * $3) / $4 * 1e-9}')

    # 输出当前结果
    echo "TileLang Latency: ${tilelang_latency} ms | ${tilelang_tflops} TFLOPS"
    echo ""

    # 写入CSV
    echo "$m,$n,$k,$tilelang_latency,$tilelang_tflops" >> "$result_file"
done

echo "========================================"
echo "✅ 测试完成！结果已保存至 $result_file"