# GEMM + Reduce-Scatter

当前入口：`example_gemm_rs_push_waves.py`。

- **GEMM + push**：persistent CTA 内 8 个 compute wave、4 个 push wave；通过复用 LDS 将完整 FP16 输出交给 push 寄存器，在下一 tile 的计算间隙分四批发送。
- **waiting_reduce**：等待各来源 producer 的 ready 轮号，再在本卡使用 FP32 累加 inbox，输出 FP16。
- **RCCL 尾屏障**：保护下一轮 inbox 复用；没有 entry_barrier。双卡使用原始顺序，四卡按 rank 旋转 tile 顺序，分散推送目标。

payload 和 ready 使用 fine-grained HIP IPC 内存；保留 system-scope release/acquire，使用标准阻塞 signal 等待；不再保留自定义 inline C++ 或设备超时诊断。信号轮号在设备端推进，Graph replay 无需 CPU 重置。

```bash
# 先确认容器内设备空闲；四卡将可见设备及 -np 对应改成 4。
HIP_VISIBLE_DEVICES=0,1 TILELANG_CACHE_DIR=/tmp/tl-rs-example \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
  mpirun -np 2 -x HIP_VISIBLE_DEVICES -x TILELANG_CACHE_DIR \
  -x MASTER_ADDR -x MASTER_PORT \
  python examples/distributed/gemm_reduce_scatter/example_gemm_rs_push_waves.py \
  --size 4096 --benchmark
```

默认先验证随机输入、连续复用、延迟发送与 Graph；`--benchmark` 再测当前完整流程及 `torch.mm + RCCL reduce_scatter_tensor` 基线。两者使用相同输入、预热次数、重复次数和 GPU Event 计时，分别报告普通调用与 Graph replay 的中位耗时。`timings` 打印各 rank 结果，`summary` 取各指标的最慢 rank；随后按 `mode=eager/graph` 打印 TileLang/Torch 耗时与 `Torch / TileLang` 加速比。TileLang 耗时包含自身所需的尾屏障，Torch 基线包含 matmul 和 reduce-scatter，不额外添加尾屏障。需要单机 2 或 4 张互通的 GPU，以及支持 RCCL 的 PyTorch。

当前实现采用已选定的配置：双卡原始顺序、四卡 rank swizzle、grid 80。以下为同批 Torch MM + RCCL RS 对比：连续 8 次 Graph replay 的每次平均时间，取 120 轮中位数；包含 reduce 和尾屏障。K 为每 rank 本地 K。

| 卡数 | M=N=K | 最新方案（ms） | Torch（ms） | 耗时降低 |
|---|---:|---:|---:|---:|
| 2 | 4096 | 0.5640 | 0.6331 | 10.9% |
| 4 | 4096 | 0.6088 | 0.6684 | 8.9% |
| 2 | 6144 | 1.4597 | 1.6230 | 10.1% |
| 4 | 6144 | 1.4670 | 1.7285 | 15.1% |
| 2 | 8192 | 4.0476 | 4.1468 | 2.4% |
| 4 | 8192 | 4.0198 | 4.3554 | 7.7% |

当前标准阻塞版已复测四卡 4096³，随机输入、连续 20 轮复用、延迟发送及 Graph 均通过。其他行保留此前所选配置的实验结果，并非本次代码重新测量；各行只与各自同批 Torch 对比。
