# GEMM + ReduceScatter：通信 Block Push

入口：[example_gemm_reducescatter_block_push.py](example_gemm_reducescatter_block_push.py)。

## 实现简介

每个 rank 输入 FP16 的 `A[M, K]` 和 `B[K, N]`，先计算本卡 GEMM，再对各卡的对应输出求和。`K` 是**每卡本地 K**，不会再除以卡数。每个 rank 输出 `[M / world, N]`，对应完整规约结果的连续行区间 `[rank * M / world, (rank + 1) * M / world)`，与 Torch 的 `reduce_scatter_tensor` 布局一致。

- **计算与通信分工**：默认启动 80 个 persistent Block，每个 Block 512 个线程。64 个计算 Block 搭配 16 个通信 Block，或使用 72/8 配置；各 Block 循环处理分配到的任务。
- **GEMM**：输出 tile 为 `256×256`，K 步长为 16，使用手写四阶段流水线和两组 wave 交错执行搬运、计算。FP32 累加结果转换为 FP16，写入普通本卡显存中的 `partial`，然后发布 `ready`。
- **owner Push**：通信 Block 等待本卡对应 tile 就绪，将其贡献从 `partial` 读入寄存器，再写入 owner 的 IPC 接收缓冲区。每个 tile 默认拆成两个 `128×256` 条带，由两个通信任务处理；搬运和规约以 `16×256` chunk 为单位。条带发送完成后发布 `arrived`，owner 等待其他 rank 的贡献，与本卡贡献一起用 FP32 累加，再转换为 FP16。owner 将规约结果写入本卡输出分片，不再广播。
- **tile 调度**：为让连续产出的 tile 分散到不同 owner，GEMM 按 owner 交错调度：`physical_tile = (s % world) * owned + s // world`，其中 `owned` 是每卡负责规约的 tile 数。规约后的结果仍写入本卡连续行分片。
- **调用间同步**：`ready / arrived / done` 等信号配合设备端生产者、消费者计数器区分调用轮次，仅在初始化时清零。调用顺序为融合 Kernel → 完成等待 Kernel → 流上的 RCCL barrier，保护下一次调用前的缓冲区复用；共享同一组缓冲区的调用不能并发执行。

主 Kernel 参数默认 `BM=256, BN=256, BK=16, BLOCKS=80`。目前手写 GEMM 流水线只支持这一 tile 几何，不能仅修改 BM/BN/BK 就使用其他尺寸。

## 运行


```bash
# 双卡：eager / Graph 随机输入和连续调用正确性检查
torchrun --standalone --nproc_per_node=2 \
  examples/distributed/gemm_reduce_scatter/example_gemm_reducescatter_block_push.py \
  --M 4096 --N 4096 --K 4096 --compute-blocks 64

# 八卡：检查正确性，并比较 TileLang 与 Torch 串行版的 eager / Graph 性能
torchrun --standalone --nproc_per_node=8 \
  examples/distributed/gemm_reduce_scatter/example_gemm_reducescatter_block_push.py \
  --M 6144 --N 6144 --K 6144 --compute-blocks 72 --benchmark --repeats 30
```

`--M/--N/--K` 各自默认 4096；M/N 必须是正的 256 倍数，K 必须是正的 64 倍数。ReduceScatter 还要求 `M` 能被 `256 * world` 整除。 `--compute-blocks` 仅接受 64 或 72，分别对应 64/16、72/8。`--dump-source <目录>` 可额外导出各 rank 的 HIP 源码文件。

当前 `--benchmark` 用 GPU Event 分别测量普通调用和 Graph replay，报告各 rank 的中位数，并取其中最大值；融合路径包含完成等待及复用 barrier。Torch 基线为 `torch.mm` 后接 RCCL `reduce_scatter_tensor`，使用预分配输出，不额外添加复用 barrier。

## 性能

测量环境为 BW1000 / gfx936，每卡 80 CU。固定 80 Block、512 线程、`16×256` chunk、每 tile 两个通信条带，FP16 输入/输出、FP32 累加。表中 `M=N=K` 均指每卡尺寸。历史测法为每个 Graph 包含 5 次调用，每轮 20 个样本，共 5 轮：Graph 总时间除以 5，各 rank 分别取中位数后取最大值，再对 5 轮取中位数。

时间单位为 **ms**；加速比为 `Torch 串行 / 融合`，大于 1 表示融合更快。Torch 单 GEMM 和单通信单独测量，两者之和不要求严格等于串行耗时。表中保留全部配置，不只列最快值。

| M=N=K | 卡数 | 计算/通信 Block | 融合 | Torch 串行 | 加速比 | 
| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 2 | 64/16 | 0.5230 | 0.6571 | 1.257× | 
| 4096 | 4 | 64/16 | 0.5447 | 0.6928 | 1.272× | 
| 4096 | 8 | 64/16 | 0.5875 | 0.7314 | 1.245× | 
| 6144 | 2 | 64/16 | 1.4912 | 1.6968 | 1.138× | 
| 6144 | 2 | 72/8 | 1.4507 | 1.6966 | 1.170× | 
| 6144 | 4 | 64/16 | 1.5128 | 1.8006 | 1.190× | 
| 6144 | 4 | 72/8 | 1.4000 | 1.8021 | 1.287× | 
| 6144 | 8 | 64/16 | 1.5775 | 1.8435 | 1.169× | 
| 6144 | 8 | 72/8 | 1.4497 | 1.8437 | 1.272× | 
| 8192 | 2 | 64/16 | 4.1490 | 4.3565 | 1.050× | 
| 8192 | 2 | 72/8 | 4.0595 | 4.3563 | 1.073× | 
| 8192 | 4 | 64/16 | 4.0967 | 4.5614 | 1.113× | 
| 8192 | 4 | 72/8 | 4.0218 | 4.5609 | 1.134× | 
| 8192 | 8 | 64/16 | 4.2187 | 4.6459 | 1.101× | 
| 8192 | 8 | 72/8 | 4.1204 | 4.6466 | 1.128× | 
