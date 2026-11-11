# AllGather + GEMM：通信 Block Pull

入口：[example_allgather_gemm_block_pull.py](example_allgather_gemm_block_pull.py)。使用与 GEMM＋AllReduce / ReduceScatter 一致的普通 TileLang 启动路径。

## 实现简介

每个 rank 输入 FP16 的 `A_local[M / world, K]` 和 `B[K, N]`，按 rank 顺序沿行拼接各卡的 A，再计算 `C = A_gathered @ B`。**M 是 AllGather 后的全局行数**，每卡输入只有 `M / world` 行；N 和 K 不按卡数拆分。每卡输出完整的 `C[M, N]`，与 Torch 的 `all_gather_into_tensor` 后接 `torch.mm` 语义一致。各 rank 的 B 可以不同。

- **计算与通信分工**：固定启动 80 个 persistent Block，每个 Block 512 个线程。默认初始分配为 64 个计算 Block＋16 个通信 Block。每个通信 Block 完成自己的全部搬运后，即可独立领取剩余 GEMM 任务；后期最多 80 个 Block 参与计算。
- **GEMM**：输出 tile 为 `256×256`，K 步长为 16，使用手写四阶段流水线和两组 wave 交错执行搬运、计算。FP32 累加，FP16 输出。逻辑任务从本卡拥有的 A 行开始，优先直接读取 `A_local` 计算，不等待这部分数据复制到接收区；其余任务等待对应 A 行的所有 K 面板就绪后，从 `A_gathered` 读取。最终输出仍保持原始行顺序。
- **小批次 Pull**：每个通信任务处理一个 `256×256` A 面板。Block 按 **`16×256`** 子块从源卡普通显存读入寄存器，再写入本卡普通显存接收区；每个 FP16 批次为 8 KiB，一个面板分 16 批搬完。整个面板写完，执行 fence 与 Block 同步后才发布 `ready`。本卡面板也保留实际复制，因此接收区含完整 AllGather 结果。信号使用 fine-grained 显存；源数据与接收区使用普通显存。
- **tile 调度**：所有 Block 从同一个动态任务队列独立领取一个 tile，完成后立即领取下一张任务票，没有固定搭档或跨 Block 的配对等待。保留原来的 tile 发放顺序：`(行0,列0) → (行1,列0) → (行0,列1) → (行1,列1) → …`，发完这两行后再处理下一组两行。顺序只约束队列中的任务排列，不要求两个 Block 同时开始或完成。每个 rank 将逻辑行循环旋转到本卡输入在前；写输出时映射回真实物理行。
- **纯 TileLang 任务领取**：`claim_tile` 是 `T.macro`。线程 0 通过 `T.atomic_add(..., return_prev=True)` 领取票号并按原顺序映射到 tile；借用 `A_shared_0` 起始位置作为 `int32` 共享槽位，经两次 `T.sync_threads()` 完成广播并保护 LDS 复用。没有内嵌 HIP helper，也没有额外分配共享内存。生成汇编仍包含原子加、LDS 读写、Block barrier，以及编译器自动插入的 `v_readfirstlane_b32`。
- **调用间同步**：每次调用先执行 `dist.barrier(group=rccl, device_ids=[device], async_op=True)`，并通过 `work.block_current_stream()` 让当前 GPU 流等待屏障完成，再递增设备端 `epoch`、清零任务队列并启动融合 Kernel。`ready` 只在初始化时清零，通过 epoch 区分调用。共享缓冲区的调用不能并发执行；修改 `A_local` 前必须先确认所有 rank 完成上一轮读取；修改后也要确认所有 rank 的写入完成，再统一进入下一轮。示例的 `random_inputs()` 在修改前后均执行 `torch.cuda.synchronize()` 和跨 rank barrier。入口 barrier用于调用间缓冲区复用，不能据此省略源输入更新所需的完成同步。
- **普通启动**：通过 `tilelang.compile(...)` 编译，执行 `kernel.initialize(signals)` 初始化 IPC 元数据，再直接调用 `kernel(...)`。没有自定义 HIP 模块加载、cooperative 启动器或手工 Graph 节点操作。普通路径已针对本示例的固定 80 Block、gfx936 / 80 CU 配置做过 eager、Graph 和连续调用验证。

目前固定使用 `BM=256, BN=256, BK=16, BLOCKS=80` 和 `16×256` 读取批次。没有保留 naive、大批次、限速、跳过本地复制或插桩等实验开关。

## 运行

```bash
# 双卡：eager / Graph 随机输入和连续调用正确性检查
# 每卡 A_local 为 [2048, 4096]，输出为 [4096, 4096]
torchrun --standalone --nproc_per_node=2 \
  examples/distributed/allgather_gemm/example_allgather_gemm_block_pull.py \
  --M 4096 --N 4096 --K 4096 --compute-blocks 64

# 八卡：检查正确性，并比较 TileLang 与 Torch 串行版的 eager / Graph 性能
# 每卡 A_local 为 [1024, 8192]，输出为 [8192, 8192]
torchrun --standalone --nproc_per_node=8 \
  examples/distributed/allgather_gemm/example_allgather_gemm_block_pull.py \
  --M 8192 --N 8192 --K 8192 --compute-blocks 64 --benchmark --repeats 30
```

需要本仓库已构建的 HCU/IPC Python 环境，以及支持对应 DCU 指令的 DTK/AICC 工具链。示例面向单节点 2/4/8 卡 gfx936（每卡 80 CU），HIP 库路径为 `/opt/dtk/hip/lib/libamdhip64.so`。

`--M/--N/--K` 各自默认 4096；M 必须是正的 `256 × world` 倍数，N 和 K 必须是正的 256 倍数。支持满足这些条件的矩形尺寸。`--compute-blocks` 接受 64 或 72，分别对应初始 64/16、72/8；本版本下表性能只测了默认 64/16，不代表 72/8 更快。

默认执行三组 eager、三组 Graph 随机输入检查，并分别保存、校验连续八次调用的输出，验证完整 AllGather、GEMM 数值、ready 和任务队列状态。检查过程包含 rank 启动延迟。随机输入检查之外的连续复用测试保持源数据不变。

`--benchmark` 用 GPU Event 测量普通调用和 Graph replay，报告各 rank 的中位数，并取其中最大值。`--graph-batch` 默认 5，即每个计时 Graph 内包含五次调用，时间除以 5；融合路径包含 RCCL barrier、epoch 更新和队列清零。Torch 基线使用预分配输出，执行 AllGather 后接 GEMM。eager 测量可能包含主机提交不及时形成的空档，比较融合性能时优先看 Graph。

默认不生成结果或 Trace 文件。`--dump-source <目录>` 可额外导出各 rank 的 HIP 源码；编译过程可能使用 TileLang/TVM 临时目录与缓存，可通过 `TILELANG_CACHE_DIR`、`TILELANG_TMP_DIR`、`TVM_FFI_CACHE_DIR` 和 `TMPDIR` 指定位置。

## 性能

时间单位为 **ms**；加速比为 `Torch 串行 / 当前融合`。

| M=N=K | 卡数 | 初始计算/通信 Block | 改屏障前融合 | 原 HIP helper | Torch 串行 | 加速比 |
| ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| 4096 | 2 | 64/16 | 0.4615 | 0.4620 | 0.6361 | 1.378× |
| 4096 | 4 | 64/16 | 0.4653 | 0.4673 | 0.6758 | 1.452× |
| 4096 | 8 | 64/16 | 0.5361 | 0.5380 | 0.7204 | 1.344× |
| 6144 | 2 | 64/16 | 1.3569 | 1.3583 | 1.6439 | 1.212× |
| 6144 | 4 | 64/16 | 1.3676 | 1.3679 | 1.7670 | 1.292× |
| 6144 | 8 | 64/16 | 1.4804 | 1.4962 | 1.8325 | 1.238× |
| 8192 | 2 | 64/16 | 3.9695 | 3.9660 | 4.3105 | 1.086× |
| 8192 | 4 | 64/16 | 3.9897 | 3.9906 | 4.5173 | 1.132× |
| 8192 | 8 | 64/16 | 4.0370 | 4.0456 | 4.6299 | 1.147× |

每种实现均检查三组随机输入的 eager / Graph 结果，以及连续 20 次 eager 调用和 20 次五调用 Graph replay；连续复用期间源输入保持不变。验证完整 AllGather、GEMM、epoch、任务队列、信号 padding 和缓冲区哨兵。92 个实际编译产物均导出并检查 HIP、汇编及资源元数据：共享缓冲区保持 64 KiB，未出现寄存器溢出。

Block 独立领取和原 tile 发放顺序均保持不变。此前删除 Block 配对的实验中，直接改成普通行优先领取会让三个 8192³ 配置慢约 3.6%～4.4%；当前版本继续按同列、相邻两行交替发放，不设置固定搭档或搭档等待。

更换输入时仍需遵守前述同步协议。此前输入更新对照中，矩形配置若省略输入修改后的完成同步，普通启动与 cooperative 启动都曾出现错误；示例的 `random_inputs()` 在输入修改前后均完成跨 rank 同步。改用纯 TileLang 领取任务不改变这一要求。

调用间屏障现已统一为与 GEMM+AllReduce / ReduceScatter 相同的异步 RCCL `dist.barrier` 加 `work.block_current_stream()`，并移除了显式 token 张量。AG 的屏障仍位于每次调用入口，保护上一轮读取完成后再复用缓冲区。更改后，八卡 8192³、双卡 4096³，以及四卡 `2048×768×512`、72/8 分工均通过三组随机输入的 eager / Graph 检查、连续八次普通调用和连续八次 Graph replay；五调用 Graph 的测速及测速后检查也通过。单轮 Graph 测得约 4.012 ms、0.459 ms、0.0767 ms；这些是功能验证时的测速，不是与旧 token 路径随机交错的性能对照。
