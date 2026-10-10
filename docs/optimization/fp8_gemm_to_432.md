# HCU fp8 GEMM:从 232 到 432 TFLOPS 的优化过程

> 目标:让 HCU(gfx938)上原生 tilelang 编译的 fp8 channelwise GEMM,尽量逼近
> lightop / hipblasLt w8a8 的 ~560 TFLOPS 峰值。最终在 **8192³** 达到 **432 TFLOPS**
> (占参考峰值 77%),且结果与精确 fp32 参考**逐元素相等**。

一键复现:

```bash
HIP_VISIBLE_DEVICES=2 python perf/gemm/repro_fp8_432.py
# 期望:max_rel=0.00e+00, TFLOPS ~= 430
```

正确性测试:

```bash
HIP_VISIBLE_DEVICES=2 pytest testing/python/hcu/test_hcu_gemm_fp8_mls_channelwise.py -x -q
```

---

## 起点与参照

- 硬件:Hygon DCU gfx938,64 CU,warp size 64。
- 参考峰值:lightop / hipblasLt w8a8 在 8192³ ≈ **560 TFLOPS**;单看 MMAC 指令
  (`v_mmac_f32_16x16x32_fp8_fp8`)理论 ≈ 529 TFLOPS。
- 我们最初的 fp8 channelwise 手写异步拷贝内核只有 **232 TFLOPS**。

## 关键定位:差距不在 MMA,也不在 global 拷贝

对比我方内核与 lightop/hipblaslt 反汇编后,发现**两者用的是同一条 MMAC 指令、
同一种 16 字节异步 global→shared 拷贝**(`cp_async_gs<16>` / `buffer_load_dwordx4`)。

真正的差异在 **shared→register(fragment)这一读的形态**:

| 路径 | shared 读指令 | 结果 |
|---|---|---|
| 早期:scalar LDS | 逐元素 `ds_read_b8` 散读 | 232 / 332 TFLOPS |
| 最终:MLS 矩阵 LDS | `ds_read_format_tensor_a/b` → HCU `ds_read_m32x32_b8` | **432 TFLOPS** |

也就是说:global 搬运和 MMA 都不是瓶颈,**LDS 读是不是"矩阵形态"并和 MMA 重叠**
才是决定性能的那一环。lightop/hipblaslt 用的正是硬件矩阵 LDS 读。

## 怎么把 LDS 读切到矩阵形态(MLS 路径)

tilelang 的 MLS 生产者会把 `T.copy(..., prefer_instruction="matrix_load")`
降低成 `ds_read_format_tensor_a/b`(硬件矩阵 LDS 读)。让 A 和 B **都**走这条路径,
`T.gemm` 就会通过 `_gemm_mls_mls` lowering 生成矩阵读 + MMAC:

```python
T.copy(A[...], A_shared, prefer_instruction="matrix_load")   # A 走矩阵 LDS
T.copy(B[...], B_shared, prefer_instruction="matrix_load")   # B 走矩阵 LDS
T.gemm(A_shared, B_shared, C_local, False, True, k_pack=1)   # -> ds_read_format + v_mmac
```

> 这条指令链正是 hipblaslt fp8 内核用的读法,是"400+ 唯一剩路",现在打通了。

### 两个必须注意的约束(踩过)

1. **`block_K` 必须是 64 的倍数。**
   fp8 MLS 在 `trans_B=True` 时,B 侧用 trans b8 MLS atom(见
   `src/hcu/op/mls.cc` 的 `kMlsTileConfigsB8Trans`,tile_k = 64/128),
   要求 `block_K % 64 == 0`。`block_K=32` 会报 "No valid MLS tile config"。
   参考内核正是 `block_K=64`。

2. **`num_stages==0` 时手动补异步等待。**
   MLS 的 global→shared 拷贝是异步的;不分段(ns=0)时 pipeline pass 不会自动发
   wait,必须显式 `T.ptx_wait_group(0); T.sync_warp()`,否则读到未落地的数据。
   ns=1 时 pipeline pass 会自动插入 wait。

## 最优配置与 sweep 结果(8192³, block_K=64)

| 配置 (bm×bn×64, ns, threads) | TFLOPS |
|---|---|
| **256×256, ns=1, 512** | **432.6** ✅ 最优 |
| 256×128, ns=1, 256 | 351 |
| 256×256, ns=2, 512 | 340 |
| ns=0 各配置 | ≈ 245 |

两个决定性因素:**矩阵 LDS 读** + **单段异步重叠(ns=1)**。

## 走过的弯路(负结果,别重复)

- **循环内 full-K 寄存器重叠(double-buffer fragment 读)**:在 gfx938 fp8 上
  **零收益**(225.7 vs 225.8)。瓶颈是每线程 256 宽 fp8 fragment 的寄存器压力,
  不是 load-vs-MMAC 的重叠,该改动已回退。
- **scalar LDS 路径加线程数**:256→512→1024 能涨到 332,但 1536/2048 会寄存器
  溢出,是 scalar 路径的上限,仍远低于 MLS 路径。

## 复现脚本说明

- `perf/gemm/repro_fp8_432.py` — **单文件自包含**,编译算子 + 正确性校验 + 打印 TFLOPS。
  直接跑即可,是给你手动验证用的入口。
- `perf/gemm/fp8_mls_channelwise.py` — 算子模块(默认就是 432 配置),供别处 import。
- `perf/gemm/fp8_mls_channelwise_bench.py` — (bm,bn)×ns×threads 的 sweep 脚本。
- `perf/gemm/native_fp8_channelwise.py` — 300+ 的 scalar-LDS cp.async 对照版本。
- `perf/gemm/fp8_asm_dump.py` — 导出内核源码、grep 关键指令(`ds_read_m32x32_b8`、
  `v_mmac`、`cp_async`),用于定位/验证。
- `testing/python/hcu/test_hcu_gemm_fp8_mls_channelwise.py` — 正确性 + 路径断言测试。

### 复现注意事项

- 所有 GPU 运行统一用 `HIP_VISIBLE_DEVICES=2`。
- 这些交付物**全部在 DSL 层,没有任何 C++ 改动**,不需要重新编译仓库。
- **缓存陷阱**:若你改过 C++,`~/.tilelang/cache/` 的 key 只含 git hash、不含未提交
  的 C++ 改动,须 `rm -rf ~/.tilelang/cache/*` 再跑,否则会命中旧缓存。纯 Python
  改动同理,清缓存更保险。
