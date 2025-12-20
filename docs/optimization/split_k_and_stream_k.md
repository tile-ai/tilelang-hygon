# Split-K and Stream-K Optimization

This document explains **Split-K** and **Stream-K**, two advanced work distribution techniques used in TileLang to optimize reductions (like Matrix Multiplication or FlashAttention) when standard parallelization strategies fail to saturate the GPU or suffer from load imbalance.

## 1. Split-K Optimization

### 1.1 The Problem
In a standard Matrix Multiplication (GEMM) $C = A \times B$, we typically parallelize over the output matrix dimensions $M$ and $N$. Each thread block computes one or more output tiles of $C$.
*   **Problem:** If $M$ and $N$ are small but $K$ (the reduction dimension) is very large, the grid size (number of output tiles) might be too small to fill the GPU's Streaming Multiprocessors (SMs).
*   **Consequence:** "Tail Effects" and low occupancy. The GPU is underutilized because there isn't enough independent parallel work in the output space.

### 1.2 The Solution
**Split-K** divides the reduction dimension $K$ into multiple chunks (splits). Instead of one thread block computing the full dot product for a tile $C_{i,j}$, multiple thread blocks work on partial dot products for the same tile in parallel.
*   **Block 1:** Computes $A \times B$ for $k \in [0, K/2)$
*   **Block 2:** Computes $A \times B$ for $k \in [K/2, K)$
*   **Final Step:** An atomic reduction (or a separate reduction kernel) sums up the partial results from Block 1 and Block 2 to produce the final $C_{i,j}$.

### 1.3 When to Use
*   **Large K, Small M/N:** Typical in the "prefill" phase of LLMs or batch-1 inference where batch size is small.
*   **Low Occupancy:** When `(M / block_M) * (N / block_N)` is significantly smaller than the number of SMs on your GPU.

### 1.4 Implementation in TileLang

In TileLang, you can implement Split-K by adding a `split_k` dimension to your grid and using `T.atomic_add` for the final write-back.

```python
import tilelang.language as T

@T.prim_func
def matmul_splitk(A: T.Tensor, B: T.Tensor, C: T.Tensor):
    # Standard grid (M/block_M, N/block_N) + Split-K dimension
    with T.Kernel(N_tiles, M_tiles, split_k_slices) as (bx, by, bz):
        # bz is the split index (0 to split_k-1)
        
        # Calculate K range for this block
        k_step = K // split_k_slices
        k_start = bz * k_step
        
        # ... Perform GEMM on partial K range ...
        
        # Final accumulation requires atomics because multiple blocks write to C[by, bx]
        for i, j in T.Parallel(block_M, block_N):
             T.atomic_add(C[by * block_M + i, bx * block_N + j], C_local[i, j])
```

## 2. Stream-K Optimization

### 2.1 The Problem
**Split-K** fixes the "not enough work" problem but introduces a new one: **Load Imbalance (Quantization Efficiency)**.
*   If you simply divide $K$ by an integer factor, some blocks might finish much faster than others if the work doesn't divide evenly.
*   More importantly, standard tiling creates a "wave quantization" effect. If your grid size is 81 and you have 80 SMs, 1 SM will have to do a second pass while 79 sit idle.

### 2.2 The Solution
**Stream-K** treats the entire GEMM computation (all output tiles $\times$ all K iterations) as a single linear stream of work.
1.  **Linearize Work:** Calculate total work units (e.g., total tiles).
2.  **Even Distribution:** Divide this total work *exactly* evenly among all available SMs.
3.  **Hybrid Execution:**
    *   Most blocks compute full output tiles (standard data reuse).
    *   "fixup" blocks handle the boundaries where one output tile is split across two SMs.

**Benefit:** Perfect load balancing. Every SM does exactly the same amount of work, regardless of grid dimensions.

### 2.3 When to Use
*   **Extreme Irregular Shapes:** When $M, N, K$ sizes cause severe wave quantization (tail effects) with standard tiling.
*   **Strict Latency Requirements:** When predictable execution time is critical.

### 2.4 Implementation in TileLang

Stream-K is more complex to implement manually. It generally involves:
1.  Launching exactly `#SM` persistent thread blocks.
2.  Each block calculating its global start and end offset in the linearized "iteration space" $(m, n, k)$.
3.  Logic to determine if a block is computing a "full tile" or a "partial tile" (requiring atomic accumulation or peer-to-peer communication).

*(Note: Full Stream-K implementation in TileLang typically leverages the `T.Persistent` primitive combined with complex index math to map linear IDs to $(m, n, k)$ coordinates).*

## 3. Summary Recommendation

| Technique | Best For | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Standard Tiling** | Large M, N | Simple, max L2 reuse | Poor occupancy for small M/N |
| **Split-K** | Large K, Small M/N | Increases parallelism, fills GPU | Atomic overhead, requires tuning splits |
| **Stream-K** | Odd shapes, "Tail" problems | Perfect load balance | Complex index logic, potential overhead |

### Heuristic Rule of Thumb
1.  Start with **Standard Tiling**.
2.  If occupancy is low (< 4 waves) and $K$ is large, try **Split-K** with `split=2` or `4`.
3.  If performance is still limited by the "tail" (e.g., grid size slightly larger than #SMs), consider **Stream-K** or **Persistent** kernels.

