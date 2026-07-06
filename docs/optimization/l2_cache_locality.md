# L2 Cache Locality Optimization Techniques

This document details the optimization techniques used in TileLang to maximize GPU L2 cache locality. These techniques are critical for memory-bound and compute-bound kernels (like GEMM, Convolution, and FlashAttention) to minimize DRAM access and avoid "partition camping."

## 1. Why L2 Locality Matters

On modern GPUs (NVIDIA Ampere/Hopper, AMD CDNA), the L2 cache is the last level of on-chip memory before accessing global memory (HBM/DRAM).
- **High Latency:** DRAM access is 100x slower than L2 access.
- **Partition Camping:** DRAM is divided into "partitions." If multiple thread blocks try to access addresses that map to the same partition simultaneously, they serialize, causing massive performance drops.
- **Reuse:** If Block A loads data (e.g., a tile of Matrix B), and Block B (scheduled immediately after) needs the same data, we want Block B to find it in L2 before it gets evicted.

## 2. Global Memory Traversal (Inter-Block Locality)

TileLang employs two primary strategies to enforcing spatial locality at the grid level.

### 2.1 Thread Block Swizzling (Rasterization)

**Target:** Standard "One-to-One" Kernels (Non-Persistent)

In a standard grid launch, `blockIdx.x` increases linearly. This causes a "scan-line" effect where the kernel finishes one full row of tiles before moving to the next. By the time it wraps around to the start of the next row, the "left-side" data of the B-matrix has likely been evicted from L2.

**Solution: Rasterization (Zig-Zag)**
We remap the linear `blockIdx` to a 2D coordinate that follows a "Snake" or "Zig-Zag" pattern. This ensures that consecutively executed blocks are physically close in 2D space.

- **API:** `tl.use_swizzle(panel_size=10, order="row")`
- **Underlying C++:** `tl::rasterization2DRow<panel_width>()`

**Visual Pattern:**
```
Standard:
-> -> -> -> (Jump)
-> -> -> ->

Swizzled (Panel Width = 2):
|  /  |  /
v /   v /
|/    |/
```

**Code Example:**
```python
@T.prim_func
def main(...):
    with T.Kernel(...):
        # Enables swizzling for this kernel
        T.use_swizzle(panel_size=10)
        # ... logic using standard bx, by ...
```

---

### 2.2 Group Swizzling (Persistent)

**Target:** Persistent Kernels

Persistent kernels launch a fixed number of blocks (e.g., equal to #SMs) that loop over the problem size. Since the software controls the loop order, we can't rely on `blockIdx` remapping.

**Solution: Explicit Grouping**
Inside the persistent loop, we calculate 2D coordinates `(bx, by)` using a "Group Size" logic. This forces the loop to process a "thick" strip (group) of tiles before moving to the next strip.

- **Key Parameter:** `group_size` (typically 8)
- **Logic:**
    1. Divide grid into macro-groups of size `group_size`.
    2. Iterate fully within a macro-group (local reuse) before moving to the next.

**Code Logic:**
```python
# Typically inside a persistent loop
group_size = 8
tile_id = start_id + iteration_step

# Map linear tile_id to 2D coordinates with locality
bx = (tile_id // group_size) % m_blocks
by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size
```

### 2.3 Comparison

| Feature | Thread Block Swizzling | Group Swizzling |
| :--- | :--- | :--- |
| **Kernel Type** | Standard (Dynamic Grid) | Persistent (Fixed Grid) |
| **Implementation** | C++ Device Function (Macro) | Python/TIR Loop Arithmetic |
| **Overhead** | Near Zero (Static calculation) | Low (Integer math per loop) |
| **Flexibility** | Limited to predefined patterns | Fully programmable |

## 3. Shared Memory Layout (Intra-Block Locality)

While grid swizzling improves **L2** hit rates, optimizing **L1/Shared Memory** access is equally critical to prevent bank conflicts and maximize bandwidth from L1 to Registers.

### 3.1 Swizzled Shared Memory Layout

**Target:** Tensor Core Operations (MMA) & Vectorized Loads

When multiple threads in a warp access shared memory, bank conflicts occur if they access different addresses falling into the same memory bank. This is especially common with standard Row-Major or Column-Major layouts during Tensor Core loads.

**Solution: XOR Swizzling**
TileLang supports specialized layouts (like `swizzled` or `cutlass` compatible layouts) that permute the address bits using XOR operations.

- **Logic:** `bank_id = (logical_addr ^ (logical_addr >> shift)) % num_banks`
- **Effect:** Consecutive threads access permuted banks, preventing conflicts even when accessing a strided column.

**Usage:**
This is typically handled automatically by TileLang's `T.copy()` or intrinsic layouts when using Tensor Cores, but manual layouts can be defined.

```python
# Automatic swizzling during copy to shared memory for Tensor Cores
T.copy(Global_A, Shared_A, coalesced_width=8)
```

## 4. Pipeline Prefetching (L2 Latency Hiding)

**Target:** All Memory-Bound Kernels

Even with perfect locality, L2 access still has latency (~200 cycles).

**Solution: Multi-Stage Pipelining**
TileLang uses `T.Pipelined` to prefetch data for future iterations into registers/L1 while computing the current iteration.

- **Logic:** Load `Tile[k+1]` from Global $\to$ Shared while computing `Tile[k]`.
- **Benefit:** Hides the global memory latency behind the compute math.

```python
for k in T.Pipelined(K_iters, num_stages=3):
    T.copy(Global[k], Shared)  # Prefetch
    T.gemm(Shared, Registers)  # Compute
```

## 5. When to Use Which?

1. **Use Thread Block Swizzling** for most standard kernels (`matmul`, `conv`). It is the safest default.
2. **Use Group Swizzling** only when writing **Persistent Kernels**. It's also worth a try when thread block swizzling yields suboptimal results.
3. **Use Swizzled Layouts** whenever targeting **Tensor Cores** (usually implicit in TileLang templates).
4. **Use Pipelining** (`num_stages >= 2`) for almost every GEMM-like kernel to maximize memory throughput.

## 6. Tuning `panel_size` / `group_size`

- **Default:** `8` or `10`.
- **Why?** This roughly matches the number of thread blocks that can run concurrently on an SM or a cluster of SMs (GPC/TPC) sharing an L2 slice.
- **Too Small (1):** Degrades to standard row-major (bad locality).
- **Too Large (>16):** May lose locality in the *other* dimension (vertical vs horizontal).
