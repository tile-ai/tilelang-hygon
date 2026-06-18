# Thread Block Swizzling Optimization

This tutorial explains the **Thread Block Swizzling** (or Rasterization) techniques used in TileLang templates to improve GPU L2 cache locality.

## 1. The Problem: Standard Grid Traversal

By default, GPUs launch thread blocks in a predictable "Row-Major" order (increasing Block X, then Block Y).

**Standard Order:**
```
Block(0,0) -> Block(1,0) -> Block(2,0) ... -> Block(N,0)
      ^ (Huge jump in memory address)
Block(0,1) -> Block(1,1) -> Block(2,1) ...
```

**Issue:**
When a kernel computes a large matrix multiplication (GEMM) or convolution, it typically tiles the input matrices.
- **Block(N,0)** uses data from the far right of Matrix A and top of Matrix B.
- **Block(0,1)** (the very next block executed) uses data from the far left of Matrix A and slightly lower in Matrix B.
- **Result:** Data cached in L2 for the right side of the grid is flushed before it can be reused by the next row, leading to **poor L2 cache hit rates** and lower performance ("DRAM bound").

## 2. The Solution: Swizzled (Zig-Zag) Traversal

**Thread Block Swizzling** re-maps the linear block execution index into a custom 2D coordinate $(X_{new}, Y_{new})$. The goal is to maximize spatial locality: blocks that execute close together in time should be close together in 2D space.

**Swizzled Order:**
```
Start -> -> -> \
               | (Short jump)
/ <- <- <- <- /
|
\ -> -> -> -> ...
```
This "Snake" or "Zig-Zag" pattern ensures that when one row of blocks finishes, the next block executed is physically adjacent (just below or above), allowing it to reuse the data just loaded into L2 cache.

### 2.1 Why Remap the Linear Block Index?

**The Core Problem: Hardware Controls Block Scheduling**

The fundamental reason we remap `blockIdx` is that **we cannot control how the GPU hardware schedules thread blocks**, but we can simulate scheduling control by remapping what `blockIdx` means.

#### How GPU Hardware Schedules Blocks

When you launch a kernel with a 2D grid:
```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), ...) as (bx, by):
```

The GPU hardware assigns `blockIdx.x` and `blockIdx.y` values and schedules blocks in a **fixed, predictable order**:

1. **Row-Major Order (Default):** The GPU scheduler processes blocks in row-major order:
   - `(0,0) → (1,0) → (2,0) → ... → (N-1,0)`
   - Then moves to the next row: `(0,1) → (1,1) → (2,1) → ...`
   - This continues until all blocks are scheduled

2. **Hardware-Controlled:** This scheduling order is determined by the GPU's **work distribution unit (WDU)** or **global scheduler**, which is:
   - **Not programmable** - you cannot change it via software
   - **Fixed at hardware level** - determined by GPU architecture
   - **Optimized for general workloads** - but not necessarily for your specific data access pattern

3. **Linear Block Index:** Internally, the hardware converts the 2D `(blockIdx.x, blockIdx.y)` into a linear index:
   ```
   linear_block_idx = blockIdx.x + blockIdx.y * gridDim.x
   ```
   Blocks are scheduled in order of increasing `linear_block_idx`.

#### Why This Causes Poor Locality

The hardware's row-major scheduling creates a "scan-line" effect:

```
Memory Access Pattern (Standard Scheduling):
Block(0,0) → Accesses: A[0:block_M, :], B[:, 0:block_N]
Block(1,0) → Accesses: A[0:block_M, :], B[:, block_N:2*block_N]  (Different B region)
Block(2,0) → Accesses: A[0:block_M, :], B[:, 2*block_N:3*block_N]  (Different B region)
...
Block(N-1,0) → Accesses: A[0:block_M, :], B[:, (N-1)*block_N:N*block_N]  (Far right)
      ↓ (HUGE JUMP - L2 cache likely evicted)
Block(0,1) → Accesses: A[block_M:2*block_M, :], B[:, 0:block_N]  (Back to left, different A)
```

**The Problem:**
- Block `(N-1,0)` loads data from the **right side** of Matrix B into L2 cache
- The very next block `(0,1)` needs data from the **left side** of Matrix B
- By the time `(0,1)` executes, the right-side data has been **evicted from L2**
- Result: **Cache misses** and **DRAM access** (100x slower than L2)

#### The Solution: Remapping `blockIdx` to Simulate Better Scheduling

Since we **cannot control the hardware scheduler**, we use a clever workaround: **remap what `blockIdx` means** so that when the hardware schedules blocks in linear order, they actually compute work in a zig-zag pattern.

**The Key Insight:**
- Hardware schedules: `linear_idx = 0, 1, 2, 3, 4, 5, ...` (row-major)
- We remap: `(bx, by) = f(linear_idx)` to create a zig-zag pattern
- Hardware still schedules linearly, but each block computes different work

**How It Works:**

1. **Hardware assigns `blockIdx`:** When block with `linear_idx = 5` is scheduled, the hardware provides `blockIdx.x` and `blockIdx.y` based on row-major mapping.

2. **We remap at kernel entry:** Before using `bx` and `by`, we call the rasterization function:
   ```cpp
   dim3 swizzled = tl::rasterization2DRow<panel_width>();
   bx = swizzled.x;  // Remapped!
   by = swizzled.y;  // Remapped!
   ```

3. **The remapping function:**
   - Takes the hardware-provided `blockIdx` (which represents row-major order)
   - Calculates a new `(bx, by)` that follows a zig-zag pattern
   - Returns coordinates that are spatially adjacent to recently executed blocks

**Example with Panel Width = 2:**

```
Hardware Scheduling Order (Linear):
0  1  2  3  4  5  6  7  8  9  10 11
↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
Remapped Coordinates (Zig-Zag):
0  2  4  6  |  12 10 8  7  |  15 13 11 9
1  3  5  7  |  14 12 9  8  |  16 14 12 10
```

When hardware schedules `linear_idx = 5`, instead of computing tile `(5, 0)`, we remap it to compute tile `(1, 2)`, which is spatially adjacent to the previously scheduled tiles.

#### Why This Works

**Spatial Locality:** By remapping to adjacent tiles, consecutive hardware-scheduled blocks now:
- Access **overlapping or nearby memory regions**
- Find data **still in L2 cache** from the previous block
- Reduce **DRAM accesses** and improve performance

**No Hardware Changes Required:** This is purely a **software remapping** - we don't need to modify GPU hardware or drivers. The remapping happens at the very start of each kernel execution, with near-zero overhead.

**Visual Analogy:**
Think of it like a **conveyor belt** (hardware scheduler) that moves items in a fixed order. We can't change the belt's direction, but we can **relabel the items** on the belt so that when they arrive, they're processed in the order we want. The belt still moves the same way, but the work done at each position is different.

#### Summary

- **Hardware limitation:** GPU scheduler processes blocks in row-major order (fixed, unchangeable)
- **Software solution:** Remap `blockIdx` at kernel entry to create zig-zag execution pattern
- **Result:** Hardware still schedules linearly, but blocks compute spatially adjacent tiles
- **Benefit:** Improved L2 cache hit rates, reduced DRAM access, better performance

This is why rasterization is so powerful - it gives us **scheduling control** without requiring hardware changes or persistent kernels.

## 3. Implementations in TileLang

TileLang provides two primary swizzling templates in `src/tl_templates/hip/threadblock_swizzle.h`.

### 3.1 `rasterization2DRow`

This function creates **horizontal panels** (strips) and zig-zags within them.

- **Best for:** Workloads where reusing data along the row (horizontal axis) is critical, or generally for standard GEMM where $M$ dimension tiling favors horizontal traversal.
- **Logic:**
    1. Divide the grid into horizontal strips of height `panel_width`.
    2. Inside a strip, fill columns top-to-bottom.
    3. Traverse strips Left $\to$ Right for even panels, Right $\to$ Left for odd panels.

```cpp
template <int panel_width> TL_DEVICE dim3 rasterization2DRow() {
  // 1. Calculate linear ID
  const unsigned int block_idx = blockIdx.x + blockIdx.y * gridDim.x;

  // 2. Determine which "Panel" (horizontal strip) we are in
  const unsigned int panel_size = panel_width * gridDim.x;
  const unsigned int panel_idx = block_idx / panel_size;

  // 3. Calculate internal coordinates
  // ... (See source for full stride math) ...

  // 4. Zig-Zag Logic: Flip Column direction if panel_idx is Odd
  const unsigned int col_idx = (panel_idx & 1)
      ? gridDim.x - 1 - panel_offset / stride
      : panel_offset / stride;

  return {col_idx, row_idx, blockIdx.z};
}
```

### 3.2 `rasterization2DColumn`

This function creates **vertical panels** (strips) and zig-zags within them.

- **Best for:** Workloads where reusing data along the column (vertical axis) is more important.
- **Logic:**
    1. Divide the grid into vertical strips of width `panel_width`.
    2. Inside a strip, fill rows left-to-right.
    3. Traverse strips Top $\to$ Bottom for even panels, Bottom $\to$ Top for odd panels.

```cpp
template <int panel_width> TL_DEVICE dim3 rasterization2DColumn() {
  // ...
  // Zig-Zag Logic: Flip Row direction if panel_idx is Odd
  const unsigned int row_idx = (panel_idx & 1)
      ? gridDim.y - 1 - panel_offset / stride
      : panel_offset / stride;
  // ...
}
```

## 4. Visual Comparison

| **Feature** | **rasterization2DRow** | **rasterization2DColumn** |
| :--- | :--- | :--- |
| **Panel Orientation** | **Horizontal** (Rows) | **Vertical** (Columns) |
| **Traversal Axis** | Moves **Horizontal** (Left $\leftrightarrow$ Right) | Moves **Vertical** (Top $\leftrightarrow$ Bottom) |
| **Inner Loop** | Fills vertical column inside panel | Fills horizontal row inside panel |
| **Visual Pattern** | 🐍 Snake moves sideways | 🐍 Snake moves up/down |

### Example: rasterization2DRow (Panel Width = 2)

Imagine a grid of blocks. The numbers represent the execution order.

**Standard:**
```
0  1  2  3
4  5  6  7
8  9  10 11
```

**Swizzled (Row):**
- Panel 0 (Rows 0-1): Left->Right
- Panel 1 (Rows 2-3): Right->Left

```
0  2  4  6   (Inner order fills vertical pairs first)
1  3  5  7
---------
15 13 11 9   (Next panel reverses direction)
14 12 10 8
```
*(Note: The exact inner ordering depends on the stride calculation, but the macro-movement zig-zags across the grid).*

## 5. TileLang Python API

In TileLang, you don't always need to write the C++ code manually. You can use the `use_swizzle` primitive:

```python
import tilelang.language as tl

# In your TileLang program
tl.use_swizzle(panel_size=10, order="row", enable=True)
```

- **`panel_size`**: Corresponds to `panel_width` in the C++ template.
- **`order="row"`**: Selects `rasterization2DRow`.
- **`order="col"`**: Selects `rasterization2DColumn`.

This annotation injects the C++ call `tl::rasterization2D{Row|Column}<panel_size>()` at the start of the generated kernel, automatically overriding `blockIdx`.

## 6. Thread Block Swizzling vs. Group Swizzling

TileLang provides two complementary swizzling techniques for improving L2 cache locality. Understanding when to use each is crucial for optimal performance.

### 6.1 Overview

**Thread Block Swizzling** (also called Rasterization) is designed for **standard kernels** with dynamic grid launches, where the GPU hardware controls block execution order. It remaps `blockIdx` at runtime using C++ device functions.

**Group Swizzling** is designed for **persistent kernels** with fixed grid sizes, where software controls the loop iteration order. It calculates tile coordinates using arithmetic inside the loop.

### 6.2 Detailed Comparison

| Feature | Thread Block Swizzling | Group Swizzling |
| :--- | :--- | :--- |
| **Kernel Type** | Standard (Dynamic Grid) | Persistent (Fixed Grid) |
| **Grid Launch** | One block per tile (grid size = total tiles) | Fixed number of blocks (e.g., #SMs) |
| **Execution Control** | Hardware-controlled (GPU scheduler) | Software-controlled (loop iteration) |
| **Implementation** | C++ device function (macro) | Python/TIR loop arithmetic |
| **When Applied** | At kernel entry (remaps `blockIdx`) | Inside loop (calculates `bx`, `by` per iteration) |
| **Overhead** | Near zero (static calculation at kernel start) | Low (integer math per loop iteration) |
| **Flexibility** | Limited to predefined patterns (row/column zig-zag) | Fully programmable (custom coordinate mapping) |
| **API** | `T.use_swizzle(panel_size=10, order="row")` | Manual calculation with `group_size` parameter |

### 6.3 Implementation Details

#### Thread Block Swizzling

**How it works:**
1. GPU launches blocks in standard order (row-major by default)
2. At kernel entry, `T.use_swizzle()` injects a C++ device function call
3. The function remaps `blockIdx.x` and `blockIdx.y` to new coordinates
4. Creates zig-zag patterns within horizontal or vertical panels

**Code Pattern:**
```python
@T.prim_func
def main(...):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), ...) as (bx, by):
        T.use_swizzle(panel_size=10, order="row")  # Remaps bx, by
        # Use bx, by as normal - they're already swizzled
        ...
```

**Underlying C++:**
```cpp
// Injected at kernel start
dim3 swizzled = tl::rasterization2DRow<10>();
// bx = swizzled.x, by = swizzled.y
```

#### Group Swizzling

**How it works:**
1. Kernel launches with fixed grid size (e.g., 80 blocks = #SMs)
2. Each block loops over multiple tiles
3. Inside the loop, calculate `(bx, by)` from linear `tile_id` using group arithmetic
4. Groups tiles into "thick strips" to maximize locality

**Code Pattern:**
```python
@T.prim_func
def main(...):
    grid_size = factor * num_sms
    waves = T.ceildiv(total_tiles, grid_size)
    group_size = 8

    with T.Kernel(grid_size, ...) as (block_id,):
        for w in T.serial(waves):
            tile_id = grid_size * w + block_id

            # Group swizzling: map linear tile_id to 2D coordinates
            bx = (tile_id // group_size) % m_blocks
            by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

            # Process tile at (bx, by)
            ...
```

**Key Formula:**
```python
bx = (tile_id // group_size) % m_blocks
by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size
```

This formula ensures that:
- Consecutive `tile_id` values process tiles in groups of `group_size`
- Within a group, tiles are vertically adjacent (same `bx`, different `by`)
- Groups are processed horizontally before moving to the next row of groups

### 6.4 Visual Pattern Comparison

**Thread Block Swizzling (Panel Width = 4):**
```
Execution order (zig-zag within panels):
0  1  2  3  |  12 11 10 9
4  5  6  7  |  15 14 13 8
            |
Panel 0     |  Panel 1 (reversed)
```

**Group Swizzling (Group Size = 4):**
```
Tile processing order (thick vertical strips):
0  4  8  12  |  16 20 24 28
1  5  9  13  |  17 21 25 29
2  6  10 14  |  18 22 26 30
3  7  11 15  |  19 23 27 31
            |
Group 0     |  Group 1
```

### 6.5 When to Use Which?

**Use Thread Block Swizzling when:**
- Writing standard (non-persistent) kernels
- Grid size equals problem size (one block per tile)
- You want automatic, zero-overhead swizzling
- Working with standard GEMM, convolution, or similar workloads

**Use Group Swizzling when:**
- Writing persistent kernels (fixed grid, looping over tiles)
- Need fine-grained control over tile processing order
- Want to experiment with custom locality patterns
- Thread block swizzling doesn't apply (no `blockIdx` remapping possible)

**Can both be used together?**
- No, they target different kernel types
- Thread block swizzling requires standard grid launches
- Group swizzling requires persistent loops
- Choose based on your kernel architecture

### 6.6 Performance Considerations

**Thread Block Swizzling:**
- **Pros:** Zero runtime overhead, automatic, proven patterns
- **Cons:** Limited to predefined zig-zag patterns
- **Best for:** Most standard kernels as a safe default

**Group Swizzling:**
- **Pros:** Fully customizable, works with persistent kernels
- **Cons:** Small arithmetic overhead per iteration, requires manual implementation
- **Best for:** Persistent kernels where thread block swizzling isn't applicable

**Tuning Parameters:**
- **`panel_size`** (Thread Block): Typically 8-16, matches concurrent blocks per SM cluster
- **`group_size`** (Group): Typically 4-16, balances vertical vs. horizontal locality

Both techniques aim to maximize L2 cache reuse by ensuring spatially adjacent tiles are processed temporally close together, reducing DRAM access and improving overall kernel performance.
