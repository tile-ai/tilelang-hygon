# Persistent Kernel Optimization

This document explains the **Persistent Kernel** optimization technique in TileLang and how to implement it to improve GPU performance.

## 1. What is a Persistent Kernel?

In a standard GPU kernel launch ("One-to-One" mapping), you typically launch a grid of thread blocks where each block computes one output tile. If the problem size is large, the number of blocks can far exceed the number of physical Streaming Multiprocessors (SMs) on the GPU. The GPU scheduler manages these blocks, swapping them in and out as resources become available.

**Persistent Kernel** changes this paradigm:
1. **Fixed Grid Size**: You launch a fixed number of thread blocks, typically equal to the number of SMs (or a small multiple).
2. **Work Loop**: These blocks do not exit after computing a single tile. Instead, they stay "alive" (persistent) and run a loop to fetch and process work tiles from a global work queue (or a calculated index sequence) until all work is completed.

## 2. Why Use Persistent Kernels?

Persistent kernels offer several performance advantages, especially for memory-bound or latency-sensitive workloads like GEMM or FlashAttention:

- **Reduced Launch Overhead**: Eliminates the hardware overhead of repeatedly scheduling and retiring thousands of short-lived thread blocks.
- **Improved L2 Cache Locality**: Since the same physical SM processes multiple tiles in sequence, data loaded into the L2 cache for one tile (e.g., weights in a matrix multiplication) might be reused by the next tile processed by the same block.
- **Better Load Balancing (Tail Effect)**: In a standard launch, if one block is slow, the entire wave waits. With persistent kernels, faster SMs can pick up more work from the queue, reducing the "tail effect" where the GPU is underutilized waiting for the last few blocks to finish.

## 3. Implementation in TileLang

TileLang supports persistent kernels through two main approaches: Manual Calculation and the `T.Persistent` Primitive.

### 3.1 Method 1: The `T.Persistent` Primitive (Recommended)

TileLang provides a high-level primitive `T.Persistent` that automatically handles the loop logic and work distribution.

**Syntax:**
```python
for indices in T.Persistent(problem_sizes, grid_size, block_id):
    # Kernel logic
```

**Example:**
```python
import tilelang as tl
from tilelang import tvm as tvm
from tilelang.language import T

@tl.jit(out_idx=[-1])
def matmul_persistent(M, N, K, block_M, block_N, block_K, dtype="float16"):
    # 1. Determine Grid Size (usually match #SMs)
    # This example assumes we pass sm_num as an argument or query it
    sm_num = 80  # Example for A100, or use torch.cuda.get_device_properties...

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # 2. Launch fixed number of blocks (sm_num)
        with T.Kernel(sm_num, threads=128) as (block_id):

            # 3. Use T.Persistent to iterate over output tiles
            # Problem size is [M/block_M, N/block_N]
            for bx, by in T.Persistent(
                [T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
                sm_num,
                block_id
            ):
                # ... Initialize shared memory & registers ...
                # ... Compute GEMM for tile (bx, by) ...
                pass
```

### 3.2 Method 2: Manual Loop Implementation

You can also manually implement the loop logic. This gives you full control over the traversal order (swizzling) inside the persistent loop.

**Logic:**
1. Calculate total number of tiles (`total_tiles`).
2. Each persistent block processes tiles with ID `tid = block_id + i * grid_size`.
3. Map linear `tid` back to 2D coordinates `(bx, by)`.

**Example:**
```python
@T.prim_func
def main_manual(
    A: T.Tensor((M, K), dtype),
    B: T.Tensor((K, N), dtype),
    C: T.Tensor((M, N), dtype),
):
    # Calculate grid dimensions
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    total_tiles = m_blocks * n_blocks

    # Calculate how many "waves" of work each block needs to do
    waves = T.ceildiv(total_tiles, sm_num)

    with T.Kernel(sm_num, threads=128) as (block_id):
        # ... Allocations ...

        # Loop over waves
        for w in T.serial(waves):
            # Calculate linear tile ID
            tile_id = sm_num * w + block_id

            # Boundary check: Ensure we don't process invalid tiles in the last wave
            if tile_id < total_tiles:
                # Map linear ID to 2D (Row-Major example)
                bx = tile_id // n_blocks
                by = tile_id % n_blocks

                # ... Compute tile (bx, by) ...
```

## 4. Best Practices

1. **Grid Size Selection**:
   - Set the grid size to be equal to the number of concurrent blocks on the target GPU. It is basically
    the number of SMs multiplied by the number of active blocks per SM.
   - In PyTorch: `torch.cuda.get_device_properties(device).multi_processor_count`.
   - If your kernel uses very high register counts, you might want to launch slightly fewer blocks to ensure occupancy, generally the numbler of block per SM is a hyperparameter that needs to be tuned.

2. **Global Memory Atomicity**:
   - Since persistent blocks reuse the same hardware resources, be careful with global synchronization barriers. Standard `__syncthreads()` only synchronizes threads *within* a block.
   - If you need global synchronization across all blocks, persistent kernels require special "Device Scope" barriers (often implemented via atomic counters), which are advanced and not standard in basic persistent loops.

3. **L2 Cache Swizzling**:
   - The traversal order inside the persistent loop matters.
   - Instead of a simple row-major mapping (`bx = tile_id // n_blocks`), you can implement swizzling logic (like Hilbert curve or Z-order) *inside* the loop to maximize L2 hit rates.

## 5. Complete Example Code

See `tilelang/perf/gemm/persistent_gemm.py` for a runnable example comparing standard and persistent GEMM implementations.
