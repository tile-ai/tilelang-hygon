<img src=./images/logo-row.svg />

<div align="center">

# tilelang-hygon

</div>

Tile Language (**tile-lang**) is a concise domain-specific language for high-performance GPU kernels (GEMM, FlashAttention, LinearAttention, etc.). It uses a Pythonic syntax and a [TVM](https://tvm.apache.org/)-based compiler stack so developers can focus on operator logic while the compiler handles low-level optimizations.

**tilelang-hygon** is a Hygon HCU–focused fork of [TileLang](https://github.com/tile-ai/tilelang). We adapt the compilation pipeline (passes, codegen, and runtime) for the Hygon toolchain, so operators written in the TileLang DSL can run on Hygon HCU with minimal changes.

tilelang-hygon supports almost all upstream TileLang syntax.

<img src=./images/MatmulExample.png />

## Latest News

- **v0.1.12-release** — Based on TileLang v0.1.12 for Hygon HCU. Use `build.sh` for source builds.
- **v0.1.9-release** — Based on TileLang v0.1.9 for Hygon HCU. Use `build.sh` for source builds.

## Tested Devices

tilelang-hygon has been tested on **Hygon HCU-2G** and **HCU-3G**.

**Toolchain**: requires **Hygon DTK**. Source the DTK environment before `build.sh` or compiling kernels.

**Target**: leave unset / `auto`, or set `target="hcu"` explicitly (e.g. `@tilelang.jit(target="hcu")`).

## Build from Source

```bash
git clone --recursive https://github.com/tile-ai/tilelang-hygon.git
cd tilelang-hygon

# Default: install deps + editable install (Debug)
bash build.sh

# Skip dependency install when deps are already present
bash build.sh --no-deps

# Build a release wheel
bash build.sh --wheel
```

`build.sh` installs `requirements-dev.txt` and `requirements.txt`, then runs `pip install -e . -v --no-build-isolation`. The first full build compiles TVM and may take several minutes; incremental rebuilds are much faster.

For manual install:

```bash
pip install -r requirements-dev.txt
pip install -r requirements.txt
pip install -e . -v --no-build-isolation
```

## Quick Start

Below is the upstream GEMM example with layout annotation, pipelining, and optional L2 swizzle.

```python
import tilelang
import tilelang.language as T

# Default / auto selects HCU on Hygon HCU; or pass target="hcu" explicitly.
@tilelang.jit
def matmul_relu(
    A, B,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 64,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    M, N, K = T.const('M, N, K')

    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty([M, N], dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        # T.use_swizzle(panel_size=10, enable=True)

        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


import torch

M, N, K = 1024, 1024, 1024
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c_ref = torch.relu(a @ b)

c = matmul_relu(a, b)
torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)

kernel = matmul_relu.compile(a, b)
print(kernel.get_kernel_source())
profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
print(f"Latency: {profiler.do_bench()} ms")
```

## Examples

### Performance-tuned kernels

These are the kernels we actively optimize on Hygon HCU; use them as references for production-oriented tuning:

- [GEMM](./perf/gemm/) — vanilla / persistent / split-K / Stream-K variants with benchmarking
- [Sparse MLA forward](./perf/sparse_mla/) — tuned sparse multi-head latent attention forward kernel

### Upstream reference examples

The [`examples/`](./examples/) tree is inherited from upstream TileLang and is a good place to learn DSL patterns. For now, we have not yet dedicated performance tuning to these cases on Hygon HCU.

## Acknowledgments

We thank the [TileLang](https://github.com/tile-ai/tilelang) and [TVM](https://github.com/apache/tvm) communities. tilelang-hygon is derived from TileLang and adapted for Hygon HCU.
