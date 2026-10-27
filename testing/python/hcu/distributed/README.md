# HCU distributed

Currently IPC is the only implemented communication backend.

## Register a backend

Implement native `HcuDistributedBackend` lowering/preamble and a Python
`DistributedAllocator` with metadata and lifecycle support first. Registration
alone does not implement communication. See `src/hcu/distributed/ipc/ipc_backend.cc`
and `tilelang/distributed/backends/ipc.py` for the working IPC implementation.

For an already implemented backend, register it during native initialization:

```cpp
namespace hcu = tvm::tl::hcu;
hcu::RegisterHcuDistributedBackend(
    "my_backend", 42, GetMyBackend,
    {hcu::DistributedCapability::kRankAndWorldSize,
     hcu::DistributedCapability::kBlockRemoteGet});
```

Register its allocator in Python before compilation or cache loading:

```python
from tilelang.distributed import register_distributed_backend
from my_backend import MyAllocator  # Your DistributedAllocator implementation.

register_distributed_backend("my_backend", 42, MyAllocator)
```

Native/Python names and nonzero uint32 IDs must agree; IPC reserves ID 1.
Duplicate registration/replacement is rejected. Declare capabilities only in
native registration. Select the backend with target attribute
`dist_backend="my_backend"`. Native implementations must be built/loaded before
use; dynamically loaded native plugins must remain loaded.

## Examples and tests

- Example: `examples/distributed/gemm_allreduce/example_gemm_ar_ipc.py`
  (all-peer GEMM all-reduce, two or four ranks).
- `testing/python/hcu/distributed/test_integration.py`: real IPC reduction,
  MPI/torchrun remote get and GEMM, tensor lifetime/close, and launcher timeout.
  The `ipc_*_smoke.py` files are child programs invoked by these tests.

Run from the repository root:

```bash
TILELANG_CACHE_DIR=$(mktemp -d /tmp/tl-dist-XXXXXX) \
python -m pytest testing/python/hcu/distributed \
  examples/distributed/gemm_allreduce/test_example_gemm_ar_ipc.py -q -s
```

## Notes

- Device tests check HCU/P2P availability and idle cards automatically, wait up
  to 60 seconds for enough idle cards, then skip with a reason if unavailable.
  Only `HIP_VISIBLE_DEVICES` selects cards; each rank gets a fresh cache.
- Call `allocator.initialize()`, then `kernel.initialize(allocator)` before
  distributed launches. HCU distributed currently uses the TVM-FFI launcher.
- Callers guarantee valid peers, arena-owned sources, symmetric peer layouts,
  and producer/consumer synchronization; launch does not check source ownership.
- Release tensor views/aliases and complete peer reads before collective
  `allocator.close()`. CLOSED is terminal; no context-manager API is provided.
- `HSA_USE_SVM` and `HSA_FORCE_FINE_GRAIN_PCIE` are not set by the launcher.
  They were historical environment workarounds, not current API requirements.
