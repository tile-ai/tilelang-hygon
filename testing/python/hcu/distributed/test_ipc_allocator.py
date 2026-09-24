import pytest

import tilelang
from tilelang.distributed.backends.ipc import IpcAllocator, IpcMetadataInitializer, tvm_hcu_stream
from tilelang.jit.kernel import JITKernel
from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter


class FakeRuntime:
    def __init__(self): self.closed = []; self.freed = []
    def malloc(self, n): return 0x1000
    def free(self, p): self.freed.append(p)
    def create_handle(self, p): return b"local"
    def open_handle(self, h): return 0x2000 if h == b"peer" else 0x1000
    def close_handle(self, p): self.closed.append(p)
    def can_access_peer(self, d, p): return True


def test_ipc_arena_offsets_metadata_generation_and_close():
    runtime = FakeRuntime()
    allocator = IpcAllocator(1024, rank=0, world_size=2, device=0, runtime=runtime, synchronize=lambda: None)
    allocator.initialize(handles=[b"local", b"peer"], device_ids=[0, 1])
    first, second = allocator.allocate(3), allocator.allocate(5)
    assert (first.offset, second.offset) == (0, 256)
    assert allocator.metadata == (0, 2, 0x1000, 0x2000)
    assert allocator.contains(first.ptr, first.nbytes)
    generation = allocator.generation
    allocator.close(); allocator.close()
    assert runtime.closed == [0x2000] and runtime.freed == [0x1000]
    assert allocator.generation == generation + 1


def test_ipc_collective_close_orders_sync_barrier_unmap_and_free():
    events = []

    class Runtime(FakeRuntime):
        def close_handle(self, ptr):
            events.append(f"close:{ptr:#x}")
            super().close_handle(ptr)

        def free(self, ptr):
            events.append(f"free:{ptr:#x}")
            super().free(ptr)

    class Group:
        def barrier(self):
            events.append("barrier")

    allocator = IpcAllocator(
        1024, rank=0, world_size=2, device=0, group=Group(), runtime=Runtime(),
        synchronize=lambda: events.append("sync"),
    )
    allocator.initialize(handles=[b"local", b"peer"], device_ids=[0, 1])
    allocator.close()
    assert events == ["sync", "barrier", "close:0x2000", "free:0x1000"]


def test_ipc_arena_rejects_bad_requests():
    allocator = IpcAllocator(128, rank=0, world_size=1, device=0, runtime=FakeRuntime())
    allocator.initialize()
    with pytest.raises(ValueError, match="positive"): allocator.allocate(0)
    with pytest.raises(MemoryError, match="exhausted"): allocator.allocate(129)


def test_ipc_allocator_factory_is_exported_from_tilelang():
    with pytest.raises(ValueError, match="Unsupported"):
        tilelang.get_distributed_allocator("unknown")


def test_metadata_initializer_is_generation_aware():
    class Module:
        def __init__(self): self.calls = []
        def get_function(self, name, query_imports=False):
            assert name == "__tilelang_init_ipc_metadata" and not query_imports
            return lambda tensor, count: self.calls.append((tensor, count))

    allocator = IpcAllocator(128, rank=0, world_size=1, device=0, runtime=FakeRuntime())
    allocator.initialize()
    module, initializer = Module(), IpcMetadataInitializer()
    assert initializer.initialize(module, allocator, metadata_tensor="metadata")
    assert not initializer.initialize(module, allocator, metadata_tensor="metadata")
    allocator._generation += 1
    assert initializer.initialize(module, allocator, metadata_tensor="replacement")
    assert module.calls == [("metadata", 3), ("replacement", 3)]


@pytest.mark.parametrize(
    "metadata, message",
    [
        ((0, 1), "count"),
        ((1, 1, 0x1000), "rank/world_size"),
        ((0, 2, 0x1000), "count"),
        ((0, 1023, *(0x1000 for _ in range(1023))), "capacity"),
    ],
)
def test_metadata_initializer_rejects_inconsistent_payload(metadata, message):
    class Module:
        def get_function(self, name, query_imports=False):
            return lambda tensor, count: pytest.fail("invalid metadata must not launch the helper")

    allocator = IpcAllocator(128, rank=0, world_size=1, device=0, runtime=FakeRuntime())
    allocator.initialize()
    allocator._peer_bases = list(metadata[2:])
    allocator.rank, allocator.world_size = metadata[:2]
    with pytest.raises(RuntimeError, match=message):
        IpcMetadataInitializer().initialize(Module(), allocator, metadata_tensor="metadata")


def test_jit_kernel_initialize_delegates_to_distributed_adapter():
    class Adapter:
        def initialize_ipc_metadata(self, allocator, stream=None):
            return allocator == "allocator" and stream == "stream"

    kernel = JITKernel.__new__(JITKernel)
    kernel.adapter = Adapter()
    assert kernel.initialize("allocator", stream="stream")


def test_tvm_hcu_stream_restores_previous_stream():
    calls = []
    current = {0: 7}
    def get_stream(device): return current[device]
    def set_stream(device, stream): calls.append((device, stream)); current[device] = stream
    with tvm_hcu_stream(0, 42, get_stream=get_stream, set_stream=set_stream):
        assert current[0] == 42
    assert current[0] == 7
    assert calls == [(0, 42), (0, 7)]


def test_ipc_adapter_initialization_is_idempotent_per_allocator_generation():
    class Kind:
        name = "hcu"

    class Target:
        kind = Kind()
        attrs = {"dist_backend": "ipc"}

    class Allocator:
        device = 0
        generation = 3

    allocator = Allocator()
    adapter = TVMFFIKernelAdapter.__new__(TVMFFIKernelAdapter)
    adapter.target = Target()
    adapter._ipc_allocator = allocator
    adapter._ipc_initialized_generation = allocator.generation
    adapter.rt_mod = None
    adapter.executable = None

    assert not adapter.initialize_ipc_metadata(allocator)


def test_ipc_launch_rejects_missing_or_stale_allocator_generation():
    class Kind: name = "hcu"
    class Target: kind = Kind(); attrs = {"dist_backend": "ipc"}
    class Allocator: generation = 3
    adapter = TVMFFIKernelAdapter.__new__(TVMFFIKernelAdapter)
    adapter.target, adapter._ipc_allocator, adapter._ipc_initialized_generation = Target(), None, None
    with pytest.raises(RuntimeError, match="must be initialized"):
        adapter.validate_ipc_launch()
    adapter._ipc_allocator, adapter._ipc_initialized_generation = Allocator(), 2
    with pytest.raises(RuntimeError, match="generation changed"):
        adapter.validate_ipc_launch()
