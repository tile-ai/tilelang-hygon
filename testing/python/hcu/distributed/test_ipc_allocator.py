import pytest

import tilelang
from tilelang.distributed.backends.ipc import IpcAllocator, IpcMetadataInitializer, tvm_hcu_stream
from tilelang.jit.kernel import JITKernel


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
    allocator = IpcAllocator(1024, rank=0, world_size=2, device=0, runtime=runtime)
    allocator.initialize(handles=[b"local", b"peer"], device_ids=[0, 1])
    first, second = allocator.allocate(3), allocator.allocate(5)
    assert (first.offset, second.offset) == (0, 256)
    assert allocator.metadata == (0, 2, 0x1000, 0x2000)
    assert allocator.contains(first.ptr, first.nbytes)
    generation = allocator.generation
    allocator.close(); allocator.close()
    assert runtime.closed == [0x2000] and runtime.freed == [0x1000]
    assert allocator.generation == generation + 1


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
