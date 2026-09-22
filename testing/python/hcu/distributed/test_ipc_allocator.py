import pytest

from tilelang.distributed.backends.ipc import IpcAllocator


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
