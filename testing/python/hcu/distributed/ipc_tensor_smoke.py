"""Actual tensor storage/alias checks and collective close rejection on HCU."""

import gc
import os
import traceback

import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
from tilelang.distributed import release_tensor
from tilelang.distributed.backends.ipc import IpcAllocator


def _local_arena(device):
    allocator = IpcAllocator(65536, rank=0, world_size=1, device=device)
    allocator.initialize()
    return allocator


def _assert_denied(allocator):
    base, generation, metadata = allocator.base_ptr, allocator.generation, allocator.metadata
    try:
        allocator.close()
    except RuntimeError as exc:
        assert "active tensor view" in str(exc)
    else:
        raise AssertionError("close accepted live tensor storage")
    assert not allocator._closed
    assert allocator.base_ptr == base and allocator.generation == generation and allocator.metadata == metadata


def check_local_views(device):
    # Base remains usable after a rejected close, and new views are allowed.
    allocator = _local_arena(device)
    base = tilelang.tensor((16,), torch.float32, allocator=allocator)
    base.fill_(3)
    _assert_denied(allocator)
    another = tilelang.tensor((16,), torch.float32, allocator=allocator)
    another.copy_(base)
    torch.cuda.synchronize(device)
    assert torch.all(another == 3)
    del base
    gc.collect()
    _assert_denied(allocator)
    del another
    gc.collect()
    allocator.close()

    factories = {
        "view": lambda tensor: tensor.view(16),
        "detach": lambda tensor: tensor.detach(),
        "storage": lambda tensor: tensor.untyped_storage(),
    }
    for name, factory in factories.items():
        allocator = _local_arena(device)
        base = tilelang.tensor((4, 4), torch.float32, allocator=allocator)
        base.fill_(5)
        torch.cuda.synchronize(device)
        alias = factory(base)
        del base
        gc.collect()
        _assert_denied(allocator)
        if name != "storage":
            assert torch.all(alias == 5)
        del alias
        gc.collect()
        allocator.close()
        print(f"tensor-alias device={device} kind={name} passed", flush=True)

    for dtype in (torch.float32, torch.bfloat16, torch.int64):
        allocator = _local_arena(device)
        base = tilelang.tensor((4, 4), dtype, allocator=allocator)
        base.fill_(3)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(base, torch.full_like(base, 3))
        release_tensor(base)
        release_tensor(base)
        assert base.numel() == 0
        allocator.close()

    allocator = _local_arena(device)
    base = tilelang.tensor((16,), torch.float32, allocator=allocator)
    base.fill_(7)
    alias = base.detach()
    release_tensor(base)
    _assert_denied(allocator)
    assert torch.all(alias == 7)
    release_tensor(alias)
    release_tensor(alias)
    gc.collect()
    allocator.close()


def remote_read_kernel():
    @T.prim_func
    def main(out: T.Tensor((128,), "float32"), src: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=64):
            T.get_block(T.address_of(src[0]), T.address_of(out[0]), 128, T.get_rank() ^ 1)

    return main


def check_collective_close(rank, world_size, device):
    allocator = IpcAllocator(65536, rank=rank, world_size=world_size, device=device, group=dist.group.WORLD)
    allocator.initialize()
    base = tilelang.tensor((128,), torch.float32, allocator=allocator)
    if rank == 1:
        release_tensor(base)
        del base
        gc.collect()
    dist.barrier()
    original_base, generation, metadata = allocator.base_ptr, allocator.generation, allocator.metadata
    try:
        allocator.close()
    except RuntimeError as exc:
        assert "rank 0" in str(exc) and "1 active tensor view" in str(exc)
    else:
        raise AssertionError("collective close accepted rank 0's live view")
    assert not allocator._closed and allocator.base_ptr == original_base
    assert allocator.generation == generation and allocator.metadata == metadata

    # Rejected close must not invalidate initialization or further allocation
    # and launch. Both ranks allocate the next symmetric slot, even though
    # rank 1 has already dropped its first view.
    src = tilelang.tensor((128,), torch.float32, allocator=allocator)
    # The sync/barrier below completes all producers before the peer pull.
    src.fill_(rank + 2)
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
    kernel = tilelang.compile(remote_read_kernel(), out_idx=[0], target={"kind": "hcu", "mcpu": arch, "dist_backend": "ipc"})
    kernel.initialize(allocator)
    torch.cuda.synchronize(device)
    dist.barrier()
    out = kernel(src)
    torch.cuda.synchronize(device)
    assert torch.all(out == float((rank ^ 1) + 2))
    del out, src
    if rank == 0:
        release_tensor(base)
        del base
    gc.collect()
    dist.barrier()
    allocator.close()
    allocator.close()
    assert allocator._closed and allocator._base == 0
    try:
        kernel(torch.empty((128,), device=f"cuda:{device}"))
    except RuntimeError as error:
        assert "closed" in str(error)
    else:
        raise AssertionError("initialized kernel launched after allocator close")
    del kernel
    print(f"tensor-collective rank={rank} denied_without_free then launch/read/retry-close/closed-launch-rejected passed", flush=True)


def main():
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    device = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    torch.cuda.set_device(device)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        error = None
        try:
            check_local_views(device)
        except Exception:
            error = traceback.format_exc()
        errors = [None] * world_size
        dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError(f"local tensor storage checks failed: {errors}")
        check_collective_close(rank, world_size, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
