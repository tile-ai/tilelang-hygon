"""Single-rank device smoke for the IPC metadata initialization helper."""

import gc

import torch

import tilelang
import tilelang.language as T
from tilelang.distributed.backends.ipc import IpcAllocator


def metadata_probe_kernel():
    @T.prim_func
    def main(out: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=1):
            out[0] = T.get_rank()
            out[1] = T.get_num_ranks()

    return main


def main() -> None:
    device = 0
    torch.cuda.set_device(device)
    allocator = IpcAllocator(4096, rank=0, world_size=1, device=device)
    try:
        allocator.initialize()
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
        probe = tilelang.compile(
            metadata_probe_kernel(),
            out_idx=[0],
            target={"kind": "hcu", "mcpu": arch, "dist_backend": "ipc"},
        )
        if not probe.initialize(allocator):
            raise RuntimeError("metadata helper was not launched")
        if probe.initialize(allocator):
            raise RuntimeError("metadata helper relaunched for the same allocator generation")

        out = probe()
        torch.cuda.synchronize(device)
        actual = out.cpu().tolist()
        if actual != [0, 1]:
            raise RuntimeError(f"metadata probe returned {actual}, expected [0, 1]")
        print("ipc-metadata-launch rank=0 world_size=1", flush=True)

        del out, probe
        gc.collect()
        torch.cuda.synchronize(device)
    finally:
        allocator.close()


if __name__ == "__main__":
    main()
