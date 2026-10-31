"""Shared validation and complete-call timing for the two GEMM push examples."""

import argparse
import gc
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

import tilelang as tl
import tilelang.language as T
from perf.gemm.kernel_registry import dispatch_kernel, get_kernel_config
from tilelang.distributed.backends.ipc import IpcAllocator, ipc_metadata_layout


def _reduce_local(elements, world_size):
    @T.prim_func
    def main(inbox: T.Tensor((world_size, elements), "float16"), output: T.Tensor((elements,), "float16")):
        with T.Kernel(elements // 256, threads=256) as block:
            values = T.alloc_fragment((256,), "float32")
            T.clear(values)
            for peer in T.serial(world_size):
                for i in T.Parallel(256):
                    values[i] += inbox[peer, block * 256 + i]
            T.copy(values, output[block * 256 : (block + 1) * 256])

    return main


def _rank_env(name):
    mpi_name = {"RANK": "OMPI_COMM_WORLD_RANK", "WORLD_SIZE": "OMPI_COMM_WORLD_SIZE", "LOCAL_RANK": "OMPI_COMM_WORLD_LOCAL_RANK"}[name]
    return int(os.environ.get(name) or os.environ[mpi_name])


def run_benchmark(kernel_factory, variant):
    parser = argparse.ArgumentParser(description=f"{variant} GEMM + reduce-scatter")
    parser.add_argument("--size", type=int, default=4096, help="M and N")
    parser.add_argument("--k", type=int, help="K (defaults to --size)")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--dump-source", type=Path)
    args = parser.parse_args()
    rank, world, device = (_rank_env(key) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    n = args.size
    k = args.k if args.k is not None else n
    if world not in (2, 4, 8) or n <= 0 or n % (256 * world) or k < 64 or k % 64 or args.repeats < 1:
        raise ValueError("use two, four or eight ranks, positive size divisible by 256*world_size, K divisible by 64, and positive repeats")
    os.environ["TILELANG_CACHE_DIR"] = str(Path(os.environ.get("TILELANG_CACHE_DIR", f"/tmp/tl-gemm-rs-{variant}")) / f"rank-{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=180))
    rccl = dist.new_group(backend="nccl", timeout=timedelta(seconds=180))
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
    target = {"kind": "hcu", "mcpu": arch}
    elements = n * n // world
    allocator = IpcAllocator(n * n * 2 + 256, rank=rank, world_size=world, device=device, group=dist.group.WORLD)
    allocator.initialize()
    inbox = tl.tensor((world, elements), torch.float16, allocator=allocator)
    inbox.fill_(float("nan"))
    offset = inbox.data_ptr() - allocator.base_ptr
    bases = allocator.metadata[ipc_metadata_layout()["peer_base"] :]
    peers = torch.tensor([base + offset for base in bases], dtype=torch.uint64, device="cuda")
    config = get_kernel_config(dispatch_kernel("async_copy", transpose_B=False), n, n, k)
    passes = {tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True}
    kernels = {
        variant: tl.compile(
            kernel_factory(n, n, k, rank=rank, world_size=world, steady_wait=8, shared_0_wait=6, **config),
            out_idx=[],
            target=target,
            pass_configs=passes,
        )
    }
    reducer = tl.compile(_reduce_local(elements, world), out_idx=[], target=target)
    if args.dump_source:
        args.dump_source.mkdir(parents=True, exist_ok=True)
        for name, kernel in kernels.items():
            (args.dump_source / f"{name}.rank-{rank}.hip").write_text(kernel.get_kernel_source())
    a = torch.empty((n, k), dtype=torch.float16, device="cuda")
    b = torch.empty((k, n), dtype=torch.float16, device="cuda")
    partial = torch.empty((n, n), dtype=torch.float16, device="cuda")
    output = torch.empty((elements,), dtype=torch.float16, device="cuda")
    torch_output = torch.empty((n // world, n), dtype=torch.float16, device="cuda")
    token = torch.zeros(1, dtype=torch.float32, device="cuda")

    def enqueue(kernel):
        # Same two stream-ordered RCCL rendezvous as the original example:
        # previous readers finish before reuse, then peer stores finish before reduce.
        dist.all_reduce(token, group=rccl)
        kernel(a, b, peers)
        dist.all_reduce(token, group=rccl)
        reducer(inbox, output)

    def torch_path():
        torch.mm(a, b, out=partial)
        dist.reduce_scatter_tensor(torch_output, partial, group=rccl)

    def random_inputs():
        a.normal_().mul_(k**-0.5)
        b.normal_()

    def reference():
        full = (a.float() @ b.float()).cpu()
        dist.all_reduce(full)
        return full[rank * (n // world) : (rank + 1) * (n // world)].to(device="cuda", dtype=torch.float16)

    torch.manual_seed(1234 + rank)
    torch.cuda.current_stream().synchronize()
    dist.barrier()
    for trial in range(3):
        random_inputs()
        expected = reference()
        torch_path()
        torch.testing.assert_close(torch_output, expected, atol=1e-2, rtol=1e-2)
        for name, kernel in kernels.items():
            if trial == 1 and rank == world - 1:
                time.sleep(0.05)
            enqueue(kernel)
            actual = output.reshape_as(expected)
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(actual, torch_output, atol=1e-2, rtol=1e-2)
            print(
                f"random-check rank={rank}/{world} trial={trial} path={name} arch={arch} "
                f"size={n} k={k} max_abs={(actual - expected).abs().max().item():.6g} "
                f"sample={actual[0, 0].item():.6g} torch_sample={torch_output[0, 0].item():.6g} passed",
                flush=True,
            )
    for name, kernel in kernels.items():
        for _ in range(20):
            random_inputs()
            enqueue(kernel)
        expected = reference()
        torch.testing.assert_close(output.reshape_as(expected), expected, atol=1e-2, rtol=1e-2)
        print(f"queued-random-check rank={rank}/{world} path={name} calls=20 passed", flush=True)

    def capture(fn):
        for _ in range(3):
            fn()
        torch.cuda.current_stream().synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        return graph

    def wall_ms(fn):
        values = []
        for iteration in range(3 + args.repeats):
            dist.barrier()
            start = time.perf_counter()
            fn()
            torch.cuda.current_stream().synchronize()
            if iteration >= 3:
                values.append((time.perf_counter() - start) * 1000)
        return statistics.median(values)

    # Fixed replay counts on all ranks, including all captured collectives.
    paths = {name: lambda k=kernel: enqueue(k) for name, kernel in kernels.items()}
    paths["torch"] = torch_path
    results = {}
    graphs = []
    for name, fn in paths.items():
        graph = capture(fn)
        graphs.append(graph)
        graph.replay()
        torch.cuda.current_stream().synchronize()
        actual = torch_output if name == "torch" else output.reshape_as(expected)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        print(f"graph-check rank={rank}/{world} path={name} passed", flush=True)
        if args.benchmark:
            results[name] = {"call_ms": wall_ms(fn), "graph_call_ms": wall_ms(graph.replay)}
    if args.benchmark:
        # Production-only timing has no reduction/rendezvous; ranks own disjoint
        # sender slots. It measures GEMM+epilogue, not a complete RS invocation.
        for name, kernel in kernels.items():
            graph = capture(lambda k=kernel: k(a, b, peers))
            events = []
            dist.barrier()
            for _ in range(3 + args.repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                events.append((start, end))
            torch.cuda.current_stream().synchronize()
            dist.barrier()
            results[name]["production_ms"] = statistics.median(s.elapsed_time(e) for s, e in events[3:])
            graph.reset()
        all_results = [None] * world
        dist.all_gather_object(all_results, results)
        if rank == 0:
            print(f"timings world_size={world} size={n} k={k} by_rank={all_results}", flush=True)
            worst = {name: {key: max(row[name][key] for row in all_results) for key in timings} for name, timings in results.items()}
            print(f"summary world_size={world} size={n} k={k} max_rank_ms={worst}", flush=True)
        for name, kernel in kernels.items():
            enqueue(kernel)
            torch.testing.assert_close(output.reshape_as(expected), expected, atol=1e-2, rtol=1e-2)
            print(f"post-benchmark-check rank={rank}/{world} path={name} passed", flush=True)
    torch.cuda.current_stream().synchronize()
    dist.barrier()
    for graph in graphs:
        graph.reset()
    del graph, graphs
    inbox = None
    peers = None
    gc.collect()
    allocator.close()
    dist.destroy_process_group(rccl)
    dist.destroy_process_group()
