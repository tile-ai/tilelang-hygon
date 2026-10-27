"""MPI/torchrun helpers with automatic idle-HCU selection."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Literal

_EXPORTED_ENV = (
    "PYTHONPATH", "LD_LIBRARY_PATH",
    "MASTER_ADDR", "MASTER_PORT", "TILELANG_CACHE_DIR", "HIP_VISIBLE_DEVICES",
)
_DEVICE_WAIT_SECONDS = 60.0
_DEVICE_POLL_SECONDS = 5.0


def _terminate_process_group(pid):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with suppress(ProcessLookupError):
            os.killpg(pid, sig)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                return
            time.sleep(0.05)
    raise TimeoutError(f"HCU launcher {pid} was not reaped after SIGKILL")


def _run(command, env, timeout, output=None):
    """Spawn without forking a HIP-initialized parent; bound timeout cleanup."""
    actions = [] if output is None else [
        (os.POSIX_SPAWN_DUP2, output.fileno(), fd) for fd in (1, 2)
    ]
    pid = os.posix_spawnp(command[0], command, env, file_actions=actions, setpgroup=0)
    deadline = time.monotonic() + timeout
    while True:
        waited, status = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            return os.waitstatus_to_exitcode(status)
        if time.monotonic() >= deadline:
            _terminate_process_group(pid)
            # A SIGTERM handler returning zero does not make a timeout a pass.
            raise TimeoutError(f"HCU process exceeded {timeout:g}s: {' '.join(command)}")
        time.sleep(0.05)


def _check_hcu(world_size):
    try:
        import torch
    except ImportError:
        return "HCU preflight: PyTorch is not available"
    if not torch.version.hip:
        return "HCU preflight: PyTorch has no HIP runtime"
    if not torch.cuda.is_available():
        return "HCU preflight: no HCU devices available"
    count = torch.cuda.device_count()
    if count < world_size:
        return f"HCU preflight: needs {world_size} HCU devices, found {count}"
    try:
        from tilelang import tvm
    except ImportError:
        return "HCU preflight: TileLang HCU runtime is not available"
    if "hcu" not in tvm.target.Target.list_kinds():
        return "HCU preflight: HCU target is not available in this build"
    from tilelang.hcu.target import HCU_MCPU_SET, normalize_hcu_arch
    for device in range(world_size):
        arch = normalize_hcu_arch(getattr(torch.cuda.get_device_properties(device), "gcnArchName", None))
        if arch not in HCU_MCPU_SET:
            return f"HCU preflight: device {device} is not a supported HCU device (arch={arch})"
    can_access = tvm.ffi.get_global_func("tl.hcu.ipc.can_access_peer", allow_missing=True)
    if can_access is None:
        return "HCU preflight: tl.hcu.ipc.can_access_peer is not available"
    for device in range(world_size):
        for peer in range(world_size):
            if device == peer:
                continue
            try:
                supported = can_access(device, peer)
            except RuntimeError as error:
                return f"HCU preflight: P2P probe {device}->{peer} failed: {error}"
            if not supported:
                return f"HCU preflight: P2P unavailable from HCU device {device} to {peer}"
    return None


def _select_devices(world_size, env):
    import pytest

    deadline = time.monotonic() + _DEVICE_WAIT_SECONDS
    while True:
        try:
            result = subprocess.run(["hy-smi"], capture_output=True, text=True, check=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as error:
            pytest.skip(f"HCU preflight: cannot query idle devices with hy-smi: {error}")
        lines = result.stdout.splitlines()
        try:
            header = next(i for i, line in enumerate(lines)
                          if line.strip().startswith("HCU") and "VRAM%" in line.split())
            column = lines[header].split().index("VRAM%")
            usage = {int(row[0]): float(row[column].rstrip("%"))
                     for row in (line.split() for line in lines[header + 1:])
                     if row and row[0].isdigit()}
        except (StopIteration, ValueError, IndexError):
            pytest.skip("HCU preflight: cannot parse the hy-smi HCU/VRAM% table")
        mask = env.get("HIP_VISIBLE_DEVICES")
        try:
            visible = sorted(usage) if mask is None else (
                [] if mask.strip() in ("", "-1") else [int(i) for i in mask.split(",")]
            )
            if len(set(visible)) != len(visible) or any(i not in usage for i in visible):
                raise ValueError
        except ValueError:
            pytest.skip(f"HCU preflight: invalid HIP_VISIBLE_DEVICES={mask!r}")
        if len(visible) < world_size:
            pytest.skip(f"HCU preflight: needs {world_size} visible HCU devices, found {len(visible)}")
        idle = [device for device in visible if usage[device] == 0]
        if len(idle) >= world_size:
            selected = idle[:world_size]
            env["HIP_VISIBLE_DEVICES"] = ",".join(map(str, selected))
            print(f"HCU preflight: using idle HCU devices {selected} for {world_size} rank(s)", flush=True)
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pytest.skip(f"HCU preflight: needs {world_size} idle HCU devices, found {len(idle)} after waiting {_DEVICE_WAIT_SECONDS:g}s")
        print(f"HCU preflight: needs {world_size} idle HCU devices, found {len(idle)}; waiting ({remaining:.0f}s left)", flush=True)
        time.sleep(min(_DEVICE_POLL_SECONDS, remaining))


def run_hcu_script(
    script: Path, *, world_size: int, port: int,
    launcher: Literal["mpi", "torchrun"] = "mpi", timeout: float = 240,
) -> None:
    """Run an IPC test with a fresh cache for every rank."""
    import pytest

    if world_size <= 0 or launcher not in ("mpi", "torchrun"):
        raise ValueError("world_size must be positive and launcher must be mpi or torchrun")
    env = os.environ.copy()
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = str(port)
    _select_devices(world_size, env)
    helper = str(Path(__file__).resolve())
    with tempfile.TemporaryFile(mode="w+") as output:
        status = _run([sys.executable, helper, "preflight", str(world_size)], env, 60, output)
        if status:
            output.seek(0)
            message = output.read()
            if status == 77:
                reason = next((line for line in reversed(message.splitlines())
                               if line.startswith("HCU preflight:")), message.strip())
                pytest.skip(reason)
            raise RuntimeError(f"HCU preflight failed (exit {status}): {message}")
    env["TILELANG_CACHE_DIR"] = tempfile.mkdtemp(prefix=f"tilelang-hcu-{script.stem}-w{world_size}-")
    entry = [sys.executable, helper, "run", launcher, str(script.resolve())]
    if launcher == "mpi":
        command = ["mpirun", "--allow-run-as-root", "-np", str(world_size)]
        for name in _EXPORTED_ENV:
            if name in env:
                command.extend(["-x", name])
        command.extend(entry)
    else:
        command = ["torchrun", "--nnodes=1", f"--nproc_per_node={world_size}",
                   f"--master_addr={env['MASTER_ADDR']}", f"--master_port={port}", *entry[1:]]
    status = _run(command, env, timeout)
    if status:
        raise RuntimeError(f"{launcher} HCU test exited with status {status}: {script}")


if __name__ == "__main__":
    if sys.argv[1] == "preflight":
        reason = _check_hcu(int(sys.argv[2]))
        if reason is not None:
            print(reason, flush=True)
            sys.exit(77)
    elif sys.argv[1] == "run":
        import runpy

        launcher, script = sys.argv[2], Path(sys.argv[3]).resolve()
        if launcher == "mpi":
            for name, mpi_name in (("RANK", "RANK"), ("LOCAL_RANK", "LOCAL_RANK"),
                                   ("WORLD_SIZE", "SIZE")):
                os.environ[name] = os.environ[f"OMPI_COMM_WORLD_{mpi_name}"]
        cache = Path(os.environ["TILELANG_CACHE_DIR"]) / f"rank-{os.environ['RANK']}"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["TILELANG_CACHE_DIR"] = str(cache)
        sys.argv = [str(script), *sys.argv[4:]]
        sys.path[0] = str(script.parent)
        runpy.run_path(str(script), run_name="__main__")
    else:
        raise ValueError(f"Unknown HCU test helper mode: {sys.argv[1]}")
