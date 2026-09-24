"""Process launch helpers for opt-in HCU distributed tests."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path
from typing import Literal


_EXPORTED_ENV = (
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "HSA_USE_SVM",
    "HSA_FORCE_FINE_GRAIN_PCIE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def run_hcu_script(
    script: Path,
    *,
    world_size: int,
    port: int,
    launcher: Literal["mpi", "torchrun", "direct"] = "mpi",
    timeout: float = 240,
) -> None:
    """Spawn a test without forking a torch/HIP-initialized pytest parent."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if launcher == "direct" and world_size != 1:
        raise ValueError("direct launcher only supports world_size=1")

    env = os.environ.copy()
    env.setdefault("HSA_USE_SVM", "0")
    env.setdefault("HSA_FORCE_FINE_GRAIN_PCIE", "1")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = str(port)

    if launcher == "mpi":
        command = [
            "mpirun",
            "--allow-run-as-root",
            "-np",
            str(world_size),
        ]
        for name in _EXPORTED_ENV:
            command.extend(["-x", name])
        command.extend([sys.executable, str(script)])
    elif launcher == "torchrun":
        command = [
            "torchrun",
            "--nnodes=1",
            "--node_rank=0",
            f"--nproc_per_node={world_size}",
            f"--master_addr={env['MASTER_ADDR']}",
            f"--master_port={port}",
            str(script),
        ]
    elif launcher == "direct":
        command = [sys.executable, str(script)]
    else:
        raise ValueError(f"unsupported launcher: {launcher}")

    pid = os.posix_spawnp(command[0], command, env, setpgroup=0)
    deadline = time.monotonic() + timeout
    while True:
        waited_pid, status = os.waitpid(pid, os.WNOHANG)
        if waited_pid == pid:
            exit_code = os.waitstatus_to_exitcode(status)
            if exit_code != 0:
                raise RuntimeError(
                    f"{launcher} HCU test exited with status {exit_code}: {' '.join(command)}"
                )
            return
        if time.monotonic() >= deadline:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)
            raise TimeoutError(f"{launcher} HCU test exceeded {timeout} seconds: {script}")
        time.sleep(0.1)
