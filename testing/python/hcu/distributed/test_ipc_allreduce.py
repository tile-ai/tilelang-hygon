import os
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("TILELANG_RUN_HCU_IPC_TESTS") != "1",
    reason="requires two or more P2P HCU devices",
)
def test_ipc_allpeer_allreduce():
    world_size = int(os.environ.get("TILELANG_HCU_IPC_WORLD_SIZE", "2"))
    if world_size < 2:
        pytest.skip("TILELANG_HCU_IPC_WORLD_SIZE must be at least 2")
    script = Path(__file__).with_name("ipc_allpeer_allreduce_smoke.py")
    env = os.environ.copy()
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29644")
    command = [
        "mpirun", "--allow-run-as-root", "-np", str(world_size),
        "-x", "PYTHONPATH", "-x", "LD_LIBRARY_PATH",
        "-x", "HSA_USE_SVM", "-x", "HSA_FORCE_FINE_GRAIN_PCIE",
        "-x", "MASTER_ADDR", "-x", "MASTER_PORT",
        sys.executable, str(script),
    ]
    pid = os.posix_spawnp(command[0], command, env)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
