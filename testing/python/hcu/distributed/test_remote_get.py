import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("TILELANG_RUN_HCU_IPC_TESTS") != "1", reason="requires two P2P HCU devices")
def test_ipc_remote_global_to_global():
    script = Path(__file__).with_name("ipc_remote_get_smoke.py")
    env = os.environ.copy()
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29642")
    result = subprocess.run(
        ["mpirun", "--allow-run-as-root", "-np", "2", "-x", "PYTHONPATH", "-x", "LD_LIBRARY_PATH",
         "-x", "HSA_USE_SVM", "-x", "HSA_FORCE_FINE_GRAIN_PCIE", "-x", "MASTER_ADDR", "-x", "MASTER_PORT",
         sys.executable, str(script)],
        check=False, capture_output=True, text=True, timeout=180, env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
