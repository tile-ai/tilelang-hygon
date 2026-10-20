import os
from pathlib import Path

import pytest

from testing.python.hcu.distributed._utils import run_hcu_script


@pytest.mark.skipif(os.environ.get("TILELANG_RUN_HCU_IPC_TESTS") != "1", reason="requires two P2P HCU devices")
def test_ipc_remote_get_and_gemm_allreduce():
    script = Path(__file__).with_name("ipc_remote_get_smoke.py")
    run_hcu_script(script, world_size=2, port=29642)
