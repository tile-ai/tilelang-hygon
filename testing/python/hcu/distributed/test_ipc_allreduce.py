import os
from pathlib import Path

import pytest

from testing.python.hcu.distributed._utils import run_hcu_script


@pytest.mark.skipif(
    os.environ.get("TILELANG_RUN_HCU_IPC_TESTS") != "1",
    reason="requires two or more P2P HCU devices",
)
def test_ipc_allpeer_allreduce():
    world_size = int(os.environ.get("TILELANG_HCU_IPC_WORLD_SIZE", "2"))
    if world_size < 2:
        pytest.skip("TILELANG_HCU_IPC_WORLD_SIZE must be at least 2")
    script = Path(__file__).with_name("ipc_allpeer_allreduce_smoke.py")
    run_hcu_script(script, world_size=world_size, port=29644)
