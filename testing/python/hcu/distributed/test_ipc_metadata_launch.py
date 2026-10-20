import os
from pathlib import Path

import pytest

from testing.python.hcu.distributed._utils import run_hcu_script


@pytest.mark.skipif(
    os.environ.get("TILELANG_RUN_HCU_IPC_TESTS") != "1",
    reason="requires an HCU device",
)
def test_ipc_metadata_helper_device_launch():
    script = Path(__file__).with_name("ipc_metadata_launch_smoke.py")
    run_hcu_script(script, world_size=1, port=29645, launcher="direct")
