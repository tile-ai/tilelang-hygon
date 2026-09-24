import os
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("TILELANG_RUN_HCU_IPC_TESTS") != "1",
    reason="requires an HCU device",
)
def test_ipc_metadata_helper_device_launch():
    script = Path(__file__).with_name("ipc_metadata_launch_smoke.py")
    command = [sys.executable, str(script)]
    pid = os.posix_spawn(command[0], command, os.environ.copy())
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
