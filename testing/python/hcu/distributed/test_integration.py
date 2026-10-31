"""IPC device integration entry points; unit coverage stays in domain-specific files."""

import os
from pathlib import Path

import pytest

from testing.python.hcu.distributed._utils import run_hcu_script


def test_allreduce():
    world_size = int(os.environ.get("TILELANG_HCU_IPC_WORLD_SIZE", "2"))
    if world_size < 2:
        pytest.skip("TILELANG_HCU_IPC_WORLD_SIZE must be at least 2")
    script = Path(__file__).with_name("ipc_allpeer_allreduce_smoke.py")
    run_hcu_script(script, world_size=world_size, port=29644)


@pytest.mark.parametrize("launcher", ["mpi", "torchrun"])
def test_remote_get_and_gemm(launcher):
    script = Path(__file__).with_name("ipc_remote_get_smoke.py")
    run_hcu_script(script, world_size=2, port=29642 if launcher == "mpi" else 29646, launcher=launcher)


def test_tensor_views():
    run_hcu_script(Path(__file__).with_name("ipc_tensor_smoke.py"), world_size=2, port=29648, timeout=300)


def test_timeout(tmp_path):
    import sys
    from testing.python.hcu.distributed._utils import _run

    handled = tmp_path / "handled"
    code = (
        "import signal,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, lambda *_: "
        "(Path(sys.argv[1]).write_text('exit 0'), sys.exit(0))); "
        "time.sleep(60)"
    )
    with pytest.raises(TimeoutError):
        _run([sys.executable, "-c", code, str(handled)], os.environ.copy(), 2)
    assert handled.read_text() == "exit 0"
