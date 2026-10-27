from pathlib import Path

import pytest

from testing.python.hcu.distributed._utils import run_hcu_script


@pytest.mark.parametrize("world_size", [2, 4])
def test_example_gemm_ar_ipc(world_size):
    script = Path(__file__).with_name("example_gemm_ar_ipc.py")
    run_hcu_script(script, world_size=world_size, port=29643)
