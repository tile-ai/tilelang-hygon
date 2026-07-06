import os
import sys

import pytest

_HCU_TEST_DIR = os.path.dirname(__file__)
if _HCU_TEST_DIR not in sys.path:
    sys.path.insert(0, _HCU_TEST_DIR)


def _precompile_only():
    return os.getenv("TILEKERNELS_PRECOMPILE_ONLY", "").lower() in ("1", "true", "yes", "on")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if _precompile_only() and report.when == "call":
        report.outcome = "skipped"
        report.longrepr = "precompile-only: kernel compilation only"
