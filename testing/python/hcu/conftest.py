import os
import sys

import pytest

_HCU_TEST_DIR = os.path.dirname(__file__)
if _HCU_TEST_DIR not in sys.path:
    sys.path.insert(0, _HCU_TEST_DIR)


def _sanitize_hcu_visible_text(text: str) -> str:
    if not text:
        return text
    text = text.replace("amdgcn-amd-amdhsa", "<hcu-target-triple>")
    text = text.replace("__builtin_amdgcn", "<hcu-builtin>")
    return text


def _precompile_only():
    return os.getenv("TILEKERNELS_PRECOMPILE_ONLY", "").lower() in ("1", "true", "yes", "on")


def _sanitize_longrepr(longrepr):
    if isinstance(longrepr, str):
        return _sanitize_hcu_visible_text(longrepr)
    if isinstance(longrepr, tuple):
        return tuple(_sanitize_hcu_visible_text(value) if isinstance(value, str) else value for value in longrepr)
    return longrepr


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    report.longrepr = _sanitize_longrepr(report.longrepr)

    if hasattr(report, "sections"):
        report.sections = [(name, _sanitize_hcu_visible_text(content)) for name, content in report.sections]

    if _precompile_only() and report.when == "call":
        report.outcome = "skipped"
        _, lineno, _ = item.location
        report.longrepr = (
            str(item.fspath),
            lineno + 1,
            "Skipped: precompile-only: kernel compilation only",
        )
