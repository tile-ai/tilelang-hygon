import os
import random
import pytest

os.environ["PYTHONHASHSEED"] = "0"

random.seed(0)

try:
    import torch
except ImportError:
    pass
else:
    torch.manual_seed(0)

try:
    import numpy as np
except ImportError:
    pass
else:
    np.random.seed(0)


# ---------------------------------------------------------------------------
# CuTeDSL backend: auto-mark known failures / unsupported tests
# ---------------------------------------------------------------------------

# Known failures when running with TILELANG_TARGET=cutedsl.
# These are marked as xfail(strict=False) so unexpected passes are reported.
CUTEDSL_KNOWN_FAILURES = {
    # Unimplemented sparse ops: tl.tl_gemm_sp
    "sparse_tensorcore/test_example_sparse_tensorcore.py::test_tilelang_example_sparse_tensorcore",
    "gemm_sp/test_example_gemm_sp.py::test_example_gemm_sp",
    # Flaky — passes when run in isolation, fails under parallel execution
    "minference/test_vs_sparse_attn.py::test_vs_sparse_attn",
}

# HCU gfx validated for examples that rely on FP8/MMAC lowering.
_FP8_MMAC_SUPPORTED_HCU_ARCHES = frozenset({"gfx938", "gfx92a", "gfx946"})

FP8_MMAC_HCU_SKIP_PATTERNS = {
    "cast/test_example_cast.py",
    "gemm_fp8/test_example_gemm_fp8.py",
    "deepseek_deepgemm/test_example_deepgemm_fp8_2xAcc.py",
    "deepseek_v32/test_tilelang_example_deepseek_v32.py::test_example_fp8_lighting_indexer",
}


def _match_any(nodeid, patterns):
    """Return True if *nodeid* contains any of the *patterns*."""
    return any(p in nodeid for p in patterns)


def _unsupported_hcu_fp8_mmac_target():
    try:
        from tilelang import tvm
        from tilelang.utils.target import determine_target, target_is_hcu

        target = tvm.target.Target(determine_target("auto"))
        mcpu = target.attrs.get("mcpu") if target_is_hcu(target) else None
        return mcpu is not None and str(mcpu) not in _FP8_MMAC_SUPPORTED_HCU_ARCHES
    except Exception:
        return False


def _precompile_only():
    return os.getenv("TILEKERNELS_PRECOMPILE_ONLY", "").lower() in ("1", "true", "yes", "on")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if _precompile_only() and report.when == "call":
        report.outcome = "skipped"
        report.longrepr = "precompile-only: kernel compilation only"


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """When TILELANG_TARGET=cutedsl, annotate known-bad tests automatically."""
    skip_unsupported_hcu_fp8 = _unsupported_hcu_fp8_mmac_target()

    for item in items:
        nid = item.nodeid
        if os.environ.get("TILELANG_TARGET") == "cutedsl" and _match_any(nid, CUTEDSL_KNOWN_FAILURES):
            item.add_marker(
                pytest.mark.xfail(
                    reason="CuTeDSL: known limitation (unimplemented op or flaky)",
                    strict=False,
                )
            )
        if skip_unsupported_hcu_fp8 and _match_any(nid, FP8_MMAC_HCU_SKIP_PATTERNS):
            item.add_marker(pytest.mark.skip(reason="HCU FP8/MMAC example is not validated on this gfx target"))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Ensure that at least one test is collected. Error out if all tests are skipped."""
    if _precompile_only():
        return

    known_types = {
        "failed",
        "passed",
        "skipped",
        "deselected",
        "xfailed",
        "xpassed",
        "warnings",
        "error",
    }
    if sum(len(terminalreporter.stats.get(k, [])) for k in known_types.difference({"skipped", "deselected"})) == 0:
        terminalreporter.write_sep(
            "!",
            (f"Error: No tests were collected. {dict(sorted((k, len(v)) for k, v in terminalreporter.stats.items()))}"),
        )
        pytest.exit("No tests were collected.", returncode=5)
