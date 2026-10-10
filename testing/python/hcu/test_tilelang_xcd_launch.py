# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilelang as tl
import tilelang.language as T
from tilelang.contrib import hcu
from tilelang.hcu import XCDLaunchConfig
from hcu_test_utils import current_hcu_arch_string


pytestmark = pytest.mark.skipif(
    current_hcu_arch_string() != "gfx946",
    reason="HCU MultiDie launch is only validated on gfx946",
)


XCD_CONFIGS = (
    XCDLaunchConfig.linear(chunk_size=1),
    XCDLaunchConfig.linear(chunk_size=2),
    XCDLaunchConfig.block(block_x=2, block_y=2),
    XCDLaunchConfig.fixed(),
)


def make_pid_probe(grid_x: int = 8, grid_y: int = 4, threads: int = 64):
    @T.prim_func
    def pid_probe(_trigger: T.Tensor((1,), "int32"), pid: T.Tensor((grid_y, grid_x), "int32")):
        with T.Kernel(grid_x, grid_y, threads=threads) as (bx, by):
            tx = T.get_thread_binding()
            if tx == 0:
                pid[by, bx] = by * grid_x + bx

    return pid_probe


def make_pid_probe_1d(grid_x: int = 8, threads: int = 64):
    @T.prim_func
    def xcd_pid_probe_1d(_trigger: T.Tensor((1,), "int32"), pid: T.Tensor((grid_x,), "int32")):
        with T.Kernel(grid_x, threads=threads) as bx:
            tx = T.get_thread_binding()
            if tx == 0:
                pid[bx] = bx

    return xcd_pid_probe_1d


def make_multidie_stub_probe(threads: int = 64):
    @T.prim_func
    def xcd_multidie_stub_probe(_trigger: T.Tensor((1,), "int32"), value: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=threads) as bx:
            tx = T.get_thread_binding()
            if tx == 0:
                value[bx] = 7

    return xcd_multidie_stub_probe


def test_hcu_xcd_launch_preserves_raw_pid_mapping():
    kernel = tl.compile(make_pid_probe(), out_idx=[1], execution_backend="cython")
    trigger = torch.zeros(1, dtype=torch.int32, device="cuda")
    expected = torch.arange(32, dtype=torch.int32).reshape(4, 8)
    configs = (None, *XCD_CONFIGS, None)
    for xcd_config in configs:
        result = kernel(trigger, xcd_config=xcd_config)
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0, msg=str(xcd_config))


def test_hcu_xcd_fixed_rejects_degenerate_grid():
    kernel = tl.compile(make_pid_probe_1d(), out_idx=[1], execution_backend="cython")
    trigger = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(
        RuntimeError,
        match=r"Fixed XCD dispatch requires grid_x >= 2 and grid_y >= 2.*grid=\(8, 1, 1\)",
    ):
        kernel(trigger, xcd_config=XCDLaunchConfig.fixed())


def test_hcu_cython_ordinary_launch_survives_missing_multidie_api(monkeypatch):
    monkeypatch.setattr(hcu, "hcu_multidie_api_available", lambda: False)
    kernel = tl.compile(make_multidie_stub_probe(), out_idx=[1], execution_backend="cython")
    trigger = torch.zeros(1, dtype=torch.int32, device="cuda")

    result = kernel(trigger)
    torch.testing.assert_close(result.cpu(), torch.tensor([7], dtype=torch.int32), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="MultiDie launch is unavailable"):
        kernel(trigger, xcd_config=XCDLaunchConfig.linear())


def make_gemm(
    m: int = 128,
    n: int = 128,
    k: int = 32,
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 16,
    threads: int = 128,
):
    @T.prim_func
    def gemm(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((n, k), "float16"),
        C: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_k), "float16")
            B_shared = T.alloc_shared((block_n, block_k), "float16")
            C_local = T.alloc_fragment((block_m, block_n), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(k, block_k), num_stages=0):
                T.copy(A[by * block_m, ko * block_k], A_shared)
                T.copy(B[bx * block_n, ko * block_k], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_m, bx * block_n])

    return gemm


def test_hcu_xcd_gemm_matches_normal_launch():
    kernel = tl.compile(make_gemm(), out_idx=[2], execution_backend="cython")
    torch.manual_seed(0)
    a_cpu = torch.randn((128, 32), dtype=torch.float16)
    b_cpu = torch.randn((128, 32), dtype=torch.float16)
    expected = (a_cpu.float() @ b_cpu.float().T).half()
    a = a_cpu.cuda()
    b = b_cpu.cuda()

    baseline = kernel(a, b)
    torch.testing.assert_close(baseline.cpu(), expected, rtol=1e-2, atol=1e-2)
    for xcd_config in XCD_CONFIGS:
        result = kernel(a, b, xcd_config=xcd_config)
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2, msg=str(xcd_config))


def test_xcd_config_validation():
    with pytest.raises(TypeError, match="die_mask must be an int"):
        XCDLaunchConfig.linear(die_mask=True)
    with pytest.raises(ValueError, match="does not accept block_x"):
        XCDLaunchConfig(mode="linear", block_x=2)
    with pytest.raises(ValueError, match="does not accept die_mask"):
        XCDLaunchConfig(mode="fixed", die_mask=1)
