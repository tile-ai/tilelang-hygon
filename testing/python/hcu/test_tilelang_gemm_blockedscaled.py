# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""HCU gemm_blockscaled (FP4xFP4 + E8M0) tests.

Numerical check: decode E2M1 / E8M0 on CPU and compare against kernel output
(perf-model friendly; golden stays on CPU).
"""

import os
from pathlib import Path

# Keep generated .cu under the mounted tilelang tree (host: ~/tilelang/...).
# Must be set before tilelang.env reads cache paths at compile time.
_CACHE_DIR = Path(__file__).resolve().parents[3] / ".tilelang_cache" / "blockscaled"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TILELANG_CACHE_DIR"] = str(_CACHE_DIR)
os.environ.setdefault("TILELANG_DISABLE_CACHE", "0")

import pytest
import tilelang as tl
import tilelang.language as T
import tilelang.testing
import torch
import tvm
from hcu_test_utils import target_supports_blockscaled

tilelang.testing.set_random_seed(0)

requires_blockscaled = pytest.mark.skipif(
    not target_supports_blockscaled(),
    reason="gemm_blockscaled requires an HCU gfx946 target (auto)",
)


def _fp4_e2m1fn_decode(logical: torch.Tensor) -> torch.Tensor:
    """Decode E2M1 nibbles to float32 (CPU/GPU)."""
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=logical.device,
        dtype=torch.float32,
    )
    return table[(logical & 0x0F).long()]


def _pack_fp4_last_dim(logical: torch.Tensor) -> torch.Tensor:
    logical = logical.to(torch.uint8)
    return (logical[..., 0::2] | (logical[..., 1::2] << 4)).contiguous().view(torch.int8)


def _e8m0_to_float(scale_u8: torch.Tensor) -> torch.Tensor:
    """UE8M0 exponent byte -> float scale ``2^(e - 127)``."""
    return torch.pow(2.0, scale_u8.to(torch.float32) - 127.0)


def _expand_scale_mn(scale_1d: torch.Tensor, length: int, gran_mn: int) -> torch.Tensor:
    """Map scale along MN (length // gran_mn points) to per-element scales."""
    assert length % gran_mn == 0
    assert scale_1d.numel() == length // gran_mn
    return scale_1d.to(torch.float32).repeat_interleave(gran_mn)


def blockscaled_fp4_cpu_ref(
    A_logical: torch.Tensor,
    B_logical: torch.Tensor,
    ScaleA: torch.Tensor,
    ScaleB: torch.Tensor,
    *,
    sf_a_granularity_m: int,
    sf_a_granularity_k: int,
    sf_b_granularity_n: int,
    sf_b_granularity_k: int,
    a_scale_k_major: bool = False,
    b_scale_k_major: bool = False,
    transpose_B: bool = True,
) -> torch.Tensor:
    """CPU soft-sim: decode FP4 + E8M0, then block-scaled GEMM (A @ B^T)."""
    assert transpose_B
    A_f = _fp4_e2m1fn_decode(A_logical.cpu())
    B_f = _fp4_e2m1fn_decode(B_logical.cpu())
    M, K = A_f.shape
    N, K2 = B_f.shape
    assert K == K2

    sfa = _e8m0_to_float(ScaleA.cpu())
    sfb = _e8m0_to_float(ScaleB.cpu())
    # Normalize to [sf_k, sf_mn]
    if a_scale_k_major:
        sfa = sfa.T.contiguous()
    if b_scale_k_major:
        sfb = sfb.T.contiguous()

    sf_ka = (K + sf_a_granularity_k - 1) // sf_a_granularity_k
    sf_kb = (K + sf_b_granularity_k - 1) // sf_b_granularity_k
    assert sfa.shape == (sf_ka, M // sf_a_granularity_m)
    assert sfb.shape == (sf_kb, N // sf_b_granularity_n)
    assert sf_a_granularity_k == sf_b_granularity_k

    c = torch.zeros(M, N, dtype=torch.float32)
    for ki in range(sf_ka):
        k0 = ki * sf_a_granularity_k
        k1 = min(k0 + sf_a_granularity_k, K)
        sa = _expand_scale_mn(sfa[ki], M, sf_a_granularity_m)
        sb = _expand_scale_mn(sfb[ki], N, sf_b_granularity_n)
        a_block = A_f[:, k0:k1] * sa[:, None]
        b_block = B_f[:, k0:k1] * sb[:, None]
        c += a_block @ b_block.T
    return c


def _make_fp4_inputs(M, N, K, device="cuda"):
    a_rows = torch.arange(M, device=device, dtype=torch.uint8)[:, None]
    a_cols = torch.arange(K, device=device, dtype=torch.uint8)[None, :]
    b_rows = torch.arange(N, device=device, dtype=torch.uint8)[:, None]
    b_cols = torch.arange(K, device=device, dtype=torch.uint8)[None, :]
    # Positive E2M1 nibbles only (0..7) for a well-conditioned CPU golden.
    A_logical = (a_rows * 3 + a_cols * 5 + a_rows // 7 + a_cols // 11) & 0x07
    B_logical = (b_rows * 7 + b_cols * 2 + b_rows // 5 + b_cols // 13) & 0x07
    return A_logical, B_logical, _pack_fp4_last_dim(A_logical), _pack_fp4_last_dim(B_logical)


def _make_e8m0_scales(sf_k, sf_m, sf_n, device="cuda", *, non_unit_scale=True):
    if non_unit_scale:
        mn_a = torch.arange(sf_m, device=device, dtype=torch.int32)
        mn_b = torch.arange(sf_n, device=device, dtype=torch.int32)
        ScaleA = (125 + (mn_a * 3 + 1) % 5).to(torch.uint8).view(1, sf_m).expand(sf_k, sf_m).contiguous()
        ScaleB = (125 + (mn_b * 2 + 3) % 5).to(torch.uint8).view(1, sf_n).expand(sf_k, sf_n).contiguous()
    else:
        ScaleA = torch.full((sf_k, sf_m), 127, device=device, dtype=torch.uint8)
        ScaleB = torch.full((sf_k, sf_n), 127, device=device, dtype=torch.uint8)
    return ScaleA, ScaleB


def _check_blockscaled_out(lib_out, ref):
    assert torch.isfinite(lib_out).all(), "non-finite outputs in kernel result"
    # Not bit-exact vs CPU soft-sim: mmac vs torch matmul differ in K-reduction
    # order / intermediate rounding even when E2M1 and E8M0 are exactly
    # representable in f32.
    torch.testing.assert_close(lib_out, ref, rtol=1e-2, atol=1e-2)


def _interleave_scale_k_for_copy(scale, rows_per_instr):
    """Pack logical [K, MN] as [K-group, MN, K-inner] for op_ctrl>0."""
    scale_k, scale_mn = scale.shape
    assert scale_k % rows_per_instr == 0
    return scale.reshape(scale_k // rows_per_instr, rows_per_instr, scale_mn).permute(0, 2, 1).contiguous()


def _interleave_scale_k2_mn2_for_copy(scale):
    """Pack [K,MN] as [K2-group,MN32-block,lane16,MN-half,K2]."""
    scale_k, scale_mn = scale.shape
    assert scale_k % 2 == 0 and scale_mn % 32 == 0
    return scale.reshape(scale_k // 2, 2, scale_mn // 32, 2, 16).permute(0, 2, 4, 3, 1).contiguous()


def _interleave_scale_mn_for_copy(scale, mn_group):
    """Pack MN16 rows into the MN2/MN4 atom consumed by scale MMAC."""
    scale_k, scale_mn = scale.shape
    assert scale_mn % (16 * mn_group) == 0
    return (
        scale.reshape(scale_k, scale_mn // 16 // mn_group, mn_group, 16)
        .permute(0, 1, 3, 2)
        .reshape(scale_k, scale_mn // mn_group, mn_group)
        .contiguous()
    )


def matmul_blockscaled(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    *,
    mn_pad=0,
    k_pad=0,
    op_ctrl=0,
    sf_a_granularity_m=1,
    sf_a_granularity_k=64,
    sf_b_granularity_n=1,
    sf_b_granularity_k=64,
    threads=128,
    num_stages=0,
    use_mls=False,
    k_pack=1,
    scale_format_name=None,
    scale_stages=1,
    annotate_scale_stage_layout=False,
    explicit_unpacked_ds_read=False,
    a_dtype="float4_e2m1fn",
    b_dtype=None,
):
    """Block-scaled GEMM with optional ScaleView slicing and MLS operands.

    ScaleA/B LDS include optional leading MN/K space, while the real scale tile
    is copied directly at the corresponding storage-format coordinates.
    ``scale_buffer`` stays the 2D tile shape. ``mn_pad`` and ``k_pad`` select
    the explicit logical origin inside the parent plane. ``use_mls`` loads
    both FP4 operands through ``T.matrix_load``.
    """
    if b_dtype is None:
        b_dtype = a_dtype
    unpack_a = explicit_unpacked_ds_read is True or explicit_unpacked_ds_read == "a"
    unpack_b = explicit_unpacked_ds_read is True or explicit_unpacked_ds_read == "b"
    out_dtype = "float32"
    accum_dtype = "float32"
    scale_dtype = "uint8"

    scale_m = block_M // sf_a_granularity_m
    scale_n = block_N // sf_b_granularity_n
    scale_ka = block_K // sf_a_granularity_k
    scale_kb = block_K // sf_b_granularity_k
    assert scale_ka == scale_kb
    if scale_format_name is None:
        assert op_ctrl == 0, "interleaved copy_scale cases must select scale_format_name explicitly"
        scale_format_name = "identity"
    selected_scale_format_name = scale_format_name
    scale_format = {
        "identity": T.scale_identity,
        "k2": T.scale_k2_interleaved,
        "k4": T.scale_k4_interleaved,
        "k2mn2": T.scale_k2mn2_interleaved,
        "mn2": T.scale_mn2_interleaved,
        "mn4": T.scale_mn4_interleaved,
    }[selected_scale_format_name]()

    ScaleA_tile = (scale_ka, scale_m)
    ScaleB_tile = (scale_kb, scale_n)
    ScaleA_logical_parent = (scale_ka + k_pad, scale_m + mn_pad)
    ScaleB_logical_parent = (scale_kb + k_pad, scale_n + mn_pad)

    def physical_scale_shape(logical_shape):
        sk, smn = logical_shape
        if selected_scale_format_name == "identity":
            return (sk, smn)
        if selected_scale_format_name == "k2":
            assert sk % 2 == 0
            return (sk // 2, smn, 2)
        if selected_scale_format_name == "k4":
            assert sk % 4 == 0
            return (sk // 4, smn, 4)
        if selected_scale_format_name == "mn2":
            assert smn % 2 == 0
            return (sk, smn // 2, 2)
        if selected_scale_format_name == "mn4":
            assert smn % 4 == 0
            return (sk, smn // 4, 4)
        assert selected_scale_format_name == "k2mn2"
        assert sk % 2 == 0 and smn % 32 == 0
        return (sk // 2, smn // 32, 16, 2, 2)

    ScaleA_plane_shape = physical_scale_shape(ScaleA_logical_parent)
    ScaleB_plane_shape = physical_scale_shape(ScaleB_logical_parent)
    ScaleA_lds_shape = ScaleA_plane_shape if scale_stages == 1 else (scale_stages, *ScaleA_plane_shape)
    ScaleB_lds_shape = ScaleB_plane_shape if scale_stages == 1 else (scale_stages, *ScaleB_plane_shape)

    A_shape = (M, K)
    B_shape = (N, K)
    ScaleA_logical_shape = (K // sf_a_granularity_k, M // sf_a_granularity_m)
    ScaleB_logical_shape = (K // sf_b_granularity_k, N // sf_b_granularity_n)
    ScaleA_shape = physical_scale_shape(ScaleA_logical_shape)
    ScaleB_shape = physical_scale_shape(ScaleB_logical_shape)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, a_dtype),
        B: T.Tensor(B_shape, b_dtype),
        ScaleA: T.Tensor(ScaleA_shape, scale_dtype),
        ScaleB: T.Tensor(ScaleB_shape, scale_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), a_dtype)
            B_shared = T.alloc_shared((block_N, block_K), b_dtype)
            A_fragment = T.alloc_fragment(
                (block_M, block_K),
                T.float4_e2m1_unpacked if unpack_a else a_dtype,
            )
            B_fragment = T.alloc_fragment(
                (block_N, block_K),
                T.float4_e2m1_unpacked if unpack_b else b_dtype,
            )
            ScaleA_lds = T.alloc_shared(ScaleA_lds_shape, scale_dtype)
            ScaleB_lds = T.alloc_shared(ScaleB_lds_shape, scale_dtype)
            if annotate_scale_stage_layout:
                assert scale_stages == 2 and selected_scale_format_name == "identity"
                if annotate_scale_stage_layout == "mixed":
                    T.annotate_layout(
                        {
                            ScaleA_lds: T.Layout(ScaleA_lds_shape, lambda s, scale_k, scale_mn: [s, scale_k, scale_mn ^ s]),
                            ScaleB_lds: T.Layout(ScaleB_lds_shape, lambda s, scale_k, scale_mn: [s, scale_k, scale_mn ^ s]),
                        }
                    )
                else:
                    T.annotate_layout(
                        {
                            ScaleA_lds: T.Layout(ScaleA_lds_shape, lambda s, scale_k, scale_mn: [s, scale_k, scale_mn]),
                            ScaleB_lds: T.Layout(ScaleB_lds_shape, lambda s, scale_k, scale_mn: [s, scale_k, scale_mn]),
                        }
                    )
            ScaleA_sbuf = T.alloc_scale_buffer(ScaleA_tile, scale_dtype)
            ScaleB_sbuf = T.alloc_scale_buffer(ScaleB_tile, scale_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                scale_stage = (k + 1) % scale_stages
                if use_mls:
                    T.matrix_load(A[by * block_M, k * block_K], A_shared)
                    T.matrix_load(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                if selected_scale_format_name == "identity":
                    if scale_stages == 1:
                        T.copy(ScaleA, ScaleA_lds[k_pad:, mn_pad:])
                        T.copy(ScaleB, ScaleB_lds[k_pad:, mn_pad:])
                    else:
                        T.copy(ScaleA, ScaleA_lds[scale_stage, k_pad:, mn_pad:])
                        T.copy(ScaleB, ScaleB_lds[scale_stage, k_pad:, mn_pad:])
                elif selected_scale_format_name == "k2":
                    if scale_stages == 1:
                        T.copy(ScaleA, ScaleA_lds[k_pad // 2 :, mn_pad:, :])
                        T.copy(ScaleB, ScaleB_lds[k_pad // 2 :, mn_pad:, :])
                    else:
                        T.copy(ScaleA, ScaleA_lds[scale_stage, k_pad // 2 :, mn_pad:, :])
                        T.copy(ScaleB, ScaleB_lds[scale_stage, k_pad // 2 :, mn_pad:, :])
                elif selected_scale_format_name == "k4":
                    if scale_stages == 1:
                        T.copy(ScaleA, ScaleA_lds[k_pad // 4 :, mn_pad:, :])
                        T.copy(ScaleB, ScaleB_lds[k_pad // 4 :, mn_pad:, :])
                    else:
                        T.copy(ScaleA, ScaleA_lds[scale_stage, k_pad // 4 :, mn_pad:, :])
                        T.copy(ScaleB, ScaleB_lds[scale_stage, k_pad // 4 :, mn_pad:, :])
                else:
                    assert k_pad == 0 and mn_pad == 0
                    if scale_stages == 1:
                        T.copy(ScaleA, ScaleA_lds)
                        T.copy(ScaleB, ScaleB_lds)
                    elif selected_scale_format_name == "k2mn2":
                        T.copy(ScaleA, ScaleA_lds[scale_stage, :, :, :, :, :])
                        T.copy(ScaleB, ScaleB_lds[scale_stage, :, :, :, :, :])
                    else:
                        T.copy(ScaleA, ScaleA_lds[scale_stage, :, :, :])
                        T.copy(ScaleB, ScaleB_lds[scale_stage, :, :, :])
                if scale_stages == 1:
                    ScaleA_view = T.scale_view(
                        ScaleA_lds,
                        logical_shape=ScaleA_logical_parent,
                        format=scale_format,
                    )
                    ScaleB_view = T.scale_view(
                        ScaleB_lds,
                        logical_shape=ScaleB_logical_parent,
                        format=scale_format,
                    )
                elif selected_scale_format_name == "identity":
                    ScaleA_view = T.scale_view(
                        ScaleA_lds[scale_stage, :, :],
                        logical_shape=ScaleA_logical_parent,
                        format=scale_format,
                    )
                    ScaleB_view = T.scale_view(
                        ScaleB_lds[scale_stage, :, :],
                        logical_shape=ScaleB_logical_parent,
                        format=scale_format,
                    )
                elif selected_scale_format_name == "k2mn2":
                    ScaleA_view = T.scale_view(
                        ScaleA_lds[scale_stage, :, :, :, :, :],
                        logical_shape=ScaleA_logical_parent,
                        format=scale_format,
                    )
                    ScaleB_view = T.scale_view(
                        ScaleB_lds[scale_stage, :, :, :, :, :],
                        logical_shape=ScaleB_logical_parent,
                        format=scale_format,
                    )
                else:
                    ScaleA_view = T.scale_view(
                        ScaleA_lds[scale_stage, :, :, :],
                        logical_shape=ScaleA_logical_parent,
                        format=scale_format,
                    )
                    ScaleB_view = T.scale_view(
                        ScaleB_lds[scale_stage, :, :, :],
                        logical_shape=ScaleB_logical_parent,
                        format=scale_format,
                    )
                T.copy_scale(ScaleA_view[k_pad:, mn_pad:], ScaleA_sbuf, op_ctrl=op_ctrl)
                T.copy_scale(ScaleB_view[k_pad:, mn_pad:], ScaleB_sbuf, op_ctrl=op_ctrl)
                if use_mls:
                    T.ptx_wait_group(0)
                    T.sync_warp()
                if unpack_a:
                    T.ds_read_format(A_shared, A_fragment)
                if unpack_b:
                    T.ds_read_format(B_shared, B_fragment)
                T.gemm_blockscaled(
                    A_fragment if unpack_a else A_shared,
                    B_fragment if unpack_b else B_shared,
                    C_local,
                    ScaleA_sbuf,
                    ScaleB_sbuf,
                    transpose_A=False,
                    transpose_B=True,
                    k_pack=k_pack,
                    sf_a_granularity_m=sf_a_granularity_m,
                    sf_a_granularity_k=sf_a_granularity_k,
                    sf_b_granularity_n=sf_b_granularity_n,
                    sf_b_granularity_k=sf_b_granularity_k,
                    a_scale_k_major=False,
                    b_scale_k_major=False,
                )
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_blockscaled_basic_correctness(
    *,
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    gran_k,
    threads,
    use_mls,
    k_pack=1,
    non_unit_scale: bool = True,
):
    """MN-major scale MMAC correctness, optionally with both operands via MLS.

    Default uses non-unit E8M0 (exponents around 127 → scales in {0.25,0.5,1,2,4}).
    """
    gran_m = gran_n = 1
    program = matmul_blockscaled(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        sf_a_granularity_m=gran_m,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=gran_n,
        sf_b_granularity_k=gran_k,
        threads=threads,
        num_stages=0,
        use_mls=use_mls,
        k_pack=k_pack,
    )
    kernel = tl.compile(program, out_idx=[4], verbose=True)

    cache_path = getattr(kernel, "_tilelang_cache_path", None)
    print(f"[blockscaled] TILELANG_CACHE_DIR={_CACHE_DIR}")
    if cache_path:
        print(f"[blockscaled] kernel cache entry: {cache_path}")
        for name in ("device_kernel.cu", "wrapped_kernel.cu", "host_kernel.cu"):
            p = os.path.join(cache_path, name)
            if os.path.isfile(p):
                print(f"[blockscaled] wrote {p}")

    A_logical, B_logical, A, B = _make_fp4_inputs(M, N, K)
    ScaleA, ScaleB = _make_e8m0_scales(K // gran_k, M // gran_m, N // gran_n, non_unit_scale=non_unit_scale)

    profiler = kernel.get_profiler()
    torch.cuda.synchronize()
    lib_out = profiler.func(A, B, ScaleA, ScaleB)
    torch.cuda.synchronize()
    lib_out = lib_out.cpu()

    ref = blockscaled_fp4_cpu_ref(
        A_logical,
        B_logical,
        ScaleA,
        ScaleB,
        sf_a_granularity_m=gran_m,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=gran_n,
        sf_b_granularity_k=gran_k,
        a_scale_k_major=False,
        b_scale_k_major=False,
        transpose_B=True,
    )
    _check_blockscaled_out(lib_out, ref)
    return kernel, lib_out, ref


def run_blockscaled_mn_slice_correctness(*, M, N, K, block_M, block_N, block_K, gran_k, threads, mn_pad, non_unit_scale=True):
    """Same gemm as basic, but copy_scale reads an MN-sliced LDS window."""
    gran_m = gran_n = 1
    program = matmul_blockscaled(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        mn_pad=mn_pad,
        sf_a_granularity_m=gran_m,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=gran_n,
        sf_b_granularity_k=gran_k,
        threads=threads,
        num_stages=0,
    )
    kernel = tl.compile(program, out_idx=[4], verbose=True)
    cache_path = getattr(kernel, "_tilelang_cache_path", None)
    print(f"[blockscaled-mn-slice] kernel cache entry: {cache_path}")

    A_logical, B_logical, A, B = _make_fp4_inputs(M, N, K)
    ScaleA, ScaleB = _make_e8m0_scales(K // gran_k, M // gran_m, N // gran_n, non_unit_scale=non_unit_scale)

    profiler = kernel.get_profiler()
    torch.cuda.synchronize()
    lib_out = profiler.func(A, B, ScaleA, ScaleB)
    torch.cuda.synchronize()
    lib_out = lib_out.cpu()

    ref = blockscaled_fp4_cpu_ref(
        A_logical,
        B_logical,
        ScaleA,
        ScaleB,
        sf_a_granularity_m=gran_m,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=gran_n,
        sf_b_granularity_k=gran_k,
        a_scale_k_major=False,
        b_scale_k_major=False,
        transpose_B=True,
    )
    _check_blockscaled_out(lib_out, ref)
    print(f"[blockscaled-mn-slice] max_abs_diff={(lib_out - ref).abs().max().item()}")
    return kernel, lib_out, ref


def run_blockscaled_grank32_shapek4_correctness(*, M, N, K, block_M, block_N, block_K, gran_k, threads, non_unit_scale=True):
    """64x64x128, block_K=128, gran_k=32 → scaleShapeK=4, MN-major."""
    gran_m = gran_n = 1
    assert block_K // gran_k == 4
    program = matmul_blockscaled(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        sf_a_granularity_m=gran_m,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=gran_n,
        sf_b_granularity_k=gran_k,
        threads=threads,
        num_stages=0,
    )
    kernel = tl.compile(program, out_idx=[4], verbose=True)
    cache_path = getattr(kernel, "_tilelang_cache_path", None)
    print(f"[blockscaled-k4] kernel cache entry: {cache_path}")

    A_logical, B_logical, A, B = _make_fp4_inputs(M, N, K)
    ScaleA, ScaleB = _make_e8m0_scales(K // gran_k, M // gran_m, N // gran_n, non_unit_scale=non_unit_scale)
    # Vary scale along K as well so scaleShapeK>1 is actually exercised.
    if non_unit_scale:
        for ki in range(ScaleA.shape[0]):
            ScaleA[ki] = (125 + (ki * 2 + torch.arange(ScaleA.shape[1], device=ScaleA.device)) % 5).to(torch.uint8)
            ScaleB[ki] = (125 + (ki * 3 + torch.arange(ScaleB.shape[1], device=ScaleB.device) * 2) % 5).to(torch.uint8)

    profiler = kernel.get_profiler()
    torch.cuda.synchronize()
    lib_out = profiler.func(A, B, ScaleA, ScaleB)
    torch.cuda.synchronize()
    lib_out = lib_out.cpu()

    ref = blockscaled_fp4_cpu_ref(
        A_logical,
        B_logical,
        ScaleA,
        ScaleB,
        sf_a_granularity_m=gran_m,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=gran_n,
        sf_b_granularity_k=gran_k,
        a_scale_k_major=False,
        b_scale_k_major=False,
        transpose_B=True,
    )
    _check_blockscaled_out(lib_out, ref)
    print(f"[blockscaled-k4] ScaleA.shape={tuple(ScaleA.shape)} max_abs_diff={(lib_out - ref).abs().max().item()}")
    return kernel, lib_out, ref


@requires_blockscaled
@pytest.mark.parametrize(
    "M,N,K,block_M,block_N,block_K,gran_k,threads,use_mls,k_pack,expected_mmac_k",
    [
        pytest.param(
            64,
            64,
            64,
            64,
            64,
            64,
            64,
            128,
            False,
            1,
            64,
            id="lds_ab_native_fp4",
        ),
        pytest.param(
            64,
            64,
            128,
            64,
            64,
            128,
            64,
            128,
            False,
            2,
            64,
            id="lds_ab_native_fp4_kpack2",
        ),
    ],
)
def test_gemm_blockscaled_basic_correctness(M, N, K, block_M, block_N, block_K, gran_k, threads, use_mls, k_pack, expected_mmac_k):
    kernel, _, _ = run_blockscaled_basic_correctness(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        gran_k=gran_k,
        threads=threads,
        use_mls=use_mls,
        k_pack=k_pack,
        non_unit_scale=True,
    )
    source = kernel.get_kernel_source()
    assert f", {expected_mmac_k}," in source and "mmac_scale_fp4_body" in source
    cache_path = getattr(kernel, "_tilelang_cache_path", None)
    if use_mls and cache_path:
        source = (Path(cache_path) / "device_kernel.cu").read_text()
        assert source.count("async_mls_load_asm<") >= 2
        assert "ds_scale_copy<" in source
        assert "mmac_scale_fp4_body" in source


@requires_blockscaled
@pytest.mark.parametrize(
    "M,N,K,block_M,block_N,block_K,gran_k,threads,mn_pad",
    [pytest.param(64, 64, 64, 64, 64, 64, 64, 128, 64, id="mn_slice")],
)
def test_gemm_blockscaled_mn_slice_correctness(M, N, K, block_M, block_N, block_K, gran_k, threads, mn_pad):
    run_blockscaled_mn_slice_correctness(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        gran_k=gran_k,
        threads=threads,
        mn_pad=mn_pad,
    )


@requires_blockscaled
@pytest.mark.parametrize(
    "M,N,K,block_M,block_N,block_K,gran_k,threads",
    [pytest.param(64, 64, 128, 64, 64, 128, 32, 128, id="k4_linear")],
)
def test_gemm_blockscaled_grank32_shapek4_correctness(M, N, K, block_M, block_N, block_K, gran_k, threads):
    run_blockscaled_grank32_shapek4_correctness(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        gran_k=gran_k,
        threads=threads,
    )


def run_blockscaled_mn_slice_shapek4_correctness(
    *,
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    gran_k,
    threads,
    mn_pad,
    k_pad,
    non_unit_scale=True,
):
    """K4 with both K/MN sliced in a larger parent LDS plane."""
    gran_m = gran_n = 1
    assert block_K // gran_k == 4
    program = matmul_blockscaled(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        mn_pad=mn_pad,
        k_pad=k_pad,
        op_ctrl=0,
        sf_a_granularity_m=gran_m,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=gran_n,
        sf_b_granularity_k=gran_k,
        threads=threads,
        num_stages=0,
    )
    kernel = tl.compile(program, out_idx=[4], verbose=True)
    cache_path = getattr(kernel, "_tilelang_cache_path", None)
    print(f"[blockscaled-mn-slice-k4] kernel cache entry: {cache_path}")
    if cache_path:
        cu_path = Path(cache_path) / "device_kernel.cu"
        if cu_path.exists():
            src = cu_path.read_text()
            expected = f"ds_scale_copy<0, 0, {64 + mn_pad}, {4 + k_pad}, 64, 4,"
            calls = "\n".join(line for line in src.splitlines() if "ds_scale_copy<" in line)
            assert expected in src, f"expected Parent+Tile template args:\n{calls}"
            assert ", 64, 4)" in src, f"expected origin_mn=64, origin_k=4:\n{calls}"
            assert "TLGeneratedScaleLdsLayout_" in src
            assert "CalculateOffset(int i0, int i1, int i2, int i3, int i4)" in src

    A_logical, B_logical, A, B = _make_fp4_inputs(M, N, K)
    ScaleA, ScaleB = _make_e8m0_scales(K // gran_k, M // gran_m, N // gran_n, non_unit_scale=non_unit_scale)
    if non_unit_scale:
        for ki in range(ScaleA.shape[0]):
            ScaleA[ki] = (125 + (ki * 2 + torch.arange(ScaleA.shape[1], device=ScaleA.device)) % 5).to(torch.uint8)
            ScaleB[ki] = (125 + (ki * 3 + torch.arange(ScaleB.shape[1], device=ScaleB.device) * 2) % 5).to(torch.uint8)

    profiler = kernel.get_profiler()
    torch.cuda.synchronize()
    lib_out = profiler.func(A, B, ScaleA, ScaleB)
    torch.cuda.synchronize()
    lib_out = lib_out.cpu()

    ref = blockscaled_fp4_cpu_ref(
        A_logical,
        B_logical,
        ScaleA,
        ScaleB,
        sf_a_granularity_m=gran_m,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=gran_n,
        sf_b_granularity_k=gran_k,
        a_scale_k_major=False,
        b_scale_k_major=False,
        transpose_B=True,
    )
    _check_blockscaled_out(lib_out, ref)
    print(f"[blockscaled-mn-slice-k4] max_abs_diff={(lib_out - ref).abs().max().item()}")
    return kernel, lib_out, ref


@requires_blockscaled
@pytest.mark.parametrize(
    "M,N,K,block_M,block_N,block_K,gran_k,threads,mn_pad,k_pad",
    [pytest.param(64, 64, 128, 64, 64, 128, 32, 128, 64, 4, id="k4_mn_k_slice")],
)
def test_gemm_blockscaled_mn_slice_shapek4_correctness(M, N, K, block_M, block_N, block_K, gran_k, threads, mn_pad, k_pad):
    run_blockscaled_mn_slice_shapek4_correctness(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        gran_k=gran_k,
        threads=threads,
        mn_pad=mn_pad,
        k_pad=k_pad,
    )


@requires_blockscaled
@pytest.mark.parametrize(
    "M,N,K,block_M,block_N,block_K,gran_k,threads,op_ctrl,scale_format_name,mn_pad,k_pad,use_mls",
    [
        pytest.param(64, 64, 128, 64, 64, 128, 32, 128, 1, "k2", 0, 0, False, id="k2"),
        pytest.param(64, 64, 128, 64, 64, 128, 32, 128, 2, "k4", 0, 0, False, id="k4"),
        pytest.param(64, 64, 128, 64, 64, 128, 32, 128, 1, "k2", 64, 4, False, id="k2_slice"),
        pytest.param(64, 64, 128, 64, 64, 128, 32, 128, 2, "k4", 64, 4, False, id="k4_slice"),
    ],
)
def test_gemm_blockscaled_interleaved_k4_correctness(
    M, N, K, block_M, block_N, block_K, gran_k, threads, op_ctrl, scale_format_name, mn_pad, k_pad, use_mls
):
    """Global scale is pre-interleaved per design §2.5 before global -> LDS."""
    program = matmul_blockscaled(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        mn_pad=mn_pad,
        k_pad=k_pad,
        op_ctrl=op_ctrl,
        sf_a_granularity_m=1,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=1,
        sf_b_granularity_k=gran_k,
        threads=threads,
        num_stages=0,
        use_mls=use_mls,
        scale_format_name=scale_format_name,
    )
    kernel = tl.compile(program, out_idx=[4], verbose=True)
    if use_mls:
        assert ", 32>(A_local" in kernel.get_kernel_source()
    A_logical, B_logical, A, B = _make_fp4_inputs(M, N, K)
    ScaleA, ScaleB = _make_e8m0_scales(K // gran_k, M, N, non_unit_scale=True)
    rows_per_instr = 2 if op_ctrl == 1 else 4
    if ScaleA.shape[0] == 1:
        ScaleA_physical = _interleave_scale_mn_for_copy(ScaleA, rows_per_instr)
        ScaleB_physical = _interleave_scale_mn_for_copy(ScaleB, rows_per_instr)
    elif op_ctrl == 2 and ScaleA.shape[0] == 2:
        ScaleA_physical = _interleave_scale_k2_mn2_for_copy(ScaleA)
        ScaleB_physical = _interleave_scale_k2_mn2_for_copy(ScaleB)
    else:
        ScaleA_physical = _interleave_scale_k_for_copy(ScaleA, rows_per_instr)
        ScaleB_physical = _interleave_scale_k_for_copy(ScaleB, rows_per_instr)

    lib_out = kernel.get_profiler().func(A, B, ScaleA_physical, ScaleB_physical).cpu()
    ref = blockscaled_fp4_cpu_ref(
        A_logical,
        B_logical,
        ScaleA,
        ScaleB,
        sf_a_granularity_m=1,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=1,
        sf_b_granularity_k=gran_k,
        a_scale_k_major=False,
        b_scale_k_major=False,
        transpose_B=True,
    )
    _check_blockscaled_out(lib_out, ref)
    print(f"[blockscaled-interleaved] op_ctrl={op_ctrl} origin=({mn_pad}, {k_pad}) max_abs_diff={(lib_out - ref).abs().max().item()}")


@requires_blockscaled
@pytest.mark.parametrize(
    "M,N,K,block_M,block_N,block_K,gran_k,threads",
    [pytest.param(64, 64, 64, 64, 64, 64, 32, 128, id="k2mn2")],
)
def test_gemm_blockscaled_interleaved_k2_mn2_correctness(M, N, K, block_M, block_N, block_K, gran_k, threads):
    """MmacK64 op_ctrl=2 follows the explicitly selected K2MN2 format."""
    program = matmul_blockscaled(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        mn_pad=0,
        k_pad=0,
        op_ctrl=2,
        sf_a_granularity_m=1,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=1,
        sf_b_granularity_k=gran_k,
        threads=threads,
        num_stages=0,
        scale_format_name="k2mn2",
    )
    kernel = tl.compile(program, out_idx=[4], verbose=True)

    A_logical, B_logical, A, B = _make_fp4_inputs(M, N, K)
    ScaleA, ScaleB = _make_e8m0_scales(2, M, N, non_unit_scale=True)
    ScaleA_physical = _interleave_scale_k2_mn2_for_copy(ScaleA)
    ScaleB_physical = _interleave_scale_k2_mn2_for_copy(ScaleB)
    lib_out = kernel.get_profiler().func(A, B, ScaleA_physical, ScaleB_physical).cpu()
    ref = blockscaled_fp4_cpu_ref(
        A_logical,
        B_logical,
        ScaleA,
        ScaleB,
        sf_a_granularity_m=1,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=1,
        sf_b_granularity_k=gran_k,
        a_scale_k_major=False,
        b_scale_k_major=False,
        transpose_B=True,
    )
    _check_blockscaled_out(lib_out, ref)
    print(f"[blockscaled-interleaved-k2-mn2] max_abs_diff={(lib_out - ref).abs().max().item()}")


@requires_blockscaled
@pytest.mark.parametrize(
    "M,N,K,block_M,block_N,block_K,gran_k,threads,op_ctrl,scale_format_name,use_mls",
    [
        pytest.param(64, 64, 128, 64, 64, 128, 32, 128, 2, "k2mn2", False, id="k2mn2_k4"),
    ],
)
def test_gemm_blockscaled_mixed_interleave_multi_k_correctness(
    M, N, K, block_M, block_N, block_K, gran_k, threads, op_ctrl, scale_format_name, use_mls
):
    """Mixed MN/K formats support more than one logical K atom."""
    program = matmul_blockscaled(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        op_ctrl=op_ctrl,
        sf_a_granularity_m=1,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=1,
        sf_b_granularity_k=gran_k,
        threads=threads,
        num_stages=0,
        use_mls=use_mls,
        scale_format_name=scale_format_name,
    )
    kernel = tl.compile(program, out_idx=[4], verbose=True)
    A_logical, B_logical, A, B = _make_fp4_inputs(M, N, K)
    ScaleA, ScaleB = _make_e8m0_scales(K // gran_k, M, N, non_unit_scale=True)
    if scale_format_name == "k2mn2":
        ScaleA_physical = _interleave_scale_k2_mn2_for_copy(ScaleA)
        ScaleB_physical = _interleave_scale_k2_mn2_for_copy(ScaleB)
    else:
        mn_group = 2 if scale_format_name == "mn2" else 4
        ScaleA_physical = _interleave_scale_mn_for_copy(ScaleA, mn_group)
        ScaleB_physical = _interleave_scale_mn_for_copy(ScaleB, mn_group)

    lib_out = kernel.get_profiler().func(A, B, ScaleA_physical, ScaleB_physical).cpu()
    ref = blockscaled_fp4_cpu_ref(
        A_logical,
        B_logical,
        ScaleA,
        ScaleB,
        sf_a_granularity_m=1,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=1,
        sf_b_granularity_k=gran_k,
        a_scale_k_major=False,
        b_scale_k_major=False,
        transpose_B=True,
    )
    _check_blockscaled_out(lib_out, ref)


@requires_blockscaled
def test_gemm_blockscaled_rejects_mixed_scale_stage_layout():
    program = matmul_blockscaled(
        M=64,
        N=64,
        K=64,
        block_M=64,
        block_N=64,
        block_K=64,
        op_ctrl=0,
        sf_a_granularity_k=64,
        sf_b_granularity_k=64,
        threads=128,
        scale_format_name="identity",
        scale_stages=2,
        annotate_scale_stage_layout="mixed",
    )
    with pytest.raises(tvm.TVMError, match="stage dimensions must be broadcast-only"):
        tl.compile(program, out_idx=[4])


@requires_blockscaled
@pytest.mark.parametrize(
    "M,N,op_ctrl,scale_format_name,issue_kind,explicit_unpacked_ds_read",
    [
        pytest.param(32, 32, 0, "identity", "single", True, id="identity_single_issue"),
        pytest.param(128, 128, 0, "identity", "multi", True, id="identity_multi_issue"),
        pytest.param(32, 32, 1, "k2", "single", True, id="k2_single_issue"),
        pytest.param(128, 128, 1, "k2", "multi", True, id="k2_multi_issue"),
        pytest.param(32, 32, 2, "k4", "single", True, id="k4_single_issue"),
        pytest.param(128, 128, 2, "k4", "multi", True, id="k4_multi_issue"),
        pytest.param(64, 64, 2, "k2mn2", "single", True, id="k2mn2_single_issue"),
        pytest.param(128, 128, 2, "k2mn2", "multi", True, id="k2mn2_multi_issue"),
        pytest.param(32, 32, 1, "mn2", "single", True, id="mn2_single_issue"),
        pytest.param(128, 128, 1, "mn2", "multi", True, id="mn2_multi_issue"),
        pytest.param(64, 64, 2, "mn4", "single", True, id="mn4_single_issue"),
        pytest.param(128, 128, 2, "mn4", "multi", True, id="mn4_multi_issue"),
        pytest.param(32, 32, 1, "mn2", "single", False, id="mn2_auto_expand"),
        pytest.param(64, 64, 2, "mn4", "single", False, id="mn4_auto_expand"),
    ],
)
def test_gemm_blockscaled_ds_read_k32_format_issue_coverage(M, N, op_ctrl, scale_format_name, issue_kind, explicit_unpacked_ds_read):
    """Explicit representation or MN format constraints drive MMAC-K32."""
    K = block_K = 128
    gran_k = 32
    program = matmul_blockscaled(
        M=M,
        N=N,
        K=K,
        block_M=M,
        block_N=N,
        block_K=block_K,
        op_ctrl=op_ctrl,
        sf_a_granularity_m=1,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=1,
        sf_b_granularity_k=gran_k,
        threads=64,
        num_stages=0,
        use_mls=True,
        scale_format_name=scale_format_name,
        explicit_unpacked_ds_read=explicit_unpacked_ds_read,
    )
    kernel = tl.compile(program, out_idx=[4], verbose=True)
    source = kernel.get_kernel_source()
    assert ", 32," in source and "mmac_scale_fp4_body" in source
    assert "ds_read_format_tensor_a" in source and "ds_read_format_tensor_b" in source
    assert "uint8_t, 4, 8>" in source

    A_logical, B_logical, A, B = _make_fp4_inputs(M, N, K)
    ScaleA, ScaleB = _make_e8m0_scales(K // gran_k, M, N, non_unit_scale=True)
    if scale_format_name == "identity":
        ScaleA_physical, ScaleB_physical = ScaleA, ScaleB
    elif scale_format_name in ("k2", "k4"):
        atom_k = 2 if scale_format_name == "k2" else 4
        ScaleA_physical = _interleave_scale_k_for_copy(ScaleA, atom_k)
        ScaleB_physical = _interleave_scale_k_for_copy(ScaleB, atom_k)
    elif scale_format_name == "k2mn2":
        ScaleA_physical = _interleave_scale_k2_mn2_for_copy(ScaleA)
        ScaleB_physical = _interleave_scale_k2_mn2_for_copy(ScaleB)
    else:
        atom_mn = 2 if scale_format_name == "mn2" else 4
        ScaleA_physical = _interleave_scale_mn_for_copy(ScaleA, atom_mn)
        ScaleB_physical = _interleave_scale_mn_for_copy(ScaleB, atom_mn)

    lib_out = kernel.get_profiler().func(A, B, ScaleA_physical, ScaleB_physical).cpu()
    ref = blockscaled_fp4_cpu_ref(
        A_logical,
        B_logical,
        ScaleA,
        ScaleB,
        sf_a_granularity_m=1,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=1,
        sf_b_granularity_k=gran_k,
        a_scale_k_major=False,
        b_scale_k_major=False,
        transpose_B=True,
    )
    assert torch.isfinite(lib_out).all(), f"non-finite output in {issue_kind}-issue case"
    _check_blockscaled_out(lib_out, ref)


@requires_blockscaled
@pytest.mark.parametrize(
    "M,N,K,block_K,gran_k,threads,op_ctrl,scale_format_name,explicit_unpacked_ds_read",
    [
        pytest.param(64, 64, 64, 64, 64, 128, 0, "identity", False, id="identity"),
        pytest.param(64, 64, 128, 128, 32, 128, 1, "k2", False, id="k2"),
        pytest.param(64, 64, 128, 128, 32, 128, 2, "k4", False, id="k4"),
        pytest.param(64, 64, 64, 64, 32, 128, 2, "k2mn2", False, id="k2mn2"),
        pytest.param(32, 32, 128, 128, 32, 64, 1, "mn2", True, id="mn2"),
        pytest.param(64, 64, 128, 128, 32, 64, 2, "mn4", True, id="mn4"),
    ],
)
def test_gemm_blockscaled_explicit_scale_stage(M, N, K, block_K, gran_k, threads, op_ctrl, scale_format_name, explicit_unpacked_ds_read):
    """A hand-selected stage is reduced to an LDS plane base before ds_scale_copy."""
    program = matmul_blockscaled(
        M=M,
        N=N,
        K=K,
        block_M=M,
        block_N=N,
        block_K=block_K,
        op_ctrl=op_ctrl,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_k=gran_k,
        threads=threads,
        num_stages=0,
        use_mls=scale_format_name in ("mn2", "mn4"),
        scale_format_name=scale_format_name,
        scale_stages=2,
        annotate_scale_stage_layout=scale_format_name == "identity",
        explicit_unpacked_ds_read=explicit_unpacked_ds_read,
    )
    kernel = tl.compile(program, out_idx=[4], verbose=True)

    A_logical, B_logical, A, B = _make_fp4_inputs(M, N, K)
    ScaleA, ScaleB = _make_e8m0_scales(K // gran_k, M, N, non_unit_scale=True)
    if scale_format_name == "identity":
        ScaleA_physical, ScaleB_physical = ScaleA, ScaleB
    elif scale_format_name in ("k2", "k4"):
        atom_k = 2 if scale_format_name == "k2" else 4
        ScaleA_physical = _interleave_scale_k_for_copy(ScaleA, atom_k)
        ScaleB_physical = _interleave_scale_k_for_copy(ScaleB, atom_k)
    elif scale_format_name == "k2mn2":
        ScaleA_physical = _interleave_scale_k2_mn2_for_copy(ScaleA)
        ScaleB_physical = _interleave_scale_k2_mn2_for_copy(ScaleB)
    else:
        atom_mn = 2 if scale_format_name == "mn2" else 4
        ScaleA_physical = _interleave_scale_mn_for_copy(ScaleA, atom_mn)
        ScaleB_physical = _interleave_scale_mn_for_copy(ScaleB, atom_mn)

    lib_out = kernel.get_profiler().func(A, B, ScaleA_physical, ScaleB_physical).cpu()
    ref = blockscaled_fp4_cpu_ref(
        A_logical,
        B_logical,
        ScaleA,
        ScaleB,
        sf_a_granularity_m=1,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=1,
        sf_b_granularity_k=gran_k,
        a_scale_k_major=False,
        b_scale_k_major=False,
        transpose_B=True,
    )
    _check_blockscaled_out(lib_out, ref)


@requires_blockscaled
@pytest.mark.parametrize(
    "a_dtype,b_dtype,unpack_side,real_ab_type",
    [
        pytest.param("float8_e4m3fn", "float8_e4m3fn", False, 0, id="fp8_fp8"),
        pytest.param("float4_e2m1fn", "float8_e4m3fn", "a", 20, id="fp4_fp8"),
        pytest.param("float8_e4m3fn", "float4_e2m1fn", "b", 4, id="fp8_fp4"),
    ],
)
def test_gemm_blockscaled_f8f6f4_operand_combinations(a_dtype, b_dtype, unpack_side, real_ab_type):
    """Scale GEMM uses f8f6f4 for FP8 pairs and explicit FP4 expansion."""
    M = N = 32
    K = 128
    gran_k = 32
    program = matmul_blockscaled(
        M=M,
        N=N,
        K=K,
        block_M=M,
        block_N=N,
        block_K=K,
        op_ctrl=0,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_k=gran_k,
        threads=64,
        use_mls=True,
        explicit_unpacked_ds_read=unpack_side,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
    )
    kernel = tl.compile(program, out_idx=[4], verbose=True)
    source = kernel.get_kernel_source()
    assert "mmac_scale_fp4_body" in source and f", 32, {real_ab_type}>" in source

    A_logical, B_logical, A_fp4, B_fp4 = _make_fp4_inputs(M, N, K)
    A_fp8 = (((torch.arange(M, device="cuda")[:, None] * 3 + torch.arange(K, device="cuda")[None, :]) % 7) - 3).to(torch.float8_e4m3fn)
    B_fp8 = (((torch.arange(N, device="cuda")[:, None] * 5 + torch.arange(K, device="cuda")[None, :]) % 7) - 3).to(torch.float8_e4m3fn)
    A = A_fp4 if a_dtype == "float4_e2m1fn" else A_fp8
    B = B_fp4 if b_dtype == "float4_e2m1fn" else B_fp8
    A_ref = _fp4_e2m1fn_decode(A_logical).cpu() if a_dtype == "float4_e2m1fn" else A_fp8.float().cpu()
    B_ref = _fp4_e2m1fn_decode(B_logical).cpu() if b_dtype == "float4_e2m1fn" else B_fp8.float().cpu()
    ScaleA = torch.full((K // gran_k, M), 127, dtype=torch.uint8, device="cuda")
    ScaleB = torch.full((K // gran_k, N), 127, dtype=torch.uint8, device="cuda")
    lib_out = kernel.get_profiler().func(A, B, ScaleA, ScaleB).cpu()
    torch.testing.assert_close(lib_out, A_ref @ B_ref.T, rtol=1e-2, atol=1e-2)


@requires_blockscaled
@pytest.mark.parametrize(
    "M,N,K,block_M,block_N,block_K,gran_k,threads",
    [pytest.param(64, 64, 256, 64, 64, 256, 32, 128, id="k8_auto_xor")],
)
def test_gemm_blockscaled_auto_scale_lds_layout_correctness(M, N, K, block_M, block_N, block_K, gran_k, threads):
    """No annotation: K8 linear conflict should select an automatic XOR layout."""
    program = matmul_blockscaled(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        mn_pad=0,
        k_pad=0,
        op_ctrl=2,
        sf_a_granularity_m=1,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=1,
        sf_b_granularity_k=gran_k,
        threads=threads,
        num_stages=0,
        scale_format_name="k4",
    )
    kernel = tl.compile(program, out_idx=[4], verbose=True)
    cache_path = getattr(kernel, "_tilelang_cache_path", None)
    if cache_path:
        src = (Path(cache_path) / "device_kernel.cu").read_text()
        assert "TLGeneratedScaleLdsLayout_" in src
        assert " ^ " in src

    A_logical, B_logical, A, B = _make_fp4_inputs(M, N, K)
    ScaleA, ScaleB = _make_e8m0_scales(8, M, N, non_unit_scale=True)
    ScaleA_physical = _interleave_scale_k_for_copy(ScaleA, 4)
    ScaleB_physical = _interleave_scale_k_for_copy(ScaleB, 4)
    lib_out = kernel.get_profiler().func(A, B, ScaleA_physical, ScaleB_physical).cpu()
    ref = blockscaled_fp4_cpu_ref(
        A_logical,
        B_logical,
        ScaleA,
        ScaleB,
        sf_a_granularity_m=1,
        sf_a_granularity_k=gran_k,
        sf_b_granularity_n=1,
        sf_b_granularity_k=gran_k,
        a_scale_k_major=False,
        b_scale_k_major=False,
        transpose_B=True,
    )
    _check_blockscaled_out(lib_out, ref)
    print(f"[blockscaled-auto-scale-layout] max_abs_diff={(lib_out - ref).abs().max().item()}")


if __name__ == "__main__":
    tilelang.testing.main()
