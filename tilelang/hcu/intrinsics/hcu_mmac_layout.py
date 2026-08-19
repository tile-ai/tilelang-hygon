"""HCU MMAC micro-tile maps and block-level fragment layouts (mirror ``gemm_layouts.cc``)."""

from __future__ import annotations

from collections.abc import Callable

import tilelang.language as T
from tilelang.layout import Fragment
from tvm import DataType
from tvm.runtime import convert
from tvm.tirx import IndexMap


def shared_16x4_to_local_64x1_layout_A(i, j):
    thread_id = i + 16 * j
    # IndexMap needs PrimExpr for local_id; plain Python 0 is rejected.
    local = T.int32(0)
    return thread_id, local


def thread_id_shared_access_64x1_to_16x4_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = thread_id // 16
    return i, j


def shared_4x16_to_local_64x1_layout_B(i, j):
    thread_id = i * 16 + j
    local = T.int32(0)
    return thread_id, local


def thread_id_shared_access_64x1_to_4x16_layout_B(thread_id, local_id):
    i = thread_id // 16
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_C(i, j):
    thread_id = j + (i // 4) * 16
    local = i % 4
    return thread_id, local


def shared_16x16_to_ldmatrix_64x4_layout(ind):
    i, j = ind[0], ind[1]
    thread_id, local_id = shared_16x16_to_local_64x4_layout_C(i, j)
    return convert([thread_id, local_id])


def thread_id_shared_access_64x4_to_16x16_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = (thread_id // 16) * 4 + local_id
    return i, j


def shared_16x16_to_local_64x4_layout_A(i, j):
    thread_id = i + 16 * (j // 4)
    local = j % 4
    return thread_id, local


def thread_id_shared_access_64x4_to_16x16_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_B(i, j):
    thread_id = j + (i // 4) * 16
    local = i % 4
    return thread_id, local


# HCU C micro-tile (``makeGemmFragmentC16x16HCU*`` in ``gemm_layouts.cc``).
def shared_16x16_to_local_64x4_layout_C_hcu(i, j):
    thread_id = 16 * (j % 4) + i
    local = j // 4
    return thread_id, local


def shared_16x16_to_local_64x4_layout_C_hcu_lit(i, j):
    thread_id = 16 * (j // 4) + i
    local = j % 4
    return thread_id, local


def shared_16x16_to_local_64x4_layout_C_hcu_lts(i, j):
    thread_id = 16 * (i % 4) + j
    local = i // 4
    return thread_id, local


def shared_16x16_to_local_64x4_layout_C_hcu_lit_lts(i, j):
    thread_id = 16 * (i // 4) + j
    local = i % 4
    return thread_id, local


def thread_id_shared_access_64x2_to_16x8_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = (thread_id // 16) * 2 + local_id
    return i, j


def shared_16x8_to_local_64x2_layout_A(i, j):
    thread_id = i + 16 * (j // 2)
    local = j % 2
    return thread_id, local


def thread_id_shared_access_64x2_to_16x8_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 2
    j = thread_id % 16
    return i, j


def shared_16x8_to_local_64x2_layout_B(i, j):
    thread_id = j + (i // 2) * 16
    local = i % 2
    return thread_id, local


shared_16x16_to_local_64x4_layout_m_n = shared_16x16_to_local_64x4_layout_A
shared_16x16_to_local_64x4_layout_n_k = shared_16x16_to_local_64x4_layout_A
shared_16x16_to_local_64x4_layout_n_m = shared_16x16_to_local_64x4_layout_B
shared_16x16_to_local_64x4_layout_k_n = shared_16x16_to_local_64x4_layout_B


def thread_id_shared_access_64x4_to_16x16_layout_C_m_n(thread_id, local_id):
    i = thread_id % 16
    j = local_id * 4 + (thread_id // 16)
    return i, j


def thread_id_shared_access_64x4_to_16x16_layout_C_n_m(thread_id, local_id):
    i = local_id * 4 + (thread_id // 16)
    j = thread_id % 16
    return i, j


# lit: 4 interleaved
def thread_id_shared_access_64x4_to_16x16_layout_C_m_n_lit(thread_id, local_id):
    i = thread_id % 16
    j = local_id + (thread_id // 16) * 4
    return i, j


def thread_id_shared_access_64x4_to_16x16_layout_C_n_m_lit(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def thread_id_shared_access_64x4_to_16x16_layout_C_lts(thread_id, local_id):
    i = local_id * 4 + thread_id // 16
    j = thread_id % 16
    return i, j


def thread_id_shared_access_64x4_to_16x16_layout_C_lit_lts(thread_id, local_id):
    i = (thread_id // 16) * 4 + local_id
    j = thread_id % 16
    return i, j


def thread_id_shared_access_64x8_to_16x32_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = (thread_id // 16) * 8 + local_id
    return i, j


def shared_16x32_to_local_64x8_layout_A(i, j):
    thread_id = i + 16 * (j // 8)
    local = j % 8
    return thread_id, local


def thread_id_shared_access_64x8_to_16x32_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 8
    j = thread_id % 16
    return i, j


def shared_16x32_to_local_64x8_layout_B(i, j):
    thread_id = j + (i // 8) * 16
    local = i % 8
    return thread_id, local


def thread_id_shared_access_64x16_to_16x64_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = local_id + (thread_id // 16) * 16
    return i, j


def shared_16x64_to_local_64x16_layout_A(i, j):
    thread_id = i + 16 * (j // 16)
    local = j % 16
    return thread_id, local


def thread_id_shared_access_64x16_to_16x64_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 16
    j = thread_id % 16
    return i, j


def shared_16x64_to_local_64x16_layout_B(i, j):
    thread_id = i + 16 * (j // 16)
    local = j % 16
    return thread_id, local


def thread_id_shared_access_64x32_to_16x128_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = local_id + (thread_id // 16) * 32
    return i, j


def shared_16x128_to_local_64x32_layout_A(i, j):
    thread_id = i + 16 * (j // 32)
    local = j % 32
    return thread_id, local


def thread_id_shared_access_64x32_to_16x128_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 32
    j = thread_id % 16
    return i, j


def shared_16x128_to_local_64x32_layout_B(i, j):
    thread_id = j + 16 * (i // 32)
    local = i % 32
    return thread_id, local


def make_mmac_swizzle_layout(shared_buf, vecSize=8):
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    numBanks = 32
    bankBitWidth = 32
    SIMDWidth = 16

    innerDimLength = shape[-1]
    typeWidthInBit = DataType(dtype).bits

    elemsPerOneBanksRow = (numBanks * bankBitWidth) // typeWidthInBit
    perPhase = max(1, elemsPerOneBanksRow // innerDimLength)
    maxPhase = min(SIMDWidth // perPhase, innerDimLength // vecSize)

    def transform(row, col):
        phase = (row // perPhase) % maxPhase
        colOffSwizzled = ((col // vecSize) ^ phase) * vecSize
        colOffOrdered = col % vecSize
        colOff = colOffSwizzled + colOffOrdered
        return row, colOff

    return T.Layout(shape, transform)


def mmac_micro_k(element_bits: int, k_pack: int, *, mmac_k_dim: int | None = None) -> int:
    """K extent (elements) of one HCU mmac micro-tile along the reduction axis.

    When ``mmac_k_dim`` is set (from ``hcu_mmac_k_dim``), it is the per-instruction
    K for the operand dtype; otherwise dtype-default K is used.
    """
    if mmac_k_dim is not None:
        return mmac_k_dim * k_pack
    if element_bits == 16:
        return 16 * k_pack
    if element_bits == 32:
        return 8 * k_pack
    if element_bits == 8:
        return 32 * k_pack
    raise ValueError(f"unsupported element bitwidth={element_bits}")


def _micro_to_local_layout_fn(element_bits: int, micro_k: int) -> Callable:
    """Return ``(i, j) -> (thread_id, local_id)`` for one AB micro-tile (ldmatrix family)."""
    if element_bits == 16:
        if micro_k <= 16:
            return shared_16x16_to_local_64x4_layout_A
        return shared_16x32_to_local_64x8_layout_A
    if element_bits == 32:
        if micro_k <= 4:
            return shared_16x4_to_local_64x1_layout_A
        if micro_k <= 8:
            return shared_16x8_to_local_64x2_layout_A
        return shared_16x16_to_local_64x4_layout_A
    if element_bits == 8:
        if micro_k <= 32:
            return shared_16x32_to_local_64x8_layout_A
        if micro_k <= 64:
            return shared_16x64_to_local_64x16_layout_A
        if micro_k <= 128:
            return shared_16x128_to_local_64x32_layout_A
    if element_bits == 4:
        if micro_k <= 64:
            return shared_16x64_to_local_64x16_layout_A
        if micro_k <= 128:
            return shared_16x128_to_local_64x32_layout_A
    raise ValueError(f"unsupported element bitwidth={element_bits}")


def _swap_ij_layout_fn(layout_fn: Callable) -> Callable:
    def swapped(i, j):
        return layout_fn(j, i)

    return swapped


def _fragment_from_layout_fn(
    layout_fn: Callable,
    shape_i: int,
    shape_j: int,
) -> Fragment:
    """Wrap a micro-tile ``layout_fn`` as ``T.Fragment`` via ``IndexMap``."""
    index_map = IndexMap.from_func(layout_fn, index_dtype=T.int32)

    def forward_thread(i: int, j: int) -> int:
        lane_id, _ = index_map.map_indices([i, j])
        return lane_id

    def forward_index(i: int, j: int) -> int:
        _, local_id = index_map.map_indices([i, j])
        return local_id

    return T.Fragment(
        [shape_i, shape_j],
        forward_thread_fn=forward_thread,
        forward_index_fn=forward_index,
    )


def _micro_ab_fragment(element_bits: int, k_pack: int, *, spatial_leading: bool, mmac_k_dim: int | None = None) -> Fragment:
    """One warp AB micro-tile fragment; ``spatial_leading`` selects ``[M/N, K]`` vs ``[K, M/N]``."""
    micro_k = mmac_micro_k(element_bits, k_pack, mmac_k_dim=mmac_k_dim)
    layout_fn = _micro_to_local_layout_fn(element_bits, micro_k)
    if not spatial_leading:
        layout_fn = _swap_ij_layout_fn(layout_fn)
    if spatial_leading:
        return _fragment_from_layout_fn(layout_fn, 16, micro_k)
    return _fragment_from_layout_fn(layout_fn, micro_k, 16)


def _micro_c_fragment(*, lit: bool, lts: bool = False) -> Fragment:
    if lts:
        layout_fn = (
            shared_16x16_to_local_64x4_layout_C_hcu_lit_lts
            if lit
            else shared_16x16_to_local_64x4_layout_C_hcu_lts
        )
    else:
        layout_fn = shared_16x16_to_local_64x4_layout_C_hcu_lit if lit else shared_16x16_to_local_64x4_layout_C_hcu
    return _fragment_from_layout_fn(layout_fn, 16, 16)


def make_gemm_fragment_hcu(
    block_m: int,
    block_n: int,
    num_warp_m: int,
    num_warp_n: int,
    num_warp_k: int,
    element_bits: int,
    min_n_per_warp: int,
    *,
    lit: bool = False,
    lts: bool = False,
) -> Fragment:
    if element_bits == 64:
        raise ValueError("float64 C fragment is not supported for HCU MMAC")

    warp_m = block_m // num_warp_m
    num_warp_n_no_recompute = min(num_warp_n, block_n // min_n_per_warp)
    warp_n = block_n // num_warp_n_no_recompute
    n_recompute = num_warp_n // num_warp_n_no_recompute
    if num_warp_k > 1 and n_recompute != 1:
        raise ValueError("n_recompute must be 1 when num_warp_k > 1")

    if block_m % warp_m != 0 or block_n % warp_n != 0:
        raise ValueError(f"invalid block/warp tile: block=({block_m},{block_n}) warp=({warp_m},{warp_n})")
    if warp_m % 16 != 0 or warp_n % 16 != 0:
        raise ValueError(f"warp_m and warp_n must be multiples of 16, got ({warp_m}, {warp_n})")

    base = _micro_c_fragment(lit=lit, lts=lts).repeat([1, 1], repeat_on_thread=False)
    warp_layout = base.repeat([warp_m // 16, warp_n // 16], repeat_on_thread=False, lower_dim_first=True)
    block_layout = warp_layout.repeat([block_m // warp_m, block_n // warp_n], repeat_on_thread=True, lower_dim_first=False)
    if n_recompute > 1 or num_warp_k > 1:
        block_layout = block_layout.replicate(n_recompute * num_warp_k)
    return block_layout


def make_gemm_fragment_a_hcu(
    block_m: int,
    block_n: int,
    block_k: int,
    num_warp_m: int,
    num_warp_n: int,
    num_warp_k: int,
    element_bits: int,
    k_pack: int,
    transposed: bool,
    *,
    mmac_k_dim: int | None = None,
) -> Fragment:
    warp_m = block_m // num_warp_m
    warp_k = block_k // num_warp_k
    mk = mmac_micro_k(element_bits, k_pack, mmac_k_dim=mmac_k_dim)
    if block_m % warp_m != 0 or warp_m % 16 != 0:
        raise ValueError(f"invalid A warp_m={warp_m} for block_m={block_m}")
    if warp_k % mk != 0:
        raise ValueError(f"warp_k={warp_k} must be divisible by mmac_micro_k={mk}")

    spatial_leading = not transposed
    base = _micro_ab_fragment(element_bits, k_pack, spatial_leading=spatial_leading, mmac_k_dim=mmac_k_dim).repeat(
        [1, 1], repeat_on_thread=False
    )
    if transposed:
        warp_layout = base.repeat([warp_k // mk, warp_m // 16], repeat_on_thread=False, lower_dim_first=True)
        block_layout = (
            warp_layout.repeat([1, num_warp_m], repeat_on_thread=True, lower_dim_first=True)
            .replicate(num_warp_n)
            .repeat([num_warp_k, 1], repeat_on_thread=True, lower_dim_first=True)
        )
    else:
        warp_layout = base.repeat([warp_m // 16, warp_k // mk], repeat_on_thread=False, lower_dim_first=False)
        block_layout = (
            warp_layout.repeat([num_warp_m, 1], repeat_on_thread=True, lower_dim_first=True)
            .replicate(num_warp_n)
            .repeat([1, num_warp_k], repeat_on_thread=True, lower_dim_first=True)
        )
    return block_layout


def make_gemm_fragment_b_hcu(
    block_m: int,
    block_n: int,
    block_k: int,
    num_warp_m: int,
    num_warp_n: int,
    num_warp_k: int,
    element_bits: int,
    k_pack: int,
    transposed: bool,
    min_n_per_warp: int,
    *,
    mmac_k_dim: int | None = None,
) -> Fragment:
    if block_n % min_n_per_warp != 0:
        raise ValueError(f"block_n={block_n} must be divisible by min_n_per_warp={min_n_per_warp}")

    warp_n_no_recompute = min(num_warp_n, block_n // min_n_per_warp)
    warp_n = block_n // warp_n_no_recompute
    n_recompute = num_warp_n // warp_n_no_recompute
    warp_k = block_k // num_warp_k
    if num_warp_k > 1 and n_recompute != 1:
        raise ValueError("n_recompute must be 1 when num_warp_k > 1")

    if block_n % warp_n != 0 or warp_n % min_n_per_warp != 0:
        raise ValueError(f"invalid B warp_n={warp_n} for block_n={block_n}")
    mk = mmac_micro_k(element_bits, k_pack, mmac_k_dim=mmac_k_dim)
    if warp_k % mk != 0:
        raise ValueError(f"warp_k={warp_k} must be divisible by mmac_micro_k={mk}")

    # gemm ``TransposeB``: spatial-leading micro-tile when ``transposed=True`` (``gemm_layouts.cc``).
    spatial_leading = transposed
    base = _micro_ab_fragment(element_bits, k_pack, spatial_leading=spatial_leading, mmac_k_dim=mmac_k_dim).repeat(
        [1, 1], repeat_on_thread=False
    )
    if transposed:
        warp_layout = base.repeat([warp_n // 16, warp_k // mk], repeat_on_thread=False, lower_dim_first=False)
        block_layout = warp_layout.replicate(num_warp_m).repeat(
            [warp_n_no_recompute, num_warp_k], repeat_on_thread=True, lower_dim_first=False
        )
    else:
        warp_layout = base.repeat([warp_k // mk, warp_n // 16], repeat_on_thread=False, lower_dim_first=True)
        block_layout = warp_layout.replicate(num_warp_m).repeat(
            [num_warp_k, warp_n_no_recompute], repeat_on_thread=True, lower_dim_first=True
        )
    if n_recompute > 1:
        block_layout = block_layout.replicate(n_recompute)
    return block_layout


def make_ds_read_format_fragment_hcu(
    block_mn: int,
    block_k: int,
    num_warp_mn: int,
    num_warp_k: int,
    element_bits: int,
    num_warp_mn_no_recompute: int,
    trans: bool,
) -> Fragment:
    n = max(1, num_warp_mn_no_recompute)
    warp_mn = block_mn // n
    mn_recompute = num_warp_mn // n
    warp_k = block_k // num_warp_k
    k_pack = 1
    mk = mmac_micro_k(element_bits, k_pack)

    spatial_leading = trans
    base = _micro_ab_fragment(element_bits, k_pack, spatial_leading=spatial_leading).repeat([1, 1], repeat_on_thread=False)
    if trans:
        warp_layout = base.repeat([warp_mn // 16, warp_k // mk], repeat_on_thread=False, lower_dim_first=False)
        block_layout = warp_layout.repeat([n, num_warp_k], repeat_on_thread=True, lower_dim_first=False)
    else:
        warp_layout = base.repeat([warp_k // mk, warp_mn // 16], repeat_on_thread=False, lower_dim_first=True)
        block_layout = warp_layout.repeat([num_warp_k, n], repeat_on_thread=True, lower_dim_first=True)
    if mn_recompute > 1:
        block_layout = block_layout.replicate(mn_recompute)
    return block_layout
