from __future__ import annotations

from types import SimpleNamespace

from tilelang import _ffi_api
from tilelang import language as T
from tilelang.hcu.intrinsics.hcu_mmac_layout import (
    make_gemm_fragment_a_hcu,
    make_gemm_fragment_b_hcu,
    make_gemm_fragment_hcu,
)
from tilelang.hcu.intrinsics.hcu_mmac_macro_generator import HCUMatrixCoreIntrinEmitter
from tilelang.layout.swizzle import make_hcu_swizzled_layout
from tilelang.transform.simplify import _Simplify
from tilelang.utils.language import is_fragment, is_full_region, is_shared, is_shared_dynamic
from tilelang.hcu.intrinsics.hcu_mmac_emitter_utils import hcu_mmac_k_dim_for_operand
from tilelang.hcu.target import target_has_mmac_lit_lts, target_is_hcu
from tvm import DataType, tirx
from tvm.ir import Range
from tvm.target import Target

from tilelang.tileop.gemm.gemm_base import GemmBase

GEMM_INST_HCU_MMAC = "hcu.mmac"


def _int_annotation(annotations, key: str, default: int = 0) -> int:
    if not annotations:
        return default
    try:
        val = annotations[key]
    except (KeyError, TypeError):
        return default
    if isinstance(val, tirx.IntImm):
        return int(val.value)
    if isinstance(val, (int, bool)):
        return int(val)
    attr_val = getattr(val, "value", None)
    if attr_val is not None:
        return int(attr_val)
    return default


def _is_shared_like(buf) -> bool:
    return is_shared(buf) or is_shared_dynamic(buf)


def _is_fp4_dtype(dtype) -> bool:
    return DataType(dtype).is_float4()


def _is_f8f6f4_operand_dtype(dtype) -> bool:
    dtype_str = str(dtype)
    return _is_fp4_dtype(dtype) or "float8_e4m3" in dtype_str or "float8_e5m2" in dtype_str


def _hcu_layout_bits_for_dtype(dtype, *, fp4_mmac_mode: str) -> int:
    dtype = DataType(dtype)
    if fp4_mmac_mode == "f8f6f4" and _is_f8f6f4_operand_dtype(dtype):
        return 8
    return int(dtype.bits)


def _mls_local_dtype(dtype, *, fp4_mmac_mode: str, use_tf32: bool = False) -> str:
    dtype = DataType(dtype)
    if use_tf32 and str(dtype) == "float32":
        return "int32"
    if fp4_mmac_mode == "f8f6f4" and _is_f8f6f4_operand_dtype(dtype):
        return "uint8"
    return str(dtype)


def _resolve_hcu_mls_meta(gemm_node, A, B, block_size: int, target: Target):
    """Resolve MLS physical representation and MMAC mode through C++."""
    annotations = getattr(gemm_node, "annotations", None) or {}
    a_from_mls = _int_annotation(annotations, "tl.a_from_mls")
    b_from_mls = _int_annotation(annotations, "tl.b_from_mls")
    trans_a = bool(gemm_node.transA)
    trans_b = bool(gemm_node.transB)
    a_mls_trans = not trans_a
    b_mls_trans = trans_b

    mode = _ffi_api.ResolveHcuMmacMode(
        DataType(A.dtype),
        DataType(B.dtype),
        bool(is_fragment(A)),
        bool(is_fragment(B)),
        bool(getattr(gemm_node, "sfaRegion", None) is not None and getattr(gemm_node, "sfbRegion", None) is not None),
        int(gemm_node.k),
        _int_annotation(annotations, "tl.scale_a_format", 0),
        _int_annotation(annotations, "tl.scale_b_format", 0),
        target,
    )
    mmac_mode = "f8f6f4" if int(mode[0]) == 1 else "native"

    return SimpleNamespace(
        a_from_mls=a_from_mls,
        b_from_mls=b_from_mls,
        a_mls_trans=int(a_mls_trans),
        b_mls_trans=int(b_mls_trans),
        mmac_mode=mmac_mode,
        gemm_element_bits=int(mode[1]),
        mmac_k=int(mode[2]),
        real_ab_type=int(mode[3]),
    )


def _compute_hcu_warp_partition(gemm, thread_nums: int, target: Target, meta) -> tuple[int, int, int, int, int]:
    """Return (m_warp, n_warp, k_warp, m_per_warp, n_per_warp).

    Floors come from C++ ``ComputeWarpPartitionHCU`` (MLS ∪ scale extra).
    """
    element_bits = int(meta.gemm_element_bits)
    annotations = getattr(gemm.gemm_node, "annotations", None) or {}
    min_m = _int_annotation(annotations, "tl.scale_min_m_per_warp", 0)
    min_n = _int_annotation(annotations, "tl.scale_min_n_per_warp", 0)
    floors = _ffi_api.GemmWarpPolicyComputeWarpPartitionHCU(
        gemm.policy,
        int(gemm.M),
        int(gemm.N),
        int(gemm.K),
        int(gemm.k_pack),
        int(element_bits),
        int(thread_nums),
        target,
        0,  # gemm_inst unused; HCU warp partition is target-gated in C++
        bool(meta.a_from_mls),
        bool(meta.b_from_mls),
        bool(meta.a_mls_trans),
        bool(meta.b_mls_trans),
        int(min_m),
        int(min_n),
    )
    m_per_warp = int(floors[0])
    n_per_warp = int(floors[1])
    return (
        int(gemm.policy.m_warp),
        int(gemm.policy.n_warp),
        int(gemm.policy.k_warp),
        m_per_warp,
        n_per_warp,
    )


def _thread_bounds_min(thread_bounds: Range) -> int:
    if isinstance(thread_bounds.min, tirx.IntImm):
        return int(thread_bounds.min.value)
    return 0


def _make_hcu_emitter(
    gemm: GemmHCUMMAC,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    target: Target,
    thread_var: tirx.Var,
    thread_bounds_min: int = 0,
    fp4_mmac_mode: str = "native",
    min_n_per_warp: int = 16,
) -> HCUMatrixCoreIntrinEmitter:
    return HCUMatrixCoreIntrinEmitter(
        a_dtype=gemm.a_dtype,
        b_dtype=gemm.b_dtype,
        accum_dtype=gemm.accum_dtype,
        a_transposed=gemm.trans_A,
        b_transposed=gemm.trans_B,
        block_row_warps=warp_m,
        block_col_warps=warp_n,
        block_m=int(gemm.M),
        block_n=int(gemm.N),
        chunk=gemm.chunk,
        k_pack=gemm.k_pack,
        block_k_warps=warp_k,
        target=target,
        thread_var=thread_var,
        thread_bounds_min=thread_bounds_min,
        min_n_per_warp=min_n_per_warp,
        use_tf32=gemm.use_tf32,
        fp4_mmac_mode=fp4_mmac_mode,
    )


def _ann_int(annotations, key: str, default: int | None = None) -> int:
    if not annotations:
        if default is None:
            raise KeyError(key)
        return default
    try:
        val = annotations[key]
    except (KeyError, TypeError) as err:
        if default is None:
            raise KeyError(key) from err
        return default
    if isinstance(val, tirx.IntImm):
        return int(val.value)
    if isinstance(val, (int, bool)):
        return int(val)
    attr_val = getattr(val, "value", None)
    if attr_val is not None:
        return int(attr_val)
    if default is None:
        raise KeyError(key)
    return default


def _ann_expr(annotations, key: str) -> tirx.PrimExpr:
    if not annotations:
        raise KeyError(key)
    val = annotations[key]
    if isinstance(val, tirx.PrimExpr):
        return val
    raise TypeError(f"annotation {key} must be PrimExpr, got {type(val)}")


def _scale_rows_mn(scale_shape_mn: int, gran_mn: int) -> int:
    if gran_mn >= 16:
        return scale_shape_mn
    return scale_shape_mn * gran_mn // 16


def _scale_phys_k(scale_shape_k: int, gran_k: int, mmac_k: int = 64) -> int:
    k_dup = 2 if (mmac_k == 64 and gran_k >= 64) else 1
    return scale_shape_k * k_dup


def _scale_aligned_rows_per_wave(
    scale_shape_mn: int, scale_shape_k: int, gran_mn: int, gran_k: int, mn_warps: int, mmac_k: int = 64
) -> int:
    rows_mn = _scale_rows_mn(scale_shape_mn, gran_mn)
    assert rows_mn % mn_warps == 0, f"rows_mn={rows_mn} not divisible by mn_warps={mn_warps}"
    phys_k = _scale_phys_k(scale_shape_k, gran_k, mmac_k)
    logical = (rows_mn // mn_warps) * phys_k
    return (logical + 7) // 8 * 8


def _scale_mn_to_row(mn_idx: tirx.PrimExpr | int, gran_mn: int) -> tirx.PrimExpr:
    if gran_mn >= 16:
        return tirx.Cast("int32", mn_idx) if not isinstance(mn_idx, tirx.PrimExpr) else mn_idx
    return (tirx.Cast("int32", mn_idx) if not isinstance(mn_idx, int) else mn_idx) * gran_mn // 16


def _parse_scale_shape(shape, k_major: bool) -> tuple[int, int]:
    """Return (scale_shape_mn, scale_shape_k)."""
    if k_major:
        return int(shape[-2]), int(shape[-1])
    return int(shape[-1]), int(shape[-2])


class GemmHCUMMAC(GemmBase):
    """HCU matrix core GEMM: layout and lowering via Python ``HCUMatrixCoreIntrinEmitter``."""

    @property
    def allow_f8f6f4_mixed_dtypes(self) -> bool:
        return True

    def infer_layout(self, target: Target, thread_nums: int):
        if not target_is_hcu(target):
            raise ValueError("GemmHCUMMAC requires an HCU target")
        meta = _resolve_hcu_mls_meta(self.gemm_node, self.A, self.B, int(thread_nums), target)
        block_size = int(thread_nums)
        fp4_mmac_mode = meta.mmac_mode
        warp_m, warp_n, warp_k, _m_per_warp, min_n_per_warp = _compute_hcu_warp_partition(self, block_size, target, meta)
        elem_bits_c = int(DataType(self.C.dtype).bits)
        elem_bits_a = _hcu_layout_bits_for_dtype(self.A.dtype, fp4_mmac_mode=fp4_mmac_mode)
        elem_bits_b = _hcu_layout_bits_for_dtype(self.B.dtype, fp4_mmac_mode=fp4_mmac_mode)
        mmac_k_a = hcu_mmac_k_dim_for_operand(target, self.A.dtype, operand_mode=fp4_mmac_mode, use_tf32=self.use_tf32)
        mmac_k_b = hcu_mmac_k_dim_for_operand(target, self.B.dtype, operand_mode=fp4_mmac_mode, use_tf32=self.use_tf32)
        frag_c = make_gemm_fragment_hcu(
            int(self.M),
            int(self.N),
            warp_m,
            warp_n,
            warp_k,
            elem_bits_c,
            min_n_per_warp,
            lit=target_has_mmac_lit_lts(target),
        )
        out = {self.C: frag_c}
        if _is_shared_like(self.A):
            out[self.A] = make_hcu_swizzled_layout(self.A, int(self.k_pack))
        elif is_fragment(self.A):
            out[self.A] = make_gemm_fragment_a_hcu(
                int(self.M),
                int(self.N),
                int(self.K),
                warp_m,
                warp_n,
                warp_k,
                elem_bits_a,
                int(self.k_pack),
                bool(self.trans_A),
                mmac_k_dim=mmac_k_a,
            )
        else:
            raise ValueError(f"Unsupported A scope for HCU gemm: {self.A.scope()}")
        if _is_shared_like(self.B):
            out[self.B] = make_hcu_swizzled_layout(self.B, int(self.k_pack))
        elif is_fragment(self.B):
            out[self.B] = make_gemm_fragment_b_hcu(
                int(self.M),
                int(self.N),
                int(self.K),
                warp_m,
                warp_n,
                warp_k,
                elem_bits_b,
                int(self.k_pack),
                bool(self.trans_B),
                min_n_per_warp,
                mmac_k_dim=mmac_k_b,
            )
        else:
            raise ValueError(f"Unsupported B scope for HCU gemm: {self.B.scope()}")
        return out

    def _lower_blockscaled(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tirx.Var,
    ):
        """Lower FP4×FP4 + E8M0 gemm_blockscaled via mmac_scale_fp4_body."""
        _ = layout_map
        annotations = getattr(self.gemm_node, "annotations", None) or {}
        a_dtype = str(self.a_dtype)
        b_dtype = str(self.b_dtype)
        if not (_is_f8f6f4_operand_dtype(self.a_dtype) and _is_f8f6f4_operand_dtype(self.b_dtype)):
            raise ValueError(f"HCU gemm_blockscaled requires f8/f6/f4 operands, got A={a_dtype}, B={b_dtype}")
        if not target_has_mmac_lit_lts(target):
            raise ValueError("HCU gemm_blockscaled requires lit/lts target support")

        gran_m = _ann_int(annotations, "sf_a_granularity_m", 1)
        gran_ka = _ann_int(annotations, "sf_a_granularity_k")
        gran_n = _ann_int(annotations, "sf_b_granularity_n", 1)
        gran_kb = _ann_int(annotations, "sf_b_granularity_k")
        a_k_major = bool(_ann_int(annotations, "a_scale_k_major", 0))
        b_k_major = bool(_ann_int(annotations, "b_scale_k_major", 0))
        row_base_a = _ann_expr(annotations, "tl.scale_a_row_base")
        row_base_b = _ann_expr(annotations, "tl.scale_b_row_base")
        scale_format_a = _ann_int(annotations, "tl.scale_a_format")
        scale_format_b = _ann_int(annotations, "tl.scale_b_format")

        thread_nums = int(thread_bounds.extent)
        meta = _resolve_hcu_mls_meta(self.gemm_node, self.A, self.B, thread_nums, target)
        fp4_mmac_mode = meta.mmac_mode
        mmac_k = meta.mmac_k
        k_per_scale_row = 32
        if gran_ka % k_per_scale_row != 0 or gran_kb % k_per_scale_row != 0:
            raise ValueError(f"sf_*_granularity_k must be divisible by {k_per_scale_row}, got {gran_ka}, {gran_kb}")
        if gran_m >= 16 and gran_m % 16 != 0:
            raise ValueError(f"sf_a_granularity_m={gran_m} must be divisible by 16 when >= 16")
        if gran_m < 16 and 16 % gran_m != 0:
            raise ValueError(f"sf_a_granularity_m={gran_m} must divide 16 when < 16")
        if gran_n >= 16 and gran_n % 16 != 0:
            raise ValueError(f"sf_b_granularity_n={gran_n} must be divisible by 16 when >= 16")
        if gran_n < 16 and 16 % gran_n != 0:
            raise ValueError(f"sf_b_granularity_n={gran_n} must divide 16 when < 16")

        sfa = self.SFARegion.buffer
        sfb = self.SFBRegion.buffer
        scale_m, scale_ka = _parse_scale_shape(sfa.shape, a_k_major)
        scale_n, scale_kb = _parse_scale_shape(sfb.shape, b_k_major)

        a_from_mls = int(meta.a_from_mls)
        b_from_mls = int(meta.b_from_mls)
        warp_m, warp_n, warp_k, _m_per_warp, min_n_per_warp = _compute_hcu_warp_partition(self, thread_nums, target, meta)
        if warp_k != 1:
            raise ValueError("HCU gemm_blockscaled currently requires warp_k == 1")
        use_gemm_mls = (a_from_mls and not is_fragment(self.A)) or (b_from_mls and not is_fragment(self.B))
        if use_gemm_mls and int(self.k_pack) != 1:
            raise ValueError("HCU gemm_blockscaled MLS operands require k_pack == 1")

        emitter = _make_hcu_emitter(
            self,
            warp_m,
            warp_n,
            warp_k,
            target,
            thread_var,
            _thread_bounds_min(thread_bounds),
            fp4_mmac_mode=fp4_mmac_mode,
            min_n_per_warp=min_n_per_warp,
        )
        # Force full-K local layout so one call_extern covers all K atoms.
        emitter._a_full_k = True
        emitter._b_full_k = True

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion
        A_buf = A_region.buffer
        B_buf = B_region.buffer
        C_buf = C_region.buffer
        clear_accum = self.clear_accum
        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        a_local_dtype = _mls_local_dtype(a_dtype, fp4_mmac_mode=fp4_mmac_mode)
        b_local_dtype = _mls_local_dtype(b_dtype, fp4_mmac_mode=fp4_mmac_mode)
        assert is_full_region(C_region), "Fragment output C must be a full region"

        thread_binding = thread_var

        def _build_m0():
            tx, warp_n_idx, warp_m_idx = emitter.extract_thread_binding(thread_binding)
            _ = tx
            # Per-wave segment bases (non-recompute MN), matching ds_scale_copy.
            m_seg = int(emitter.block_row_warps)
            n_seg = int(emitter.block_col_warps_no_recompute)
            a_align = _scale_aligned_rows_per_wave(scale_m, scale_ka, gran_m, gran_ka, m_seg, mmac_k)
            b_align = _scale_aligned_rows_per_wave(scale_n, scale_kb, gran_n, gran_kb, n_seg, mmac_k)
            a_row = row_base_a + warp_m_idx * a_align
            # N-recompute waves share the same physical ScaleB segment.
            if emitter.block_col_warps_no_recompute != emitter.block_col_warps:
                warp_n_idx = warp_n_idx % n_seg
            b_row = row_base_b + warp_n_idx * b_align
            return (b_row << 16) | a_row

        if use_gemm_mls and a_from_mls and b_from_mls and _is_shared_like(self.A) and _is_shared_like(self.B):

            @T.prim_func
            def _gemm_bs_mls_mls() -> None:
                A_local = T.alloc_local(emitter.local_elems_a(full_k=True), a_local_dtype)
                B_local = T.alloc_local(emitter.local_elems_b(full_k=True), b_local_dtype)
                if clear_accum:
                    T.clear(C_buf)
                emitter.ldmatrix_mls_a(A_local, A_region)
                emitter.ldmatrix_mls_b(B_local, B_region)
                emitter.mmac_scale_fp4(
                    A_local,
                    B_local,
                    C_buf,
                    m0_wave_base=_build_m0(),
                    gran_m=gran_m,
                    gran_ka=gran_ka,
                    gran_n=gran_n,
                    gran_kb=gran_kb,
                    scale_shape_m=scale_m,
                    scale_shape_ka=scale_ka,
                    scale_shape_n=scale_n,
                    scale_shape_kb=scale_kb,
                    scale_format_a=scale_format_a,
                    scale_format_b=scale_format_b,
                    real_ab_type=meta.real_ab_type,
                )

            return _Simplify(_gemm_bs_mls_mls, inline_let=True)

        if use_gemm_mls and b_from_mls and is_fragment(self.A) and _is_shared_like(self.B):
            assert is_full_region(A_region), "Fragment input A must be a full region"

            @T.prim_func
            def _gemm_bs_r_mls() -> None:
                B_local = T.alloc_local(emitter.local_elems_b(full_k=True), b_local_dtype)
                if clear_accum:
                    T.clear(C_buf)
                emitter.ldmatrix_mls_b(B_local, B_region)
                emitter.mmac_scale_fp4(
                    A_buf,
                    B_local,
                    C_buf,
                    m0_wave_base=_build_m0(),
                    gran_m=gran_m,
                    gran_ka=gran_ka,
                    gran_n=gran_n,
                    gran_kb=gran_kb,
                    scale_shape_m=scale_m,
                    scale_shape_ka=scale_ka,
                    scale_shape_n=scale_n,
                    scale_shape_kb=scale_kb,
                    scale_format_a=scale_format_a,
                    scale_format_b=scale_format_b,
                    real_ab_type=meta.real_ab_type,
                )

            return _Simplify(_gemm_bs_r_mls, inline_let=True)

        if use_gemm_mls and a_from_mls and _is_shared_like(self.A) and is_fragment(self.B):
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_bs_mls_r() -> None:
                A_local = T.alloc_local(emitter.local_elems_a(full_k=True), a_local_dtype)
                if clear_accum:
                    T.clear(C_buf)
                emitter.ldmatrix_mls_a(A_local, A_region)
                emitter.mmac_scale_fp4(
                    A_local,
                    B_buf,
                    C_buf,
                    m0_wave_base=_build_m0(),
                    gran_m=gran_m,
                    gran_ka=gran_ka,
                    gran_n=gran_n,
                    gran_kb=gran_kb,
                    scale_shape_m=scale_m,
                    scale_shape_ka=scale_ka,
                    scale_shape_n=scale_n,
                    scale_shape_kb=scale_kb,
                    scale_format_a=scale_format_a,
                    scale_format_b=scale_format_b,
                    real_ab_type=meta.real_ab_type,
                )

            return _Simplify(_gemm_bs_mls_r, inline_let=True)

        if use_gemm_mls and b_from_mls and _is_shared_like(self.A) and _is_shared_like(self.B):
            inner_k = emitter.inner_k_per_warp()

            @T.prim_func
            def _gemm_bs_s_mls() -> None:
                A_local = T.alloc_local(emitter.local_elems_a(full_k=True), a_local_dtype)
                B_local = T.alloc_local(emitter.local_elems_b(full_k=True), b_local_dtype)
                if clear_accum:
                    T.clear(C_buf)
                emitter.ldmatrix_mls_b(B_local, B_region)
                for ki in T.serial(0, inner_k):
                    emitter.ldmatrix_a_fullk(A_local, A_region, ki)
                emitter.mmac_scale_fp4(
                    A_local,
                    B_local,
                    C_buf,
                    m0_wave_base=_build_m0(),
                    gran_m=gran_m,
                    gran_ka=gran_ka,
                    gran_n=gran_n,
                    gran_kb=gran_kb,
                    scale_shape_m=scale_m,
                    scale_shape_ka=scale_ka,
                    scale_shape_n=scale_n,
                    scale_shape_kb=scale_kb,
                    scale_format_a=scale_format_a,
                    scale_format_b=scale_format_b,
                    real_ab_type=meta.real_ab_type,
                )

            return _Simplify(_gemm_bs_s_mls, inline_let=True)

        if _is_shared_like(self.A) and _is_shared_like(self.B):
            inner_k = emitter.inner_k_per_warp()

            @T.prim_func
            def _gemm_bs_ss() -> None:
                A_local = T.alloc_local(emitter.local_elems_a(full_k=True), a_local_dtype)
                B_local = T.alloc_local(emitter.local_elems_b(full_k=True), b_local_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, inner_k):
                    emitter.ldmatrix_a_fullk(A_local, A_region, ki)
                    emitter.ldmatrix_b_fullk(B_local, B_region, ki)
                emitter.mmac_scale_fp4(
                    A_local,
                    B_local,
                    C_buf,
                    m0_wave_base=_build_m0(),
                    gran_m=gran_m,
                    gran_ka=gran_ka,
                    gran_n=gran_n,
                    gran_kb=gran_kb,
                    scale_shape_m=scale_m,
                    scale_shape_ka=scale_ka,
                    scale_shape_n=scale_n,
                    scale_shape_kb=scale_kb,
                    scale_format_a=scale_format_a,
                    scale_format_b=scale_format_b,
                    real_ab_type=meta.real_ab_type,
                )

            return _Simplify(_gemm_bs_ss, inline_let=True)

        if _is_shared_like(self.A) and is_fragment(self.B):
            assert is_full_region(B_region), "Fragment input B must be a full region"
            inner_k = emitter.inner_k_per_warp()

            @T.prim_func
            def _gemm_bs_sr() -> None:
                A_local = T.alloc_local(emitter.local_elems_a(full_k=True), a_local_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, inner_k):
                    emitter.ldmatrix_a_fullk(A_local, A_region, ki)
                emitter.mmac_scale_fp4(
                    A_local,
                    B_buf,
                    C_buf,
                    m0_wave_base=_build_m0(),
                    gran_m=gran_m,
                    gran_ka=gran_ka,
                    gran_n=gran_n,
                    gran_kb=gran_kb,
                    scale_shape_m=scale_m,
                    scale_shape_ka=scale_ka,
                    scale_shape_n=scale_n,
                    scale_shape_kb=scale_kb,
                    scale_format_a=scale_format_a,
                    scale_format_b=scale_format_b,
                    real_ab_type=meta.real_ab_type,
                )

            return _Simplify(_gemm_bs_sr, inline_let=True)

        if is_fragment(self.A) and _is_shared_like(self.B):
            assert is_full_region(A_region), "Fragment input A must be a full region"
            inner_k = emitter.inner_k_per_warp()

            @T.prim_func
            def _gemm_bs_rs() -> None:
                B_local = T.alloc_local(emitter.local_elems_b(full_k=True), b_local_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, inner_k):
                    emitter.ldmatrix_b_fullk(B_local, B_region, ki)
                emitter.mmac_scale_fp4(
                    A_buf,
                    B_local,
                    C_buf,
                    m0_wave_base=_build_m0(),
                    gran_m=gran_m,
                    gran_ka=gran_ka,
                    gran_n=gran_n,
                    gran_kb=gran_kb,
                    scale_shape_m=scale_m,
                    scale_shape_ka=scale_ka,
                    scale_shape_n=scale_n,
                    scale_shape_kb=scale_kb,
                    scale_format_a=scale_format_a,
                    scale_format_b=scale_format_b,
                    real_ab_type=meta.real_ab_type,
                )

            return _Simplify(_gemm_bs_rs, inline_let=True)

        if is_fragment(self.A) and is_fragment(self.B):
            assert is_full_region(A_region), "Fragment input A must be a full region"
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_bs_rr() -> None:
                if clear_accum:
                    T.clear(C_buf)
                emitter.mmac_scale_fp4(
                    A_buf,
                    B_buf,
                    C_buf,
                    m0_wave_base=_build_m0(),
                    gran_m=gran_m,
                    gran_ka=gran_ka,
                    gran_n=gran_n,
                    gran_kb=gran_kb,
                    scale_shape_m=scale_m,
                    scale_shape_ka=scale_ka,
                    scale_shape_n=scale_n,
                    scale_shape_kb=scale_kb,
                    scale_format_a=scale_format_a,
                    scale_format_b=scale_format_b,
                    real_ab_type=meta.real_ab_type,
                )

            return _Simplify(_gemm_bs_rr, inline_let=True)

        raise ValueError(f"Unsupported HCU gemm_blockscaled operand scopes: A={self.A.scope()}, B={self.B.scope()}")

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tirx.Var,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        _ = layout_map
        _ = mbar_phase_expr
        if not target_is_hcu(target):
            raise ValueError("GemmHCUMMAC lowering requires an HCU target")

        if self.is_blockscaled:
            return self._lower_blockscaled(layout_map, target, thread_bounds, thread_var)

        thread_nums = int(thread_bounds.extent)
        meta = _resolve_hcu_mls_meta(self.gemm_node, self.A, self.B, thread_nums, target)
        a_from_mls = int(meta.a_from_mls)
        b_from_mls = int(meta.b_from_mls)
        fp4_mmac_mode = meta.mmac_mode
        warp_m, warp_n, warp_k, _m_per_warp, min_n_per_warp = _compute_hcu_warp_partition(self, thread_nums, target, meta)
        emitter = _make_hcu_emitter(
            self,
            warp_m,
            warp_n,
            warp_k,
            target,
            thread_var,
            _thread_bounds_min(thread_bounds),
            fp4_mmac_mode=fp4_mmac_mode,
            min_n_per_warp=min_n_per_warp,
        )

        k_pack = emitter.k_pack
        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        a_mls_local_dtype = _mls_local_dtype(a_dtype, fp4_mmac_mode=fp4_mmac_mode, use_tf32=self.use_tf32)
        b_mls_local_dtype = _mls_local_dtype(b_dtype, fp4_mmac_mode=fp4_mmac_mode, use_tf32=self.use_tf32)
        inner_k = emitter.inner_k_per_warp()
        block_K = emitter.chunk
        micro_size_k = emitter.micro_size_k
        local_size_a = emitter.local_size_a
        local_size_b = emitter.local_size_b

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion
        A_buf = A_region.buffer
        B_buf = B_region.buffer
        C_buf = C_region.buffer
        clear_accum = self.clear_accum

        assert block_K >= micro_size_k * k_pack, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k}) * k_pack ({k_pack})"
        assert block_K % (micro_size_k * k_pack) == 0, (
            f"block_K ({block_K}) must be divisible by micro_size_k ({micro_size_k}) * k_pack ({k_pack})"
        )
        assert is_full_region(C_region), "Fragment output C must be a full region"

        use_gemm_mls = (a_from_mls and not is_fragment(self.A)) or (b_from_mls and not is_fragment(self.B))
        if use_gemm_mls:
            if k_pack != 1:
                raise ValueError("gemm_mls does not support kPack > 1")
            if warp_k != 1:
                raise ValueError("gemm_mls does not support warp on K")

        if use_gemm_mls and a_from_mls and b_from_mls:

            @T.prim_func
            def _gemm_mls_mls() -> None:
                A_local = T.alloc_local(emitter.local_elems_a(full_k=True), a_mls_local_dtype)
                B_local = T.alloc_local(emitter.local_elems_b(full_k=True), b_mls_local_dtype)
                if clear_accum:
                    T.clear(C_buf)
                emitter.ldmatrix_mls_a(A_local, A_region)
                emitter.ldmatrix_mls_b(B_local, B_region)
                for ki in T.serial(0, inner_k):
                    emitter.mmac(A_local, B_local, C_buf, ki)

            return _Simplify(_gemm_mls_mls, inline_let=True)

        elif use_gemm_mls and b_from_mls:
            if a_from_mls:
                raise NotImplementedError("gemm_mls_s (A mls, B r or s) not implemented")

            if is_fragment(self.A):
                assert is_full_region(A_region), "Fragment input A must be a full region"

                @T.prim_func
                def _gemm_r_mls() -> None:
                    B_local = T.alloc_local(emitter.local_elems_b(full_k=True), b_mls_local_dtype)
                    if clear_accum:
                        T.clear(C_buf)
                    emitter.ldmatrix_mls_b(B_local, B_region)
                    for ki in T.serial(0, inner_k):
                        emitter.mmac(A_buf, B_local, C_buf, ki)

                return _Simplify(_gemm_r_mls, inline_let=True)

            elif _is_shared_like(self.A):

                @T.prim_func
                def _gemm_s_mls() -> None:
                    A_local = T.alloc_local(emitter.local_elems_a(), a_dtype)
                    B_local = T.alloc_local(emitter.local_elems_b(full_k=True), b_mls_local_dtype)
                    if clear_accum:
                        T.clear(C_buf)
                    emitter.ldmatrix_mls_b(B_local, B_region)
                    for ki in T.serial(0, inner_k):
                        emitter.ldmatrix_a(A_local, A_region, ki)
                        emitter.mmac(A_local, B_local, C_buf, ki)

                return _Simplify(_gemm_s_mls, inline_let=True)

            raise ValueError(f"Unsupported A scope for HCU gemm_mls: {self.A.scope()}")

        elif _is_shared_like(self.A) and _is_shared_like(self.B):

            @T.prim_func
            def _gemm_ss() -> None:
                A_local = T.alloc_local(emitter.warp_rows * local_size_a * k_pack, a_dtype)
                B_local = T.alloc_local(emitter.warp_cols * local_size_b * k_pack, b_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, inner_k):
                    emitter.ldmatrix_a(A_local, A_region, ki)
                    emitter.ldmatrix_b(B_local, B_region, ki)
                    emitter.mmac(A_local, B_local, C_buf, ki)

            return _Simplify(_gemm_ss, inline_let=True)

        elif _is_shared_like(self.A) and is_fragment(self.B):
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_sr() -> None:
                A_local = T.alloc_local(emitter.warp_rows * local_size_a * k_pack, a_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, inner_k):
                    emitter.ldmatrix_a(A_local, A_region, ki)
                    emitter.mmac(A_local, B_buf, C_buf, ki)

            return _Simplify(_gemm_sr, inline_let=True)

        elif is_fragment(self.A) and _is_shared_like(self.B):
            assert is_full_region(A_region), "Fragment input A must be a full region"

            @T.prim_func
            def _gemm_rs() -> None:
                B_local = T.alloc_local(emitter.warp_cols * local_size_b * k_pack, b_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, inner_k):
                    emitter.ldmatrix_b(B_local, B_region, ki)
                    emitter.mmac(A_buf, B_local, C_buf, ki)

            return _Simplify(_gemm_rs, inline_let=True)

        elif is_fragment(self.A) and is_fragment(self.B):
            assert is_full_region(A_region), "Fragment input A must be a full region"
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_rr() -> None:
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, inner_k):
                    emitter.mmac(A_buf, B_buf, C_buf, ki)

            return _Simplify(_gemm_rr, inline_let=True)

        raise ValueError(f"Unsupported HCU gemm operand scopes: A={self.A.scope()}, B={self.B.scope()}")

    def is_gemm_ss(self) -> bool:
        return _is_shared_like(self.A) and _is_shared_like(self.B)

    def is_gemm_sr(self) -> bool:
        return _is_shared_like(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and _is_shared_like(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)
