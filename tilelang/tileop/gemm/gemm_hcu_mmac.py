from __future__ import annotations

from types import SimpleNamespace

from tilelang import _ffi_api
from tilelang import language as T
from tilelang.intrinsics.hcu_mmac_layout import (
    make_gemm_fragment_a_hcu,
    make_gemm_fragment_b_hcu,
    make_gemm_fragment_hcu,
)
from tilelang.intrinsics.hcu_mmac_macro_generator import HCUMatrixCoreIntrinEmitter
from tilelang.layout.swizzle import make_hcu_swizzled_layout
from tilelang.transform.simplify import _Simplify
from tilelang.utils.language import is_fragment, is_full_region, is_shared, is_shared_dynamic
from tilelang.utils.target import hcu_mmac_k_dim, target_has_mmac_lit_lts, target_is_hcu
from tvm import DataType, tir
from tvm.ir import Range
from tvm.target import Target

from .gemm_base import GemmBase
from .inst import GemmInst


def _int_annotation(annotations, key: str, default: int = 0) -> int:
    if not annotations:
        return default
    try:
        val = annotations[key]
    except (KeyError, TypeError):
        return default
    if isinstance(val, tir.IntImm):
        return int(val.value)
    if isinstance(val, (int, bool)):
        return int(val)
    attr_val = getattr(val, "value", None)
    if attr_val is not None:
        return int(attr_val)
    return default


def _is_shared_like(buf) -> bool:
    return is_shared(buf) or is_shared_dynamic(buf)


def _mls_block_dims(shape, mls_trans: bool) -> tuple[int, int]:
    if mls_trans:
        return int(shape[-2]), int(shape[-1])
    return int(shape[-1]), int(shape[-2])


def _compute_mls_tiles(trans: bool, block_mn: int, block_k: int, block_size: int, target: Target, elem_bits: int) -> tuple[int, int]:
    warp_mn, warp_k, tile_mn, tile_k = _ffi_api.ComputeMlsWarpPartition(
        bool(trans), int(block_mn), int(block_k), int(block_size), target, int(elem_bits)
    )
    _ = warp_mn
    _ = warp_k
    return int(tile_mn), int(tile_k)


def _resolve_hcu_mls_meta(gemm_node, A, B, block_size: int, target: Target):
    """Read AnnotateMlsGemmDep flags from gemm.annotations and derive MLS tiles."""
    annotations = getattr(gemm_node, "annotations", None) or {}
    a_from_mls = _int_annotation(annotations, "tl.a_from_mls")
    b_from_mls = _int_annotation(annotations, "tl.b_from_mls")
    trans_a = bool(gemm_node.transA)
    trans_b = bool(gemm_node.transB)
    a_mls_trans = not trans_a
    b_mls_trans = trans_b

    mls_tile_m = mls_tile_ka = mls_tile_n = mls_tile_kb = -1
    if a_from_mls:
        block_mn, block_k = _mls_block_dims(A.shape, a_mls_trans)
        mls_tile_m, mls_tile_ka = _compute_mls_tiles(a_mls_trans, block_mn, block_k, block_size, target, DataType(A.dtype).bits)
    if b_from_mls:
        block_mn, block_k = _mls_block_dims(B.shape, b_mls_trans)
        mls_tile_n, mls_tile_kb = _compute_mls_tiles(b_mls_trans, block_mn, block_k, block_size, target, DataType(B.dtype).bits)

    return SimpleNamespace(
        a_from_mls=a_from_mls,
        b_from_mls=b_from_mls,
        a_mls_trans=int(a_mls_trans),
        b_mls_trans=int(b_mls_trans),
        mls_tile_m=mls_tile_m,
        mls_tile_ka=mls_tile_ka,
        mls_tile_n=mls_tile_n,
        mls_tile_kb=mls_tile_kb,
    )


def _compute_hcu_warp_partition(gemm, thread_nums: int, target: Target, meta) -> tuple[int, int, int]:
    element_byte_size = DataType(gemm.in_dtype).bits // 8
    _ffi_api.GemmWarpPolicyComputeWarpPartitionHCU(
        gemm.policy,
        int(gemm.M),
        int(gemm.N),
        int(gemm.K),
        int(gemm.k_pack),
        int(element_byte_size),
        int(thread_nums),
        target,
        int(GemmInst.HCUMMAC),
        bool(meta.a_from_mls),
        bool(meta.b_from_mls),
        bool(meta.a_mls_trans),
        bool(meta.b_mls_trans),
    )
    return int(gemm.policy.m_warp), int(gemm.policy.n_warp), int(gemm.policy.k_warp)


def _make_hcu_emitter(
    gemm: GemmHCUMMAC,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    meta,
    target: Target,
    thread_var: tir.Var,
) -> HCUMatrixCoreIntrinEmitter:
    min_n_per_warp = 32 if (meta.b_from_mls and not meta.b_mls_trans) else 16
    return HCUMatrixCoreIntrinEmitter(
        a_dtype=gemm.in_dtype,
        b_dtype=gemm.in_dtype,
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
        min_n_per_warp=min_n_per_warp,
        use_tf32=gemm.use_tf32,
    )


class GemmHCUMMAC(GemmBase):
    """HCU matrix core GEMM: layout and lowering via Python ``HCUMatrixCoreIntrinEmitter``."""

    def infer_layout(self, target: Target, thread_nums: int):
        if not target_is_hcu(target):
            raise ValueError("GemmHCUMMAC requires an HCU target")
        meta = _resolve_hcu_mls_meta(self.gemm_node, self.A, self.B, int(thread_nums), target)
        b_from_mls = int(meta.b_from_mls)
        b_mls_trans = bool(meta.b_mls_trans)
        block_size = int(thread_nums)
        warp_m, warp_n, warp_k = _compute_hcu_warp_partition(self, block_size, target, meta)
        min_n_per_warp = 32 if (b_from_mls and not b_mls_trans) else 16
        elem_bits_c = int(DataType(self.C.dtype).bits)
        elem_bits_a = int(DataType(self.A.dtype).bits)
        elem_bits_b = int(DataType(self.B.dtype).bits)
        mmac_k_a = hcu_mmac_k_dim(target, elem_bits_a, use_tf32=self.use_tf32)
        mmac_k_b = hcu_mmac_k_dim(target, elem_bits_b, use_tf32=self.use_tf32)
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

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tir.Var,
        mbar_phase_expr: tir.PrimExpr | None = None,
    ):
        _ = layout_map
        _ = mbar_phase_expr
        if not target_is_hcu(target):
            raise ValueError("GemmHCUMMAC lowering requires an HCU target")

        thread_nums = int(thread_bounds.extent)
        meta = _resolve_hcu_mls_meta(self.gemm_node, self.A, self.B, thread_nums, target)
        a_from_mls = int(meta.a_from_mls)
        b_from_mls = int(meta.b_from_mls)
        warp_m, warp_n, warp_k = _compute_hcu_warp_partition(self, thread_nums, target, meta)
        emitter = _make_hcu_emitter(self, warp_m, warp_n, warp_k, meta, target, thread_var)

        k_pack = emitter.k_pack
        in_dtype = self.in_dtype
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
            if self.use_tf32:
                raise ValueError("HCU gemm: use_tf32=True is not supported for gemm_mls (MLS) path")

        if use_gemm_mls and a_from_mls and b_from_mls:

            @T.prim_func
            def _gemm_mls_mls() -> None:
                A_local = T.alloc_local(emitter.local_elems_a(full_k=True), in_dtype)
                B_local = T.alloc_local(emitter.local_elems_b(full_k=True), in_dtype)
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
                    B_local = T.alloc_local(emitter.local_elems_b(full_k=True), in_dtype)
                    if clear_accum:
                        T.clear(C_buf)
                    emitter.ldmatrix_mls_b(B_local, B_region)
                    for ki in T.serial(0, inner_k):
                        emitter.mmac(A_buf, B_local, C_buf, ki)

                return _Simplify(_gemm_r_mls, inline_let=True)

            elif _is_shared_like(self.A):

                @T.prim_func
                def _gemm_s_mls() -> None:
                    A_local = T.alloc_local(emitter.local_elems_a(), in_dtype)
                    B_local = T.alloc_local(emitter.local_elems_b(full_k=True), in_dtype)
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
                A_local = T.alloc_local(emitter.warp_rows * local_size_a * k_pack, in_dtype)
                B_local = T.alloc_local(emitter.warp_cols * local_size_b * k_pack, in_dtype)
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
                A_local = T.alloc_local(emitter.warp_rows * local_size_a * k_pack, in_dtype)
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
                B_local = T.alloc_local(emitter.warp_cols * local_size_b * k_pack, in_dtype)
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
