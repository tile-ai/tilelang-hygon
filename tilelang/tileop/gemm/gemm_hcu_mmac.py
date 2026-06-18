from __future__ import annotations

from types import SimpleNamespace

from tilelang import _ffi_api
from tilelang import language as T
from tilelang.layout.swizzle import make_hcu_swizzled_layout
from tilelang.transform.simplify import _Simplify
from tilelang.utils.language import is_fragment, is_shared, is_shared_dynamic, retrieve_ptr
from tilelang.utils.target import get_hcu_arch_string, target_has_mmac_lit_lts, target_is_hcu
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


def _hcu_mls_ab_dtype_str(dtype) -> str:
    """Map TIR / buffer dtype to tl C++ type names for MLS templates.

    Note: ``from tvm import DataType`` is the tvm_ffi lightweight ``dtype`` object
    (not the legacy IR helper with ``is_bfloat16()``), so classify by string.
    """
    s = str(dtype).lower()
    if "e4m3" in s:
        return "fp8_e4_t"
    if "e5m2" in s:
        return "fp8_e5_t"
    if "bfloat16" in s or s == "bf16":
        return "bfloat16_t"
    if ("float16" in s or s in ("fp16", "half")) and "float8" not in s:
        return "half_t"
    raise ValueError(f"gemm_mls unsupported dtype for HCU template: {dtype}")


def _clear_accum_template_flag(expr) -> str:
    if isinstance(expr, tir.IntImm):
        if expr.dtype == "bool":
            return "true" if expr.value else "false"
        return "true" if expr.value != 0 else "false"
    raise ValueError(f"clear_accum must be a constant for HCU tl_gemm, got {expr}")


class GemmHCUMMAC(GemmBase):
    """HCU matrix core GEMM: layout in Python, lowering via ``tl.tl_gemm`` + C++ templates."""

    def infer_layout(self, target: Target, thread_nums: int):
        if not target_is_hcu(target):
            raise ValueError("GemmHCUMMAC requires an HCU target")
        meta = _resolve_hcu_mls_meta(self.gemm_node, self.A, self.B, int(thread_nums), target)
        a_from_mls = int(meta.a_from_mls)
        b_from_mls = int(meta.b_from_mls)
        a_mls_trans = bool(meta.a_mls_trans)
        b_mls_trans = bool(meta.b_mls_trans)
        block_size = int(thread_nums)
        element_byte_size = DataType(self.in_dtype).bits // 8
        _ffi_api.GemmWarpPolicyComputeWarpPartitionHCU(
            self.policy,
            int(self.M),
            int(self.N),
            int(self.K),
            int(self.k_pack),
            int(element_byte_size),
            int(block_size),
            target,
            int(GemmInst.HCUMMAC),
            bool(a_from_mls),
            bool(b_from_mls),
            bool(a_mls_trans),
            bool(b_mls_trans),
        )
        warp_m = int(self.policy.m_warp)
        warp_n = int(self.policy.n_warp)
        warp_k = int(self.policy.k_warp)
        min_n_per_warp = 32 if (b_from_mls and not b_mls_trans) else 16
        elem_bits_c = int(DataType(self.C.dtype).bits)
        if target_has_mmac_lit_lts(target):
            frag_c = _ffi_api.make_gemm_fragment_hcu_lit(int(self.M), int(self.N), warp_m, warp_n, warp_k, elem_bits_c, min_n_per_warp)
        else:
            frag_c = _ffi_api.make_gemm_fragment_hcu(int(self.M), int(self.N), warp_m, warp_n, warp_k, elem_bits_c, min_n_per_warp)
        out = {self.C: frag_c}
        if _is_shared_like(self.A):
            out[self.A] = make_hcu_swizzled_layout(self.A, int(self.k_pack))
        elif is_fragment(self.A):
            out[self.A] = _ffi_api.make_gemm_fragment_a_hcu(
                int(self.M),
                int(self.N),
                int(self.K),
                warp_m,
                warp_n,
                warp_k,
                int(DataType(self.A.dtype).bits),
                int(self.k_pack),
                bool(self.trans_A),
            )
        else:
            raise ValueError(f"Unsupported A scope for HCU gemm: {self.A.scope()}")
        if _is_shared_like(self.B):
            out[self.B] = make_hcu_swizzled_layout(self.B, int(self.k_pack))
        elif is_fragment(self.B):
            out[self.B] = _ffi_api.make_gemm_fragment_b_hcu(
                int(self.M),
                int(self.N),
                int(self.K),
                warp_m,
                warp_n,
                warp_k,
                int(DataType(self.B.dtype).bits),
                int(self.k_pack),
                bool(self.trans_B),
                min_n_per_warp,
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
        _ = thread_var
        _ = mbar_phase_expr
        if not target_is_hcu(target):
            raise ValueError("GemmHCUMMAC lowering requires an HCU target")
        thread_nums = int(thread_bounds.extent)
        meta = _resolve_hcu_mls_meta(self.gemm_node, self.A, self.B, thread_nums, target)
        a_from_mls = int(meta.a_from_mls)
        b_from_mls = int(meta.b_from_mls)
        b_mls_trans = bool(meta.b_mls_trans)
        element_byte_size = DataType(self.in_dtype).bits // 8
        _ffi_api.GemmWarpPolicyComputeWarpPartitionHCU(
            self.policy,
            int(self.M),
            int(self.N),
            int(self.K),
            int(self.k_pack),
            int(element_byte_size),
            int(thread_nums),
            target,
            int(GemmInst.HCUMMAC),
            bool(a_from_mls),
            bool(b_from_mls),
            bool(meta.a_mls_trans),
            bool(b_mls_trans),
        )
        warp_m = int(self.policy.m_warp)
        warp_n = int(self.policy.n_warp)
        warp_k = int(self.policy.k_warp)

        use_gemm_mls = (a_from_mls and not is_fragment(self.A)) or (b_from_mls and not is_fragment(self.B))
        if use_gemm_mls:
            if int(self.k_pack) != 1:
                raise ValueError("gemm_mls does not support kPack > 1")
            if warp_k != 1:
                raise ValueError("gemm_mls does not support warp on K")

        if use_gemm_mls:
            if a_from_mls and b_from_mls:
                op_name = "tl::gemm_mls_mls"
            elif b_from_mls:
                op_name = "tl::gemm_r_mls" if is_fragment(self.A) else "tl::gemm_s_mls"
            elif a_from_mls:
                raise NotImplementedError("gemm_mls_s (A mls, B r or s) not implemented")
            else:
                raise RuntimeError("internal: use_gemm_mls but no MLS inputs")
        elif is_fragment(self.A):
            op_name = "tl::gemm_rr" if is_fragment(self.B) else "tl::gemm_rs"
        elif is_fragment(self.B):
            op_name = "tl::gemm_sr"
        elif _is_shared_like(self.A) and _is_shared_like(self.B):
            op_name = "tl::gemm_ss"
        else:
            raise ValueError("Unsupported HCU gemm operand scopes for tl_gemm")

        parts: list[str] = []
        parts.append(f"{op_name}<{self.M}, {self.N}, {self.K}, {warp_m}, {warp_n}, ")
        if warp_k > 1:
            parts.append(f"tl::WarpKParam<{warp_k}>, ")
        parts.append(f"{int(self.trans_A)}, {int(self.trans_B)}")

        if use_gemm_mls:
            tm = tn = tka = tkb = 0
            if a_from_mls:
                tm = int(meta.mls_tile_m)
                tka = int(meta.mls_tile_ka)
            if b_from_mls:
                tn = int(meta.mls_tile_n)
                tkb = int(meta.mls_tile_kb)
            parts.append(f", {int(self.k_pack)}")
            if a_from_mls and b_from_mls:
                parts.append(f", tl::sequence<{tm}, {tka}>, tl::sequence<{tn}, {tkb}>, 1, 1")
            elif b_from_mls:
                parts.append(f", tl::sequence<{tn}, {tkb}>, 1")
            else:
                parts.append(f", tl::sequence<{tm}, {tka}>, tl::sequence<0, 0>, 1")
            parts.append(", ")
            parts.append(_hcu_mls_ab_dtype_str(self.A.dtype))
            parts.append(", ")
            parts.append(_hcu_mls_ab_dtype_str(self.B.dtype))
            parts.append(", float, float, tl::hcu_target_enum::")
            parts.append(get_hcu_arch_string(target))
        else:
            parts.append(", ")
            parts.append(_clear_accum_template_flag(self.clear_accum))
            parts.append(f", {int(self.k_pack)}")
            if b_from_mls and not b_mls_trans:
                parts.append(", 32")
            if int(self.wg_wait) != 0:
                raise ValueError("wg_wait must be 0 for HCU gemm")

        use_tf32 = self.use_tf32
        if use_tf32:
            if use_gemm_mls:
                raise ValueError("HCU gemm: use_tf32=True is not supported for gemm_mls (MLS) path")
            if self.A.dtype != T.float32 or self.B.dtype != T.float32:
                raise ValueError(
                    f"HCU gemm: use_tf32=True requires float32 A and B dtypes, got A.dtype={self.A.dtype}, B.dtype={self.B.dtype}"
                )

        parts.append(">")
        template = "".join(parts)

        Aptr = retrieve_ptr(self.ARegion, "r")
        Bptr = retrieve_ptr(self.BRegion, "r")
        Cptr = retrieve_ptr(self.CRegion, "rw")
        gemm_op = tir.op.Op.get("tl.tl_gemm")

        @T.prim_func
        def _hcu_tl_gemm() -> None:
            tf32_ann = {"tl.hcu_tf32_ab": 1} if use_tf32 else None
            T.call_intrin(
                "handle",
                gemm_op,
                template,
                Aptr,
                Bptr,
                Cptr,
                annotations=tf32_ann,
            )

        return _Simplify(_hcu_tl_gemm, inline_let=True)
