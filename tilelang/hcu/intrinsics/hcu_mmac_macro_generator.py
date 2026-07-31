from __future__ import annotations


import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.hcu.intrinsics.hcu_mmac_emitter_utils import (
    block_col_warps_no_recompute,
    build_ds_read_format_tensor_a_template,
    build_ds_read_format_tensor_b_template,
    check_mls_slice_aligned_to_tile,
    compute_mls_tiles,
    elem_bits,
    hcu_mls_ds_read_dtype_str,
    min_n_per_warp_for_b,
    mls_block_mn_k_from_region,
    mls_full_and_read_mn_k,
    retrieve_mls_lds_base_ptr,
)
from tilelang.language.utils import get_buffer_region_from_load
from tilelang.utils.language import is_fragment, retrieve_ptr
from tilelang.hcu.intrinsics.hcu_mmac_emitter_utils import hcu_mmac_k_dim
from tilelang.backend.target import determine_target
from tilelang.hcu.target import target_has_mmac_lit_lts
from tvm import DataType, tirx
from tvm.ir import Range
from tvm.runtime import convert
from tvm.target import Target
from tvm.tirx import Buffer, BufferLoad, BufferRegion, PrimExpr, Var

lift = convert


def _normalize_dtype_str(dtype) -> str:
    """Return a plain Python ``str`` dtype name.

    ``tvm.DataType`` / ``tvm_ffi`` dtype subclasses ``str``, so ``isinstance(x, str)``
    is True but the object is still a ``DataType``. Passing that into ``tirx.Call``
    args fails (``Array[index N: DataType]``). Always coerce to a builtin ``str``.
    """
    if type(dtype) is str:
        return dtype
    try:
        return str(DataType(dtype))
    except Exception:
        s = str(dtype)
        if s.startswith("dtype(") and "'" in s:
            return s.split("'")[1]
        return s


class HCUMatrixCoreIntrinEmitter:
    """
    HCU MMAC macro emitter: swizzle ``ldmatrix_*``, MLS ``ldmatrix_mls_*`` (C
    ``ds_read_format_tensor_*`` into 1D ``local``), and ``mmac``.

    Kernel-side usage (emitter holds tile/warp/transpose config)::

        emitter = HCUMatrixCoreIntrinEmitter(
            ..., block_m=M, block_n=N, block_col_warps=2, b_transposed=True
        )
        A_local = T.alloc_local(emitter.local_elems_a(full_k=True), dtype)
        B_local = T.alloc_local(emitter.local_elems_b(full_k=True), dtype)
        emitter.ldmatrix_mls_b(B_local, B_shared[ko % num_stages, :, :])
        for ki in T.serial(emitter.inner_k_per_warp()):
            emitter.ldmatrix_a(A_local, A_shared[ko % num_stages, :, :], ki)
            emitter.mmac(A_local, B_local, C_local, ki)
    """

    M_DIM = 16
    N_DIM = 16
    WARP_SIZE = 64
    dtype_abbrv = {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "int8": "int8",
        "int32": "int32",
        "float8_e4m3": "e4m3",
        "float8_e5m2": "e5m2",
        "float8_e4m3fnuz": "e4m3fnuz",
        "float8_e4m3fn": "e4m3fn",
        "float4_e2m1fn": "fp4",
    }

    k_pack = 1
    # Represent the thread binding in the form of (tx, warp_n, warp_m)
    is_m_first = False

    def __init__(
        self,
        a_dtype: str = "float16",
        b_dtype: str = "float16",
        accum_dtype: str = "float16",
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        block_m: int = 0,
        block_n: int = 0,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        k_pack: int | None = None,
        is_m_first: bool | None = False,
        b_preshuffle: bool | None = False,
        block_k_warps: int = 1,
        target: Target | None = None,
        thread_var: Var | None = None,
        thread_bounds_min: int = 0,
        min_n_per_warp: int | None = None,
        use_tf32: bool = False,
    ):
        self.a_dtype = _normalize_dtype_str(a_dtype)
        self.b_dtype = _normalize_dtype_str(b_dtype)
        self.accum_dtype = _normalize_dtype_str(accum_dtype)
        self.a_transposed = a_transposed
        self.b_transposed = b_transposed
        self.block_row_warps = block_row_warps
        self.block_col_warps = block_col_warps
        self.chunk = chunk
        self.block_k_warps = int(block_k_warps)
        self.thread_var = thread_var
        self.thread_bounds_min = int(thread_bounds_min)
        self.use_tf32 = bool(use_tf32)
        if self.use_tf32:
            if a_dtype != "float32" or b_dtype != "float32":
                raise ValueError("use_tf32=True requires float32 A and B dtypes")
            if accum_dtype != "float32":
                raise ValueError("use_tf32=True requires float32 accumulator dtype")
        if target is None:
            target = Target(determine_target("auto", return_object=True))
        self.target = target
        self._initialize_k_dim(a_dtype)
        self._initialize_abbrev(self.a_dtype, self.b_dtype, self.accum_dtype)
        self._initialize_local_size(self.M_DIM, self.N_DIM, self.k_dim, self.WARP_SIZE)
        self._initialize_mmac_prefix(self.k_dim)
        self._initialize_micro_size(self.M_DIM, self.N_DIM, self.k_dim)
        self._initialize_k_pack(k_pack)
        self._initialize_is_m_first(is_m_first)
        self._initialize_b_preshuffle(b_preshuffle)

        if min_n_per_warp is None:
            min_n_per_warp = min_n_per_warp_for_b(b_mls=False, b_mls_trans=b_transposed)
        self.min_n_per_warp = int(min_n_per_warp)
        self._initialize_block_warp_tiles(block_m, block_n)

        self.reduce_k = reduce_k
        self.threads = self.WARP_SIZE * block_row_warps * block_col_warps * self.block_k_warps * reduce_k
        self.num_elems_per_byte = num_elems_per_byte
        self._a_full_k = False
        self._b_full_k = False

    def _initialize_block_warp_tiles(self, block_m: int, block_n: int) -> None:
        """Derive per-warp tile sizes from block tile and N-side recompute."""
        if block_m <= 0 or block_n <= 0:
            raise ValueError(f"block_m and block_n must be positive, got block_m={block_m}, block_n={block_n}")
        self.block_m = int(block_m)
        self.block_n = int(block_n)
        self.block_col_warps_no_recompute = block_col_warps_no_recompute(self.block_n, self.block_col_warps, self.min_n_per_warp)
        self.warp_row_tiles = self.block_m // self.block_row_warps
        self.warp_col_tiles = self.block_n // self.block_col_warps_no_recompute
        self.warp_rows = self.block_m // (self.block_row_warps * self.micro_size_x)
        self.warp_cols = self.block_n // (self.block_col_warps_no_recompute * self.micro_size_y)

    def _initialize_k_dim(self, a_dtype="float16"):
        if isinstance(a_dtype, str):
            a_dtype = DataType(a_dtype)
        self.k_dim = hcu_mmac_k_dim(self.target, a_dtype.bits, use_tf32=self.use_tf32)

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16, warp_size=32):
        self.local_size_a = (m_dim * k_dim) // warp_size
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

    def _mmac_ab_operand(self, scalar_dtype: str, local_size: int) -> tuple[str, int]:
        """Map scalar local/fragment layout to mmac operand dtype and index divisor.

        ``local_size`` is the per-thread scalar element count (same as ``local_size_a/b``).
        TF32 reinterprets float32 fragment lanes as ``int32`` and uses the same packed
        vector indexing as ``float32x{local_size}`` (``index_div = local_size``).
        """
        if self.use_tf32:
            scalar_dtype = "int32"
        scalar_dtype = str(scalar_dtype)
        if local_size == 1:
            return scalar_dtype, 1
        return f"{scalar_dtype}x{local_size}", local_size

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self.dtype_abbrv[a_dtype]
        self.b_dtype_abbrv = self.dtype_abbrv[b_dtype]
        self.accum_dtype_abbrv = self.dtype_abbrv[accum_dtype]

    def _initialize_mmac_prefix(self, k_dim=16):
        in_dtype, out_dtype = self.a_dtype, self.accum_dtype
        M_DIM, N_DIM = self.M_DIM, self.N_DIM
        use_tf32 = self.use_tf32
        target = self.target
        has_lit = target is not None and target_has_mmac_lit_lts(target)

        if in_dtype == "float32" and not use_tf32:
            suffix = f"16x16x{self.k_dim}_f32"
            if has_lit:
                suffix += "_lit_lts"
            self.mmac_suffix = suffix
            return

        out_dtype_abbrv = {"float16": "f16", "float32": "f32", "int8": "i8", "int32": "i32"}[out_dtype]

        in_dtype_abbrv = {
            "float16": "f16",
            "bfloat16": "bf16",
            "float32": "f32",
            "int8": "i8",
            "int32": "i32",
            "float8_e4m3": "fp8",
            "float8_e4m3fnuz": "fp8",
            "float8_e4m3fn": "fp8",
            "float8_e5m2": "bf8",
            "float4_e2m1fn": "fp4",
        }[in_dtype]

        target = self.target
        if use_tf32:
            in_abbr = "tf32"
        else:
            in_abbr = in_dtype_abbrv

        if target is not None and target_has_mmac_lit_lts(target):
            if in_abbr == "fp8":
                self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_fp8_fp8_lit_lts"
            elif in_abbr == "bf8":
                self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_bf8_bf8_lit_lts"
            elif in_abbr == "fp4":
                if out_dtype_abbrv != "f32":
                    raise AssertionError("HCU fp4 MMAC currently only supports float32 accum")
                self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_fp4_lit_lts"
            elif in_abbr == "i8":
                self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_i8_lit_clamp_lts"
            elif in_abbr == "tf32":
                self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_tf32_lit_lts"
            elif in_abbr == "f16":
                self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_f16_lit_lts"
            elif in_abbr == "bf16":
                self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_bf16_lit_lts"
            elif in_abbr == "f32":
                self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_f32_lit_lts"
            else:
                raise AssertionError(f"Unsupported in_abbr = {in_abbr}")
        else:
            if in_abbr == "fp8":
                self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_fp8_fp8"
            elif in_abbr == "bf8":
                self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_bf8_bf8"
            elif in_abbr == "fp4":
                raise AssertionError("HCU fp4 MMAC requires lit/lts target support")
            elif in_abbr == "i8":
                self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_i8"
            elif in_abbr == "tf32":
                self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_tf32"
            elif in_abbr == "f16":
                self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_f16"
            elif in_abbr == "bf16":
                self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_bf16"
            elif in_abbr == "f32":
                self.mmac_suffix = f"{out_dtype_abbrv}_{M_DIM}x{N_DIM}x{k_dim}_f32"
            else:
                raise AssertionError(f"Unsupported in_abbr = {in_abbr}")

    def _initialize_micro_size(self, m_dim=16, n_dim=16, k_dim=16):
        self.micro_size_x = m_dim
        self.micro_size_y = n_dim
        self.micro_size_k = k_dim

    def _initialize_k_pack(self, k_pack: int | None = None):
        if k_pack is not None:
            self.k_pack = k_pack

    def _initialize_is_m_first(self, is_m_first: bool | None = False):
        if is_m_first is None:
            is_m_first = False
        if is_m_first:
            raise ValueError("HCU MMAC emitter: is_m_first=True is not supported; warp order is fixed M->N->K to match layout inference")
        self.is_m_first = False

    def _initialize_b_preshuffle(self, b_preshuffle: bool | None = False):
        if b_preshuffle is not None:
            self.b_preshuffle = b_preshuffle

    @staticmethod
    def _legalize_to_buffer_region(obj: Buffer | BufferLoad | BufferRegion) -> BufferRegion:
        if isinstance(obj, BufferRegion):
            return obj
        if isinstance(obj, Buffer):
            mins = [tirx.IntImm("int32", 0) for _ in obj.shape]
            ranges = [Range.from_min_extent(m, e) for m, e in zip(mins, obj.shape)]
            return BufferRegion(obj, ranges)
        if isinstance(obj, BufferLoad):
            region = get_buffer_region_from_load(obj)
            if region is not None:
                return region
            mins = [idx for idx in obj.indices]
            ones = [tirx.IntImm("int32", 1) for _ in obj.indices]
            ranges = [Range.from_min_extent(m, e) for m, e in zip(mins, ones)]
            return BufferRegion(obj.buffer, ranges)
        raise ValueError(f"Unsupported argument type for BufferRegion: {type(obj)}")

    def get_thread_binding(self):
        if self.thread_var is None:
            current_frame = T.KernelLaunchFrame.Current()
            assert current_frame is not None, "Must be called in a T.Kernel frame"
            return current_frame.get_thread_binding()
        return self.thread_var

    def scoped_warp_id_offset(self) -> int:
        if self.thread_bounds_min <= 0:
            return 0
        if self.thread_bounds_min % self.WARP_SIZE != 0:
            raise ValueError(f"HCU gemm scoped lowering requires thread_bounds.min to be warp-aligned, got min={self.thread_bounds_min}")
        return self.thread_bounds_min // self.WARP_SIZE

    def scoped_thread_id(self, thread_id: PrimExpr) -> PrimExpr:
        if self.thread_bounds_min > 0:
            return thread_id - self.thread_bounds_min
        return thread_id

    def inner_k_per_warp(self) -> int:
        warp_k_tile = self.chunk // self.block_k_warps
        return warp_k_tile // (self.micro_size_k * self.k_pack)

    def local_elems_a(self, *, full_k: bool = False) -> int:
        if full_k:
            return self.inner_k_per_warp() * self.warp_rows * self.k_pack * self.local_size_a
        return self.warp_rows * self.k_pack * self.local_size_a

    def local_elems_b(self, *, full_k: bool = False) -> int:
        if full_k:
            return self.inner_k_per_warp() * self.warp_cols * self.k_pack * self.local_size_b
        return self.warp_cols * self.k_pack * self.local_size_b

    def configure_b_mls(self, *, b_mls: bool = True) -> None:
        """Update N-side recompute metadata when B is loaded from MLS LDS."""
        self.min_n_per_warp = min_n_per_warp_for_b(
            b_mls=b_mls,
            b_mls_trans=self.b_transposed,
            element_bits=DataType(self.b_dtype).bits,
        )
        self._initialize_block_warp_tiles(self.block_m, self.block_n)

    def _shared_block_mn_k(self, src: Buffer | BufferLoad | BufferRegion, mls_trans: bool) -> tuple[int, int]:
        region = self._legalize_to_buffer_region(src)
        return mls_block_mn_k_from_region(region, mls_trans)

    def _resolve_mls_tiles(self, mls_trans: bool, block_mn: int, block_k: int, dtype) -> tuple[int, int]:
        return compute_mls_tiles(
            mls_trans,
            block_mn,
            block_k,
            self.threads,
            self.target,
            elem_bits(dtype),
        )

    def get_ldmatrix_index_map(self, is_b=False):
        from .hcu_mmac_layout import (
            shared_16x4_to_local_64x1_layout_A,
            shared_4x16_to_local_64x1_layout_B,
            shared_16x16_to_local_64x4_layout_A,
            shared_16x16_to_local_64x4_layout_B,
            shared_16x32_to_local_64x8_layout_A,
            shared_16x32_to_local_64x8_layout_B,
            shared_16x8_to_local_64x2_layout_A,
            shared_16x8_to_local_64x2_layout_B,
            shared_16x64_to_local_64x16_layout_A,
            shared_16x64_to_local_64x16_layout_B,
            thread_id_shared_access_64x1_to_16x4_layout_A,
            thread_id_shared_access_64x1_to_4x16_layout_B,
            thread_id_shared_access_64x2_to_16x8_layout_A,
            thread_id_shared_access_64x2_to_16x8_layout_B,
            thread_id_shared_access_64x4_to_16x16_layout_A,
            thread_id_shared_access_64x4_to_16x16_layout_B,
            thread_id_shared_access_64x8_to_16x32_layout_A,
            thread_id_shared_access_64x8_to_16x32_layout_B,
            thread_id_shared_access_64x16_to_16x64_layout_A,
            thread_id_shared_access_64x16_to_16x64_layout_B,
        )

        k_dim = self.k_dim * self.k_pack
        transposed = self.a_transposed if not is_b else self.b_transposed
        if k_dim == 4:
            index_map = shared_4x16_to_local_64x1_layout_B if transposed else shared_16x4_to_local_64x1_layout_A
            reverse_index_map = (
                thread_id_shared_access_64x1_to_4x16_layout_B if transposed else thread_id_shared_access_64x1_to_16x4_layout_A
            )
            if is_b:
                index_map = shared_16x4_to_local_64x1_layout_A if transposed else shared_4x16_to_local_64x1_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x1_to_16x4_layout_A if transposed else thread_id_shared_access_64x1_to_4x16_layout_B
                )
        elif k_dim == 8:
            index_map = shared_16x8_to_local_64x2_layout_B if transposed else shared_16x8_to_local_64x2_layout_A
            reverse_index_map = (
                thread_id_shared_access_64x2_to_16x8_layout_B if transposed else thread_id_shared_access_64x2_to_16x8_layout_A
            )
            if is_b:
                index_map = shared_16x8_to_local_64x2_layout_A if transposed else shared_16x8_to_local_64x2_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x2_to_16x8_layout_A if transposed else thread_id_shared_access_64x2_to_16x8_layout_B
                )
        elif k_dim == 16:
            index_map = shared_16x16_to_local_64x4_layout_B if transposed else shared_16x16_to_local_64x4_layout_A
            reverse_index_map = (
                thread_id_shared_access_64x4_to_16x16_layout_B if transposed else thread_id_shared_access_64x4_to_16x16_layout_A
            )

            if is_b:
                index_map = shared_16x16_to_local_64x4_layout_A if transposed else shared_16x16_to_local_64x4_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x4_to_16x16_layout_A if transposed else thread_id_shared_access_64x4_to_16x16_layout_B
                )
        elif k_dim == 32:
            index_map = shared_16x32_to_local_64x8_layout_B if transposed else shared_16x32_to_local_64x8_layout_A
            reverse_index_map = (
                thread_id_shared_access_64x8_to_16x32_layout_B if transposed else thread_id_shared_access_64x8_to_16x32_layout_A
            )

            if is_b:
                index_map = shared_16x32_to_local_64x8_layout_A if transposed else shared_16x32_to_local_64x8_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x8_to_16x32_layout_A if transposed else thread_id_shared_access_64x8_to_16x32_layout_B
                )
        elif k_dim == 64:
            index_map = shared_16x64_to_local_64x16_layout_B if transposed else shared_16x64_to_local_64x16_layout_A
            reverse_index_map = (
                thread_id_shared_access_64x16_to_16x64_layout_B if transposed else thread_id_shared_access_64x16_to_16x64_layout_A
            )

            if is_b:
                index_map = shared_16x64_to_local_64x16_layout_A if transposed else shared_16x64_to_local_64x16_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x16_to_16x64_layout_A if transposed else thread_id_shared_access_64x16_to_16x64_layout_B
                )
        else:
            raise ValueError("k_dim must be 4 or 16 or 32 or 64 currently")

        return index_map, reverse_index_map

    def get_store_index_map(self):
        from .hcu_mmac_layout import (
            thread_id_shared_access_64x4_to_16x16_layout_C_m_n,
            thread_id_shared_access_64x4_to_16x16_layout_C_m_n_lit,
        )

        # use target on current device
        target = self.target
        # check if target has mmac lit lts
        if target is not None and target_has_mmac_lit_lts(target):
            return thread_id_shared_access_64x4_to_16x16_layout_C_m_n_lit
        else:
            return thread_id_shared_access_64x4_to_16x16_layout_C_m_n

    def extract_thread_binding(self, thread_id, is_m_first=None) -> tuple[PrimExpr, PrimExpr, PrimExpr]:
        """Decode ``thread_id`` into ``(lane_id, warp_n, warp_m)``.

        Return order is always ``(lane_id, warp_n, warp_m)`` for ldmatrix / stmatrix
        macros. Only the **warp_id linearization** differs with ``is_m_first`` and
        ``block_k_warps``.

        ``is_m_first`` (ignored when ``block_k_warps > 1``)
            Controls how ``warp_id = thread_id // WARP_SIZE`` maps to
            ``(warp_m, warp_n)``. Must stay consistent with ``tl_templates/hcu/gemm.h`` when lowering via C++.

            * ``is_m_first=False`` (HCU ``gemm_ss`` default): **M-inner, N-outer**

              ``warp_m = warp_id % block_row_warps`` (fast),
              ``warp_n = warp_id // block_row_warps`` (slow).
              Equivalent to ``warp_id = warp_n * block_row_warps + warp_m``.

            * ``is_m_first=True``: **M-outer, N-inner** ("m first" = M is the
              major warp-grid axis, not "warp_m appears first in the return tuple")

              ``warp_n = warp_id % block_col_warps`` (fast),
              ``warp_m = warp_id // block_col_warps`` (slow).
              Equivalent to ``warp_id = warp_m * block_col_warps + warp_n``.

        ``block_k_warps > 1`` (K-partition / warp_k)
            Same traversal as ``GemmTensorOpKPartition`` in ``gemm.h``:
            ``warp_m -> warp_n -> warp_k`` from innermost to outermost:

            ``warp_m = warp_id % block_row_warps``,
            ``warp_n = (warp_id // block_row_warps) % block_col_warps``,
            ``warp_k = warp_id // (block_row_warps * block_col_warps)``.

            Per-warp K offset is applied separately via :meth:`_k_base`.

        N recompute (``block_col_warps_no_recompute != block_col_warps``)
            When the N tile is smaller than ``block_col_warps * min_n_per_warp``
            (e.g. MLS B with ``min_n_per_warp=32``), only
            ``block_col_warps_no_recompute = min(block_col_warps, block_n //
            min_n_per_warp)`` distinct N warps are needed. Extra warps fold back:

            ``warp_n = warp_n % block_col_warps_no_recompute``.

            ``warp_cols`` and fragment layouts also use
            ``block_col_warps_no_recompute`` so recompute warps share the same
            N slice (see ``gemm.h`` ``body`` / ``gemm_mls.h`` ``ds_read_format_tensor_b``).
        """
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
        block_col_no_recompute = self.block_col_warps_no_recompute
        if is_m_first is None:
            is_m_first = self.is_m_first

        lane_id = self.scoped_thread_id(thread_id) % WARP_SIZE
        warp_id = self.scoped_thread_id(thread_id) // WARP_SIZE

        if self.block_k_warps > 1:
            warp_m = warp_id % block_row_warps
            warp_n = (warp_id // block_row_warps) % block_col_warps
        elif is_m_first:
            warp_n = warp_id % block_col_warps
            warp_m = (warp_id // block_col_warps) % block_row_warps
        else:
            warp_m = warp_id % block_row_warps
            warp_n = (warp_id // block_row_warps) % block_col_warps

        if block_col_no_recompute != block_col_warps:
            warp_n = warp_n % block_col_no_recompute
        return lane_id, warp_n, warp_m

    def _k_base(self, thread_id) -> PrimExpr:
        """K-tile base offset for the current warp when ``block_k_warps > 1``."""
        if self.block_k_warps <= 1:
            return 0
        warp_id = self.scoped_thread_id(thread_id) // self.WARP_SIZE
        warp_k_idx = warp_id // (self.block_row_warps * self.block_col_warps)
        warp_k_tile = self.chunk // self.block_k_warps
        return warp_k_idx * warp_k_tile

    def ldmatrix_a(self, A_local_buf, A_shared_buf: Buffer | BufferLoad | BufferRegion, ki, rk=0):
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        k_pack = self.k_pack
        is_transposed = self.a_transposed
        thread_binding = self.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=False)

        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_other = [r.min for r in A_region.region[:-2]]
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            k_base = rk * chunk + ki * (k_pack * micro_size_k) + self._k_base(thread_binding)
            if is_transposed:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (k_base, warp_m * warp_row_tiles + i * micro_size_x)
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[tuple(A_other) + (A_base0 + l + row, A_base1 + r + col)]
            else:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (warp_m * warp_row_tiles + i * micro_size_x, k_base)
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[tuple(A_other) + (A_base0 + l + row, A_base1 + r + col)]

        return _warp_ldmatrix_a(A_local_buf, A_shared_buf, ki, thread_binding, rk)

    def ldmatrix_b(self, B_local_buf, B_shared_buf: Buffer | BufferLoad | BufferRegion, ki, rk=0):
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        k_pack = self.k_pack
        is_transposed = self.b_transposed
        thread_binding = self.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=True)

        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_other = [r.min for r in B_region.region[:-2]]
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min

        @T.macro
        def _warp_ldmatrix_b(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            k_base = rk * chunk + ki * (k_pack * micro_size_k) + self._k_base(thread_binding)
            if is_transposed:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (warp_n * warp_col_tiles + j * micro_size_y, k_base)
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[tuple(B_other) + (B_base0 + l + row, B_base1 + r + col)]
            else:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (k_base, warp_n * warp_col_tiles + j * micro_size_y)
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[tuple(B_other) + (B_base0 + l + row, B_base1 + r + col)]

        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_binding, rk)

    def ldmatrix_mls_a(self, A_local_buf, A_mls_src: Buffer | BufferLoad | BufferRegion):
        """MLS LDS -> 1D ``local`` in gemm ``body_rr`` layout (C ``ds_read_format_tensor_a``)."""
        if self.block_k_warps != 1:
            raise ValueError("ldmatrix_mls_a requires block_k_warps == 1 (MLS ds_read path)")
        mls_trans = not self.a_transposed
        lds_block_mn, lds_block_k, ds_read_mn, ds_read_k, origin_mn, origin_k = mls_full_and_read_mn_k(A_mls_src, mls_trans)
        # Mls tile / LdsDesc follow the full write shape.
        tile_mn, tile_k = compute_mls_tiles(mls_trans, lds_block_mn, lds_block_k, self.threads, self.target, elem_bits(A_local_buf.dtype))
        check_mls_slice_aligned_to_tile(
            origin_mn=origin_mn,
            origin_k=origin_k,
            read_mn=ds_read_mn,
            read_k=ds_read_k,
            tile_mn=tile_mn,
            tile_k=tile_k,
            what="ldmatrix_mls_a",
        )
        template = build_ds_read_format_tensor_a_template(
            lds_block_mn=lds_block_mn,
            lds_block_k=lds_block_k,
            ds_read_mn=ds_read_mn,
            ds_read_k=ds_read_k,
            tile_mn=tile_mn,
            tile_k=tile_k,
            warp_m=self.block_row_warps,
            warp_k=self.block_k_warps,
            mls_trans=mls_trans,
            dtype_str=hcu_mls_ds_read_dtype_str(A_local_buf.dtype),
            target=self.target,
        )
        src_ptr = retrieve_mls_lds_base_ptr(A_mls_src, "r")
        dst_ptr = retrieve_ptr(A_local_buf, "rw")
        warp_id_offset = self.scoped_warp_id_offset()
        self._a_full_k = True

        @T.macro
        def _warp_ldmatrix_mls_a(A_local_buf, A_mls_src):
            T.evaluate(
                tirx.call_extern(
                    "handle",
                    template,
                    src_ptr,
                    dst_ptr,
                    warp_id_offset,
                    origin_mn,
                    origin_k,
                )
            )

        return _warp_ldmatrix_mls_a(A_local_buf, A_mls_src)

    def ldmatrix_mls_b(self, B_local_buf, B_mls_src: Buffer | BufferLoad | BufferRegion):
        """MLS LDS -> 1D ``local`` in gemm ``body_rr`` layout (C ``ds_read_format_tensor_b``)."""
        if self.block_k_warps != 1:
            raise ValueError("ldmatrix_mls_b requires block_k_warps == 1 (MLS ds_read path)")
        self.configure_b_mls(b_mls=True)
        mls_trans = self.b_transposed
        lds_block_mn, lds_block_k, ds_read_mn, ds_read_k, origin_mn, origin_k = mls_full_and_read_mn_k(B_mls_src, mls_trans)
        tile_mn, tile_k = compute_mls_tiles(mls_trans, lds_block_mn, lds_block_k, self.threads, self.target, elem_bits(B_local_buf.dtype))
        check_mls_slice_aligned_to_tile(
            origin_mn=origin_mn,
            origin_k=origin_k,
            read_mn=ds_read_mn,
            read_k=ds_read_k,
            tile_mn=tile_mn,
            tile_k=tile_k,
            what="ldmatrix_mls_b",
        )
        total_warp = self.threads // self.WARP_SIZE
        template = build_ds_read_format_tensor_b_template(
            lds_block_mn=lds_block_mn,
            lds_block_k=lds_block_k,
            ds_read_mn=ds_read_mn,
            ds_read_k=ds_read_k,
            tile_mn=tile_mn,
            tile_k=tile_k,
            total_warp=total_warp,
            warp_n=self.block_col_warps,
            warp_k=self.block_k_warps,
            mls_trans=mls_trans,
            dtype_str=hcu_mls_ds_read_dtype_str(B_local_buf.dtype),
            target=self.target,
        )
        src_ptr = retrieve_mls_lds_base_ptr(B_mls_src, "r")
        dst_ptr = retrieve_ptr(B_local_buf, "rw")
        warp_id_offset = self.scoped_warp_id_offset()
        self._b_full_k = True

        @T.macro
        def _warp_ldmatrix_mls_b(B_local_buf, B_mls_src):
            T.evaluate(
                tirx.call_extern(
                    "handle",
                    template,
                    src_ptr,
                    dst_ptr,
                    warp_id_offset,
                    origin_mn,
                    origin_k,
                )
            )

        return _warp_ldmatrix_mls_b(B_local_buf, B_mls_src)

    def mmac(self, A_local_buf, B_local_buf, C_local_buf, k_inner: PrimExpr | int = 0):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        k_pack = self.k_pack
        mmac_suffix = self.mmac_suffix
        out_dtype = self.accum_dtype
        compute_a_dtype, a_index_div = self._mmac_ab_operand(self.a_dtype, local_size_a)
        compute_b_dtype, b_index_div = self._mmac_ab_operand(self.b_dtype, local_size_b)
        compute_out_dtype = out_dtype if local_size_out == 1 else f"{out_dtype}x{local_size_out}"
        tf32_ann = {"tl.hcu_tf32_ab": 1} if self.use_tf32 else None
        mmac_op = tirx.op.Op.get("tl.tvm_mfma")

        a_is_fragment = is_fragment(A_local_buf)
        b_is_fragment = is_fragment(B_local_buf)
        a_full_k = self._a_full_k or a_is_fragment
        b_full_k = self._b_full_k or b_is_fragment
        a_local_stride: PrimExpr = k_inner * warp_rows * k_pack * local_size_a if a_full_k else 0
        b_local_stride: PrimExpr = k_inner * warp_cols * k_pack * local_size_b if b_full_k else 0

        # tirx.Call args must be PrimExpr; wrap dtype/layout strings as StringImm
        # (same pattern as tl.infinity). Return dtype stays a plain str for Call(dtype, ...).
        mmac_suffix_imm = tirx.StringImm(str(mmac_suffix))
        layout_imm = tirx.StringImm("row")
        compute_a_dtype_imm = tirx.StringImm(str(compute_a_dtype))
        compute_b_dtype_imm = tirx.StringImm(str(compute_b_dtype))
        compute_out_dtype_imm = tirx.StringImm(str(compute_out_dtype))

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for kp, i, j in T.grid(k_pack, warp_rows, warp_cols):
                T.call_intrin(
                    str(compute_out_dtype),
                    mmac_op,
                    mmac_suffix_imm,
                    layout_imm,
                    layout_imm,
                    compute_a_dtype_imm,
                    compute_b_dtype_imm,
                    compute_out_dtype_imm,
                    A_local_buf.data,
                    (a_local_stride + (i * k_pack + kp) * local_size_a) // a_index_div,
                    B_local_buf.data,
                    (b_local_stride + (j * k_pack + kp) * local_size_b) // b_index_div,
                    C_local_buf.data,
                    (i * warp_cols * local_size_out + j * local_size_out) // local_size_out,
                    annotations=tf32_ann,
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    def stmatrix(self, C_local_buf, C_buf, pid_m=None, pid_n=None):
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_out = self.local_size_out
        thread_binding = self.get_thread_binding()
        is_global = pid_m is not None and pid_n is not None
        BLOCK_M = block_row_warps * warp_rows
        BLOCK_N = block_col_warps * warp_cols
        M_DIM, N_DIM = self.M_DIM, self.N_DIM
        C_buf_dims = len(C_buf.shape)
        assert C_buf_dims in {2, 4}, "C_buf should be 2D or 4D"
        mmac_store_index_map = self.get_store_index_map()

        # STS
        # MMA Store must be in simulated instead of TVM Intrins
        # As TVM Intrins is like a hack that the threadIdx.x should be always
        # equal to the warp_size
        @T.macro
        def _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id in T.vectorized(local_size_out):
                    row, col = T.meta_var(mmac_store_index_map(tx, local_id))
                    if C_buf_dims == 2:
                        C_buf[(warp_m * warp_rows + i) * M_DIM + row, (warp_n * warp_cols + j) * N_DIM + col] = C_local_buf[
                            i * (warp_cols * local_size_out) + j * local_size_out + local_id
                        ]
                    else:
                        C_buf[warp_m * warp_rows + i, warp_n * warp_cols + j, row, col] = C_local_buf[
                            i * warp_cols * local_size_out + j * local_size_out + local_id
                        ]

        @T.macro
        def _warp_stmatrix_global(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id in T.vectorized(local_size_out):
                    row, col = T.meta_var(mmac_store_index_map(tx, local_id))
                    C_buf[
                        (pid_m * BLOCK_M + warp_m * warp_rows + i) * M_DIM + row, (pid_n * BLOCK_N + warp_n * warp_cols + j) * N_DIM + col
                    ] = C_local_buf[i * warp_cols * local_size_out + j * local_size_out + local_id]

        return (
            _warp_stmatrix_global(C_local_buf, C_buf, thread_binding)
            if is_global
            else _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding)
        )


class HCUMatrixCorePreshuffleIntrinEmitter(HCUMatrixCoreIntrinEmitter):
    def __init__(
        self,
        a_dtype: str = "float16",
        b_dtype: str = "float16",
        accum_dtype: str = "float16",
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        block_m: int = 0,
        block_n: int = 0,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        k_pack: int | None = None,
        is_m_first: bool | None = False,
        a_preshuffle: bool | None = False,
        b_preshuffle: bool | None = False,
        block_k_warps: int = 1,
        target: Target | None = None,
        thread_var: Var | None = None,
        min_n_per_warp: int | None = None,
        use_tf32: bool = False,
    ):
        super().__init__(
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            accum_dtype=accum_dtype,
            a_transposed=a_transposed,
            b_transposed=b_transposed,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            block_m=block_m,
            block_n=block_n,
            chunk=chunk,
            reduce_k=reduce_k,
            num_elems_per_byte=num_elems_per_byte,
            k_pack=k_pack,
            is_m_first=is_m_first,
            b_preshuffle=b_preshuffle,
            block_k_warps=block_k_warps,
            target=target,
            thread_var=thread_var,
            min_n_per_warp=min_n_per_warp,
            use_tf32=use_tf32,
        )
        self._initialize_preshuffle(a_preshuffle, b_preshuffle)

    def _initialize_preshuffle(self, a_preshuffle: bool, b_preshuffle: bool):
        if a_preshuffle is not None:
            self.a_preshuffle = a_preshuffle
        if b_preshuffle is not None:
            self.b_preshuffle = b_preshuffle

    def ldmatrix_a(self, A_local_buf, A_buf, ki, rk=0, pid_m=None, pid_n=None):
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        k_pack = self.k_pack
        is_transposed = self.a_transposed
        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=False)
        is_global = pid_m is not None and pid_n is not None

        # no preshuffle, use the default implementation
        if self.a_preshuffle is False:
            return super().ldmatrix_a(A_local_buf, A_buf, ki, rk)

        def _warp_ldmatrix_a_global(
            A_local_buf,
            A_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            (pid_m * self.block_row_warps + warp_m) * warp_rows + i,
                        )
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[l, r, row, col]
            else:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            (pid_m * self.block_row_warps + warp_m) * warp_rows + i,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[l, r, row, col]

        @T.macro
        def _warp_ldmatrix_a_shared(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            warp_m * warp_rows + i,
                        )
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_shared_buf[l, r, row, col]
            else:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (warp_m * warp_rows + i, rk * (chunk // micro_size_k) + ki)
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_shared_buf[l, r, row, col]

        return (
            _warp_ldmatrix_a_global(A_local_buf, A_buf, ki, thread_binding, rk)
            if is_global
            else _warp_ldmatrix_a_shared(A_local_buf, A_buf, ki, thread_binding, rk)
        )

    def ldmatrix_b(self, B_local_buf, B_buf, ki, rk=0, pid_m=None, pid_n=None):
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        k_pack = self.k_pack
        is_transposed = self.b_transposed
        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=True)
        is_global = pid_m is not None and pid_n is not None

        if self.b_preshuffle is False:
            return super().ldmatrix_b(B_local_buf, B_buf, ki, rk, pid_m, pid_n)

        @T.macro
        def _warp_ldmatrix_b_global(
            B_local_buf,
            B_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            (pid_n * self.block_col_warps + warp_n) * warp_cols + j,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[l, r, row, col]
            else:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            (pid_n * self.block_col_warps + warp_n) * warp_cols + j,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[l, r, row, col]

        @T.macro
        def _warp_ldmatrix_b_shared(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            warp_n * warp_cols + j,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_shared_buf[l, r, row, col]
            else:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            warp_n * warp_cols + j,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_shared_buf[l, r, row, col]

        return (
            _warp_ldmatrix_b_global(B_local_buf, B_buf, ki, thread_binding, rk)
            if is_global
            else _warp_ldmatrix_b_shared(B_local_buf, B_buf, ki, thread_binding, rk)
        )
