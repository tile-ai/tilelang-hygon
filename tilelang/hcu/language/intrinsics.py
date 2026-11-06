"""HCU-owned intrinsic frontends and compatibility spellings."""

from tvm import DataType, tirx

from tilelang.language.builtin import (
    annotate_consumer_reg_alloc,
    annotate_producer_reg_dealloc,
    disable_warp_group_reg_alloc,
    no_set_max_nreg,
)


def _optional_intrin(name, *args):
    return tirx.call_intrin("int32", tirx.op.Op.get(name), *args)


def get_lane_idx(warp_size=None):
    args = () if warp_size is None else (warp_size,)
    return _optional_intrin("tl.hcu_get_lane_idx", *args)


def get_warp_idx(warp_size=None):
    args = () if warp_size is None else (warp_size,)
    return _optional_intrin("tl.hcu_get_wave_idx", *args)


def get_warp_idx_sync(warp_size=None):
    args = () if warp_size is None else (warp_size,)
    return _optional_intrin("tl.hcu_get_wave_idx_sync", *args)


def get_warp_group_idx(warp_size=None, warps_per_group=None):
    args = tuple(arg for arg in (warp_size, warps_per_group) if arg is not None)
    return _optional_intrin("tl.hcu_get_wave_group_idx", *args)


def set_max_nreg(reg_count, is_inc):
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.hcu_set_max_nreg"), reg_count, is_inc)


def set_max_nreg_inc(reg_count):
    return set_max_nreg(reg_count, 1)


def set_max_nreg_dec(reg_count):
    return set_max_nreg(reg_count, 0)


def ieee_fmaf(x, y, z, rounding_mode="rn"):
    return tirx.call_intrin(x.dtype, tirx.op.Op.get("tl.hcu_ieee_fmaf"), x, y, z, rounding_mode)


def scaled_cast(value, scale, dtype, *, clamp=False):
    """Convert between low-precision floats and float16/bfloat16/float32.

    ``scale`` must have dtype ``float8_e8m0fnu``. A scalar scale broadcasts;
    vector scales must match the source lanes. Raw scale bytes must be
    explicitly reinterpreted as E8M0, not numerically cast.

    ``clamp=True`` requests the instruction's [0, 1] clamp, not finite
    saturation. This is one scaled conversion, not two casts and a multiply.
    Widening multiplies by the decoded E8M0 scale; narrowing divides by it
    before rounding to the destination format. Stochastic rounding is
    not supported. ``clamp`` is available only for widening conversions.
    """
    value, scale = tirx.convert(value), tirx.convert(scale)
    src, sf, dst = DataType(value.dtype), DataType(scale.dtype), DataType(dtype)
    src_element = str(src).split("x")[0]
    sf_element = str(sf).split("x")[0]
    dst_element = str(dst).split("x")[0]
    low = ("float8_e4m3fn", "float8_e5m2", "float4_e2m1fn")
    high = ("float16", "bfloat16", "float32")
    if not ((src_element in low and dst_element in high) or (src_element in high and dst_element in low)):
        raise TypeError(f"scaled_cast unsupported conversion: {src} -> {dst}")
    if sf_element != "float8_e8m0fnu":
        raise TypeError(f"scaled_cast scale must be float8_e8m0fnu, got {sf}")
    if dst.lanes not in (1, src.lanes):
        raise ValueError("scaled_cast output lanes must match the source")
    if sf.lanes not in (1, src.lanes):
        raise ValueError("scaled_cast scale must be scalar or match the source lanes")
    if not isinstance(clamp, bool):
        raise TypeError("scaled_cast clamp must be a compile-time bool")
    if clamp and dst_element in low:
        raise ValueError("scaled_cast clamp is only supported for widening conversions")
    result_dtype = dst_element if src.lanes == 1 else f"{dst_element}x{src.lanes}"
    return tirx.call_intrin(result_dtype, tirx.op.Op.get("tl.scaled_cast"), value, scale, tirx.const(clamp, "bool"))


def abarrier_init(abar_id, arrive_waves):
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.abarrier_init"), abar_id, arrive_waves)


def abarrier_inv(abar_id):
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.abarrier_inv"), abar_id)


def abarrier_arrive(abar_id, wave_count=1):
    return tirx.call_intrin("int32", tirx.op.Op.get("tl.abarrier_arrive"), abar_id, wave_count)


def abarrier_try_wait(abar_id, phase):
    return tirx.call_intrin("int32", tirx.op.Op.get("tl.abarrier_try_wait"), abar_id, phase)


def abarrier_wait(abar_id, phase):
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.abarrier_wait"), abar_id, phase)


def abarrier_test_wait(abar_id, phase):
    return tirx.call_intrin("int32", tirx.op.Op.get("tl.abarrier_test_wait"), abar_id, phase)


def abarrier_seq(abar_id):
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.abarrier_seq"), abar_id)


def abarrier_expect_tx(abar_id, num_bytes):
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.abarrier_expect_tx"), abar_id, num_bytes)


def abarrier_complete_tx(abar_id, num_bytes):
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.abarrier_complete_tx"), abar_id, num_bytes)


def ebarrier_sync(ebar_id):
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.ebarrier_sync"), ebar_id)


def ebarrier_sync_cnt(ebar_id, wave_count):
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.ebarrier_sync_cnt"), ebar_id, wave_count)


def ebarrier_arrive(ebar_id, wave_count=1):
    return tirx.call_intrin("handle", tirx.op.Op.get("tl.ebarrier_arrive"), ebar_id, wave_count)


def _pack_s_waitcnt_imm(cnt: int, flag: str) -> int:
    if not isinstance(cnt, int):
        raise TypeError(f"Expect cnt to be int, but got {type(cnt)}.")
    if flag == "vmcnt":
        if not 0 <= cnt <= 63:
            raise ValueError(f"vmcnt must be in [0, 63], but got {cnt}.")
        return (cnt & 0xF) | (7 << 4) | (15 << 8) | (3 << 12) | ((cnt & 0x30) << 10)
    if flag == "lgkmcnt":
        if not 0 <= cnt <= 15:
            raise ValueError(f"lgkmcnt must be in [0, 15], but got {cnt}.")
        return 0xF | (7 << 4) | (cnt << 8) | (3 << 12) | (3 << 14)
    if flag == "expcnt":
        if not 0 <= cnt <= 7:
            raise ValueError(f"expcnt must be in [0, 7], but got {cnt}.")
        return 0xF | (cnt << 4) | (15 << 8) | (3 << 12) | (3 << 14)
    raise ValueError(f"Unsupported s_waitcnt flag: {flag}. Expected one of vmcnt, lgkmcnt, expcnt.")


def s_waitcnt(cnt: int = 0, flag: str = "vmcnt"):
    imm = _pack_s_waitcnt_imm(cnt, flag)
    return tirx.call_extern("int32", "__builtin_amdgcn_s_waitcnt", tirx.IntImm("int32", imm))


def sched_barrier(mask: int = 0):
    """Insert the HCU scheduler barrier used to constrain instruction issue."""
    return tirx.call_extern("void", "__builtin_amdgcn_sched_barrier", tirx.IntImm("int32", mask))


__all__ = (
    "abarrier_init",
    "abarrier_inv",
    "abarrier_arrive",
    "abarrier_try_wait",
    "abarrier_wait",
    "abarrier_test_wait",
    "abarrier_seq",
    "abarrier_expect_tx",
    "abarrier_complete_tx",
    "ebarrier_sync",
    "ebarrier_sync_cnt",
    "ebarrier_arrive",
    "get_lane_idx",
    "get_warp_idx",
    "get_warp_idx_sync",
    "get_warp_group_idx",
    "annotate_consumer_reg_alloc",
    "annotate_producer_reg_dealloc",
    "disable_warp_group_reg_alloc",
    "no_set_max_nreg",
    "set_max_nreg",
    "set_max_nreg_inc",
    "set_max_nreg_dec",
    "ieee_fmaf",
    "scaled_cast",
    "s_waitcnt",
    "sched_barrier",
)
