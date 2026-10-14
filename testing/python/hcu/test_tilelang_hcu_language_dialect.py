import inspect

import tilelang as tl
import tilelang.language as default_language
import tilelang.language.common as common_language
from tilelang.hcu import language as hcu_language


HCU_COMPATIBILITY_MARKERS = {
    "annotate_consumer_reg_alloc",
    "annotate_producer_reg_dealloc",
    "disable_warp_group_reg_alloc",
    "no_set_max_nreg",
}

HCU_OVERRIDE_NAMES = {
    "Kernel",
    "Parallel",
    "copy",
    "gemm",
    "gemm_blockscaled",
}

HCU_COMPATIBILITY_DEFAULTS = {
    ("gemm_blockscaled", "k_start"): 0,
}


def test_hygon_default_facade_is_hcu_language():
    assert default_language.__tilelang_dialect__ == "hcu"
    assert set(default_language.__all__) == set(hcu_language.__all__)
    for name in default_language.__all__:
        assert getattr(default_language, name) is getattr(hcu_language, name)


def test_hcu_language_contains_the_complete_common_surface():
    assert set(common_language.__all__) <= set(hcu_language.__all__)
    for name in set(common_language.__all__) - HCU_OVERRIDE_NAMES:
        assert getattr(hcu_language, name) is getattr(common_language, name)


def test_hcu_overrides_preserve_every_common_parameter():
    for name in HCU_OVERRIDE_NAMES:
        common_signature = inspect.signature(getattr(common_language, name))
        hcu_signature = inspect.signature(getattr(hcu_language, name))
        for parameter_name, common_parameter in common_signature.parameters.items():
            assert parameter_name in hcu_signature.parameters
            hcu_parameter = hcu_signature.parameters[parameter_name]
            compatibility_default = HCU_COMPATIBILITY_DEFAULTS.get((name, parameter_name), common_parameter.default)
            assert hcu_parameter.default == compatibility_default


def test_hcu_exports_supported_compatibility_markers():
    assert set(hcu_language.__all__) >= HCU_COMPATIBILITY_MARKERS
    for name in HCU_COMPATIBILITY_MARKERS:
        assert callable(getattr(hcu_language, name))


def test_hcu_compatibility_register_markers_compile_to_no_device_instruction():
    T = hcu_language

    @T.prim_func
    def main(A: T.Tensor((64,), T.float32), B: T.Tensor((64,), T.float32)):
        with T.Kernel(1, threads=64):
            T.no_set_max_nreg()
            T.disable_warp_group_reg_alloc()
            T.annotate_producer_reg_dealloc()
            T.annotate_consumer_reg_alloc()
            for i in T.Parallel(64):
                B[i] = A[i]

    source = tl.compile(main, out_idx=[1]).get_kernel_source()
    assert "__builtin_hcu_s_set_vgpr_size" not in source
