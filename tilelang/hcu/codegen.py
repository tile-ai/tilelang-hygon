from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen

from . import compile as _compile_mod  # noqa: F401


register_device_codegen(
    "hcu",
    DeviceCodegen(
        "hcu",
        build=global_func_device_codegen("target.build.tilelang_hcu"),
        build_without_compile=global_func_device_codegen("target.build.tilelang_hcu_without_compile"),
    ),
    override=True,
)
