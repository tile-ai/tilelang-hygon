"""HCU backend manifest for the upstream backend-module architecture."""

from __future__ import annotations

from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.module import BackendModule, register_backend
from tilelang.contrib.hcu import tilelang_callback_hcu_compile

from . import codegen, execution_backend, pipeline


BACKEND = register_backend(
    BackendModule(
        name="hcu",
        target_kinds=("hcu",),
        pipelines={"hcu": pipeline.HCU_PIPELINE},
        device_codegens={"hcu": codegen.HCU_DEVICE_CODEGEN},
        execution_backends=execution_backend.EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
        callbacks={"tilelang_callback_hcu_compile": tilelang_callback_hcu_compile},
    )
)
