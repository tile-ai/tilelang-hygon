import pytest

import tilelang.language as T
from tilelang import tvm
from tilelang.engine.lower import lower


def _program():
    @T.prim_func
    def main(out: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            out[0] = T.get_rank() + T.get_num_ranks()

    return main


def test_ipc_metadata_source_is_emitted_only_for_distributed_hcu_module():
    artifact = lower(
        _program().with_attr("global_symbol", "main"),
        target={"kind": "hcu", "mcpu": "gfx938", "dist_backend": "ipc"},
    )

    assert "tl_templates/hcu/distributed/distributed.h" in artifact.kernel_source
    assert "__tilelang_ipc_metadata" in artifact.kernel_source
    assert "__tilelang_init_ipc_metadata" in artifact.kernel_source
    assert "tl::ipc_get_rank()" in artifact.kernel_source
    assert "tl::ipc_get_num_ranks()" in artifact.kernel_source


def test_rank_intrinsic_requires_explicit_distributed_backend():
    with pytest.raises(Exception, match="requires a non-empty dist_backend"):
        lower(
            _program().with_attr("global_symbol", "main"),
            target={"kind": "hcu", "mcpu": "gfx938"},
        )


def test_normal_hcu_module_does_not_emit_ipc_metadata():
    @T.prim_func
    def main(out: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            out[0] = 1

    artifact = lower(
        main.with_attr("global_symbol", "main"),
        target={"kind": "hcu", "mcpu": "gfx938"},
    )
    assert "__tilelang_ipc_metadata" not in artifact.kernel_source


def test_ipc_metadata_helper_is_registered_in_module_function_map():
    artifact = lower(
        _program().with_attr("global_symbol", "main"),
        target={"kind": "hcu", "mcpu": "gfx938", "dist_backend": "ipc"},
    )
    build = tvm.ffi.get_global_func("target.build.tilelang_hcu_without_compile")
    module = build(artifact.device_mod, artifact.target)

    assert module.get_function("__tilelang_init_ipc_metadata", query_imports=False) is not None


def test_ipc_template_handles_full_3d_block_and_invalid_peer_guard():
    from pathlib import Path

    template = (Path(__file__).parents[4] / "src/tl_templates/hcu/distributed/distributed.h").read_text()
    assert "blockDim.y) * threadIdx.z" in template
    assert "index += thread_count" in template
    assert "src_pe < 0 || src_pe >= world_size" in template
    assert "dst[index] = DstT{}" in template


def test_ipc_get_block_accepts_buffer_address_expressions():
    @T.prim_func
    def main(out: T.Tensor((16,), "float32"), src: T.Tensor((16,), "float32")):
        with T.Kernel(1, threads=32):
            peer_tile = T.alloc_shared((16,), "float32")
            T.get_block(
                T.address_of(src[0]),
                T.address_of(peer_tile[0]),
                16,
                T.get_rank() ^ 1,
            )

    artifact = lower(
        main.with_attr("global_symbol", "main"),
        target={"kind": "hcu", "mcpu": "gfx938", "dist_backend": "ipc"},
    )
    assert "// tilelang_ipc_remote_source: src" in artifact.kernel_source
    assert "tl::ipc_get_block" in artifact.kernel_source
