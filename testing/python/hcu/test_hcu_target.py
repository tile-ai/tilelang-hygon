from tilelang.contrib.hcu import get_hcu_compile_flags
from tilelang.hcu.target import target_get_lds_size_bytes, with_hcu_target_attrs
from tvm.target import Target


def test_hcu_target_lds_size_by_arch():
    gfx946 = with_hcu_target_attrs(Target({"kind": "hcu", "mcpu": "gfx946"}))
    gfx948 = with_hcu_target_attrs(Target({"kind": "hcu", "mcpu": "gfx948"}))

    assert int(gfx946.attrs["max_shared_memory_per_block"]) == 128 * 1024
    assert int(gfx948.attrs["max_shared_memory_per_block"]) == 160 * 1024
    assert target_get_lds_size_bytes(gfx946) == 128 * 1024
    assert target_get_lds_size_bytes(gfx948) == 160 * 1024


def test_hcu_target_explicit_lds_size_takes_precedence():
    target = with_hcu_target_attrs(
        Target(
            {
                "kind": "hcu",
                "mcpu": "gfx948",
                "max_shared_memory_per_block": 96 * 1024,
            }
        )
    )

    assert int(target.attrs["max_shared_memory_per_block"]) == 96 * 1024
    assert target_get_lds_size_bytes(target) == 96 * 1024


def test_hcu_vgpr_compile_flags_by_arch(monkeypatch):
    monkeypatch.setattr("tilelang.contrib.hcu.get_hcu_compiler", lambda: "aicc")
    for arch in ("gfx936", "gfx938"):
        flags = get_hcu_compile_flags(arch)
        assert "-mllvm=-support-768-vgprs=true" in flags
        assert "-mllvm=-support-512-vgprs=true" not in flags

    for arch in ("gfx928", "gfx92a", "gfx946", "gfx948"):
        flags = get_hcu_compile_flags(arch)
        assert "-mllvm=-support-512-vgprs=true" in flags
        assert "-mllvm=-support-768-vgprs=true" not in flags
