from tilelang.backend.target import determine_target
from tilelang.hcu.target import with_hcu_target_attrs


def test_hcu_dist_backend_target_attr_is_preserved():
    target = determine_target(
        {"kind": "hcu", "mcpu": "gfx938", "dist_backend": "ipc"},
        return_object=True,
    )

    assert target.kind.name == "hcu"
    assert str(target.attrs["dist_backend"]) == "ipc"


def test_hcu_target_normalization_keeps_dist_backend():
    target = determine_target(
        {"kind": "hcu", "mcpu": "gfx938", "dist_backend": "ipc"},
        return_object=True,
    )
    normalized = with_hcu_target_attrs(target)

    assert str(normalized.attrs["dist_backend"]) == "ipc"
    assert int(normalized.attrs["thread_warp_size"]) == 64
