import pytest

from tilelang import tvm
from tilelang.backend.target import determine_target


def _target(dist_backend=None):
    target = {"kind": "hcu", "mcpu": "gfx938"}
    if dist_backend is not None:
        target["dist_backend"] = dist_backend
    return determine_target(target, return_object=True)


def test_ipc_backend_is_selected_and_declares_capabilities():
    target = _target("ipc")
    backend_name = tvm.ffi.get_global_func("tl.GetHcuDistributedBackendName")
    supports_rank = tvm.ffi.get_global_func("tl.HcuDistributedBackendSupportsRankWorldSize")
    supports_remote_get = tvm.ffi.get_global_func("tl.HcuDistributedBackendSupportsBlockRemoteGet")

    assert str(backend_name(target)) == "ipc"
    assert supports_rank(target)
    assert supports_remote_get(target)


def test_distributed_backend_requires_explicit_known_target_attribute():
    backend_name = tvm.ffi.get_global_func("tl.GetHcuDistributedBackendName")

    with pytest.raises(Exception, match="requires a non-empty dist_backend"):
        backend_name(_target())
    with pytest.raises(Exception, match="Unsupported HCU distributed backend: mori"):
        backend_name(_target("mori"))
