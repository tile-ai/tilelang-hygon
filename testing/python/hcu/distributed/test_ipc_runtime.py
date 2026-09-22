import pytest

from tilelang import tvm


IPC_FFI_NAMES = (
    "tl.hcu.ipc.malloc",
    "tl.hcu.ipc.free",
    "tl.hcu.ipc.create_handle",
    "tl.hcu.ipc.open_handle",
    "tl.hcu.ipc.close_handle",
    "tl.hcu.ipc.can_access_peer",
)


def test_ipc_runtime_ffi_is_registered():
    for name in IPC_FFI_NAMES:
        assert tvm.ffi.get_global_func(name, allow_missing=True) is not None


def test_ipc_runtime_rejects_invalid_arguments_without_a_device():
    malloc = tvm.ffi.get_global_func("tl.hcu.ipc.malloc")
    create_handle = tvm.ffi.get_global_func("tl.hcu.ipc.create_handle")
    open_handle = tvm.ffi.get_global_func("tl.hcu.ipc.open_handle")
    close_handle = tvm.ffi.get_global_func("tl.hcu.ipc.close_handle")

    with pytest.raises(Exception, match="positive"):
        malloc(0)
    with pytest.raises(Exception, match="null pointer"):
        create_handle(0)
    with pytest.raises(Exception, match="expected"):
        open_handle(b"invalid")
    with pytest.raises(Exception, match="null HIP IPC mapping"):
        close_handle(0)
