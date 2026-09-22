/*!
 * \file hcu/distributed/ipc/ipc_runtime.cc
 * \brief Python FFI bindings for HIP inter-process memory handles.
 */
#include "ipc_runtime.h"

#include <tvm/ffi/error.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstring>
#include <string>

namespace tvm {
namespace tl {
namespace hcu {
namespace ipc {
namespace {

void CheckHip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    TVM_FFI_THROW(RuntimeError) << operation << " failed: " << hipGetErrorString(status)
                                 << " (HIP error " << static_cast<int>(status) << ")";
  }
}

hipIpcMemHandle_t DecodeHandle(ffi::Bytes bytes) {
  if (bytes.size() != sizeof(hipIpcMemHandle_t)) {
    TVM_FFI_THROW(ValueError) << "HIP IPC handle has " << bytes.size() << " bytes; expected "
                              << sizeof(hipIpcMemHandle_t);
  }
  hipIpcMemHandle_t handle{};
  std::memcpy(&handle, bytes.data(), sizeof(handle));
  return handle;
}

} // namespace

int64_t Malloc(int64_t nbytes) {
  if (nbytes <= 0) {
    TVM_FFI_THROW(ValueError) << "IPC allocation size must be positive, got " << nbytes;
  }
  void* ptr = nullptr;
  CheckHip(hipMalloc(&ptr, static_cast<size_t>(nbytes)), "hipMalloc");
  return static_cast<int64_t>(reinterpret_cast<intptr_t>(ptr));
}

void Free(int64_t ptr) {
  if (ptr == 0) {
    TVM_FFI_THROW(ValueError) << "Cannot free a null IPC allocation";
  }
  CheckHip(hipFree(reinterpret_cast<void*>(static_cast<intptr_t>(ptr))), "hipFree");
}

ffi::Bytes CreateHandle(int64_t ptr) {
  if (ptr == 0) {
    TVM_FFI_THROW(ValueError) << "Cannot create a HIP IPC handle for a null pointer";
  }
  hipIpcMemHandle_t handle{};
  CheckHip(hipIpcGetMemHandle(&handle, reinterpret_cast<void*>(static_cast<intptr_t>(ptr))),
           "hipIpcGetMemHandle");
  return ffi::Bytes(reinterpret_cast<const char*>(&handle), sizeof(handle));
}

int64_t OpenHandle(ffi::Bytes handle) {
  void* ptr = nullptr;
  CheckHip(hipIpcOpenMemHandle(&ptr, DecodeHandle(handle), hipIpcMemLazyEnablePeerAccess),
           "hipIpcOpenMemHandle");
  return static_cast<int64_t>(reinterpret_cast<intptr_t>(ptr));
}

void CloseHandle(int64_t ptr) {
  if (ptr == 0) {
    TVM_FFI_THROW(ValueError) << "Cannot close a null HIP IPC mapping";
  }
  CheckHip(hipIpcCloseMemHandle(reinterpret_cast<void*>(static_cast<intptr_t>(ptr))),
           "hipIpcCloseMemHandle");
}

bool CanAccessPeer(int device, int peer_device) {
  int can_access = 0;
  CheckHip(hipDeviceCanAccessPeer(&can_access, device, peer_device), "hipDeviceCanAccessPeer");
  return can_access != 0;
}

} // namespace ipc
} // namespace hcu
} // namespace tl
} // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.hcu.ipc.malloc", tvm::tl::hcu::ipc::Malloc)
      .def("tl.hcu.ipc.free", tvm::tl::hcu::ipc::Free)
      .def("tl.hcu.ipc.create_handle", tvm::tl::hcu::ipc::CreateHandle)
      .def("tl.hcu.ipc.open_handle", tvm::tl::hcu::ipc::OpenHandle)
      .def("tl.hcu.ipc.close_handle", tvm::tl::hcu::ipc::CloseHandle)
      .def("tl.hcu.ipc.can_access_peer", tvm::tl::hcu::ipc::CanAccessPeer);
}
