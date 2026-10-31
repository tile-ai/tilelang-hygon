/*!
 * \file hcu/distributed/ipc/ipc_runtime.h
 * \brief HIP IPC helpers used by the HCU IPC distributed backend.
 */
#ifndef TILELANG_HCU_DISTRIBUTED_IPC_IPC_RUNTIME_H_
#define TILELANG_HCU_DISTRIBUTED_IPC_IPC_RUNTIME_H_

#include <hip/hip_runtime_api.h>
#include <tvm/ffi/string.h>

#include <cstdint>

namespace tvm {
namespace tl {
namespace hcu {
namespace ipc {

int64_t Malloc(int64_t nbytes);
void Free(int64_t ptr);
ffi::Bytes CreateHandle(int64_t ptr);
int64_t OpenHandle(ffi::Bytes handle);
void CloseHandle(int64_t ptr);
bool CanAccessPeer(int device, int peer_device);

} // namespace ipc
} // namespace hcu
} // namespace tl
} // namespace tvm

#endif // TILELANG_HCU_DISTRIBUTED_IPC_IPC_RUNTIME_H_
