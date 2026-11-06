/*!
 * \file hcu/distributed/stream.cc
 * \brief TVM ROCm stream bridge for HCU IPC launches.
 */
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/reflection/registry.h>

#include <dlpack/dlpack.h>

#include <cstdint>

namespace tvm {
namespace tl {
namespace hcu {
namespace stream_bridge {

int64_t GetTvmStream(int device_id) {
  return static_cast<int64_t>(
      reinterpret_cast<intptr_t>(TVMFFIEnvGetStream(kDLROCM, device_id)));
}

void SetTvmStream(int device_id, int64_t stream) {
  TVMFFIEnvSetStream(
      kDLROCM, device_id,
      reinterpret_cast<TVMFFIStreamHandle>(static_cast<intptr_t>(stream)),
      nullptr);
}

} // namespace stream_bridge
} // namespace hcu
} // namespace tl
} // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.hcu.distributed.get_tvm_stream",
           tvm::tl::hcu::stream_bridge::GetTvmStream)
      .def("tl.hcu.distributed.set_tvm_stream",
           tvm::tl::hcu::stream_bridge::SetTvmStream);
}
