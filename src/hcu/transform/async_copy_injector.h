#pragma once

#include <tvm/ir/expr.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

struct HCUAsyncCopyInjectResult {
  tvm::tirx::Stmt stmt;
  bool injected_hcu_async_copy{false};
};

/*! \brief Inject HCU async-copy lowering patterns into a statement. */
HCUAsyncCopyInjectResult
InjectHCUAsyncCopy(const tvm::tirx::Stmt &body,
                   bool async_without_async_commit_wait = false,
                   tvm::ffi::Map<tvm::ffi::String, tvm::ffi::ObjectRef>
                       call_annotations = {});

} // namespace tl
} // namespace tvm
