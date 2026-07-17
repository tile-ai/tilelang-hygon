/*!
 * \file extern_call_checker.h
 * \brief Recognize call_extern symbols in lowered TIR.
 */
#ifndef TVM_TL_HCU_UTILS_EXTERN_CALL_CHECKER_H_
#define TVM_TL_HCU_UTILS_EXTERN_CALL_CHECKER_H_

#include <string>

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tl {

inline bool IsCallExternWithPrefix(const tirx::CallNode *call,
                                   const char *prefix) {
  if (!call->op.same_as(tirx::builtin::call_extern()) || call->args.empty()) {
    return false;
  }
  if (const auto *name = call->args[0].as<tirx::StringImmNode>()) {
    return std::string(name->value).find(prefix) == 0;
  }
  return false;
}

inline bool IsMlsLoadTileExternCall(const tirx::CallNode *call) {
  return IsCallExternWithPrefix(call, "tl::mls::mls_load_tile");
}

inline bool IsMlsAsyncLoadExternCall(const tirx::CallNode *call) {
  return IsCallExternWithPrefix(call, "tl::mls::async_load");
}

inline bool IsDsReadFormatExternCall(const tirx::CallNode *call) {
  return IsCallExternWithPrefix(call, "tl::mls::ds_read_format_tensor");
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_UTILS_EXTERN_CALL_CHECKER_H_
