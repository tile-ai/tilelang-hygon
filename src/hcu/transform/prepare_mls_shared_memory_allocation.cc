/*!
 * \file prepare_mls_shared_memory_allocation.cc
 * \brief Convert lowered HCU MLS LDS-size metadata into a generic shared-memory
 * allocation-size override consumed by MergeSharedMemoryAllocations.
 */

#include <tvm/ir/transform.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_map>

#include "hcu/target_utils.h"
#include "hcu/utils/extern_call_checker.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

class MlsSharedMemoryAllocationCollector : public StmtExprVisitor {
public:
  static Map<String, PrimExpr> Collect(const Stmt &stmt) {
    MlsSharedMemoryAllocationCollector collector;
    collector(stmt);

    Map<String, PrimExpr> result;
    for (const auto &[name, size_bytes] : collector.buffer_size_bytes_) {
      result.Set(String(name), size_bytes);
    }
    return result;
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == attr::kMlsActualSizeBytesMap) {
      if (auto map = op->node.as<Map<String, PrimExpr>>()) {
        for (const auto &[var_name, actual_size_bytes] : map.value()) {
          pending_buffer_size_bytes_[std::string(var_name)] = actual_size_bytes;
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr || !IsMlsLoadTileExternCall(call)) {
      return StmtExprVisitor::VisitStmt_(op);
    }

    ICHECK(call->args.size() == 8U || call->args.size() == 9U)
        << "mls_load_tile extern expects symbol, src, stride, mn_len, k_len, "
           "mn_base, k_base, dst[, warp_id_offset]";

    const PrimExpr &dst_ptr_expr = call->args[7];
    const auto *dst_ptr_call = dst_ptr_expr.as<CallNode>();
    if (dst_ptr_call == nullptr ||
        !dst_ptr_call->op.same_as(builtin::tvm_access_ptr()) ||
        dst_ptr_call->args.size() < 2) {
      LOG(DEBUG) << "MlsSharedMemoryAllocationCollector: unexpected dst_ptr "
                 << "for lowered MLS extern call, value=" << dst_ptr_expr;
      return StmtExprVisitor::VisitStmt_(op);
    }

    const auto *buffer_var = dst_ptr_call->args[1].as<VarNode>();
    if (buffer_var == nullptr) {
      LOG(DEBUG) << "MlsSharedMemoryAllocationCollector: failed to recover dst "
                 << "buffer var from lowered MLS extern call, dst_ptr="
                 << dst_ptr_expr;
      return StmtExprVisitor::VisitStmt_(op);
    }

    auto name = std::string(buffer_var->name_hint);
    auto it = pending_buffer_size_bytes_.find(name);
    if (it != pending_buffer_size_bytes_.end()) {
      LOG(DEBUG) << "MlsSharedMemoryAllocationCollector: found lowered MLS "
                    "size for "
                 << buffer_var->name_hint
                 << ", actual_size_bytes=" << it->second;
      buffer_size_bytes_[name] = it->second;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  std::unordered_map<std::string, PrimExpr> buffer_size_bytes_;
  std::unordered_map<std::string, PrimExpr> pending_buffer_size_bytes_;
};

} // namespace

PrimFunc ApplyPrepareMlsSharedMemoryAllocation(PrimFunc f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  if (!target.defined() || !TargetIsHCU(target.value())) {
    return f;
  }

  Map<String, PrimExpr> size_bytes =
      MlsSharedMemoryAllocationCollector::Collect(f->body);
  if (size_bytes.empty()) {
    return f;
  }

  if (auto existing = f->GetAttr<Map<String, PrimExpr>>(
          attr::kSharedMemoryAllocationSizeBytesMap)) {
    Map<String, PrimExpr> merged = existing.value();
    for (const auto &[name, size] : size_bytes) {
      merged.Set(name, size);
    }
    size_bytes = merged;
  }

  return WithAttr(std::move(f), attr::kSharedMemoryAllocationSizeBytesMap,
                  size_bytes);
}

tvm::transform::Pass PrepareMlsSharedMemoryAllocation() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    (void)m;
    (void)ctx;
    return tl::ApplyPrepareMlsSharedMemoryAllocation(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PrepareMlsSharedMemoryAllocation",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.PrepareMlsSharedMemoryAllocation",
                        PrepareMlsSharedMemoryAllocation);
}

} // namespace tl
} // namespace tvm
