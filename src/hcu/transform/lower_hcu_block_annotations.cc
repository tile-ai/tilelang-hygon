/*!
 * \file lower_hcu_block_annotations.cc
 * \brief Lower HCU-owned block annotations before the backend-neutral opaque
 * block lowering discards them.
 */
#include "hcu/op/builtin.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;

class HcuBlockAnnotationLower : public StmtExprMutator {
public:
  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    SBlockRealize realize =
        Downcast<SBlockRealize>(StmtExprMutator::VisitStmt_(op));
    const SBlock &block = realize->block;
    Stmt body = block->body;

    if (auto opt = block->annotations.Get(attr::kHcuDirectToLds)) {
      ffi::Map<Var, PrimExpr> input =
          Downcast<ffi::Map<Var, PrimExpr>>(opt.value());
      ffi::Map<ffi::String, PrimExpr> by_name;
      for (const auto &[buffer_var, enabled] : input) {
        by_name.Set(buffer_var->name_hint, enabled);
      }
      body =
          AttrStmt(by_name, attr::kHcuDirectToLds, Integer(0), std::move(body));
    }

    if (auto opt = block->annotations.Get(attr::kHcuBufferOpsRebaseMap)) {
      ffi::Map<Var, PrimExpr> rebase_map =
          Downcast<ffi::Map<Var, PrimExpr>>(opt.value());
      for (const auto &[buffer_var, enabled] : rebase_map) {
        body = AttrStmt(buffer_var, attr::kHcuBufferOpsRebaseMap, enabled,
                        std::move(body));
      }
    }

    if (body.same_as(block->body)) {
      return realize;
    }
    SBlock updated = block;
    updated.CopyOnWrite()->body = std::move(body);
    auto node = realize.CopyOnWrite();
    node->block = std::move(updated);
    return realize;
  }
};

tvm::transform::Pass LowerHcuBlockAnnotations() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    auto node = f.CopyOnWrite();
    node->body = HcuBlockAnnotationLower()(std::move(node->body));
    return f;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.LowerHcuBlockAnnotations", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerHcuBlockAnnotations",
                        LowerHcuBlockAnnotations);
}

} // namespace tl
} // namespace tvm
