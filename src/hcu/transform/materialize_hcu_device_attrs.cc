/*!
 * \file materialize_hcu_device_attrs.cc
 * \brief Materialize HCU body markers as split device-function attributes.
 */
#include "hcu/op/builtin.h"
#include "hcu/target_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;

class HcuDeviceAttrExtractor : public StmtMutator {
public:
  ffi::Optional<Integer> scale_buffer_size;

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == attr::kHcuScaleBufferSize) {
      Integer slots = Downcast<Integer>(op->value);
      if (scale_buffer_size.defined()) {
        ICHECK_EQ(scale_buffer_size.value()->value, slots->value)
            << "Conflicting HCU scale-buffer size markers";
      } else {
        scale_buffer_size = slots;
      }
      return VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }
};

tvm::transform::Pass MaterializeHcuDeviceAttrs() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsHCU(target.value())) {
      return f;
    }

    HcuDeviceAttrExtractor extractor;
    Stmt body = extractor(f->body);
    if (!extractor.scale_buffer_size.defined()) {
      return f;
    }
    f.CopyOnWrite()->body = std::move(body);
    return WithAttr(std::move(f), attr::kHcuScaleBufferSize,
                    extractor.scale_buffer_size.value());
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.MaterializeHcuDeviceAttrs", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MaterializeHcuDeviceAttrs",
                        MaterializeHcuDeviceAttrs);
}

} // namespace tl
} // namespace tvm
