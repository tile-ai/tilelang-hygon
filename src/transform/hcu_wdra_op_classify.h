/*!
 * \file hcu_wdra_op_classify.h
 * \brief Classify TileLang / TIR ops for HCU WDRA prologue validation.
 */
#ifndef TVM_TL_TRANSFORM_HCU_WDRA_OP_CLASSIFY_H_
#define TVM_TL_TRANSFORM_HCU_WDRA_OP_CLASSIFY_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tl {

/// Return true if evaluating this Call may touch VGPR-class state on HCU.
bool IsHcuWdraVgprCall(const tir::CallNode *call);

/// Return true if this stmt performs VGPR-class work (loads/stores/compute).
bool IsHcuWdraVgprStmt(const tir::Stmt &stmt);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_HCU_WDRA_OP_CLASSIFY_H_
