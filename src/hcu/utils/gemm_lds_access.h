/*!
 * \file gemm_lds_access.h
 * \brief Operand-independent HCU GEMM LDS access classification.
 */
#ifndef TVM_TL_HCU_UTILS_GEMM_LDS_ACCESS_H_
#define TVM_TL_HCU_UTILS_GEMM_LDS_ACCESS_H_

#include "op/gemm.h"

namespace tvm {
namespace tl {

enum class HcuGemmLdsAccessKind {
  kAtBn,
  kAnBt,
};

inline HcuGemmLdsAccessKind GetHcuGemmLdsAccessKind(const GemmNode &gemm,
                                                    bool feeds_a) {
  const bool is_at_bn = feeds_a ? !gemm.transA_ : gemm.transB_;
  return is_at_bn ? HcuGemmLdsAccessKind::kAtBn
                  : HcuGemmLdsAccessKind::kAnBt;
}

inline int GetHcuGemmLdsMnExtent(const GemmNode &gemm, bool feeds_a) {
  return feeds_a ? gemm.m_ : gemm.n_;
}

inline DataType GetHcuGemmOperandDType(const GemmNode &gemm, bool feeds_a) {
  return feeds_a ? gemm.a_->dtype : gemm.b_->dtype;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_UTILS_GEMM_LDS_ACCESS_H_
