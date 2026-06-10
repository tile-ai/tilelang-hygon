/*!
 * \file ds_read_format.h
 * \brief DsReadFormat operator: read MLS-formatted shared memory into register.
 */

#ifndef TVM_TL_OP_DS_READ_FORMAT_H_
#define TVM_TL_OP_DS_READ_FORMAT_H_

#include "mls_gemm_dep.h"
#include "operator.h"

namespace tvm {
namespace tl {

using namespace tir;

class DsReadFormatNode : public TileOperatorNode {
 public:
  Buffer src, dst;
  Array<Range> src_ranges;
  Array<Range> dst_ranges;
  /// Consumer GEMM facts from AnnotateMlsGemmDep (optional).
  Optional<MlsGemmDepMeta> gemm_dep_;
  mutable bool completed_ = false;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.DsReadFormat", DsReadFormatNode,
                                    TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  TileOperator Clone() const override;
};

class DsReadFormat : public TileOperator {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DsReadFormat, TileOperator,
                                             DsReadFormatNode);
  TVM_DLL DsReadFormat(Array<PrimExpr> args,
                       Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_OP_DS_READ_FORMAT_H_
