/*!
 * \file ds_read_format.h
 * \brief DsReadFormat operator: read MLS-formatted shared memory into register.
 */

#ifndef TVM_TL_OP_DS_READ_FORMAT_H_
#define TVM_TL_OP_DS_READ_FORMAT_H_

#include "hcu/utils/mls_gemm_dep.h"
#include "op/operator.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class DsReadFormatNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_ranges;
  Array<Range> dst_ranges;
  /// Consumer GEMM facts from AnnotateMlsGemmDep (optional).
  Optional<MlsGemmDepMeta> gemm_dep_;
  bool hcu_linear_ds_read_{false};
  bool hcu_layout_ds_read_{false};
  mutable bool completed_ = false;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.DsReadFormat", DsReadFormatNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<DsReadFormatNode>()
        .def_ro("src", &DsReadFormatNode::src)
        .def_ro("dst", &DsReadFormatNode::dst)
        .def_ro("src_ranges", &DsReadFormatNode::src_ranges)
        .def_ro("dst_ranges", &DsReadFormatNode::dst_ranges)
        .def_ro("gemm_dep_", &DsReadFormatNode::gemm_dep_);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  TileOperator Clone() const override;
};

class DsReadFormat : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DsReadFormat, TileOperator,
                                             DsReadFormatNode);
  TVM_DLL
  DsReadFormat(Array<PrimExpr> args,
               Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_DS_READ_FORMAT_H_
