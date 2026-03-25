/*!
 * \file ds_read_format.h
 * \brief DsReadFormat operator: read MLS-formatted shared memory into register.
 */

#ifndef TVM_TL_OP_DS_READ_FORMAT_H_
#define TVM_TL_OP_DS_READ_FORMAT_H_

#include "operator.h"

namespace tvm {
namespace tl {

using namespace tir;

class DsReadFormatNode : public TileOperatorNode {
 public:
  Buffer src, dst;
  Array<Range> src_ranges;
  Array<Range> dst_ranges;
  /// From InferLayout: mls tile (derived via ComputeMlsWarpPartition)
  mutable int mls_tile_mn = 0;
  mutable int mls_tile_k = 0;
  mutable bool trans = true;
  mutable int warp_mn = 0;
  mutable int warp_k = 0;
  mutable bool completed_ = false;

  static constexpr const char *_type_key = "tl.DsReadFormat";
  TVM_DECLARE_FINAL_OBJECT_INFO(DsReadFormatNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  TileOperator Clone() const override;
  Array<Buffer> GetOutBuffers() const override { return {dst}; }
  Array<Buffer> GetInBuffers() const override { return {src}; }
};

class DsReadFormat : public TileOperator {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(DsReadFormat, TileOperator, DsReadFormatNode);
  TVM_DLL DsReadFormat(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_OP_DS_READ_FORMAT_H_
