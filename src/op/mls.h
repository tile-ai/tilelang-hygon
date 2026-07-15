/*!
 * \file tl/op/mls.h
 * \brief MLS (Matrix Load Store) operator for HCU.
 */

#ifndef TVM_TL_OP_MLS_H_
#define TVM_TL_OP_MLS_H_

#include "operator.h"

namespace tvm {
namespace tl {
using namespace tir;

/// Derive MLS warp division (warp_mn, warp_k) from block shape and num_warps.
/// num_warps = block_size / TargetGetWarpSize(target); warp_mn * warp_k =
/// num_warps.
void ComputeMlsWarpPartition(bool trans, int block_mn, int block_k,
                             int block_size, Target target, int elem_bits,
                             int &warp_mn, int &warp_k, int &mls_tile_mn,
                             int &mls_tile_k);

/// Logical warp index base for scoped MLS: thread_bounds.min / warp_size.
int MlsScopedWarpIdOffset(const Range &thread_bounds, Target target);

class MatrixLoadNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range>
      src_ranges; // from src region, last 2 dims = MN,K (order from Gemm)
  Array<Range>
      dst_ranges; // leading dims select ping-pong slice; last 2 = MN,K tile
  bool check_last_load;
  bool last_load;
  /// Validated by AnnotateMlsGemmDep and stored as tl.mls_trans on the call.
  bool mls_trans_{true};

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.MatrixLoad", MatrixLoadNode,
                                    TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  TileOperator Clone() const override;
};

class MatrixLoad : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(MatrixLoad, TileOperator,
                                             MatrixLoadNode);
  TVM_DLL
  MatrixLoad(Array<PrimExpr> args,
             Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_MLS_H_
