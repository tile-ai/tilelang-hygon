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
/// num_warps = block_size / TargetGetWarpSize(target); warp_mn * warp_k = num_warps.
void ComputeMlsWarpPartition(bool trans, int block_mn, int block_k, int block_size,
                            Target target, int elem_bits, int &warp_mn,
                            int &warp_k, int &mls_tile_mn, int &mls_tile_k);

class MatrixLoadNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_ranges;  // from src region, last 2 dims = MN,K (order from Gemm)
  bool check_last_load;
  bool last_load;
  /// MLS tile from InferLayout, written into IR by layout_inference. 0 = not set.
  mutable int mls_tile_mn = 0;
  mutable int mls_tile_k = 0;
  mutable int warp_mn = 0;  // MLS warp division (not Gemm's), derived in InferLayout
  mutable int warp_k = 0;   // MLS warp division (not Gemm's), derived in InferLayout
  mutable bool trans = true;  // K-major (true) or MN-major (false), from consumer gemm or default
  mutable bool completed_ = false;  // InferLayout done, skip re-run

  static constexpr const char *_type_key = "tl.MatrixLoad";
  TVM_DECLARE_FINAL_OBJECT_INFO(MatrixLoadNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  TileOperator Clone() const override;
  Array<Buffer> GetOutBuffers() const override { return {dst}; }
  Array<Buffer> GetInBuffers() const override { return {src}; }
};

class MatrixLoad : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(MatrixLoad, TileOperator, MatrixLoadNode);
  TVM_DLL MatrixLoad(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_MLS_H_
