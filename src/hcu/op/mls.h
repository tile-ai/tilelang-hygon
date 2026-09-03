/*!
 * \file tl/op/mls.h
 * \brief MLS (Matrix Load Store) operator for HCU.
 */

#ifndef TVM_TL_OP_MLS_H_
#define TVM_TL_OP_MLS_H_

#include "op/operator.h"

namespace tvm {
namespace tl {
using namespace tirx;
using namespace ffi;

class CopyNode;
class MatrixLoad;

/// Whether a Copy explicitly requests HCU MatrixLoad lowering.
bool IsMatrixLoadPreferredCopy(const CopyNode &copy);

/// Adapt a Copy to MatrixLoad while preserving its regions and annotations.
MatrixLoad MakeMatrixLoadFromCopy(const CopyNode &copy);

/// Derive MLS warp division (warp_mn, warp_k) from block shape and num_warps.
/// num_warps = block_size / TargetHcuGetWarpSize(target); warp_mn * warp_k =
/// num_warps. Pack along the storage-major axis first (K if trans, MN
/// otherwise); leftover warps go to the non-major axis, and surplus warps
/// repeat on the non-major axis.
void ComputeMlsWarpPartition(bool trans, int block_mn, int block_k,
                             int block_size, Target target, int elem_bits,
                             int &warp_mn, int &warp_k, int &mls_tile_mn,
                             int &mls_tile_k, int requested_lds_bits = 0);

/// Logical warp index base for scoped MLS: thread_bounds.min / warp_size.
int MlsScopedWarpIdOffset(const Range &thread_bounds, Target target);

/// Last-2 buffer dims as (MN, K) according to mls_trans.
std::pair<int64_t, int64_t> MlsBlockDims(const Buffer &buf, bool mls_trans);

Optional<Integer> TryGetMlsDstActualSizeBytes(const Buffer &dst,
                                              int mls_tile_mn, int mls_tile_k,
                                              bool trans, Target target,
                                              int lds_physical_bits = -1);

Optional<PrimExpr> TryGetMlsPackedLeadingElemOffset(
    const Buffer &buffer, const Array<Range> &ranges, int mls_tile_mn,
    int mls_tile_k, bool trans, Target target, int lds_physical_bits = -1);

class MatrixLoadNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range>
      src_ranges; // from src region, last 2 dims = MN,K (order from Gemm)
  Array<Range>
      dst_ranges; // leading dims select ping-pong slice; last 2 = MN,K tile
  /// Per-axis policy: -1 analyze, 0 skip, 1 refresh (see MlsBoundaryMode).
  int mn_boundary;
  int k_boundary;
  bool no_implicit_async_commit_wait_{false};
  /// Validated by AnnotateMlsGemmDep and stored as tl.mls_trans on the call.
  bool mls_trans_{true};

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.MatrixLoad", MatrixLoadNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<MatrixLoadNode>()
        .def_ro("src", &MatrixLoadNode::src)
        .def_ro("dst", &MatrixLoadNode::dst)
        .def_ro("src_ranges", &MatrixLoadNode::src_ranges)
        .def_ro("dst_ranges", &MatrixLoadNode::dst_ranges)
        .def_ro("mn_boundary", &MatrixLoadNode::mn_boundary)
        .def_ro("k_boundary", &MatrixLoadNode::k_boundary)
        .def_ro("no_implicit_async_commit_wait",
                 &MatrixLoadNode::no_implicit_async_commit_wait_);
  }

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
