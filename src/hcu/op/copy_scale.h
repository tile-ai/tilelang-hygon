// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file copy_scale.h
 * \brief CopyScale tileop: LDS scale -> HCU scale_buffer
 * (ds_scale_copy_ds2buf).
 */

#ifndef TVM_TL_OP_COPY_SCALE_H_
#define TVM_TL_OP_COPY_SCALE_H_

#include "hcu/op/scale_format.h"
#include "op/operator.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class CopyScaleNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_ranges;
  Array<Range> dst_ranges;
  int op_ctrl_{0};
  // Optional logical descriptor over the physical LDS buffer.
  bool has_scale_view_{false};
  int scale_format_{0};
  PrimExpr parent_k_, parent_mn_, origin_k_, origin_mn_, tile_k_, tile_mn_;
  // Filled by AnnotateScaleGemmDep (optional until annotated).
  int granularity_mn_{1};
  int granularity_k_{1};
  int scale_k_major_{0};
  int role_{0}; // 0=A(M), 1=B(N)
  // Deferred ComputeWarpPartitionHCU clues (not precomputed WarpM/N).
  int gemm_m_{0};
  int gemm_n_{0};
  int gemm_k_{0};
  int gemm_policy_{0};
  int gemm_k_pack_{1};
  int gemm_elem_bits_{4};
  int a_from_mls_{0};
  int b_from_mls_{0};
  int a_mls_trans_{1};
  int b_mls_trans_{1};
  int min_m_per_warp_{0};
  int min_n_per_warp_{0};
  PrimExpr row_base_{IntImm(DataType::Int(32), 0)};
  mutable bool completed_ = false;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.CopyScale", CopyScaleNode,
                                    TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<CopyScaleNode>()
        .def_ro("src", &CopyScaleNode::src)
        .def_ro("dst", &CopyScaleNode::dst)
        .def_ro("src_ranges", &CopyScaleNode::src_ranges)
        .def_ro("dst_ranges", &CopyScaleNode::dst_ranges)
        .def_ro("op_ctrl_", &CopyScaleNode::op_ctrl_)
        .def_ro("has_scale_view_", &CopyScaleNode::has_scale_view_)
        .def_ro("scale_format_", &CopyScaleNode::scale_format_)
        .def_ro("granularity_mn_", &CopyScaleNode::granularity_mn_)
        .def_ro("granularity_k_", &CopyScaleNode::granularity_k_)
        .def_ro("scale_k_major_", &CopyScaleNode::scale_k_major_)
        .def_ro("role_", &CopyScaleNode::role_)
        .def_ro("row_base_", &CopyScaleNode::row_base_);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  TileOperator Clone() const override;
};

class CopyScale : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(CopyScale, TileOperator,
                                             CopyScaleNode);
  TVM_DLL
  CopyScale(Array<PrimExpr> args,
            Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_COPY_SCALE_H_
