// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file copy_scale.cc
 * \brief CopyScale: LDS scale -> HCU scale_buffer (ds_scale_copy).
 */

#include "copy_scale.h"

#include "hcu/op/gemm_partition.h"
#include "hcu/target_utils.h"
#include "hcu/utils/layout_functor.h"
#include "hcu/utils/scale_gemm_dep.h"
#include "hcu/utils/scale_lds_layout.h"
#include "layout/layout.h"
#include "op/builtin.h"
#include "op/gemm.h"
#include "op/region.h"
#include "op/utils.h"

#include <algorithm>
#include <sstream>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

int GetIntAnnOrDefault(const Map<String, ObjectRef> &annotations,
                       const char *key, int default_val) {
  if (auto val = annotations.Get(key)) {
    if (const auto *imm = val->as<IntImmNode>()) {
      return static_cast<int>(imm->value);
    }
    LOG(FATAL) << "Annotation `" << key << "` expects IntImm, got "
               << val->GetTypeKey();
  }
  return default_val;
}

PrimExpr GetRowBaseFromAnnotations(const Map<String, ObjectRef> &annotations) {
  if (auto val = annotations.Get(attr::kScaleRowBase)) {
    if (const auto *vnode = val->as<VarNode>()) {
      return GetRef<Var>(vnode);
    }
    if (const auto *imm = val->as<IntImmNode>()) {
      return IntImm(imm->dtype, imm->value);
    }
    LOG(FATAL) << "Annotation `" << attr::kScaleRowBase
               << "` expects Var or IntImm, got " << val->GetTypeKey();
  }
  return IntImm(DataType::Int(32), 0);
}

size_t ScaleFormatPhysicalRank(ScaleLdsFormat format) {
  if (format == ScaleLdsFormat::kIdentity) {
    return 2;
  }
  if (format == ScaleLdsFormat::kK2MN2Interleave) {
    return 5;
  }
  return 3;
}

PrimExpr FlattenLayoutOffset(const Layout &layout,
                             const Array<PrimExpr> &inputs) {
  Array<PrimExpr> physical = layout->Forward(inputs);
  Array<PrimExpr> shape = layout->OutputShape();
  ICHECK_EQ(physical.size(), shape.size());
  PrimExpr offset = make_zero(DataType::Int(32));
  PrimExpr stride = make_const(DataType::Int(32), 1);
  for (int i = static_cast<int>(physical.size()) - 1; i >= 0; --i) {
    offset = offset + Cast(DataType::Int(32), physical[i]) * stride;
    stride = stride * Cast(DataType::Int(32), shape[i]);
  }
  return offset;
}

void ValidateScaleStageBroadcastLayout(const Layout &layout,
                                       size_t leading_stage_dims,
                                       arith::Analyzer *analyzer) {
  if (leading_stage_dims == 0) {
    return;
  }
  const size_t rank = layout->InputDim();
  ICHECK_LT(leading_stage_dims, rank);
  Array<PrimExpr> sx, s0, zx, z0;
  for (size_t i = 0; i < rank; ++i) {
    PrimExpr value =
        Var(i < leading_stage_dims ? "scale_stage_" + std::to_string(i)
                                   : "scale_plane_" + std::to_string(i),
            DataType::Int(32));
    PrimExpr zero = make_zero(DataType::Int(32));
    sx.push_back(value);
    s0.push_back(i < leading_stage_dims ? value : zero);
    zx.push_back(i < leading_stage_dims ? zero : value);
    z0.push_back(zero);
  }
  PrimExpr interaction =
      FlattenLayoutOffset(layout, sx) - FlattenLayoutOffset(layout, s0) -
      FlattenLayoutOffset(layout, zx) + FlattenLayoutOffset(layout, z0);
  interaction = analyzer->Simplify(interaction);
  ICHECK(analyzer->CanProveEqual(interaction, make_zero(interaction.dtype())))
      << "copy_scale ScaleView stage dimensions must be broadcast-only in "
         "annotated layout; stage/plane coordinates may not be mixed, got "
      << interaction << " from layout " << layout->DebugOutput();
}

} // namespace

CopyScale::CopyScale(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ICHECK(args.size() >= 3)
      << "copy_scale expects at least 3 args: src_region, dst_region, op_ctrl";
  auto src_call = args[0].as<CallNode>();
  auto dst_call = args[1].as<CallNode>();
  ICHECK(src_call) << "copy_scale args[0] must be region call (src)";
  ICHECK(dst_call) << "copy_scale args[1] must be region call (dst)";

  auto src_region = RegionOp(src_call->args);
  auto dst_region = RegionOp(dst_call->args);
  auto src_ranges = src_region->GetRanges();
  auto dst_ranges = dst_region->GetRanges();

  ICHECK(src_ranges.size() >= 2) << "copy_scale src region must be at least 2D "
                                    "(optional leading stage dims)";
  ICHECK_EQ(dst_ranges.size(), 2u)
      << "copy_scale dst (scale_buffer) region must be exactly 2D, got rank="
      << dst_ranges.size();

  Buffer src_buf = src_region->GetBuffer();
  Buffer dst_buf = dst_region->GetBuffer();
  ICHECK(src_buf.scope() == "shared" || src_buf.scope() == "shared.dyn")
      << "copy_scale src must be shared memory, got scope=" << src_buf.scope();
  ICHECK(dst_buf.scope() == "shared.scale")
      << "copy_scale dst must be shared.scale, got scope=" << dst_buf.scope();
  ICHECK_EQ(dst_buf->shape.size(), 2u)
      << "copy_scale dst scale_buffer must be exactly 2D, got rank="
      << dst_buf->shape.size();

  ObjectPtr<CopyScaleNode> node = tvm::ffi::make_object<CopyScaleNode>();
  node->src = src_buf;
  node->dst = dst_buf;
  node->src_ranges = src_ranges;
  node->dst_ranges = dst_ranges;
  node->op_ctrl_ = args[2].as<IntImm>().value()->value;
  ICHECK(node->op_ctrl_ >= 0 && node->op_ctrl_ <= 2)
      << "copy_scale op_ctrl must be 0/1/2, got " << node->op_ctrl_;
  ICHECK(args.size() == 3 || args.size() == 10)
      << "copy_scale expects 3 args for a plain Buffer or 10 args for a "
         "ScaleView";
  if (args.size() == 10) {
    node->has_scale_view_ = true;
    node->scale_format_ = args[3].as<IntImm>().value()->value;
    node->parent_k_ = args[4];
    node->parent_mn_ = args[5];
    node->origin_k_ = args[6];
    node->origin_mn_ = args[7];
    node->tile_k_ = args[8];
    node->tile_mn_ = args[9];
  } else {
    ICHECK_EQ(node->op_ctrl_, 0)
        << "copy_scale op_ctrl>0 requires an explicit ScaleView/ScaleFormat";
  }

  node->granularity_mn_ =
      GetIntAnnOrDefault(annotations, attr::kScaleGranularityMN, 1);
  node->granularity_k_ =
      GetIntAnnOrDefault(annotations, attr::kScaleGranularityK, 1);
  node->scale_k_major_ = GetIntAnnOrDefault(annotations, attr::kScaleKMajor, 0);
  node->role_ = GetIntAnnOrDefault(annotations, attr::kScaleRole, 0);
  node->gemm_m_ = GetIntAnnOrDefault(annotations, attr::kScaleGemmM, 0);
  node->gemm_n_ = GetIntAnnOrDefault(annotations, attr::kScaleGemmN, 0);
  node->gemm_k_ = GetIntAnnOrDefault(annotations, attr::kScaleGemmK, 0);
  node->gemm_policy_ =
      GetIntAnnOrDefault(annotations, attr::kScaleGemmPolicy, 0);
  node->gemm_k_pack_ =
      GetIntAnnOrDefault(annotations, attr::kScaleGemmKPack, 1);
  node->gemm_elem_bits_ =
      GetIntAnnOrDefault(annotations, attr::kScaleGemmElemBits, 4);
  node->a_from_mls_ = GetIntAnnOrDefault(annotations, attr::kScaleAFromMls, 0);
  node->b_from_mls_ = GetIntAnnOrDefault(annotations, attr::kScaleBFromMls, 0);
  node->a_mls_trans_ =
      GetIntAnnOrDefault(annotations, attr::kScaleAMlsTrans, 1);
  node->b_mls_trans_ =
      GetIntAnnOrDefault(annotations, attr::kScaleBMlsTrans, 1);
  node->min_m_per_warp_ =
      GetIntAnnOrDefault(annotations, attr::kScaleMinMPerWarp, 0);
  node->min_n_per_warp_ =
      GetIntAnnOrDefault(annotations, attr::kScaleMinNPerWarp, 0);
  node->row_base_ = GetRowBaseFromAnnotations(annotations);

  AccessRegion src_access{BufferRegion(node->src, node->src_ranges),
                          kAccessRead};
  AccessRegion dst_access{BufferRegion(node->dst, node->dst_ranges),
                          kAccessWrite};
  node->SetAccessRegions({src_access, dst_access});
  data_ = std::move(node);
}

TileOperator CopyScaleNode::Clone() const {
  auto op = tvm::ffi::make_object<CopyScaleNode>(*this);
  return CopyScale(op);
}

LayoutMap CopyScaleNode::InferLayout(const LayoutInferArgs &T,
                                     InferLevel level) const {
  // Like GEMM, copy_scale is the defining consumer for its source LDS layout.
  // Select exactly once in the first strict inference; normal T.copy then
  // propagates from this anchor during the common/free BFS stages.
  if (!IsSharedBuffer(src) || completed_ || level != InferLevel::kStrict) {
    return {};
  }
  completed_ = true;
  LayoutMap result;
  // Explicit T.annotate_layout entries seed layout_map before strict inference
  // and always take precedence over automatic selection.
  if (T.layout_map.count(src)) {
    if (has_scale_view_) {
      const auto format = static_cast<ScaleLdsFormat>(scale_format_);
      const size_t format_rank = ScaleFormatPhysicalRank(format);
      ICHECK_GE(src->shape.size(), format_rank);
      ValidateScaleStageBroadcastLayout(
          T.layout_map[src], src->shape.size() - format_rank, T.analyzer);
    }
    return result;
  }
  Optional<Layout> automatic = hcu::SelectAutoScaleLdsLayout(this, T);
  result.Set(src, automatic.defined() ? automatic.value()
                                      : MakeLinearLayout(src->shape));
  return result;
}

Stmt CopyScaleNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (!TargetIsHCU(T.target)) {
    LOG(FATAL) << "copy_scale is only supported on HCU target";
  }

  ICHECK_EQ(dst->shape.size(), 2u)
      << "copy_scale dst scale_buffer must be exactly 2D";
  const int64_t *shape_mn_c = nullptr;
  const int64_t *shape_k_c = nullptr;
  if (scale_k_major_) {
    shape_mn_c = as_const_int(dst->shape[0]);
    shape_k_c = as_const_int(dst->shape[1]);
  } else {
    shape_k_c = as_const_int(dst->shape[0]);
    shape_mn_c = as_const_int(dst->shape[1]);
  }
  ICHECK(shape_mn_c && shape_k_c)
      << "copy_scale requires static dst shape dims";
  const int scale_shape_mn = static_cast<int>(*shape_mn_c);
  const int scale_shape_k = static_cast<int>(*shape_k_c);

  const auto scale_format = static_cast<ScaleLdsFormat>(scale_format_);
  ICHECK(gemm_elem_bits_ == 4 || gemm_elem_bits_ == 8)
      << "copy_scale requires consumer GEMM operand bits 4 or 8, got "
      << gemm_elem_bits_;
  const int kMmacK = gemm_elem_bits_ == 8 ? 32 : 64;
  if (op_ctrl_ != 0) {
    ICHECK_EQ(scale_k_major_, 0) << "copy_scale op_ctrl=1/2 requires MN-major "
                                    "scale (scale_k_major=false)";
    if (kMmacK == 64) {
      ICHECK_EQ(granularity_k_, 32)
          << "copy_scale mmac_k=64 op_ctrl=1/2 requires granularity_k==32, got "
          << granularity_k_;
    }
    ICHECK(granularity_k_ % 32 == 0);
    if (op_ctrl_ == 1 && kMmacK == 64) {
      ICHECK_EQ(scale_shape_k % 2, 0)
          << "copy_scale mmac_k=64 op_ctrl=1 requires even ScaleShapeK";
    } else if (kMmacK == 64) {
      ICHECK(scale_shape_k % 4 == 0 || scale_shape_k == 2)
          << "copy_scale mmac_k=64 op_ctrl=2 requires scaleShapeK %4==0 (K4) "
             "or "
             "==2 (K2MN2); ==1 is mmac_k=32-only, got "
          << scale_shape_k;
    } else {
      ICHECK(scale_shape_k % 4 == 0 || scale_shape_k == 2 || scale_shape_k == 1)
          << "copy_scale mmac_k=32 op_ctrl=2 requires scaleShapeK %4==0 / ==2 "
             "/ "
             "==1; half-superblock illegal, got "
          << scale_shape_k;
    }
  } else {
    ICHECK_EQ(granularity_k_ % 32, 0)
        << "copy_scale granularity_k must be divisible by 32, got "
        << granularity_k_;
  }

  int block_size = static_cast<int>(*as_const_int(T.thread_bounds->extent));
  int warp_size = TargetHcuGetWarpSize(T.target);
  ICHECK(block_size % warp_size == 0)
      << "copy_scale requires thread_bounds.extent divisible by warp_size="
      << warp_size;
  const int num_warps = block_size / warp_size;

  ICHECK(gemm_m_ > 0 && gemm_n_ > 0 && gemm_k_ > 0)
      << "copy_scale missing gemm M/N/K annotations from AnnotateScaleGemmDep";
  auto policy = GemmWarpPolicy(gemm_policy_);
  hcu::ScaleWarpSeg seg = hcu::ComputeScaleWarpSeg(
      *policy.get(), gemm_m_, gemm_n_, gemm_k_, gemm_k_pack_, gemm_elem_bits_,
      block_size, T.target, a_from_mls_ != 0, b_from_mls_ != 0,
      a_mls_trans_ != 0, b_mls_trans_ != 0, min_m_per_warp_, min_n_per_warp_);
  ICHECK_EQ(seg.k_warp, 1)
      << "gemm_blockscaled / copy_scale requires k_warp==1, got " << seg.k_warp;
  ICHECK_EQ(seg.total_warps, num_warps);
  const int mn_warps = (role_ == 0) ? seg.m_seg : seg.n_seg;
  ICHECK(mn_warps >= 1 && seg.total_warps % mn_warps == 0)
      << "copy_scale: TotalWarps=" << seg.total_warps
      << " must be divisible by MnWarps=" << mn_warps;

  PrimExpr warp_id = FloorDiv(T.thread_var - T.thread_bounds->min, warp_size);

  // The remapped buffer may already be flattened by LowerTileOp.  Addressing
  // metadata always comes from the logical buffer / ScaleView descriptor.
  Buffer src_phys = T.buffer_remap.count(src) ? T.buffer_remap[src] : src;

  size_t sr = src_ranges.size();
  ICHECK_EQ(src->shape.size(), sr)
      << "copy_scale physical src buffer rank must match its access region";

  PrimExpr origin_mn;
  PrimExpr origin_k;
  PrimExpr src_elem_offset;
  size_t leading_stage_dims = 0;
  const int64_t *parent_mn_c = nullptr;
  const int64_t *parent_k_c = nullptr;

  if (has_scale_view_) {
    // ScaleView keeps logical [K, MN] metadata independent of the true
    // physical Buffer rank/shape. The StorageLayout functor maps the physical
    // coordinates selected by the format policy to a byte offset.
    ICHECK_EQ(scale_k_major_, 0)
        << "ScaleView formats currently require MN-major destination scale";
    parent_k_c = as_const_int(parent_k_);
    parent_mn_c = as_const_int(parent_mn_);
    const int64_t *tile_k_c = as_const_int(tile_k_);
    const int64_t *tile_mn_c = as_const_int(tile_mn_);
    ICHECK(parent_k_c && parent_mn_c && tile_k_c && tile_mn_c)
        << "ScaleView Parent/Tile shapes must be static";
    ICHECK_EQ(*tile_k_c, scale_shape_k);
    ICHECK_EQ(*tile_mn_c, scale_shape_mn);
    size_t format_rank = 0;
    if (scale_format == ScaleLdsFormat::kIdentity) {
      format_rank = 2;
    } else if (scale_format == ScaleLdsFormat::kK2MN2Interleave) {
      format_rank = 5;
    } else {
      format_rank = 3;
    }
    ICHECK_GE(sr, format_rank);
    leading_stage_dims = sr - format_rank;
    for (size_t i = 0; i < leading_stage_dims; ++i) {
      const int64_t *extent = as_const_int(src_ranges[i]->extent);
      ICHECK(extent && *extent == 1)
          << "ScaleView leading stage dimensions must be single indices";
    }
    if (scale_format == ScaleLdsFormat::kIdentity) {
      ICHECK_EQ(op_ctrl_, 0)
          << "identity ScaleView requires copy_scale op_ctrl=0";
      ICHECK(StructuralEqual()(src->shape[sr - 2], parent_k_) &&
             StructuralEqual()(src->shape[sr - 1], parent_mn_))
          << "Identity ScaleView physical Buffer must be [ParentK, ParentMN]";
    } else if (scale_format == ScaleLdsFormat::kK2Interleave) {
      ICHECK_EQ(op_ctrl_, 1) << "K2 ScaleView requires copy_scale op_ctrl=1";
      ICHECK_EQ(scale_shape_k % 2, 0)
          << "K2 ScaleView requires tile ScaleShapeK divisible by 2";
      ICHECK_EQ(*parent_k_c % 2, 0);
    } else if (scale_format == ScaleLdsFormat::kK4Interleave) {
      ICHECK_EQ(op_ctrl_, 2) << "K4 ScaleView requires copy_scale op_ctrl=2";
      ICHECK_EQ(scale_shape_k % 4, 0)
          << "K4 ScaleView requires tile ScaleShapeK divisible by 4";
      ICHECK_EQ(*parent_k_c % 4, 0);
    } else if (scale_format == ScaleLdsFormat::kK2MN2Interleave) {
      ICHECK_EQ(op_ctrl_, 2) << "K2MN2 ScaleView requires copy_scale op_ctrl=2";
      ICHECK_EQ(scale_shape_k % 2, 0)
          << "K2MN2 ScaleView requires even tile ScaleShapeK";
      ICHECK_EQ(*parent_k_c % 2, 0);
      ICHECK_EQ(*parent_mn_c % 32, 0);
    } else if (scale_format == ScaleLdsFormat::kMN2Interleave) {
      ICHECK_EQ(op_ctrl_, 1);
      ICHECK_EQ(*parent_mn_c % 2, 0);
    } else {
      ICHECK(scale_format == ScaleLdsFormat::kMN4Interleave)
          << "unknown ScaleView format id " << scale_format_;
      ICHECK_EQ(op_ctrl_, 2);
      ICHECK_EQ(*parent_mn_c % 4, 0);
    }
    origin_k = origin_k_;
    origin_mn = origin_mn_;
    Array<PrimExpr> src_idx;
    DataType idx_dtype = src->DefaultIndexType();
    for (size_t i = 0; i < sr; ++i) {
      src_idx.push_back(i < leading_stage_dims
                            ? Cast(idx_dtype, src_ranges[i]->min)
                            : make_const(idx_dtype, 0));
    }
    Array<PrimExpr> src_offs = src.OffsetOf(src_idx);
    ICHECK_EQ(src_offs.size(), 1u);
    src_elem_offset = src_offs[0];
  } else {
    ICHECK(sr >= 2) << "copy_scale src region must be at least 2D";
    const int64_t *ext_a = as_const_int(src_ranges[sr - 2]->extent);
    const int64_t *ext_b = as_const_int(src_ranges[sr - 1]->extent);
    ICHECK(ext_a && ext_b)
        << "copy_scale requires static src region last-2 extents";
    if (scale_k_major_) {
      ICHECK_EQ(static_cast<int>(*ext_a), scale_shape_mn);
      ICHECK_EQ(static_cast<int>(*ext_b), scale_shape_k);
    } else {
      ICHECK_EQ(static_cast<int>(*ext_a), scale_shape_k);
      ICHECK_EQ(static_cast<int>(*ext_b), scale_shape_mn);
    }
    Array<PrimExpr> src_idx;
    DataType idx_dtype = src->DefaultIndexType();
    for (size_t j = 0; j < sr; ++j) {
      src_idx.push_back(j + 2 < sr ? Cast(idx_dtype, src_ranges[j]->min)
                                   : make_const(idx_dtype, 0));
    }
    Array<PrimExpr> src_offs = src.OffsetOf(src_idx);
    ICHECK_EQ(src_offs.size(), 1u);
    src_elem_offset = src_offs[0];
    const PrimExpr origin_dim0 = src_ranges[sr - 2]->min;
    const PrimExpr origin_dim1 = src_ranges[sr - 1]->min;
    origin_mn = scale_k_major_ ? origin_dim0 : origin_dim1;
    origin_k = scale_k_major_ ? origin_dim1 : origin_dim0;
    const PrimExpr parent_dim0 = src->shape[sr - 2];
    const PrimExpr parent_dim1 = src->shape[sr - 1];
    parent_mn_c = as_const_int(scale_k_major_ ? parent_dim0 : parent_dim1);
    parent_k_c = as_const_int(scale_k_major_ ? parent_dim1 : parent_dim0);
  }

  ICHECK(parent_mn_c && parent_k_c)
      << "copy_scale requires static parent last-2 shape";
  const int parent_scale_mn = static_cast<int>(*parent_mn_c);
  const int parent_scale_k = static_cast<int>(*parent_k_c);
  ICHECK_LE(scale_shape_mn, parent_scale_mn);
  ICHECK_LE(scale_shape_k, parent_scale_k);

  ICHECK(T.layout_map.count(src))
      << "copy_scale requires the resolved layout for LDS buffer `" << src->name
      << "`; LayoutInference should provide either annotate layout or linear";
  Layout resolved_layout = T.layout_map[src];
  ICHECK(StructuralEqual()(resolved_layout->InputShape(), src->shape))
      << "copy_scale resolved layout input shape must match the true physical "
         "LDS Buffer shape; layout="
      << resolved_layout->DebugOutput() << ", buffer shape=" << src->shape;
  if (has_scale_view_ && leading_stage_dims != 0) {
    Array<PrimExpr> layout_inputs;
    DataType idx_dtype = src->DefaultIndexType();
    for (size_t i = 0; i < sr; ++i) {
      layout_inputs.push_back(i < leading_stage_dims
                                  ? Cast(idx_dtype, src_ranges[i]->min)
                                  : make_const(idx_dtype, 0));
    }
    Array<PrimExpr> physical = resolved_layout->Forward(layout_inputs);
    Array<PrimExpr> physical_shape = resolved_layout->OutputShape();
    ICHECK_EQ(physical.size(), physical_shape.size());
    src_elem_offset = make_zero(idx_dtype);
    PrimExpr stride = make_const(idx_dtype, 1);
    for (int i = static_cast<int>(physical.size()) - 1; i >= 0; --i) {
      src_elem_offset = src_elem_offset + Cast(idx_dtype, physical[i]) * stride;
      stride = stride * Cast(idx_dtype, physical_shape[i]);
    }
    src_elem_offset = analyzer->Simplify(src_elem_offset);
  }
  auto src_ptr = src_phys.access_ptr(1, DataType::Handle(), 1, src_elem_offset);
  hcu::GeneratedLayoutFunctor generated_layout = hcu::MakeLayoutOffsetFunctor(
      resolved_layout, src->dtype, analyzer, "TLGeneratedScaleLdsLayout_",
      leading_stage_dims);

  std::stringstream ss;
  ss << "tl::hcu::ds_scale_copy<" << op_ctrl_ << ", " << scale_format_ << ", "
     << parent_scale_mn << ", " << parent_scale_k << ", " << scale_shape_mn
     << ", " << scale_shape_k << ", " << granularity_mn_ << ", "
     << granularity_k_ << ", " << scale_k_major_ << ", " << mn_warps << ", "
     << seg.total_warps << ", " << generated_layout.name << ", " << kMmacK
     << ">";

  Array<PrimExpr> call_args;
  call_args.push_back(StringImm(ss.str()));
  call_args.push_back(src_ptr);
  call_args.push_back(row_base_);
  call_args.push_back(Cast(DataType::Int(32), warp_id));
  call_args.push_back(Cast(DataType::Int(32), origin_mn));
  call_args.push_back(Cast(DataType::Int(32), origin_k));

  Stmt copy_call =
      Evaluate(Call(DataType::Handle(), builtin::call_extern(), call_args));
  Array<PrimExpr> register_args{
      StringImm("__tl_hcu_register_scale_layout_functor"),
      StringImm(generated_layout.name), StringImm(generated_layout.source)};
  Stmt register_functor =
      Evaluate(Call(DataType::Handle(), builtin::call_extern(), register_args));
  return SeqStmt({register_functor, copy_call});
}

TIR_REGISTER_TL_TILE_OP(CopyScale, copy_scale)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { CopyScaleNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
