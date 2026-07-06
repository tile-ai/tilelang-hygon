/*!
 * \file ds_read_format.cc
 * \brief DsReadFormat: read MLS-formatted shared memory into register.
 */

#include "ds_read_format.h"
#include "../layout/utils.h"
#include "../target/utils.h"
#include "builtin.h"
#include "gemm.h"
#include "mls.h"
#include "region.h"
#include <tvm/node/structural_equal.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
namespace tvm {
namespace tl {

using namespace tir;

namespace {

// mls_ds_traits: (mls_tile_mn, mls_tile_k, trans, alt) -> (read_tile_mn,
// read_tile_k).
using MlsReadKey = std::tuple<int, int, bool, int>;
using MlsReadMap = std::map<MlsReadKey, std::pair<int, int>>;

// b16: from mls_ds_traits.hpp (gfx938/gfx946).
static const MlsReadMap kMlsToReadTileB16 = {
    {{32, 16, false, 1}, {32, 16}}, {{32, 16, false, 2}, {32, 16}},
    {{64, 16, false, 1}, {32, 16}}, {{64, 16, false, 2}, {32, 16}},
    {{32, 32, false, 1}, {32, 16}}, {{32, 32, false, 2}, {32, 16}},
    {{16, 32, true, 1}, {32, 16}},  {{16, 32, true, 2}, {16, 32}},
    {{32, 32, true, 1}, {32, 16}},  {{32, 32, true, 2}, {16, 32}},
    {{16, 64, true, 1}, {32, 16}},  {{16, 64, true, 2}, {16, 32}},
};

// b8: from mls_ds_traits.hpp (gfx938/gfx946).
static const MlsReadMap kMlsToReadTileB8 = {
    {{64, 16, false, 1}, {32, 32}},  {{64, 16, false, 2}, {32, 32}},
    {{64, 16, false, 4}, {64, 16}},  {{128, 16, false, 1}, {32, 32}},
    {{128, 16, false, 2}, {32, 32}}, {{128, 16, false, 4}, {64, 16}},
    {{64, 32, false, 1}, {32, 32}},  {{64, 32, false, 2}, {32, 32}},
    {{64, 32, false, 4}, {64, 16}},  {{16, 64, true, 1}, {16, 64}},
    {{16, 128, true, 1}, {16, 64}},  {{16, 128, true, 2}, {32, 32}},
    {{16, 128, true, 4}, {32, 32}},  {{32, 64, true, 1}, {16, 64}},
    {{32, 64, true, 2}, {32, 32}},   {{32, 64, true, 4}, {32, 32}},
};

void GetReadTileFromMlsTile(bool trans, int mls_tile_mn, int mls_tile_k,
                            int alt, int elem_bytes, int &read_tile_mn,
                            int &read_tile_k) {
  MlsReadKey key(mls_tile_mn, mls_tile_k, trans, alt);
  const MlsReadMap *m =
      (elem_bytes == 2) ? &kMlsToReadTileB16 : &kMlsToReadTileB8;
  auto it = m->find(key);
  if (it != m->end()) {
    read_tile_mn = it->second.first;
    read_tile_k = it->second.second;
  } else {
    LOG(FATAL) << "GetReadTileFromMlsTile: no entry for (mls_mn=" << mls_tile_mn
               << ", mls_k=" << mls_tile_k << ", trans=" << trans
               << ", alt=" << alt << ", elem_bytes=" << elem_bytes
               << "). Add to kMlsToReadTileB16/B8.";
  }
}

void ComputeDsReadFormatWarpPartition(bool trans, int block_mn, int block_k,
                                      int block_size, Target target,
                                      int read_tile_mn, int read_tile_k,
                                      int &warp_mn, int &warp_k) {
  int num_warps = block_size / TargetGetWarpSize(target);
  ICHECK(num_warps >= 1);
  ICHECK(block_mn % read_tile_mn == 0 && block_k % read_tile_k == 0);
  int max_wm = std::min(block_mn / read_tile_mn, num_warps);
  int max_wk = std::min(block_k / read_tile_k, num_warps);
  warp_k = num_warps / max_wm;
  warp_k = warp_k > max_wk ? max_wk : warp_k;
  warp_mn = num_warps / warp_k;
}

LayoutMap InferLayoutWithGemmDep(const DsReadFormatNode *self,
                                 const MlsGemmDepMetaNode *meta,
                                 const LayoutInferArgs &T) {
  LayoutMap result;
  GemmWarpPolicy policy(meta->gemm_policy);
  GemmInst gemm_inst = GemmInst::kHCUMMAC;
  int block_size = *as_const_int(T.thread_bounds->extent);
  int element_byte_size = self->src->dtype.bits() / 8;
  const bool a_mls_trans = !meta->gemm_trans_a;
  const bool b_mls_trans = meta->gemm_trans_b;
  auto [warp_m, warp_n, warp_k] = policy->computeWarpPartitionHCU(
      meta->gemm_m, meta->gemm_n, meta->gemm_k, meta->gemm_k_pack,
      element_byte_size, block_size, T.target, gemm_inst, meta->a_from_mls,
      meta->b_from_mls, a_mls_trans, b_mls_trans);
  const int min_n_per_warp = (meta->feeds_slot == 1 && !b_mls_trans) ? 32 : 16;
  Fragment fragment =
      meta->feeds_slot == 0
          ? makeGemmFragmentAHCU(meta->gemm_m, meta->gemm_n, meta->gemm_k,
                                 warp_m, warp_n, warp_k,
                                 self->src->dtype.bits(), meta->gemm_k_pack,
                                 meta->gemm_trans_a)
          : makeGemmFragmentBHCU(meta->gemm_m, meta->gemm_n, meta->gemm_k,
                                 warp_m, warp_n, warp_k,
                                 self->src->dtype.bits(), meta->gemm_k_pack,
                                 meta->gemm_trans_b, min_n_per_warp);
  auto layout = fragment->BindThreadRange(T.thread_bounds);
  if (T.layout_map.count(self->dst)) {
    if (!tvm::StructuralEqual()(layout, T.layout_map[self->dst])) {
      LOG(FATAL) << "DsReadFormat layout conflict: inferred layout differs "
                    "from existing "
                    "layout for buffer "
                 << self->dst;
    }
  } else {
    result.Set(self->dst, layout);
  }
  return result;
}

} // namespace

DsReadFormat::DsReadFormat(Array<PrimExpr> args,
                           Map<String, ObjectRef> annotations) {
  ICHECK(args.size() >= 2)
      << "ds_read_format expects at least 2 args: src_region, dst_region";
  auto src_call = args[0].as<CallNode>();
  auto dst_call = args[1].as<CallNode>();
  ICHECK(src_call) << "ds_read_format args[0] must be region call (src)";
  ICHECK(dst_call) << "ds_read_format args[1] must be region call (dst)";

  auto src_region = RegionOp(src_call->args);
  auto dst_region = RegionOp(dst_call->args);
  auto src_ranges = src_region->GetRanges();
  auto dst_ranges = dst_region->GetRanges();

  ICHECK(src_ranges.size() >= 2) << "ds_read_format src region must be 2D";
  ICHECK(dst_ranges.size() >= 2) << "ds_read_format dst region must be 2D";

  Buffer src_buf = src_region->GetBuffer();
  Buffer dst_buf = dst_region->GetBuffer();
  ICHECK(src_buf.scope() == "shared" || src_buf.scope() == "shared.dyn")
      << "ds_read_format src must be shared memory, got scope="
      << src_buf.scope();
  ICHECK(dst_buf.scope() == "local.fragment")
      << "ds_read_format dst must be register (local.fragment), got scope="
      << dst_buf.scope();

  ObjectPtr<DsReadFormatNode> node = tvm::ffi::make_object<DsReadFormatNode>();
  node->src = src_buf;
  node->dst = dst_buf;
  node->src_ranges = src_ranges;
  node->dst_ranges = dst_ranges;
  AccessRegion src_access{BufferRegion(node->src, node->src_ranges),
                          kAccessRead};
  AccessRegion dst_access{BufferRegion(node->dst, node->dst_ranges),
                          kAccessWrite};
  node->SetAccessRegions({src_access, dst_access});
  node->gemm_dep_ = GetMlsGemmDepFromAnnotations(annotations);
  data_ = std::move(node);
}

TileOperator DsReadFormatNode::Clone() const {
  auto op = tvm::ffi::make_object<DsReadFormatNode>(*this);
  return DsReadFormat(op);
}

LayoutMap DsReadFormatNode::InferLayout(const LayoutInferArgs &T,
                                        InferLevel level) const {
  if (completed_)
    return {};

  LayoutMap result;
  if (gemm_dep_.defined()) {
    result = InferLayoutWithGemmDep(this, gemm_dep_.get(), T);
  } else {
    const bool trans = true;
    int64_t block_mn = *as_const_int(dst->shape[0]);
    int64_t block_k = *as_const_int(dst->shape[1]);
    int block_size = *as_const_int(T.thread_bounds->extent);
    int w_mn, w_k, t_mn, t_k;
    ComputeMlsWarpPartition(trans, static_cast<int>(block_mn),
                            static_cast<int>(block_k), block_size, T.target,
                            src->dtype.bits(), w_mn, w_k, t_mn, t_k);
    const int alt = 1;
    const int elem_bytes = src->dtype.bits() / 8;
    int read_tile_mn, read_tile_k;
    GetReadTileFromMlsTile(trans, t_mn, t_k, alt, elem_bytes, read_tile_mn,
                           read_tile_k);
    int warp_mn, warp_k;
    ComputeDsReadFormatWarpPartition(
        trans, static_cast<int>(block_mn), static_cast<int>(block_k),
        block_size, T.target, read_tile_mn, read_tile_k, warp_mn, warp_k);
    int num_warp_mn_no_recompute =
        std::min(warp_mn, static_cast<int>(block_mn / read_tile_mn));
    if (num_warp_mn_no_recompute < 1)
      num_warp_mn_no_recompute = 1;
    auto fragment = makeDsReadFormatFragmentHCU(
        static_cast<int>(block_mn), static_cast<int>(block_k), warp_mn, warp_k,
        src->dtype.bits(), num_warp_mn_no_recompute, trans);
    auto layout = fragment->BindThreadRange(T.thread_bounds);
    if (T.layout_map.count(dst)) {
      if (!tvm::StructuralEqual()(layout, T.layout_map[dst])) {
        LOG(FATAL)
            << "DsReadFormat layout conflict: inferred layout differs from "
               "existing layout for buffer "
            << dst;
      }
    } else {
      result.Set(dst, layout);
    }
  }
  this->completed_ = true;
  return result;
}

Stmt DsReadFormatNode::Lower(const LowerArgs &T,
                             arith::Analyzer *analyzer) const {
  (void)analyzer;
  if (!TargetIsHCU(T.target)) {
    LOG(FATAL) << "ds_read_format is only supported on HCU target";
  }

  int block_size = static_cast<int>(*as_const_int(T.thread_bounds->extent));
  bool ds_trans = true;
  int64_t block_mn = *as_const_int(dst->shape[0]);
  int64_t block_k = *as_const_int(dst->shape[1]);
  int tile_mn = 0;
  int tile_k = 0;
  int w_mn = 0;
  int w_k = 0;
  if (gemm_dep_.defined()) {
    const auto *meta = gemm_dep_.get();
    ds_trans = meta->trans;
    block_mn =
        ds_trans ? *as_const_int(dst->shape[0]) : *as_const_int(dst->shape[1]);
    block_k =
        ds_trans ? *as_const_int(dst->shape[1]) : *as_const_int(dst->shape[0]);
    int dummy_mn, dummy_k;
    ComputeMlsWarpPartition(ds_trans, static_cast<int>(block_mn),
                            static_cast<int>(block_k), block_size, T.target,
                            src->dtype.bits(), dummy_mn, dummy_k, tile_mn,
                            tile_k);
    GemmWarpPolicy policy(meta->gemm_policy);
    const bool a_mls_trans = !meta->gemm_trans_a;
    const bool b_mls_trans = meta->gemm_trans_b;
    auto [warp_m, warp_n, warp_k_part] = policy->computeWarpPartitionHCU(
        meta->gemm_m, meta->gemm_n, meta->gemm_k, meta->gemm_k_pack,
        src->dtype.bits() / 8, block_size, T.target, GemmInst::kHCUMMAC,
        meta->a_from_mls, meta->b_from_mls, a_mls_trans, b_mls_trans);
    w_mn = meta->feeds_slot == 0 ? warp_m : warp_n;
    w_k = warp_k_part;
  } else {
    int dummy_mn, dummy_k;
    ComputeMlsWarpPartition(ds_trans, static_cast<int>(block_mn),
                            static_cast<int>(block_k), block_size, T.target,
                            src->dtype.bits(), dummy_mn, dummy_k, tile_mn,
                            tile_k);
    const int alt = 1;
    const int elem_bytes = src->dtype.bits() / 8;
    int read_tile_mn, read_tile_k;
    GetReadTileFromMlsTile(ds_trans, tile_mn, tile_k, alt, elem_bytes,
                           read_tile_mn, read_tile_k);
    ComputeDsReadFormatWarpPartition(
        ds_trans, static_cast<int>(block_mn), static_cast<int>(block_k),
        block_size, T.target, read_tile_mn, read_tile_k, w_mn, w_k);
  }

  std::string dtype_str;
  if (src->dtype.is_bfloat16()) {
    dtype_str = "bfloat16_t";
  } else if (src->dtype.is_float16()) {
    dtype_str = "half_t";
  } else if (src->dtype.is_float8_e4m3fn() || src->dtype.is_float8_e4m3()) {
    dtype_str = "tl::fp8_t";
  } else if (src->dtype.is_float8_e5m2() || src->dtype.is_float8_e5m2fnuz()) {
    dtype_str = "tl::bf8_t";
  } else {
    LOG(FATAL) << "ds_read_format unsupported dtype: " << src->dtype;
  }

  std::stringstream ss;
  if (gemm_dep_.defined()) {
    const auto *dep = gemm_dep_.get();
    if (dep->feeds_slot == 0) {
      ss << "tl::mls::ds_read_format_tensor_a<tl::sequence<" << block_mn << ", "
         << block_k << ">, tl::sequence<" << tile_mn << ", " << tile_k << ">, "
         << w_mn << ", " << w_k << ", " << dtype_str << ", 1, "
         << (ds_trans ? "true" : "false")
         << ", tl::hcu_target_enum::" << GetHcuArchString(T.target) << ">";
    } else {
      int total_warp = block_size / TargetGetWarpSize(T.target);
      ss << "tl::mls::ds_read_format_tensor_b<tl::sequence<" << block_mn << ", "
         << block_k << ">, tl::sequence<" << tile_mn << ", " << tile_k << ">, "
         << total_warp << ", " << w_mn << ", " << w_k << ", " << dtype_str
         << ", 1, " << (ds_trans ? "true" : "false")
         << ", tl::hcu_target_enum::" << GetHcuArchString(T.target) << ">";
    }
  } else {
    ss << "tl::mls::ds_read_format_tensor_common<tl::sequence<" << block_mn
       << ", " << block_k << ">, tl::sequence<" << tile_mn << ", " << tile_k
       << ">, " << w_mn << ", " << w_k << ", " << dtype_str << ", 1, "
       << (ds_trans ? "true" : "false")
       << ", tl::hcu_target_enum::" << GetHcuArchString(T.target) << ">";
  }

  Buffer src_buf = T.buffer_remap.count(src) ? T.buffer_remap[src] : src;
  Buffer dst_buf = T.buffer_remap.count(dst) ? T.buffer_remap[dst] : dst;

  size_t sr = this->src_ranges.size();
  ICHECK(sr >= 2) << "ds_read_format src region must be at least 2D";
  PrimExpr src_leading_elem_offset = IntImm(DataType::Int(32), 0);
  if (sr > 2) {
    ICHECK_EQ(src_buf->shape.size(), sr)
        << "ds_read_format src buffer rank must match src region rank, got "
           "shape.size()="
        << src_buf->shape.size() << " vs region rank=" << sr;
    Array<PrimExpr> src_idx_leading;
    DataType idx_dtype = src_buf->DefaultIndexType();
    for (size_t j = 0; j + 2 < sr; ++j) {
      src_idx_leading.push_back(this->src_ranges[j]->min);
    }
    src_idx_leading.push_back(make_const(idx_dtype, 0));
    src_idx_leading.push_back(make_const(idx_dtype, 0));
    Array<PrimExpr> src_offs = src_buf.OffsetOf(src_idx_leading);
    ICHECK_EQ(src_offs.size(), 1u)
        << "ds_read_format src OffsetOf expects a single flat offset, got size="
        << src_offs.size();
    src_leading_elem_offset = src_offs[0];
  }

  size_t dr = this->dst_ranges.size();
  ICHECK(dr >= 2) << "ds_read_format dst region must be at least 2D";
  PrimExpr dst_leading_elem_offset = IntImm(DataType::Int(32), 0);
  if (dr > 2) {
    ICHECK_EQ(dst_buf->shape.size(), dr)
        << "ds_read_format dst buffer rank must match dst region rank, got "
           "shape.size()="
        << dst_buf->shape.size() << " vs region rank=" << dr;
    Array<PrimExpr> dst_idx_leading;
    DataType dst_idx_dtype = dst_buf->DefaultIndexType();
    for (size_t j = 0; j + 2 < dr; ++j) {
      dst_idx_leading.push_back(this->dst_ranges[j]->min);
    }
    dst_idx_leading.push_back(make_const(dst_idx_dtype, 0));
    dst_idx_leading.push_back(make_const(dst_idx_dtype, 0));
    Array<PrimExpr> dst_offs = dst_buf.OffsetOf(dst_idx_leading);
    ICHECK_EQ(dst_offs.size(), 1u)
        << "ds_read_format dst OffsetOf expects a single flat offset, got size="
        << dst_offs.size();
    dst_leading_elem_offset = dst_offs[0];
  }

  auto src_ptr =
      src_buf.access_ptr(1, DataType::Handle(), 1, src_leading_elem_offset);
  auto dst_ptr =
      dst_buf.access_ptr(2, DataType::Handle(), 1, dst_leading_elem_offset);

  Array<PrimExpr> call_args;
  call_args.push_back(StringImm(ss.str()));
  call_args.push_back(src_ptr);
  call_args.push_back(dst_ptr);

  return Evaluate(Call(DataType::Handle(), builtin::call_extern(), call_args));
}

TIR_REGISTER_TL_TILE_OP(DsReadFormat, ds_read_format)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm
