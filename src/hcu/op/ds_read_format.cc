/*!
 * \file ds_read_format.cc
 * \brief DsReadFormat: read MLS-formatted shared memory into register.
 */

#include "ds_read_format.h"
#include "gemm_partition.h"
#include "hcu/target_utils.h"
#include "layout/utils.h"
#include "mls.h"
#include "op/builtin.h"
#include "op/gemm.h"
#include "op/region.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>

#include <map>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

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
    {{16, 32, true, 1}, {16, 32}},  {{16, 32, true, 2}, {16, 32}},
    {{32, 32, true, 1}, {16, 32}},  {{32, 32, true, 2}, {16, 32}},
    {{16, 64, true, 1}, {16, 32}},  {{16, 64, true, 2}, {16, 32}},
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

// b4 format path: source and destination are both packed b4.
static const MlsReadMap kMlsToReadTileB4 = {
    {{128, 16, false, 1}, {32, 64}},
    {{256, 16, false, 1}, {32, 64}},
    {{16, 128, true, 1}, {32, 64}},
    {{16, 256, true, 1}, {16, 128}},
};

void GetReadTileFromMlsTile(bool trans, int mls_tile_mn, int mls_tile_k,
                            int alt, int element_bits, int &read_tile_mn,
                            int &read_tile_k) {
  MlsReadKey key(mls_tile_mn, mls_tile_k, trans, alt);
  const MlsReadMap *m = element_bits == 16  ? &kMlsToReadTileB16
                        : element_bits == 8 ? &kMlsToReadTileB8
                        : element_bits == 4 ? &kMlsToReadTileB4
                                            : nullptr;
  ICHECK(m != nullptr) << "GetReadTileFromMlsTile: unsupported element "
                          "bitwidth="
                       << element_bits;
  auto it = m->find(key);
  if (it != m->end()) {
    read_tile_mn = it->second.first;
    read_tile_k = it->second.second;
  } else {
    LOG(FATAL) << "GetReadTileFromMlsTile: no entry for (mls_mn=" << mls_tile_mn
               << ", mls_k=" << mls_tile_k << ", trans=" << trans
               << ", alt=" << alt << ", element_bits=" << element_bits
               << "). Add to kMlsToReadTileB16/B8/B4.";
  }
}

std::string DsReadFormatDTypeString(DataType dtype) {
  if (dtype.is_bfloat16()) {
    return "bfloat16_t";
  }
  if (dtype.is_float16()) {
    return "half_t";
  }
  if (dtype.is_float8_e4m3fn() || dtype.is_float8_e4m3()) {
    return "tl::fp8_t";
  }
  if (dtype.is_float8_e5m2() || dtype.is_float8_e5m2fnuz()) {
    return "tl::bf8_t";
  }
  if (dtype.is_float4_e2m1fn()) {
    return "tl::pk_fp4_t";
  }
  if (dtype.is_uint() && dtype.bits() == 8) {
    return "uint8_t";
  }
  LOG(FATAL) << "ds_read_format unsupported dtype: " << dtype;
  return "";
}

void ComputeDsReadFormatWarpPartition(bool trans, int block_mn, int block_k,
                                      int block_size, Target target,
                                      int read_tile_mn, int read_tile_k,
                                      int &warp_mn, int &warp_k) {
  int num_warps = block_size / TargetHcuGetWarpSize(target);
  ICHECK(num_warps >= 1);
  ICHECK(block_mn % read_tile_mn == 0 && block_k % read_tile_k == 0);
  int max_wm = std::min(block_mn / read_tile_mn, num_warps);
  int max_wk = std::min(block_k / read_tile_k, num_warps);
  warp_k = num_warps / max_wm;
  warp_k = warp_k > max_wk ? max_wk : warp_k;
  warp_mn = num_warps / warp_k;
}

struct DsReadMlsPhysicalInfo {
  int warp_mn;
  int warp_k;
  int tile_mn;
  int tile_k;
  int element_bits;
};

DsReadMlsPhysicalInfo InferDsReadMlsPhysicalInfo(const Buffer &src, bool trans,
                                                 int block_size,
                                                 Target target) {
  auto [lds_mn, lds_k] = MlsBlockDims(src, trans);
  DsReadMlsPhysicalInfo info{0, 0, 0, 0, src->dtype.bits()};
  ComputeMlsWarpPartition(trans, static_cast<int>(lds_mn),
                          static_cast<int>(lds_k), block_size, target,
                          src->dtype.bits(), info.warp_mn, info.warp_k,
                          info.tile_mn, info.tile_k);
  info.element_bits = GetMlsLdsPhysicalBits(src->dtype, trans, info.tile_mn,
                                            info.tile_k, target);
  return info;
}

LayoutMap InferLayoutWithGemmDep(const DsReadFormatNode *self,
                                 const MlsGemmDepMetaNode *meta,
                                 const LayoutInferArgs &T) {
  LayoutMap result;
  GemmWarpPolicy policy(meta->gemm_policy);
  int block_size = static_cast<int>(*as_const_int(T.thread_bounds->extent));
  int element_bits = self->dst->dtype.bits();
  if (self->src->dtype.is_float4_e2m1fn()) {
    element_bits =
        InferDsReadMlsPhysicalInfo(self->src, meta->trans, block_size, T.target)
            .element_bits;
  }
  const bool a_mls_trans = !meta->gemm_trans_a;
  const bool b_mls_trans = meta->gemm_trans_b;
  hcu::ComputeWarpPartitionHCU(*policy.get(), meta->gemm_m, meta->gemm_n,
                               meta->gemm_k, meta->gemm_k_pack, element_bits,
                               block_size, T.target, meta->a_from_mls,
                               meta->b_from_mls, a_mls_trans, b_mls_trans);
  int warp_m = policy->m_warp;
  int warp_n = policy->n_warp;
  int warp_k = policy->k_warp;
  int min_n_per_warp = 16;
  if (meta->feeds_slot == 1) {
    if (element_bits == 4) {
      // gfx946 b4 no-pad: B trans can use 16x128 reads, while B non-trans
      // only has 32x64 reads.
      min_n_per_warp = b_mls_trans ? 16 : 32;
    } else {
      min_n_per_warp = b_mls_trans ? 16 : 32;
    }
  }
  Fragment fragment =
      meta->feeds_slot == 0
          ? makeGemmFragmentAHCU(meta->gemm_m, meta->gemm_n, meta->gemm_k,
                                 warp_m, warp_n, warp_k, element_bits,
                                 meta->gemm_k_pack, meta->gemm_trans_a)
          : makeGemmFragmentBHCU(meta->gemm_m, meta->gemm_n, meta->gemm_k,
                                 warp_m, warp_n, warp_k, element_bits,
                                 meta->gemm_k_pack, meta->gemm_trans_b,
                                 min_n_per_warp);
  auto layout = fragment->BindThreadRange(T.thread_bounds);
  if (T.layout_map.count(self->dst)) {
    if (!StructuralEqual()(layout, T.layout_map[self->dst])) {
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
    int block_size = static_cast<int>(*as_const_int(T.thread_bounds->extent));
    DsReadMlsPhysicalInfo info =
        InferDsReadMlsPhysicalInfo(src, trans, block_size, T.target);
    const int alt = 1;
    int read_tile_mn, read_tile_k;
    GetReadTileFromMlsTile(trans, info.tile_mn, info.tile_k, alt,
                           info.element_bits, read_tile_mn, read_tile_k);
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
        dst->dtype.bits(), num_warp_mn_no_recompute, trans);
    auto layout = fragment->BindThreadRange(T.thread_bounds);
    if (T.layout_map.count(dst)) {
      if (!StructuralEqual()(layout, T.layout_map[dst])) {
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
  int warp_id_offset = MlsScopedWarpIdOffset(T.thread_bounds, T.target);
  bool ds_trans = true;
  if (gemm_dep_.defined()) {
    ds_trans = gemm_dep_.get()->trans;
  }

  size_t sr = this->src_ranges.size();
  ICHECK(sr >= 2) << "ds_read_format src region must be at least 2D";
  ICHECK_EQ(src->shape.size(), sr)
      << "ds_read_format src buffer rank must match src region rank, got "
         "shape.size()="
      << src->shape.size() << " vs region rank=" << sr;

  // Full LDS last-2 shape matches the matrix_load write and LdsDesc. The read
  // extent and logical origin come from the source region last-2 dims.
  auto [lds_mn, lds_k] = MlsBlockDims(src, ds_trans);
  const PrimExpr origin_dim0 = this->src_ranges[sr - 2]->min;
  const PrimExpr origin_dim1 = this->src_ranges[sr - 1]->min;
  const PrimExpr extent_dim0 = this->src_ranges[sr - 2]->extent;
  const PrimExpr extent_dim1 = this->src_ranges[sr - 1]->extent;
  PrimExpr origin_mn = ds_trans ? origin_dim0 : origin_dim1;
  PrimExpr origin_k = ds_trans ? origin_dim1 : origin_dim0;
  PrimExpr read_mn_expr = ds_trans ? extent_dim0 : extent_dim1;
  PrimExpr read_k_expr = ds_trans ? extent_dim1 : extent_dim0;

  const int64_t *read_mn_c = as_const_int(read_mn_expr);
  const int64_t *read_k_c = as_const_int(read_k_expr);
  ICHECK(read_mn_c && read_k_c)
      << "ds_read_format requires static last-2 read extents, got "
      << read_mn_expr << ", " << read_k_expr;
  int64_t read_mn = *read_mn_c;
  int64_t read_k = *read_k_c;

  // Fragment / gemm tile must match the read extent.
  int64_t frag_mn =
      ds_trans ? *as_const_int(dst->shape[0]) : *as_const_int(dst->shape[1]);
  int64_t frag_k =
      ds_trans ? *as_const_int(dst->shape[1]) : *as_const_int(dst->shape[0]);
  ICHECK_EQ(frag_mn, read_mn)
      << "ds_read_format dst fragment MN must match src read extent MN, got "
      << frag_mn << " vs " << read_mn;
  ICHECK_EQ(frag_k, read_k)
      << "ds_read_format dst fragment K must match src read extent K, got "
      << frag_k << " vs " << read_k;

  ICHECK(read_mn > 0 && read_k > 0);
  ICHECK_LE(read_mn, lds_mn);
  ICHECK_LE(read_k, lds_k);

  int tile_mn = 0;
  int tile_k = 0;
  int w_mn = 0;
  int w_k = 0;
  // Mls tile / LdsDesc must follow the *full* LDS write shape.
  DsReadMlsPhysicalInfo info =
      InferDsReadMlsPhysicalInfo(src, ds_trans, block_size, T.target);
  tile_mn = info.tile_mn;
  tile_k = info.tile_k;
  const int lds_physical_bits = info.element_bits;
  int reg_bits = dst->dtype.bits();
  if (src->dtype.is_float4_e2m1fn() && lds_physical_bits == 8) {
    reg_bits = 8;
  }

  const int64_t *origin_mn_c = as_const_int(origin_mn);
  const int64_t *origin_k_c = as_const_int(origin_k);
  ICHECK(origin_mn_c && origin_k_c)
      << "ds_read_format requires static last-2 logical origins, got "
      << origin_mn << ", " << origin_k;
  ICHECK_EQ(*origin_mn_c % tile_mn, 0)
      << "ds_read_format slice origin_mn=" << *origin_mn_c
      << " must be divisible by mls_tile_mn=" << tile_mn;
  ICHECK_EQ(*origin_k_c % tile_k, 0)
      << "ds_read_format slice origin_k=" << *origin_k_c
      << " must be divisible by mls_tile_k=" << tile_k;
  ICHECK_EQ(read_mn % tile_mn, 0)
      << "ds_read_format slice extent_mn=" << read_mn
      << " must be divisible by mls_tile_mn=" << tile_mn;
  ICHECK_EQ(read_k % tile_k, 0)
      << "ds_read_format slice extent_k=" << read_k
      << " must be divisible by mls_tile_k=" << tile_k;
  ICHECK_LE(*origin_mn_c + read_mn, lds_mn);
  ICHECK_LE(*origin_k_c + read_k, lds_k);

  if (gemm_dep_.defined()) {
    const auto *meta = gemm_dep_.get();
    GemmWarpPolicy policy(meta->gemm_policy);
    const bool a_mls_trans = !meta->gemm_trans_a;
    const bool b_mls_trans = meta->gemm_trans_b;
    hcu::ComputeWarpPartitionHCU(*policy.get(), meta->gemm_m, meta->gemm_n,
                                 meta->gemm_k, meta->gemm_k_pack, reg_bits,
                                 block_size, T.target, meta->a_from_mls,
                                 meta->b_from_mls, a_mls_trans, b_mls_trans);
    int warp_m = policy->m_warp;
    int warp_n = policy->n_warp;
    int warp_k_part = policy->k_warp;
    w_mn = meta->feeds_slot == 0 ? warp_m : warp_n;
    w_k = warp_k_part;
  } else {
    const int alt = 1;
    const int element_bits = lds_physical_bits;
    int read_tile_mn, read_tile_k;
    GetReadTileFromMlsTile(ds_trans, tile_mn, tile_k, alt, element_bits,
                           read_tile_mn, read_tile_k);
    ComputeDsReadFormatWarpPartition(
        ds_trans, static_cast<int>(read_mn), static_cast<int>(read_k),
        block_size, T.target, read_tile_mn, read_tile_k, w_mn, w_k);
  }

  std::string dtype_str = DsReadFormatDTypeString(src->dtype);
  std::string target_dtype_str = src->dtype.is_float4_e2m1fn() && reg_bits == 8
                                     ? "uint8_t"
                                     : DsReadFormatDTypeString(dst->dtype);

  std::stringstream ss;
  if (gemm_dep_.defined()) {
    const auto *dep = gemm_dep_.get();
    if (dep->feeds_slot == 0) {
      ss << "tl::mls::ds_read_format_tensor_a<tl::sequence<" << lds_mn << ", "
         << lds_k << ">, tl::sequence<" << read_mn << ", " << read_k
         << ">, tl::sequence<" << tile_mn << ", " << tile_k << ">, " << w_mn
         << ", " << w_k << ", " << dtype_str << ", 1, "
         << (ds_trans ? "true" : "false")
         << ", tl::hcu_target_enum::" << GetHcuArchString(T.target) << ", "
         << target_dtype_str << ", " << lds_physical_bits << ", " << reg_bits
         << ">";
    } else {
      int total_warp = block_size / TargetHcuGetWarpSize(T.target);
      ss << "tl::mls::ds_read_format_tensor_b<tl::sequence<" << lds_mn << ", "
         << lds_k << ">, tl::sequence<" << read_mn << ", " << read_k
         << ">, tl::sequence<" << tile_mn << ", " << tile_k << ">, "
         << total_warp << ", " << w_mn << ", " << w_k << ", " << dtype_str
         << ", 1, " << (ds_trans ? "true" : "false")
         << ", tl::hcu_target_enum::" << GetHcuArchString(T.target) << ", "
         << target_dtype_str << ", " << lds_physical_bits << ", " << reg_bits
         << ">";
    }
  } else {
    ss << "tl::mls::ds_read_format_tensor_common<tl::sequence<" << lds_mn
       << ", " << lds_k << ">, tl::sequence<" << read_mn << ", " << read_k
       << ">, tl::sequence<" << tile_mn << ", " << tile_k << ">, " << w_mn
       << ", " << w_k << ", " << dtype_str << ", 1, "
       << (ds_trans ? "true" : "false")
       << ", tl::hcu_target_enum::" << GetHcuArchString(T.target) << ", "
       << target_dtype_str << ", " << lds_physical_bits << ", " << reg_bits
       << ">";
  }

  Buffer src_buf = T.buffer_remap.count(src) ? T.buffer_remap[src] : src;
  Buffer dst_buf = T.buffer_remap.count(dst) ? T.buffer_remap[dst] : dst;

  // Leading dims only; last-2 are handled via logical origin and full LdsDesc.
  PrimExpr src_leading_elem_offset = IntImm(DataType::Int(32), 0);
  if (sr > 2) {
    ICHECK_EQ(src_buf->shape.size(), sr)
        << "ds_read_format src buffer rank must match src region rank, got "
           "shape.size()="
        << src_buf->shape.size() << " vs region rank=" << sr;
    if (auto packed_offset = TryGetMlsPackedLeadingElemOffset(
            src_buf, this->src_ranges, tile_mn, tile_k, ds_trans, T.target,
            lds_physical_bits)) {
      src_leading_elem_offset = packed_offset.value();
    } else {
      Array<PrimExpr> src_idx_leading;
      DataType idx_dtype = src_buf->DefaultIndexType();
      for (size_t j = 0; j + 2 < sr; ++j) {
        src_idx_leading.push_back(this->src_ranges[j]->min);
      }
      src_idx_leading.push_back(make_const(idx_dtype, 0));
      src_idx_leading.push_back(make_const(idx_dtype, 0));
      Array<PrimExpr> src_offs = src_buf.OffsetOf(src_idx_leading);
      ICHECK_EQ(src_offs.size(), 1u)
          << "ds_read_format src OffsetOf expects a single flat offset, got "
             "size="
          << src_offs.size();
      src_leading_elem_offset = src_offs[0];
    }
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
  call_args.push_back(IntImm(DataType::Int(32), warp_id_offset));
  call_args.push_back(Cast(DataType::Int(32), origin_mn));
  call_args.push_back(Cast(DataType::Int(32), origin_k));

  return Evaluate(Call(DataType::Handle(), builtin::call_extern(), call_args));
}

TIR_REGISTER_TL_TILE_OP(DsReadFormat, ds_read_format)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { DsReadFormatNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
