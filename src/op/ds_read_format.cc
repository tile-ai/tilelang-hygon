/*!
 * \file ds_read_format.cc
 * \brief DsReadFormat: read MLS-formatted shared memory into register.
 */

#include "ds_read_format.h"
#include "gemm.h"
#include "mls.h"
#include "propagation_util.h"
#include "region.h"
#include "../layout/utils.h"
#include "../target/utils.h"
#include "builtin.h"
#include <tvm/node/structural_equal.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <map>
#include <tuple>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

// mls_ds_traits: (mls_tile_mn, mls_tile_k, trans, alt) -> (read_tile_mn, read_tile_k).
// Key = (mls_tile_mn, mls_tile_k, trans, alt).
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

void GetReadTileFromMlsTile(bool trans, int mls_tile_mn, int mls_tile_k, int alt,
                            int elem_bytes, int &read_tile_mn, int &read_tile_k) {
  MlsReadKey key(mls_tile_mn, mls_tile_k, trans, alt);
  const MlsReadMap *m = (elem_bytes == 2) ? &kMlsToReadTileB16 : &kMlsToReadTileB8;
  auto it = m->find(key);
  if (it != m->end()) {
    read_tile_mn = it->second.first;
    read_tile_k = it->second.second;
  } else {
    LOG(FATAL) << "GetReadTileFromMlsTile: no entry for (mls_mn=" << mls_tile_mn
               << ", mls_k=" << mls_tile_k << ", trans=" << trans << ", alt=" << alt
               << ", elem_bytes=" << elem_bytes << "). Add to kMlsToReadTileB16/B8.";
  }
}

void ComputeDsReadFormatWarpPartition(bool trans, int block_mn, int block_k,
                                     int block_size, Target target, int read_tile_mn,
                                     int read_tile_k, int &warp_mn, int &warp_k) {
  int num_warps = block_size / TargetGetWarpSize(target);
  ICHECK(num_warps >= 1);
  ICHECK(block_mn % read_tile_mn == 0 && block_k % read_tile_k == 0);
  int max_wm = std::min(block_mn / read_tile_mn, num_warps);
  int max_wk = std::min(block_k / read_tile_k, num_warps);
  warp_k = num_warps / max_wm;
  warp_k = warp_k > max_wk ? max_wk : warp_k;
  warp_mn = num_warps / warp_k;
}

}  // namespace

DsReadFormat::DsReadFormat(Array<PrimExpr> args, BufferMap vmap) {
  ICHECK(args.size() >= 2) << "ds_read_format expects at least 2 args: src_region, dst_region";
  auto src_call = args[0].as<CallNode>();
  auto dst_call = args[1].as<CallNode>();
  ICHECK(src_call) << "ds_read_format args[0] must be region call (src)";
  ICHECK(dst_call) << "ds_read_format args[1] must be region call (dst)";

  auto src_region = RegionOp(src_call->args, vmap);
  auto dst_region = RegionOp(dst_call->args, vmap);
  auto src_ranges = src_region->GetRanges();
  auto dst_ranges = dst_region->GetRanges();

  ICHECK(src_ranges.size() >= 2) << "ds_read_format src region must be 2D";
  ICHECK(dst_ranges.size() >= 2) << "ds_read_format dst region must be 2D";

  Buffer src_buf = src_region->GetBuffer();
  Buffer dst_buf = dst_region->GetBuffer();
  ICHECK(src_buf.scope() == "shared" || src_buf.scope() == "shared.dyn")
      << "ds_read_format src must be shared memory, got scope=" << src_buf.scope();
  ICHECK(dst_buf.scope() == "local.fragment")
      << "ds_read_format dst must be register (local.fragment), got scope=" << dst_buf.scope();

  ObjectPtr<DsReadFormatNode> node = make_object<DsReadFormatNode>();
  node->src = src_buf;
  node->dst = dst_buf;
  node->src_ranges = src_ranges;
  node->dst_ranges = dst_ranges;
  if (args.size() >= 7) {
    node->mls_tile_mn = args[2].as<IntImmNode>()->value;
    node->mls_tile_k = args[3].as<IntImmNode>()->value;
    node->warp_mn = args[4].as<IntImmNode>()->value;
    node->warp_k = args[5].as<IntImmNode>()->value;
    node->trans = args[6].as<IntImmNode>()->value != 0;
  }
  data_ = std::move(node);
}

TileOperator DsReadFormatNode::Clone() const {
  auto op = make_object<DsReadFormatNode>(*this);
  return DsReadFormat(op);
}

LayoutMap DsReadFormatNode::InferLayout(const LayoutInferArgs &T,
                                        InferLevel level) const {
  if (completed_)
    return {};

  ICHECK(IsFromMls(src, T.tir_collector))
      << "ds_read_format input must come from matrix_load output";

  LayoutMap result;
  auto gemm_with_input = PropagateToFindGemmConsumerOpWithInput(dst, T.tir_collector);
  if (gemm_with_input) {
    auto gemm = gemm_with_input->gemm;
    auto input_buf = gemm_with_input->input;
    ICHECK(input_buf->dtype.bits() == gemm->A->dtype.bits() ||
           input_buf->dtype.bits() == gemm->B->dtype.bits())
        << "ds_read_format output dtype must match gemm input when feeding gemm";
    if (input_buf.same_as(gemm->A)) {
      this->trans = !gemm->trans_A;
    } else if (input_buf.same_as(gemm->B)) {
      this->trans = gemm->trans_B;
    } else {
      ICHECK(false) << "ds_read_format dst must feed Gemm A or B";
    }
    int64_t block_mn = trans ? *as_const_int(dst->shape[0]) : *as_const_int(dst->shape[1]);
    int64_t block_k = trans ? *as_const_int(dst->shape[1]) : *as_const_int(dst->shape[0]);
    int block_size = *as_const_int(T.thread_bounds->extent);
    int w_mn, w_k, t_mn, t_k;
    ComputeMlsWarpPartition(trans, static_cast<int>(block_mn), static_cast<int>(block_k),
                            block_size, T.target, src->dtype.bits(), w_mn, w_k, t_mn, t_k);
    this->mls_tile_mn = t_mn;
    this->mls_tile_k = t_k;
    GemmInst gemm_inst = gemm->GetGemmInst(block_size, T.target);
    int element_byte_size = gemm->A->dtype.bits() / 8;
    bool A_from_mls = IsFromMls(gemm->A, T.tir_collector);
    bool B_from_mls = IsFromMls(gemm->B, T.tir_collector);
    bool A_mls_trans = !gemm->trans_A;
    bool B_mls_trans = gemm->trans_B;
    auto [warp_m, warp_n, warp_k] = gemm->policy->ComputeWarpPartitionHCU(
        gemm->M, gemm->N, gemm->K, gemm->kPack, element_byte_size, block_size,
        T.target, gemm_inst, A_from_mls, B_from_mls, A_mls_trans, B_mls_trans);
    if (input_buf.same_as(gemm->A)) {
      this->warp_mn = warp_m;
      this->warp_k = warp_k;
    } else if (input_buf.same_as(gemm->B)) {
      this->warp_mn = warp_n;
      this->warp_k = warp_k;
    } else {
      ICHECK(false) << "ds_read_format dst must feed Gemm A or B";
    }
    // DsReadFormat must follow Gemm's warp partition when feeding Gemm.
    // MLS warp partition (w_mn, w_k) may differ; mls_tile_mn/k are used for tile shape only.
    const int min_n_per_warp = (input_buf.same_as(gemm->B) && !B_mls_trans) ? 32 : 16;
    auto fragment = input_buf.same_as(gemm->A)
                        ? makeGemmFragmentAHCU(gemm->M, gemm->N, gemm->K, warp_m, warp_n, warp_k,
                                               src->dtype.bits(), gemm->kPack, gemm->trans_A)
                        : makeGemmFragmentBHCU(gemm->M, gemm->N, gemm->K, warp_m, warp_n, warp_k,
                                               src->dtype.bits(), gemm->kPack, gemm->trans_B, min_n_per_warp);
    auto layout = fragment->BindThreadRange(T.thread_bounds);
    if (T.layout_map.count(dst)) {
      if (!tvm::StructuralEqual()(layout, T.layout_map[dst])) {
        LOG(FATAL) << "DsReadFormat layout conflict: inferred layout differs from existing "
                      "layout for buffer " << dst;
      }
      // Already consistent (thread + register layout match), no need to update
    } else {
      result.Set(dst, layout);
    }
  } else {
    // for now, default to trans=true for ds_read_format when not feeding Gemm.
    this->trans = true;
    int64_t block_mn = trans ? *as_const_int(dst->shape[0]) : *as_const_int(dst->shape[1]);
    int64_t block_k = trans ? *as_const_int(dst->shape[1]) : *as_const_int(dst->shape[0]);
    int block_size = *as_const_int(T.thread_bounds->extent);
    int w_mn, w_k, t_mn, t_k;
    ComputeMlsWarpPartition(trans, static_cast<int>(block_mn), static_cast<int>(block_k),
                            block_size, T.target, src->dtype.bits(), w_mn, w_k, t_mn, t_k);
    this->mls_tile_mn = t_mn;
    this->mls_tile_k = t_k;
    const int alt = 1;
    const int elem_bytes = src->dtype.bits() / 8;
    int read_tile_mn, read_tile_k;
    GetReadTileFromMlsTile(trans, t_mn, t_k, alt, elem_bytes, read_tile_mn, read_tile_k);
    ComputeDsReadFormatWarpPartition(trans, static_cast<int>(block_mn),
                                     static_cast<int>(block_k), block_size, T.target,
                                     read_tile_mn, read_tile_k, this->warp_mn, this->warp_k);
    int num_warp_mn_no_recompute =
        std::min(this->warp_mn, static_cast<int>(block_mn / read_tile_mn));
    if (num_warp_mn_no_recompute < 1) num_warp_mn_no_recompute = 1;
    auto fragment = makeDsReadFormatFragmentHCU(
        static_cast<int>(block_mn), static_cast<int>(block_k),
        this->warp_mn, this->warp_k, src->dtype.bits(), num_warp_mn_no_recompute,
        trans);
    auto layout = fragment->BindThreadRange(T.thread_bounds);
    if (T.layout_map.count(dst)) {
      if (!tvm::StructuralEqual()(layout, T.layout_map[dst])) {
        LOG(FATAL) << "DsReadFormat layout conflict: inferred layout differs from "
                      "existing layout for buffer " << dst;
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

  ICHECK(mls_tile_mn > 0 && mls_tile_k > 0)
      << "ds_read_format tile must be set in InferLayout";

  int64_t block_mn = trans ? *as_const_int(dst->shape[0]) : *as_const_int(dst->shape[1]);
  int64_t block_k = trans ? *as_const_int(dst->shape[1]) : *as_const_int(dst->shape[0]);

  std::string dtype_str;
  if (src->dtype.is_bfloat16()) {
    dtype_str = "ck_tile::bfloat16_t";
  } else if (src->dtype.is_float16()) {
    dtype_str = "half_t";
  } else if (src->dtype.is_float8_e4m3fn() || src->dtype.is_float8_e4m3()) {
    dtype_str = "ck_tile::fp8_t";
  } else if (src->dtype.is_float8_e5m2() || src->dtype.is_float8_e5m2fnuz()) {
    dtype_str = "ck_tile::bf8_t";
  } else {
    LOG(FATAL) << "ds_read_format unsupported dtype: " << src->dtype;
  }

  auto gemm_with_input = PropagateToFindGemmConsumerOpWithInput(dst, T.tir_collector);
  std::stringstream ss;
  if (gemm_with_input) {
    auto gemm = gemm_with_input->gemm;
    auto input_buf = gemm_with_input->input;
    if (input_buf.same_as(gemm->A)) {
      ss << "tl::mls::ds_read_format_tensor_a<ck_tile::sequence<" << block_mn
         << ", " << block_k << ">, ck_tile::sequence<" << mls_tile_mn << ", "
         << mls_tile_k << ">, " << warp_mn << ", " << warp_k << ", "
         << dtype_str << ", 1, " << (trans ? "true" : "false")
         << ", ck_tile::hcu_target_enum::" << GetHcuArchString(T.target) << ">";
    } else {
      ICHECK(input_buf.same_as(gemm->B));
      int block_size = *as_const_int(T.thread_bounds->extent);
      int total_warp = block_size / TargetGetWarpSize(T.target);
      ss << "tl::mls::ds_read_format_tensor_b<ck_tile::sequence<" << block_mn
         << ", " << block_k << ">, ck_tile::sequence<" << mls_tile_mn << ", "
         << mls_tile_k << ">, " << total_warp << ", " << warp_mn << ", " << warp_k
         << ", " << dtype_str << ", 1, " << (trans ? "true" : "false")
         << ", ck_tile::hcu_target_enum::" << GetHcuArchString(T.target) << ">";
    }
  } else {
    ss << "tl::mls::ds_read_format_tensor_common<ck_tile::sequence<"
       << block_mn << ", " << block_k << ">, ck_tile::sequence<" << mls_tile_mn
       << ", " << mls_tile_k << ">, " << warp_mn << ", " << warp_k << ", "
       << dtype_str << ", 1, " << (trans ? "true" : "false")
       << ", ck_tile::hcu_target_enum::" << GetHcuArchString(T.target) << ">";
  }

  Buffer src_buf = T.buffer_remap.count(src) ? T.buffer_remap[src] : src;
  Buffer dst_buf = T.buffer_remap.count(dst) ? T.buffer_remap[dst] : dst;
  auto src_ptr = src_buf.access_ptr(1);
  auto dst_ptr = dst_buf.access_ptr(2);

  Array<PrimExpr> call_args;
  call_args.push_back(StringImm(ss.str()));
  call_args.push_back(src_ptr);
  call_args.push_back(dst_ptr);

  return Evaluate(
      Call(DataType::Handle(), builtin::call_extern(), call_args));
}

TIR_REGISTER_TL_OP(DsReadFormat, ds_read_format)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                              Integer(CallEffectKind::kOpaque));

}  // namespace tl
}  // namespace tvm
