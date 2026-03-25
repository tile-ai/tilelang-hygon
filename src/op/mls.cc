/*!
 * \file tl/op/mls.cc
 * \brief MLS (Matrix Load Store) operator implementation.
 */

#include "mls.h"
#include "gemm.h"
#include "propagation_util.h"
#include "region.h"
#include "../target/utils.h"
#include "builtin.h"
#include <algorithm>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tl {

using namespace tir;

/// If expr is constant, return IntImm for compile-time optimization; otherwise return expr as variable.
static PrimExpr ToInt64ConstOrVar(const PrimExpr &expr) {
  if (const int64_t *p = as_const_int(expr)) {
    return IntImm(DataType::Int(64), *p);
  }
  return expr;
}

MatrixLoad::MatrixLoad(Array<PrimExpr> args, BufferMap vmap) {
  ICHECK(args.size() >= 4)
      << "matrix_load expects at least 4 args: src_region, dst_region, "
         "check_last_k_load, last_k_load";
  auto src_call = args[0].as<CallNode>();
  auto dst_call = args[1].as<CallNode>();
  ICHECK(src_call) << "matrix_load args[0] must be region call (src)";
  ICHECK(dst_call) << "matrix_load args[1] must be region call (dst)";

  auto src_region = RegionOp(src_call->args, vmap);
  auto dst_region = RegionOp(dst_call->args, vmap);
  auto src_ranges = src_region->GetRanges();
  auto dst_ranges = dst_region->GetRanges();

  ICHECK(dst_ranges.size() >= 2) << "matrix_load dst region must be 2D";
  ICHECK(src_ranges.size() >= 2)
      << "matrix_load src region must have at least 2 dims (last 2 match dst MN,K)";

  Buffer dst_buf = dst_region->GetBuffer();
  ICHECK(dst_buf.scope() == "shared" || dst_buf.scope() == "shared.dyn")
      << "matrix_load dst must be shared memory, got scope=" << dst_buf.scope();

  ObjectPtr<MatrixLoadNode> node = make_object<MatrixLoadNode>();
  node->src = src_region->GetBuffer();
  node->dst = dst_buf;
  node->src_ranges = src_ranges;
  node->check_last_load = args[2].as<IntImmNode>()->value != 0;
  node->last_load = args[3].as<IntImmNode>()->value != 0;
  if (args.size() >= 9) {
    node->mls_tile_mn = args[4].as<IntImmNode>()->value;
    node->mls_tile_k = args[5].as<IntImmNode>()->value;
    node->warp_mn = args[6].as<IntImmNode>()->value;
    node->warp_k = args[7].as<IntImmNode>()->value;
    node->trans = args[8].as<IntImmNode>()->value != 0;
  }

  data_ = std::move(node);
}

TileOperator MatrixLoadNode::Clone() const {
  auto op = make_object<MatrixLoadNode>(*this);
  return MatrixLoad(op);
}

namespace {

struct MlsTileConfig {
  int tile_mn;
  int tile_k;
  bool require_no_repeat;
};

struct MlsTileConfigSet {
  const MlsTileConfig* trans;
  int trans_count;
  const MlsTileConfig* non_trans;
  int non_trans_count;
};

constexpr MlsTileConfig kMlsTileConfigsB16Trans[] = {
    {16, 64, true},
    {32, 32, true},
    {16, 32, false},
};

constexpr MlsTileConfig kMlsTileConfigsB16NonTrans[] = {
    {64, 16, true},
    {32, 32, true},
    {32, 16, false},
};

constexpr MlsTileConfig kMlsTileConfigsB8Trans[] = {
    {16, 128, true},
    {32, 64, true},
    {16, 64, false},
};

constexpr MlsTileConfig kMlsTileConfigsB8NonTrans[] = {
    {128, 16, true},
    {64, 32, true},
    {64, 16, false},
};

constexpr MlsTileConfigSet kMlsTileConfigTable[] = {
    {nullptr, 0, nullptr, 0},
    {kMlsTileConfigsB8Trans,
     static_cast<int>(sizeof(kMlsTileConfigsB8Trans) / sizeof(kMlsTileConfigsB8Trans[0])),
     kMlsTileConfigsB8NonTrans,
     static_cast<int>(sizeof(kMlsTileConfigsB8NonTrans) / sizeof(kMlsTileConfigsB8NonTrans[0]))},
    {kMlsTileConfigsB16Trans,
     static_cast<int>(sizeof(kMlsTileConfigsB16Trans) / sizeof(kMlsTileConfigsB16Trans[0])),
     kMlsTileConfigsB16NonTrans,
     static_cast<int>(sizeof(kMlsTileConfigsB16NonTrans) / sizeof(kMlsTileConfigsB16NonTrans[0]))},
};

}  // namespace

/*
 * MLS tile size rules (MN interleave=1, b16):
 * num_warps = block_size / TargetGetWarpSize(target); warp_mn * warp_k = num_warps.
 * One warp group extent in K = warp_k * mlsTilesizeK. If > block_k, warps repeat load.
 */
void ComputeMlsWarpPartition(bool trans, int block_mn, int block_k, int block_size,
                            Target target, int elem_bits, int &warp_mn,
                            int &warp_k, int &mls_tile_mn, int &mls_tile_k) {
  const int elem_bytes = elem_bits / 8;
  ICHECK(elem_bytes == 1 || elem_bytes == 2) << "b16 or b8 only";

  int num_warps = block_size / TargetGetWarpSize(target);
  ICHECK(num_warps >= 1) << "num_warps must be >= 1";

  // require_no_repeat: warp group extent must fit in block (no repeat load).
  // When false, warp group extent may exceed block (C++ template handles repeat).
  // Always prioritize MN first (wm), then K (wk), to match GEMM K-outer/MN-inner and
  // SFC_WarpAccess MN-inner/K-outer for better locality.
  auto try_config = [&](int tile_mn, int tile_k, bool require_no_repeat) -> bool {
    if (block_k % tile_k != 0 || block_mn % tile_mn != 0) return false;
    int wm = std::min(block_mn / tile_mn, num_warps);
    if (num_warps % wm != 0) return false;
    int wk = num_warps / wm;
    if (require_no_repeat) {
      if (wm * tile_mn > block_mn || block_mn % (wm * tile_mn) != 0) return false;
      if (wk * tile_k > block_k || block_k % (wk * tile_k) != 0) return false;
    }
    warp_mn = wm;
    warp_k = wk;
    mls_tile_mn = tile_mn;
    mls_tile_k = tile_k;
    return true;
  };

  const MlsTileConfigSet& config_set = kMlsTileConfigTable[elem_bytes];
  const MlsTileConfig* configs = trans ? config_set.trans : config_set.non_trans;
  int config_count = trans ? config_set.trans_count : config_set.non_trans_count;
  for (int i = 0; i < config_count; ++i) {
    const MlsTileConfig& config = configs[i];
    if (try_config(config.tile_mn, config.tile_k, config.require_no_repeat)) return;
  }
  ICHECK(false) << "No valid MLS tile config for " << (trans ? "trans" : "non-trans")
                << ": block_mn=" << block_mn << " block_k=" << block_k
                << " num_warps=" << num_warps << " elem_bits=" << elem_bits;
}

LayoutMap MatrixLoadNode::InferLayout(const LayoutInferArgs &T,
                                      InferLevel level) const {
  if (completed_)
    return {};

  bool mls_trans = true;
  bool found_gemm = false;
  bool mls_trans_set = false;
  auto consumers = GetConsumerOpsFromTir(dst, T.tir_collector);
  for (const auto &c : consumers) {
    if (c.as<GemmNode>()) {
      found_gemm = true;
      auto gemm = Downcast<Gemm>(c);
      ICHECK(gemm->kPack == 1) << "MatrixLoad dst Gemm consumer must have kPack=1, got " << gemm->kPack;
      bool cur_trans = false;
      if (gemm->A.same_as(dst)) {
        cur_trans = !gemm->trans_A;
      } else if (gemm->B.same_as(dst)) {
        cur_trans = gemm->trans_B;
      }
      if (mls_trans_set) {
        ICHECK(cur_trans == mls_trans) << "MatrixLoad dst Gemm consumers must have consistent mls_trans";
      } else {
        mls_trans = cur_trans;
        mls_trans_set = true;
      }
    } else if (c->GetTypeKey() == std::string("tl.DsReadFormat")) {
      auto gemm_with_input = PropagateToFindGemmConsumerOpWithInput(
          c->GetOutBuffers()[0], T.tir_collector);
      if (gemm_with_input) {
        found_gemm = true;
        auto gemm = gemm_with_input->gemm;
        auto input_buf = gemm_with_input->input;
        bool cur_trans = false;
        if (input_buf.same_as(gemm->A)) {
          cur_trans = !gemm->trans_A;
        } else if (input_buf.same_as(gemm->B)) {
          cur_trans = gemm->trans_B;
        }
        if (mls_trans_set) {
          ICHECK(cur_trans == mls_trans) << "MatrixLoad dst Gemm consumers must have consistent mls_trans";
        } else {
          mls_trans = cur_trans;
          mls_trans_set = true;
        }
      }
    } else {
      ICHECK(false) << "MatrixLoad dst consumers must be Gemm or DsReadFormat, got " << c->GetTypeKey();
    }
  }
  if (!found_gemm) {
    // for now, default to trans=true for ds_read_format when not feeding Gemm.
    mls_trans = true;
  }
  this->trans = mls_trans;

  // mls_trans=true: K is major (dim_size-1), mls_trans=false: K is non-major (dim_size-2)
  ICHECK(dst->shape.size() >= 2);
  int64_t block_mn = mls_trans ? *as_const_int(dst->shape[0]) : *as_const_int(dst->shape[1]);
  int64_t block_k = mls_trans ? *as_const_int(dst->shape[1]) : *as_const_int(dst->shape[0]);
  int64_t block_size = *as_const_int(T.thread_bounds->extent);
  int elem_bits = src->dtype.bits();
  int w_mn, w_k, t_mn, t_k;
  ComputeMlsWarpPartition(mls_trans, block_mn, block_k, block_size, T.target,
                         elem_bits, w_mn, w_k, t_mn, t_k);
  this->mls_tile_mn = t_mn;
  this->mls_tile_k = t_k;
  this->warp_mn = w_mn;
  this->warp_k = w_k;
  this->completed_ = true;
  return {};
}

Stmt MatrixLoadNode::Lower(const LowerArgs &T,
                           arith::Analyzer *analyzer) const {
  if (!TargetIsHCU(T.target)) {
    LOG(FATAL) << "matrix_load is only supported on HCU target";
  }

  ICHECK(dst->shape.size() >= 2) << "dst must be 2D (MN, K)";

  ICHECK(src->shape.size() >= 2) << "src must be 2D";

  ICHECK(this->mls_tile_mn > 0 && this->mls_tile_k > 0 && this->warp_mn > 0 &&
         this->warp_k > 0)
      << "MatrixLoad tile info must be set in InferLayout; "
         "ensure layout inference runs before lower.";

  bool mls_trans = this->trans;

  // mls_trans=true: K is major (dim_size-1), mls_trans=false: K is non-major (dim_size-2)
  int64_t block_mn = mls_trans ? *as_const_int(dst->shape[0]) : *as_const_int(dst->shape[1]);
  int64_t block_k = mls_trans ? *as_const_int(dst->shape[1]) : *as_const_int(dst->shape[0]);

  size_t nr = this->src_ranges.size();
  ICHECK(nr >= 2);
  PrimExpr block_mn_base, block_k_base;
  if (mls_trans) {
    block_mn_base = this->src_ranges[nr - 2]->min;
    block_k_base = this->src_ranges[nr - 1]->min;
  } else {
    block_k_base = this->src_ranges[nr - 2]->min;
    block_mn_base = this->src_ranges[nr - 1]->min;
  }

  // src shape: mls_trans => K at shape[1], !mls_trans => K at shape[0]
  PrimExpr mn_length_raw = mls_trans ? src->shape[0] : src->shape[1];
  PrimExpr k_length_raw = mls_trans ? src->shape[1] : src->shape[0];
  PrimExpr mls_stride = mls_trans ? k_length_raw : mn_length_raw;

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
    LOG(FATAL) << "matrix_load unsupported dtype: " << src->dtype;
  }

  std::stringstream ss;
  ss << "tl::mls::mls_load_tile<ck_tile::sequence<" << block_mn << ", "
     << block_k << ">, ck_tile::sequence<" << this->mls_tile_mn << ", "
     << this->mls_tile_k << ">, " << this->warp_mn << ", " << this->warp_k
     << ", " << dtype_str << ", 1, "
     << (mls_trans ? "true" : "false")
     << ", ck_tile::hcu_target_enum::" << GetHcuArchString(T.target) << ", "
     << (check_last_load ? "true" : "false") << ", "
     << (last_load ? "true" : "false") << ">";

  Buffer src_buf = T.buffer_remap.count(src) ? T.buffer_remap[src] : src;
  Buffer dst_buf = T.buffer_remap.count(dst) ? T.buffer_remap[dst] : dst;
  auto src_ptr = src_buf.access_ptr(1);
  auto dst_ptr = dst_buf.access_ptr(2);

  Array<PrimExpr> call_args;
  call_args.push_back(StringImm(ss.str()));
  call_args.push_back(src_ptr);
  call_args.push_back(ToInt64ConstOrVar(mls_stride));
  call_args.push_back(ToInt64ConstOrVar(mn_length_raw));
  call_args.push_back(ToInt64ConstOrVar(k_length_raw));
  call_args.push_back(block_mn_base);
  call_args.push_back(block_k_base);
  call_args.push_back(dst_ptr);

  return Evaluate(
      Call(DataType::Handle(), builtin::call_extern(), call_args));
}

TIR_REGISTER_TL_OP(MatrixLoad, matrix_load)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                              Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm
