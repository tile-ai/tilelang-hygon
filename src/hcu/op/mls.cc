/*!
 * \file tl/op/mls.cc
 * \brief MLS (Matrix Load Store) operator implementation.
 */

#include "mls.h"
#include "hcu/target_utils.h"
#include "hcu/utils/mls_gemm_dep.h"
#include "op/builtin.h"
#include "op/operator.h"
#include "op/region.h"
#include "transform/common/pipeline_utils.h"
#include <algorithm>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

/// If expr is constant, return IntImm for compile-time optimization; otherwise
/// return expr as variable.
static PrimExpr ToInt64ConstOrVar(const PrimExpr &expr) {
  if (const int64_t *p = as_const_int(expr)) {
    return IntImm(DataType::Int(64), *p);
  }
  return expr;
}

MatrixLoad::MatrixLoad(Array<PrimExpr> args,
                       Map<String, ObjectRef> annotations) {
  ICHECK(args.size() >= 4)
      << "matrix_load expects at least 4 args: src_region, dst_region, "
         "check_last_k_load, last_k_load";
  auto src_call = args[0].as<CallNode>();
  auto dst_call = args[1].as<CallNode>();
  ICHECK(src_call) << "matrix_load args[0] must be region call (src)";
  ICHECK(dst_call) << "matrix_load args[1] must be region call (dst)";

  auto src_region = RegionOp(src_call->args);
  auto dst_region = RegionOp(dst_call->args);
  auto src_ranges = src_region->GetRanges();
  auto dst_ranges = dst_region->GetRanges();

  ICHECK(dst_ranges.size() >= 2) << "matrix_load dst region must be 2D";
  ICHECK(src_ranges.size() >= 2) << "matrix_load src region must have at least "
                                    "2 dims (last 2 match dst MN,K)";

  Buffer dst_buf = dst_region->GetBuffer();
  ICHECK(dst_buf.scope() == "shared" || dst_buf.scope() == "shared.dyn")
      << "matrix_load dst must be shared memory, got scope=" << dst_buf.scope();

  ObjectPtr<MatrixLoadNode> node = tvm::ffi::make_object<MatrixLoadNode>();
  node->src = src_region->GetBuffer();
  node->dst = dst_buf;
  node->src_ranges = src_ranges;
  node->dst_ranges = dst_ranges;
  AccessRegion src_access{BufferRegion(node->src, node->src_ranges),
                          kAccessRead};
  AccessRegion dst_access{BufferRegion(node->dst, node->dst_ranges),
                          kAccessWrite};
  node->SetAccessRegions({src_access, dst_access});
  node->check_last_load = args[2].as<IntImmNode>()->value != 0;
  node->last_load = args[3].as<IntImmNode>()->value != 0;
  if (auto mls_trans = GetMlsTransFromAnnotations(annotations)) {
    node->mls_trans_ = mls_trans.value();
  }

  data_ = std::move(node);
}

TileOperator MatrixLoadNode::Clone() const {
  auto op = tvm::ffi::make_object<MatrixLoadNode>(*this);
  return MatrixLoad(op);
}

namespace {

struct MlsTileConfig {
  int tile_mn;
  int tile_k;
  bool require_no_repeat;
};

struct MlsTileConfigSet {
  const MlsTileConfig *trans;
  int trans_count;
  const MlsTileConfig *non_trans;
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
     static_cast<int>(sizeof(kMlsTileConfigsB8Trans) /
                      sizeof(kMlsTileConfigsB8Trans[0])),
     kMlsTileConfigsB8NonTrans,
     static_cast<int>(sizeof(kMlsTileConfigsB8NonTrans) /
                      sizeof(kMlsTileConfigsB8NonTrans[0]))},
    {kMlsTileConfigsB16Trans,
     static_cast<int>(sizeof(kMlsTileConfigsB16Trans) /
                      sizeof(kMlsTileConfigsB16Trans[0])),
     kMlsTileConfigsB16NonTrans,
     static_cast<int>(sizeof(kMlsTileConfigsB16NonTrans) /
                      sizeof(kMlsTileConfigsB16NonTrans[0]))},
};

std::pair<int64_t, int64_t> MlsBlockDims(const Buffer &buf, bool mls_trans) {
  ICHECK(buf->shape.size() >= 2);
  size_t nd = buf->shape.size();
  if (mls_trans) {
    return {*as_const_int(buf->shape[nd - 2]),
            *as_const_int(buf->shape[nd - 1])};
  }
  return {*as_const_int(buf->shape[nd - 1]), *as_const_int(buf->shape[nd - 2])};
}

} // namespace

int MlsScopedWarpIdOffset(const Range &thread_bounds, Target target) {
  if (!is_const_int(thread_bounds->min)) {
    return 0;
  }
  int min = *as_const_int(thread_bounds->min);
  if (min == 0) {
    return 0;
  }
  int warp_size = TargetHcuGetWarpSize(target);
  ICHECK(min % warp_size == 0)
      << "MLS scoped lowering requires thread_bounds.min to be warp-aligned, "
         "got min="
      << min << " warp_size=" << warp_size;
  return min / warp_size;
}

/*
 * MLS tile size rules (MN interleave=1, b16):
 * num_warps = block_size / TargetHcuGetWarpSize(target); warp_mn * warp_k =
 * num_warps. One warp group extent in K = warp_k * mlsTilesizeK. If > block_k,
 * warps repeat load.
 */
void ComputeMlsWarpPartition(bool trans, int block_mn, int block_k,
                             int block_size, Target target, int elem_bits,
                             int &warp_mn, int &warp_k, int &mls_tile_mn,
                             int &mls_tile_k) {
  const int elem_bytes = elem_bits / 8;
  ICHECK(elem_bytes == 1 || elem_bytes == 2) << "b16 or b8 only";

  int num_warps = block_size / TargetHcuGetWarpSize(target);
  ICHECK(num_warps >= 1) << "num_warps must be >= 1";

  auto try_config = [&](int tile_mn, int tile_k,
                        bool require_no_repeat) -> bool {
    if (block_k % tile_k != 0 || block_mn % tile_mn != 0)
      return false;
    int wm = std::min(block_mn / tile_mn, num_warps);
    if (num_warps % wm != 0)
      return false;
    int wk = num_warps / wm;
    if (require_no_repeat) {
      if (wm * tile_mn > block_mn || block_mn % (wm * tile_mn) != 0)
        return false;
      if (wk * tile_k > block_k || block_k % (wk * tile_k) != 0)
        return false;
    }
    warp_mn = wm;
    warp_k = wk;
    mls_tile_mn = tile_mn;
    mls_tile_k = tile_k;
    return true;
  };

  const MlsTileConfigSet &config_set = kMlsTileConfigTable[elem_bytes];
  const MlsTileConfig *configs =
      trans ? config_set.trans : config_set.non_trans;
  int config_count =
      trans ? config_set.trans_count : config_set.non_trans_count;
  for (int i = 0; i < config_count; ++i) {
    const MlsTileConfig &config = configs[i];
    if (try_config(config.tile_mn, config.tile_k, config.require_no_repeat))
      return;
  }
  ICHECK(false) << "No valid MLS tile config for "
                << (trans ? "trans" : "non-trans") << ": block_mn=" << block_mn
                << " block_k=" << block_k << " num_warps=" << num_warps
                << " elem_bits=" << elem_bits;
}

LayoutMap MatrixLoadNode::InferLayout(const LayoutInferArgs &T,
                                      InferLevel level) const {
  (void)T;
  (void)level;
  return {};
}

Stmt MatrixLoadNode::Lower(const LowerArgs &T,
                           arith::Analyzer *analyzer) const {
  (void)analyzer;
  if (!TargetIsHCU(T.target)) {
    LOG(FATAL) << "matrix_load is only supported on HCU target";
  }

  ICHECK(dst->shape.size() >= 2)
      << "dst must have rank >= 2; MN×K tile uses the last two dimensions";
  ICHECK(src->shape.size() >= 2) << "src must be 2D";

  const bool mls_trans = mls_trans_;
  auto [block_mn, block_k] = MlsBlockDims(dst, mls_trans);
  int block_size = static_cast<int>(*as_const_int(T.thread_bounds->extent));
  int warp_id_offset = MlsScopedWarpIdOffset(T.thread_bounds, T.target);
  int tile_mn, tile_k, warp_m, warp_k;
  ComputeMlsWarpPartition(mls_trans, static_cast<int>(block_mn),
                          static_cast<int>(block_k), block_size, T.target,
                          src->dtype.bits(), warp_m, warp_k, tile_mn, tile_k);

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

  PrimExpr mn_length_raw = mls_trans ? src->shape[nr - 2] : src->shape[nr - 1];
  PrimExpr k_length_raw = mls_trans ? src->shape[nr - 1] : src->shape[nr - 2];
  PrimExpr mls_stride = mls_trans ? k_length_raw : mn_length_raw;

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
    LOG(FATAL) << "matrix_load unsupported dtype: " << src->dtype;
  }

  std::stringstream ss;
  ss << "tl::mls::mls_load_tile<tl::sequence<" << block_mn << ", " << block_k
     << ">, tl::sequence<" << tile_mn << ", " << tile_k << ">, " << warp_m
     << ", " << warp_k << ", " << dtype_str << ", 1, "
     << (mls_trans ? "true" : "false")
     << ", tl::hcu_target_enum::" << GetHcuArchString(T.target) << ", "
     << (check_last_load ? "true" : "false") << ", "
     << (last_load ? "true" : "false") << ">";

  Buffer src_buf = T.buffer_remap.count(src) ? T.buffer_remap[src] : src;
  Buffer dst_buf = T.buffer_remap.count(dst) ? T.buffer_remap[dst] : dst;

  PrimExpr leading_elem_offset = IntImm(DataType::Int(32), 0);
  if (nr > 2) {
    ICHECK_EQ(src_buf->shape.size(), nr)
        << "matrix_load src buffer rank must match src region rank, got "
           "shape.size()="
        << src_buf->shape.size() << " vs region rank=" << nr;
    Array<PrimExpr> idx_leading;
    DataType idx_dtype = src_buf->DefaultIndexType();
    for (size_t j = 0; j + 2 < nr; ++j) {
      idx_leading.push_back(this->src_ranges[j]->min);
    }
    idx_leading.push_back(make_const(idx_dtype, 0));
    idx_leading.push_back(make_const(idx_dtype, 0));
    Array<PrimExpr> offs = src_buf.OffsetOf(idx_leading);
    ICHECK_EQ(offs.size(), 1u)
        << "matrix_load src OffsetOf expects a single flat offset, got size="
        << offs.size();
    leading_elem_offset = offs[0];
  }

  auto src_ptr =
      src_buf.access_ptr(1, DataType::Handle(), 1, leading_elem_offset);

  size_t dr = this->dst_ranges.size();
  ICHECK(dr >= 2) << "matrix_load dst region must be at least 2D";
  PrimExpr dst_leading_elem_offset = IntImm(DataType::Int(32), 0);
  if (dr > 2) {
    ICHECK_EQ(dst_buf->shape.size(), dr)
        << "matrix_load dst buffer rank must match dst region rank, got "
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
        << "matrix_load dst OffsetOf expects a single flat offset, got size="
        << dst_offs.size();
    dst_leading_elem_offset = dst_offs[0];
  }

  auto dst_ptr =
      dst_buf.access_ptr(2, DataType::Handle(), 1, dst_leading_elem_offset);

  Array<PrimExpr> call_args;
  call_args.push_back(StringImm(ss.str()));
  call_args.push_back(src_ptr);
  call_args.push_back(ToInt64ConstOrVar(mls_stride));
  call_args.push_back(ToInt64ConstOrVar(mn_length_raw));
  call_args.push_back(ToInt64ConstOrVar(k_length_raw));
  call_args.push_back(block_mn_base);
  call_args.push_back(block_k_base);
  call_args.push_back(dst_ptr);
  call_args.push_back(IntImm(DataType::Int(32), warp_id_offset));

  return Evaluate(Call(DataType::Handle(), builtin::call_extern(), call_args));
}

TIR_REGISTER_TL_TILE_OP(MatrixLoad, matrix_load)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  MatrixLoadNode::RegisterReflection();
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.ComputeMlsWarpPartition",
      [](bool trans, int block_mn, int block_k, int block_size, Target target,
         int elem_bits) {
        int warp_mn = 0;
        int warp_k = 0;
        int mls_tile_mn = 0;
        int mls_tile_k = 0;
        ComputeMlsWarpPartition(trans, block_mn, block_k, block_size, target,
                                elem_bits, warp_mn, warp_k, mls_tile_mn,
                                mls_tile_k);
        return Array<Integer>{Integer(warp_mn), Integer(warp_k),
                              Integer(mls_tile_mn), Integer(mls_tile_k)};
      });
}

} // namespace tl
} // namespace tvm
