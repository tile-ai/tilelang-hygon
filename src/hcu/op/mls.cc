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
#include <optional>
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

static int64_t CeilDiv(int64_t x, int64_t y) { return (x + y - 1) / y; }

static int64_t DTypeStorageBits(DataType dtype) {
  return static_cast<int64_t>(dtype.bits()) * dtype.lanes();
}

Optional<Integer> TryGetMlsDstActualSizeBytes(const Buffer &dst,
                                              int mls_tile_mn, int mls_tile_k,
                                              bool trans, Target target,
                                              int lds_physical_bits) {
  const auto hcu_arch = TargetIsHCU(target) ? GetHcuArchString(target) : "";
  if (!TargetIsHCU(target) || (hcu_arch != "gfx946" && hcu_arch != "gfx92a") ||
      dst->shape.size() < 2) {
    return std::nullopt;
  }

  const size_t nd = dst->shape.size();
  const int64_t *dim0 = as_const_int(dst->shape[nd - 2]);
  const int64_t *dim1 = as_const_int(dst->shape[nd - 1]);
  if (dim0 == nullptr || dim1 == nullptr) {
    return std::nullopt;
  }

  int64_t leading_count = 1;
  for (size_t i = 0; i + 2 < nd; ++i) {
    const int64_t *dim = as_const_int(dst->shape[i]);
    if (dim == nullptr) {
      return std::nullopt;
    }
    leading_count *= *dim;
  }

  const int64_t block_mn = trans ? *dim0 : *dim1;
  const int64_t block_k = trans ? *dim1 : *dim0;
  if (block_mn <= 0 || block_k <= 0 || mls_tile_mn <= 0 || mls_tile_k <= 0 ||
      block_mn % mls_tile_mn != 0 || block_k % mls_tile_k != 0) {
    return std::nullopt;
  }

  const int64_t tile_issue_mn = block_mn / mls_tile_mn;
  const int64_t tile_issue_k = block_k / mls_tile_k;
  const int64_t logical_elements = block_mn * block_k;
  int64_t packed_elements = logical_elements;
  const int64_t bits_per_elem = DTypeStorageBits(dst->dtype);
  const int64_t physical_bits_per_elem =
      lds_physical_bits > 0 ? lds_physical_bits : bits_per_elem;
  int64_t packed_bytes = -1;

  auto bytes_to_elements = [&](int64_t bytes) {
    ICHECK_EQ((bytes * 8) % bits_per_elem, 0)
        << "MLS packed byte size must be divisible by element storage bits";
    return (bytes * 8) / bits_per_elem;
  };

  auto elements_to_bytes = [&](int64_t elements) {
    return CeilDiv(elements * physical_bits_per_elem, 8);
  };

  auto packed_bytes_for_4k_slots = [](int64_t contiguous_tiles,
                                      int64_t payload_bytes) {
    constexpr int64_t kSlotStrideBytes = 4096;
    constexpr int64_t kSlotsPerGroup = 4;
    if (contiguous_tiles <= 0) {
      return int64_t{0};
    }
    const int64_t groups = CeilDiv(contiguous_tiles, kSlotsPerGroup);
    const int64_t last_group_tiles =
        contiguous_tiles - (groups - 1) * kSlotsPerGroup;
    return (groups - 1) * (kSlotStrideBytes + kSlotsPerGroup * payload_bytes) +
           kSlotStrideBytes + last_group_tiles * payload_bytes;
  };

  if (dst->dtype.bits() == 4 && physical_bits_per_elem == 8) {
    packed_bytes = elements_to_bytes(logical_elements);
  } else if (dst->dtype.bits() == 16 && trans && mls_tile_mn == 16 &&
             mls_tile_k == 64) {
    packed_bytes = packed_bytes_for_4k_slots(
        tile_issue_mn * tile_issue_k,
        elements_to_bytes(mls_tile_mn * (mls_tile_k / 2)));
  } else if (dst->dtype.bits() == 16 && !trans && mls_tile_mn == 64 &&
             mls_tile_k == 16) {
    packed_bytes = packed_bytes_for_4k_slots(
        tile_issue_mn * tile_issue_k,
        elements_to_bytes((mls_tile_mn / 2) * mls_tile_k));
  } else if (dst->dtype.bits() == 8 && !trans && mls_tile_mn == 128 &&
             mls_tile_k == 16) {
    packed_bytes = packed_bytes_for_4k_slots(
        tile_issue_mn * tile_issue_k,
        elements_to_bytes((mls_tile_mn / 2) * mls_tile_k));
  } else if (dst->dtype.bits() == 8 && trans && mls_tile_mn == 16 &&
             mls_tile_k == 128) {
    packed_bytes = packed_bytes_for_4k_slots(
        tile_issue_mn * tile_issue_k,
        elements_to_bytes(mls_tile_mn * (mls_tile_k / 2)));
  } else if (dst->dtype.bits() == 4 && !trans && mls_tile_mn == 256 &&
             mls_tile_k == 16) {
    packed_bytes = packed_bytes_for_4k_slots(
        tile_issue_mn * tile_issue_k,
        elements_to_bytes((mls_tile_mn / 2) * mls_tile_k));
  } else if (dst->dtype.bits() == 4 && trans && mls_tile_mn == 16 &&
             mls_tile_k == 256) {
    packed_bytes = packed_bytes_for_4k_slots(
        tile_issue_mn * tile_issue_k,
        elements_to_bytes(mls_tile_mn * (mls_tile_k / 2)));
  }

  if (packed_bytes > 0) {
    packed_elements = bytes_to_elements(packed_bytes);
  }

  if (packed_elements <= logical_elements) {
    return std::nullopt;
  }
  return Integer(leading_count * packed_bytes);
}

Optional<PrimExpr> TryGetMlsPackedLeadingElemOffset(
    const Buffer &buffer, const Array<Range> &ranges, int mls_tile_mn,
    int mls_tile_k, bool trans, Target target, int lds_physical_bits) {
  if (ranges.size() <= 2 || buffer->shape.size() != ranges.size()) {
    return std::nullopt;
  }

  auto actual_size_bytes = TryGetMlsDstActualSizeBytes(
      buffer, mls_tile_mn, mls_tile_k, trans, target, lds_physical_bits);
  if (!actual_size_bytes) {
    return std::nullopt;
  }

  int64_t leading_count = 1;
  for (size_t i = 0; i + 2 < buffer->shape.size(); ++i) {
    const int64_t *dim = as_const_int(buffer->shape[i]);
    if (dim == nullptr) {
      return std::nullopt;
    }
    leading_count *= *dim;
  }
  if (leading_count <= 0) {
    return std::nullopt;
  }

  const int64_t *actual_bytes = as_const_int(actual_size_bytes.value());
  if (actual_bytes == nullptr) {
    return std::nullopt;
  }
  const int64_t bits_per_elem = DTypeStorageBits(buffer->dtype);
  ICHECK_EQ((*actual_bytes * 8) % (leading_count * bits_per_elem), 0)
      << "MLS packed leading offset must be representable in logical elements";
  const int64_t packed_tile_elems =
      (*actual_bytes * 8) / (leading_count * bits_per_elem);
  if (packed_tile_elems <= 0) {
    return std::nullopt;
  }

  DataType idx_dtype = buffer->DefaultIndexType();
  PrimExpr leading_index = make_const(idx_dtype, 0);
  PrimExpr stride = make_const(idx_dtype, 1);
  for (int i = static_cast<int>(ranges.size()) - 3; i >= 0; --i) {
    PrimExpr min = ranges[i]->min;
    if (min.dtype() != idx_dtype) {
      min = cast(idx_dtype, min);
    }
    leading_index = leading_index + min * stride;
    PrimExpr dim = buffer->shape[i];
    if (dim.dtype() != idx_dtype) {
      dim = cast(idx_dtype, dim);
    }
    stride = stride * dim;
  }

  return leading_index * make_const(idx_dtype, packed_tile_elems);
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
  int elem_bits;
  const MlsTileConfig *trans;
  int trans_count;
  const MlsTileConfig *non_trans;
  int non_trans_count;
};

template <int N> constexpr int ArraySize(const MlsTileConfig (&)[N]) {
  return N;
}

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

constexpr MlsTileConfig kMlsTileConfigsB4Trans[] = {
    {16, 256, false},
    {16, 128, false},
};

constexpr MlsTileConfig kMlsTileConfigsB4NonTrans[] = {
    {256, 16, false},
    {128, 16, false},
};

constexpr MlsTileConfig kMlsTileConfigsFp4B8Trans[] = {
    {16, 64, false},
};

constexpr MlsTileConfig kMlsTileConfigsFp4B8NonTrans[] = {
    {64, 16, false},
};

constexpr MlsTileConfigSet kMlsTileConfigTable[] = {
    {4, kMlsTileConfigsB4Trans, ArraySize(kMlsTileConfigsB4Trans),
     kMlsTileConfigsB4NonTrans, ArraySize(kMlsTileConfigsB4NonTrans)},
    {8, kMlsTileConfigsB8Trans, ArraySize(kMlsTileConfigsB8Trans),
     kMlsTileConfigsB8NonTrans, ArraySize(kMlsTileConfigsB8NonTrans)},
    {16, kMlsTileConfigsB16Trans, ArraySize(kMlsTileConfigsB16Trans),
     kMlsTileConfigsB16NonTrans, ArraySize(kMlsTileConfigsB16NonTrans)},
};

const MlsTileConfigSet &GetMlsTileConfigSet(int elem_bits, Target target) {
  if (elem_bits == 4) {
    ICHECK_EQ(GetHcuArchString(target), "gfx946")
        << "b4 matrix_load is only supported on gfx946";
  }
  for (const auto &config_set : kMlsTileConfigTable) {
    if (config_set.elem_bits == elem_bits) {
      return config_set;
    }
  }
  LOG(FATAL) << "matrix_load unsupported element bitwidth: " << elem_bits;
  return kMlsTileConfigTable[0];
}

bool IsFp4B8LdsTile(bool trans, int mls_tile_mn, int mls_tile_k) {
  return (!trans && mls_tile_mn == 64 && mls_tile_k == 16) ||
         (trans && mls_tile_mn == 16 && mls_tile_k == 64);
}

} // namespace

int GetMlsLdsPhysicalBits(DataType dtype, bool trans, int mls_tile_mn,
                          int mls_tile_k, Target target) {
  if (!TargetIsHCU(target) || !dtype.is_float4_e2m1fn()) {
    return dtype.bits();
  }
  const std::string arch = GetHcuArchString(target);
  if ((arch == "gfx92a" || arch == "gfx946") &&
      IsFp4B8LdsTile(trans, mls_tile_mn, mls_tile_k)) {
    return 8;
  }
  return dtype.bits();
}

std::pair<int64_t, int64_t> MlsBlockDims(const Buffer &buf, bool mls_trans) {
  ICHECK(buf->shape.size() >= 2);
  size_t nd = buf->shape.size();
  if (mls_trans) {
    return {*as_const_int(buf->shape[nd - 2]),
            *as_const_int(buf->shape[nd - 1])};
  }
  return {*as_const_int(buf->shape[nd - 1]), *as_const_int(buf->shape[nd - 2])};
}

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
 * MLS tile size rules (MN interleave=1):
 * num_warps = block_size / TargetHcuGetWarpSize(target); warp_mn * warp_k =
 * num_warps. One warp group extent in K = warp_k * mlsTilesizeK. If > block_k,
 * warps repeat load.
 */
void ComputeMlsWarpPartition(bool trans, int block_mn, int block_k,
                             int block_size, Target target, int elem_bits,
                             int &warp_mn, int &warp_k, int &mls_tile_mn,
                             int &mls_tile_k) {
  int num_warps = block_size / TargetHcuGetWarpSize(target);
  ICHECK(num_warps >= 1) << "num_warps must be >= 1";

  auto try_config = [&](const MlsTileConfig &config) -> bool {
    int tile_mn = config.tile_mn;
    int tile_k = config.tile_k;
    auto min_block_mn = [&]() -> std::optional<int> {
      if (elem_bits == 4 && trans && tile_mn == 16 && tile_k >= 128) {
        return 32;
      }
      return std::nullopt;
    }();
    auto min_block_k = [&]() -> std::optional<int> {
      if (elem_bits == 4 && !trans && tile_k == 16 && tile_mn > 64) {
        return 32;
      }
      return std::nullopt;
    }();
    if (min_block_mn && block_mn < min_block_mn.value()) {
      return false;
    }
    if (min_block_k && block_k < min_block_k.value()) {
      return false;
    }
    if (block_k % tile_k != 0 || block_mn % tile_mn != 0)
      return false;
    int wm = std::min(block_mn / tile_mn, num_warps);
    if (num_warps % wm != 0)
      return false;
    int wk = num_warps / wm;
    if (config.require_no_repeat) {
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

  auto try_configs = [&](const MlsTileConfig *configs, int config_count) {
    for (int i = 0; i < config_count; ++i) {
      if (try_config(configs[i]))
        return true;
    }
    return false;
  };

  if (elem_bits == 4) {
    const std::string arch = GetHcuArchString(target);
    if (arch == "gfx946") {
      const MlsTileConfig *b4_configs =
          trans ? kMlsTileConfigsB4Trans : kMlsTileConfigsB4NonTrans;
      int b4_config_count = trans ? ArraySize(kMlsTileConfigsB4Trans)
                                  : ArraySize(kMlsTileConfigsB4NonTrans);
      if (try_configs(b4_configs, b4_config_count)) {
        return;
      }
    } else {
      ICHECK_EQ(arch, "gfx92a")
          << "b4 matrix_load is only supported on gfx946; fp4 b8 LDS "
             "fallback is supported on gfx92a/gfx946";
    }
    const MlsTileConfig *fp4_b8_configs =
        trans ? kMlsTileConfigsFp4B8Trans : kMlsTileConfigsFp4B8NonTrans;
    int fp4_b8_config_count = trans ? ArraySize(kMlsTileConfigsFp4B8Trans)
                                    : ArraySize(kMlsTileConfigsFp4B8NonTrans);
    if (try_configs(fp4_b8_configs, fp4_b8_config_count)) {
      return;
    }
  } else {
    const MlsTileConfigSet &config_set = GetMlsTileConfigSet(elem_bits, target);
    const MlsTileConfig *configs =
        trans ? config_set.trans : config_set.non_trans;
    int config_count =
        trans ? config_set.trans_count : config_set.non_trans_count;
    if (try_configs(configs, config_count)) {
      return;
    }
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

  // MLS LdsDesc is built from the full last-2 dst shape; forbid last-2 slices.
  {
    size_t dr_chk = this->dst_ranges.size();
    ICHECK(dr_chk >= 2);
    ICHECK_EQ(dst->shape.size(), dr_chk)
        << "matrix_load dst buffer rank must match dst region rank";
    for (size_t i = dr_chk - 2; i < dr_chk; ++i) {
      const int64_t *min_c = as_const_int(this->dst_ranges[i]->min);
      const int64_t *ext_c = as_const_int(this->dst_ranges[i]->extent);
      const int64_t *shape_c = as_const_int(dst->shape[i]);
      ICHECK(min_c && ext_c && shape_c)
          << "matrix_load dst last-2 region must be static, dim=" << i;
      ICHECK_EQ(*min_c, 0) << "matrix_load forbids slicing dst last-2 dims "
                              "(min must be 0), dim="
                           << i << " min=" << *min_c;
      ICHECK_EQ(*ext_c, *shape_c)
          << "matrix_load forbids slicing dst last-2 dims (extent must equal "
             "buffer shape), dim="
          << i << " extent=" << *ext_c << " shape=" << *shape_c;
    }
  }

  const bool mls_trans = mls_trans_;
  auto [block_mn, block_k] = MlsBlockDims(dst, mls_trans);
  int block_size = static_cast<int>(*as_const_int(T.thread_bounds->extent));
  int warp_id_offset = MlsScopedWarpIdOffset(T.thread_bounds, T.target);
  int tile_mn, tile_k, warp_m, warp_k;
  ComputeMlsWarpPartition(mls_trans, static_cast<int>(block_mn),
                          static_cast<int>(block_k), block_size, T.target,
                          src->dtype.bits(), warp_m, warp_k, tile_mn, tile_k);
  const int lds_physical_bits =
      GetMlsLdsPhysicalBits(src->dtype, mls_trans, tile_mn, tile_k, T.target);

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
  } else if (src->dtype.is_float4_e2m1fn()) {
    dtype_str = "tl::pk_fp4_t";
  } else {
    LOG(FATAL) << "matrix_load unsupported dtype: " << src->dtype;
  }

  std::stringstream ss;
  ss << "tl::mls::mls_load_tile<tl::sequence<" << block_mn << ", " << block_k
     << ">, tl::sequence<" << tile_mn << ", " << tile_k << ">, " << warp_m
     << ", " << warp_k << ", " << dtype_str << ", 1, "
     << (mls_trans ? "true" : "false")
     << ", tl::hcu_target_enum::" << GetHcuArchString(T.target) << ", "
     << lds_physical_bits << ", " << (check_last_load ? "true" : "false")
     << ", " << (last_load ? "true" : "false") << ">";

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
    if (auto packed_offset = TryGetMlsPackedLeadingElemOffset(
            dst_buf, this->dst_ranges, tile_mn, tile_k, mls_trans, T.target,
            lds_physical_bits)) {
      dst_leading_elem_offset = packed_offset.value();
    } else {
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

  Stmt stmt =
      Evaluate(Call(DataType::Handle(), builtin::call_extern(), call_args));
  if (auto actual_size_bytes = TryGetMlsDstActualSizeBytes(
          dst_buf, tile_mn, tile_k, mls_trans, T.target, lds_physical_bits)) {
    Map<String, PrimExpr> actual_size_bytes_map;
    actual_size_bytes_map.Set(dst_buf->data->name_hint,
                              actual_size_bytes.value());
    stmt = AttrStmt(actual_size_bytes_map, tl::attr::kMlsActualSizeBytesMap,
                    Integer(0), std::move(stmt));
  }
  return stmt;
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
