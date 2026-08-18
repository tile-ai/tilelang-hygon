/*!
 * \file gemm_a_lds_strategy.cc
 * \brief Compiler-derived LDS strategy for ordinary HCU GEMM A copies.
 */

#include "gemm_a_lds_strategy.h"

#include "hcu/op/gemm_partition.h"
#include "hcu/target_utils.h"
#include "op/utils.h"
#include "support/check.h"

#include <tvm/arith/analyzer.h>

#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

constexpr int kStrategyVersion = 2;
// These are instruction semantics, not kernel tile parameters.
constexpr int kMmacMAtom = 16;
constexpr int kAsyncCopyTransactionBytes = 16;
constexpr int kReadBytesPerLane = 8;

struct GemmAStrategyParams {
  int block_m{0};
  int block_k{0};
  int block_threads{0};
  int warp_size{0};
  int warp_m_count{0};
  int bank_num{0};
  int bank_width_bytes{0};
  int element_bytes{0};
  int copy_bytes_per_lane{0};
  int copy_transaction_bytes{0};
  int copy_transactions_per_lane{0};
  int read_bytes_per_lane{0};
  int row_bank_stride{0};
  int row_period{0};
  int rows_per_copy_wave{0};
  int row_slab_count{0};
  int warp_tile_m{0};
  int copy_elements_per_lane{0};
  int copy_segments_per_row{0};
  int read_elements_per_lane{0};
  int read_segments_per_row{0};
  int segment_shift{0};
  int rows_per_wrap_phase{0};
  int wrap_offset{0};
  int wrap_idx_mask{0};
};

std::optional<int64_t> StaticExtent(const Array<Range> &ranges, size_t dim) {
  if (dim >= ranges.size()) {
    return std::nullopt;
  }
  const int64_t *extent = as_const_int(ranges[dim]->extent);
  if (extent == nullptr) {
    return std::nullopt;
  }
  return *extent;
}

bool IsPowerOfTwo(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

int SelectWrapOffset(const GemmAStrategyParams &params, int max_wrap_offset) {
  if (max_wrap_offset <= 0) {
    return 0;
  }
  const int banks_per_read =
      params.read_bytes_per_lane / params.bank_width_bytes;
  const int banks_per_copy =
      params.copy_transaction_bytes / params.bank_width_bytes;
  std::vector<bool> base_banks(params.bank_num, false);
  for (int row = 0; row < params.row_period; ++row) {
    const int start = (row * params.row_bank_stride) % params.bank_num;
    for (int bank = 0; bank < banks_per_read; ++bank) {
      base_banks[(start + bank) % params.bank_num] = true;
    }
  }
  for (int offset = 1;
       offset * params.wrap_idx_mask <= max_wrap_offset; ++offset) {
    const int bank_shift = (offset * banks_per_copy) % params.bank_num;
    bool overlaps = false;
    for (int bank = 0; bank < params.bank_num; ++bank) {
      if (base_banks[bank] &&
          base_banks[(bank + bank_shift) % params.bank_num]) {
        overlaps = true;
        break;
      }
    }
    if (!overlaps) {
      return offset;
    }
  }
  return 0;
}

PrimExpr MatrixARowPermute(const PrimExpr &row,
                           const GemmAStrategyParams &params) {
  PrimExpr slab = floordiv(row, Integer(params.rows_per_copy_wave));
  PrimExpr row_group = floordiv(
      floormod(row, Integer(params.rows_per_copy_wave)),
      Integer(params.row_period));
  PrimExpr inner_row = floormod(row, Integer(params.row_period));
  return row_group * (params.row_slab_count * params.row_period) +
         slab * params.row_period + inner_row;
}

Layout MakeStorageLayout(const GemmAStrategyParams &params) {
  PrimExpr row = InputPlaceholder(0);
  PrimExpr col = InputPlaceholder(1);
  PrimExpr mapped_row = MatrixARowPermute(row, params);
  PrimExpr segment = mapped_row * params.copy_segments_per_row +
                     floordiv(col, Integer(params.copy_elements_per_lane));
  PrimExpr transaction = floordiv(segment, Integer(params.block_threads));
  PrimExpr thread = floormod(segment, Integer(params.block_threads));
  PrimExpr copy_warp = floordiv(thread, Integer(params.warp_size));
  PrimExpr lane = floormod(thread, Integer(params.warp_size));
  // Reproduce the hardware wrap on each issued copy transaction so the
  // annotation and the physical LDS writes share the same mapping.
  PrimExpr wrap_offset_cur =
      floormod(copy_warp, Integer(params.wrap_idx_mask + 1)) *
      params.wrap_offset;
  PrimExpr physical_lane =
      floormod(lane + wrap_offset_cur, Integer(params.warp_size));
  PrimExpr physical_segment = transaction * params.block_threads +
                              copy_warp * params.warp_size + physical_lane;
  PrimExpr physical_linear =
      physical_segment * params.copy_elements_per_lane +
      floormod(col, Integer(params.copy_elements_per_lane));
  PrimExpr physical_row =
      floordiv(physical_linear, Integer(params.block_k));
  PrimExpr physical_col =
      floormod(physical_linear, Integer(params.block_k));
  Array<PrimExpr> input_shape = {Integer(params.block_m),
                                 Integer(params.block_k)};
  Array<PrimExpr> forward_index = {physical_row, physical_col};
  return Layout(input_shape, forward_index);
}

Fragment MakeCopyLoopLayout(const GemmAStrategyParams &params) {
  PrimExpr row = InputPlaceholder(0);
  PrimExpr col = InputPlaceholder(1);
  PrimExpr mapped_row = MatrixARowPermute(row, params);
  PrimExpr segment = mapped_row * params.copy_segments_per_row +
                     floordiv(col, Integer(params.copy_elements_per_lane));
  PrimExpr thread = floormod(segment, Integer(params.block_threads));
  PrimExpr transaction = floordiv(segment, Integer(params.block_threads));
  PrimExpr intra = floormod(col, Integer(params.copy_elements_per_lane));
  if (params.copy_transactions_per_lane == 1) {
    return Fragment({Integer(params.block_m), Integer(params.block_k)},
                    {intra}, thread, Integer(1), std::nullopt);
  }
  return Fragment({Integer(params.block_m), Integer(params.block_k)},
                  {transaction, intra}, thread, Integer(1), std::nullopt);
}

void ValidateSameLayout(const Layout &actual, const Layout &expected,
                        int block_m, int block_k, const char *kind) {
  ICHECK(actual.defined()) << kind << " layout is undefined";
  ICHECK(expected.defined()) << "Expected " << kind << " layout is undefined";
  ICHECK_EQ(actual->InputDim(), expected->InputDim())
      << kind << " layout input rank mismatch: actual=" << actual->DebugOutput()
      << ", expected=" << expected->DebugOutput();

  arith::Analyzer analyzer;
  for (int row = 0; row < block_m; ++row) {
    for (int col = 0; col < block_k; ++col) {
      Array<PrimExpr> logical = {Integer(row), Integer(col)};
      Array<PrimExpr> actual_index = actual->Forward(logical);
      Array<PrimExpr> expected_index = expected->Forward(logical);
      ICHECK_EQ(actual_index.size(), expected_index.size())
          << kind << " layout output rank mismatch";
      for (size_t dim = 0; dim < actual_index.size(); ++dim) {
        PrimExpr diff =
            analyzer.Simplify(actual_index[dim] - expected_index[dim]);
        if (!is_zero(diff)) {
          LOG(FATAL) << "HCU GEMM A " << kind
                     << " layout does not match the compiler-derived strategy"
                     << " at logical coordinate (" << row << ", " << col
                     << "), output dim " << dim
                     << ": actual=" << actual_index
                     << ", expected=" << expected_index
                     << "\nactual layout: " << actual->DebugOutput()
                     << "\nexpected layout: " << expected->DebugOutput();
        }
      }
    }
  }
}

} // namespace

void HcuGemmALdsStrategyNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<HcuGemmALdsStrategyNode>()
      .def_ro("strategy_version", &HcuGemmALdsStrategyNode::strategy_version)
      .def_ro("block_m", &HcuGemmALdsStrategyNode::block_m)
      .def_ro("block_k", &HcuGemmALdsStrategyNode::block_k)
      .def_ro("block_threads", &HcuGemmALdsStrategyNode::block_threads)
      .def_ro("warp_size", &HcuGemmALdsStrategyNode::warp_size)
      .def_ro("warp_m_count", &HcuGemmALdsStrategyNode::warp_m_count)
      .def_ro("bank_num", &HcuGemmALdsStrategyNode::bank_num)
      .def_ro("bank_width_bytes", &HcuGemmALdsStrategyNode::bank_width_bytes)
      .def_ro("element_bytes", &HcuGemmALdsStrategyNode::element_bytes)
      .def_ro("copy_bytes_per_lane",
              &HcuGemmALdsStrategyNode::copy_bytes_per_lane)
      .def_ro("copy_transaction_bytes",
              &HcuGemmALdsStrategyNode::copy_transaction_bytes)
      .def_ro("copy_transactions_per_lane",
              &HcuGemmALdsStrategyNode::copy_transactions_per_lane)
      .def_ro("read_bytes_per_lane",
              &HcuGemmALdsStrategyNode::read_bytes_per_lane)
      .def_ro("row_period", &HcuGemmALdsStrategyNode::row_period)
      .def_ro("row_bank_stride", &HcuGemmALdsStrategyNode::row_bank_stride)
      .def_ro("segment_shift", &HcuGemmALdsStrategyNode::segment_shift)
      .def_ro("rows_per_copy_wave",
              &HcuGemmALdsStrategyNode::rows_per_copy_wave)
      .def_ro("row_slab_count", &HcuGemmALdsStrategyNode::row_slab_count)
      .def_ro("warp_tile_m", &HcuGemmALdsStrategyNode::warp_tile_m)
      .def_ro("wrap_offset", &HcuGemmALdsStrategyNode::wrap_offset)
      .def_ro("wrap_idx_mask", &HcuGemmALdsStrategyNode::wrap_idx_mask)
      .def_ro("storage_layout", &HcuGemmALdsStrategyNode::storage_layout)
      .def_ro("copy_loop_layout", &HcuGemmALdsStrategyNode::copy_loop_layout);
}

Optional<HcuGemmALdsStrategy>
DeriveHcuGemmALdsStrategy(const CopyNode &copy, const GemmNode &gemm,
                          int block_threads, Target target) {
  if (!TargetIsHCU(target) || !TargetHcuHasAsyncCopy(target) ||
      block_threads <= 0) {
    return std::nullopt;
  }
  if (!IsGlobalBuffer(copy.src) || !IsSharedBuffer(copy.dst) ||
      copy.src->dtype != copy.dst->dtype ||
      !(copy.dst->dtype.is_float16() || copy.dst->dtype.is_bfloat16()) ||
      copy.src_range.size() < 2 || copy.dst_range.size() < 2) {
    return std::nullopt;
  }
  const size_t src_rank = copy.src_range.size();
  const size_t dst_rank = copy.dst_range.size();
  std::optional<int64_t> src_m = StaticExtent(copy.src_range, src_rank - 2);
  std::optional<int64_t> src_k = StaticExtent(copy.src_range, src_rank - 1);
  std::optional<int64_t> dst_m = StaticExtent(copy.dst_range, dst_rank - 2);
  std::optional<int64_t> dst_k = StaticExtent(copy.dst_range, dst_rank - 1);
  if (!src_m || !src_k || !dst_m || !dst_k || *src_m != *dst_m ||
      *src_k != *dst_k || *dst_m <= 0 || *dst_k <= 0) {
    return std::nullopt;
  }
  const int block_m = static_cast<int>(*dst_m);
  const int block_k = static_cast<int>(*dst_k);
  const int element_bits = copy.dst->dtype.bits();
  if (gemm.transA_ || gemm.m_ != block_m || gemm.k_ != block_k ||
      gemm.kPack_ != 1 || gemm.a_->dtype != copy.dst->dtype) {
    return std::nullopt;
  }

  const hcu::HcuMmacModeInfo mmac_mode = hcu::ResolveHcuMmacMode(
      gemm.a_->dtype, gemm.b_->dtype, /*a_is_fragment=*/false,
      /*b_is_fragment=*/false, /*is_blockscaled=*/false, block_k,
      ScaleLdsFormat::kIdentity, ScaleLdsFormat::kIdentity, target);
  if (element_bits <= 0 || element_bits % 8 != 0 ||
      mmac_mode.element_bits != element_bits || block_k != mmac_mode.mmac_k) {
    return std::nullopt;
  }

  GemmWarpPolicy policy(gemm.policy_->policy_type);
  hcu::ComputeWarpPartitionHCU(*policy.get(), gemm.m_, gemm.n_, gemm.k_,
                               gemm.kPack_, element_bits,
                               block_threads, target,
                               /*A_from_mls=*/false,
                               /*B_from_mls=*/false,
                               /*A_mls_trans=*/false,
                               /*B_mls_trans=*/false);
  const int warp_size = TargetHcuGetWarpSize(target);
  if (block_threads % warp_size != 0 || policy->m_warp <= 0 ||
      !IsPowerOfTwo(policy->m_warp) || policy->k_warp != 1 ||
      policy->m_warp * policy->n_warp * policy->k_warp !=
          block_threads / warp_size) {
    return std::nullopt;
  }

  GemmAStrategyParams params;
  params.block_m = block_m;
  params.block_k = block_k;
  params.block_threads = block_threads;
  params.warp_size = warp_size;
  params.warp_m_count = policy->m_warp;
  params.bank_num = TargetHcuGetLdsBankCount(target);
  params.bank_width_bytes = TargetHcuGetLdsBankWidthBytes(target);
  params.element_bytes = element_bits / 8;

  const int64_t tile_bytes =
      static_cast<int64_t>(block_m) * block_k * params.element_bytes;
  if (tile_bytes % block_threads != 0) {
    return std::nullopt;
  }
  params.copy_bytes_per_lane = static_cast<int>(tile_bytes / block_threads);
  params.copy_transaction_bytes = kAsyncCopyTransactionBytes;
  params.read_bytes_per_lane = kReadBytesPerLane;
  if (params.copy_bytes_per_lane < params.copy_transaction_bytes ||
      params.copy_bytes_per_lane % params.copy_transaction_bytes != 0 ||
      params.copy_bytes_per_lane % params.element_bytes != 0 ||
      params.read_bytes_per_lane % params.element_bytes != 0 ||
      params.copy_transaction_bytes % params.read_bytes_per_lane != 0) {
    return std::nullopt;
  }
  params.copy_transactions_per_lane =
      params.copy_bytes_per_lane / params.copy_transaction_bytes;

  // MMAC A VGPR layout: each lane reads 4 half values, i.e. one b64.
  const int row_bytes = block_k * params.element_bytes;
  if (row_bytes % params.bank_width_bytes != 0 ||
      row_bytes % params.copy_transaction_bytes != 0 ||
      row_bytes % params.read_bytes_per_lane != 0) {
    return std::nullopt;
  }
  params.row_bank_stride = row_bytes / params.bank_width_bytes;
  // Row starts advance by row_bank_stride banks and repeat after row_period
  // rows.  For the current FP16 K=16 case, these values are 8 and 4.
  params.row_period =
      params.bank_num / std::gcd(params.bank_num, params.row_bank_stride);
  params.copy_elements_per_lane =
      params.copy_transaction_bytes / params.element_bytes;
  params.copy_segments_per_row = row_bytes / params.copy_transaction_bytes;
  params.read_elements_per_lane =
      params.read_bytes_per_lane / params.element_bytes;
  params.read_segments_per_row = row_bytes / params.read_bytes_per_lane;
  params.segment_shift =
      params.copy_transaction_bytes / params.read_bytes_per_lane;
  const int64_t copy_segment_count =
      static_cast<int64_t>(block_m) * params.copy_segments_per_row;
  if (copy_segment_count % block_threads != 0 ||
      copy_segment_count / block_threads !=
          params.copy_transactions_per_lane) {
    return std::nullopt;
  }
  if (warp_size % params.copy_segments_per_row != 0) {
    return std::nullopt;
  }
  params.rows_per_copy_wave = warp_size / params.copy_segments_per_row;
  if (block_m % params.rows_per_copy_wave != 0 ||
      params.rows_per_copy_wave % params.row_period != 0 ||
      block_m % params.warp_m_count != 0 ||
      kMmacMAtom % params.row_period != 0) {
    return std::nullopt;
  }
  params.row_slab_count = block_m / params.rows_per_copy_wave;
  params.warp_tile_m = block_m / params.warp_m_count;
  // Reorganize copy waves in row_period-row phases to derive the row permute.
  params.rows_per_wrap_phase =
      params.rows_per_copy_wave / params.row_period;
  // For block_M=256, four warp-M groups are selected by wrap_idx_mask=3.
  params.wrap_idx_mask = params.warp_m_count - 1;
  // Select the smallest wrap that separates adjacent repeated-bank groups.
  // In the current case, shifting one 16-byte segment moves exactly 4 banks,
  // so wrap_offset=1.
  params.wrap_offset =
      SelectWrapOffset(params, TargetHcuGetLdsWrapMaxOffset(
                                   target, params.copy_transaction_bytes));
  if (params.wrap_offset == 0) {
    return std::nullopt;
  }

  auto node = ffi::make_object<HcuGemmALdsStrategyNode>();
  node->strategy_version = kStrategyVersion;
  node->block_m = params.block_m;
  node->block_k = params.block_k;
  node->block_threads = params.block_threads;
  node->warp_size = params.warp_size;
  node->warp_m_count = params.warp_m_count;
  node->bank_num = params.bank_num;
  node->bank_width_bytes = params.bank_width_bytes;
  node->element_bytes = params.element_bytes;
  node->copy_bytes_per_lane = params.copy_bytes_per_lane;
  node->copy_transaction_bytes = params.copy_transaction_bytes;
  node->copy_transactions_per_lane = params.copy_transactions_per_lane;
  node->read_bytes_per_lane = params.read_bytes_per_lane;
  node->row_bank_stride = params.row_bank_stride;
  node->row_period = params.row_period;
  node->segment_shift = params.segment_shift;
  node->rows_per_copy_wave = params.rows_per_copy_wave;
  node->row_slab_count = params.row_slab_count;
  node->warp_tile_m = params.warp_tile_m;
  node->wrap_offset = params.wrap_offset;
  node->wrap_idx_mask = params.wrap_idx_mask;
  // Column segment shift/carry completes the conflict-free physical layout.
  node->storage_layout = MakeStorageLayout(params);
  node->copy_loop_layout = MakeCopyLoopLayout(params);
  return HcuGemmALdsStrategy(std::move(node));
}

void ValidateHcuGemmAStorageLayout(const Layout &actual,
                                   const HcuGemmALdsStrategy &strategy) {
  ValidateSameLayout(actual, strategy->storage_layout, strategy->block_m,
                     strategy->block_k, "storage");
}

void ValidateHcuGemmACopyLayout(const Fragment &actual,
                                const HcuGemmALdsStrategy &strategy) {
  ValidateSameLayout(actual, strategy->copy_loop_layout, strategy->block_m,
                     strategy->block_k, "copy-loop local");
  arith::Analyzer analyzer;
  for (int row = 0; row < strategy->block_m; ++row) {
    for (int col = 0; col < strategy->block_k; ++col) {
      Array<PrimExpr> logical = {Integer(row), Integer(col)};
      PrimExpr actual_thread =
          actual->ForwardThread(logical, Optional<PrimExpr>());
      PrimExpr expected_thread = strategy->copy_loop_layout->ForwardThread(
          logical, Optional<PrimExpr>());
      if (!is_zero(analyzer.Simplify(actual_thread - expected_thread))) {
        LOG(FATAL) << "HCU GEMM A copy-loop layout thread mapping mismatch at "
                   << "logical coordinate (" << row << ", " << col
                   << "): actual=" << actual_thread
                   << ", expected=" << expected_thread;
      }
    }
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  HcuGemmALdsStrategyNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
