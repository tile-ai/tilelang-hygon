/*!
 * \file gemm_an_bt_lds_strategy.cc
 * \brief Compiler-derived LDS strategy for HCU GEMM AN/BT ds-read copies.
 */

#include "gemm_an_bt_lds_strategy.h"

#include "hcu/target_utils.h"
#include "hcu/utils/gemm_lds_access.h"
#include "hcu/utils/gemm_lds_strategy_utils.h"
#include "op/utils.h"

#include <tvm/arith/analyzer.h>

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

constexpr int kStrategyVersion = 4;
constexpr int kDsReadTileK = 16;
constexpr int kDsReadPanelN = 32;
constexpr int kDsReadPhaseBytes = 128;
constexpr int kDsReadPhaseCount = 8;
constexpr int kDsReadRequestsPerPhase = 8;
constexpr int kAsyncCopyBytesPerLane = 16;
constexpr int kDsReadBytesPerLane = 16;
constexpr int kDsReadWrapStepBytes = 64;
constexpr int kDsReadWrapCount = 2;

// ds_read_m32x16_b16 issues these eight lane addresses in each 128-byte phase.
constexpr std::array<std::array<int, kDsReadRequestsPerPhase>,
                     kDsReadPhaseCount>
    kDsReadPhaseLanes = {{{0, 4, 1, 5, 18, 22, 19, 23},
                          {32, 36, 33, 37, 50, 54, 51, 55},
                          {2, 6, 3, 7, 16, 20, 17, 21},
                          {40, 44, 41, 45, 58, 62, 59, 63},
                          {8, 12, 9, 13, 26, 30, 27, 31},
                          {42, 46, 43, 47, 56, 60, 57, 61},
                          {10, 14, 11, 15, 24, 28, 25, 29},
                          {34, 38, 35, 39, 48, 52, 49, 53}}};

struct AnBtStrategyParams {
  int block_k{0};
  int block_mn{0};
  int block_threads{0};
  int warp_size{0};
  int bank_num{0};
  int bank_width_bytes{0};
  int element_bytes{0};
  int copy_bytes_per_lane{0};
  int copy_transaction_bytes{0};
  int copy_transactions_per_lane{0};
  int read_bytes_per_lane{0};
  int phase_bytes{0};
  int panel_mn{0};
  int copy_elements_per_lane{0};
  int copy_segments_per_row{0};
  int wrap_offset{0};
  int wrap_idx_mask{0};
};

struct BankConflictScore {
  int max_bank_uses{0};
  int total_conflicts{0};
  int first_panel{-1};
  int first_phase{-1};
  int first_lane{-1};
  int first_bank{-1};
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

int64_t RequireConst(const PrimExpr &expr, arith::Analyzer *analyzer,
                     const char *kind, int k, int n) {
  PrimExpr simplified = analyzer->Simplify(expr);
  const int64_t *value = as_const_int(simplified);
  ICHECK(value) << "HCU GEMM AN/BT " << kind
                << " mapping must be constant at logical coordinate (" << k
                << ", " << n << "), got " << simplified;
  return *value;
}

Fragment MakeCopyLoopLayout(const AnBtStrategyParams &params,
                            bool permute_rows, int rows_per_copy_wave = 0,
                            int wrap_count = kDsReadWrapCount,
                            bool permute_across_block_k = false) {
  PrimExpr k = InputPlaceholder(0);
  PrimExpr n = InputPlaceholder(1);
  PrimExpr k_tile = floordiv(k, Integer(kDsReadTileK));
  PrimExpr k_inner = floormod(k, Integer(kDsReadTileK));
  PrimExpr n_segment =
      floordiv(n, Integer(params.copy_elements_per_lane));
  PrimExpr mapped_k_inner = k_inner;
  if (permute_rows) {
    ICHECK_GT(rows_per_copy_wave, 0);
    const int row_extent =
        permute_across_block_k ? params.block_k : kDsReadTileK;
    ICHECK_LE(rows_per_copy_wave * wrap_count, row_extent);
    PrimExpr row_index = permute_across_block_k ? k : k_inner;
    PrimExpr row_pair = floordiv(row_index, Integer(wrap_count));
    PrimExpr wrap_class = floormod(row_index, Integer(wrap_count));
    // Group adjacent K rows into the two hardware wrap classes.  For
    // rows_per_copy_wave=2 this reduces to the original 0,2,1,3 ordering.
    mapped_k_inner =
        floordiv(row_pair, Integer(rows_per_copy_wave)) *
            (wrap_count * rows_per_copy_wave) +
        wrap_class * rows_per_copy_wave +
        floormod(row_pair, Integer(rows_per_copy_wave));
  }
  PrimExpr mapped_k =
      permute_rows && permute_across_block_k
          ? mapped_k_inner
          : k_tile * kDsReadTileK + mapped_k_inner;
  PrimExpr canonical_segment =
      mapped_k * params.copy_segments_per_row + n_segment;
  PrimExpr transaction =
      floordiv(canonical_segment, Integer(params.block_threads));
  PrimExpr thread =
      floormod(canonical_segment, Integer(params.block_threads));
  PrimExpr intra = floormod(n, Integer(params.copy_elements_per_lane));
  if (params.copy_transactions_per_lane == 1) {
    return Fragment({Integer(params.block_k), Integer(params.block_mn)},
                    {intra}, thread, Integer(1), std::nullopt);
  }
  return Fragment({Integer(params.block_k), Integer(params.block_mn)},
                  {transaction, intra}, thread, Integer(1), std::nullopt);
}

Layout MakeStorageLayout(const AnBtStrategyParams &params,
                         const Fragment &copy_layout) {
  PrimExpr k = InputPlaceholder(0);
  PrimExpr n = InputPlaceholder(1);
  Array<PrimExpr> logical = {k, n};
  PrimExpr thread = copy_layout->ForwardThread(logical, std::nullopt);
  Array<PrimExpr> local = copy_layout->Forward(logical);
  ICHECK(local.size() == 1U || local.size() == 2U);
  PrimExpr transaction =
      local.size() == 2U ? local[0] : make_const(thread->dtype, 0);
  PrimExpr intra = local[local.size() - 1];

  PrimExpr copy_warp = floordiv(thread, Integer(params.warp_size));
  PrimExpr copy_lane = floormod(thread, Integer(params.warp_size));
  PrimExpr wrap = floormod(copy_warp, Integer(params.wrap_idx_mask + 1)) *
                  params.wrap_offset;
  PrimExpr physical_lane =
      floormod(copy_lane + wrap, Integer(params.warp_size));
  PrimExpr physical_offset =
      (transaction * params.block_threads + copy_warp * params.warp_size +
       physical_lane) *
          params.copy_elements_per_lane +
      intra;
  Array<PrimExpr> physical = {
      floordiv(physical_offset, Integer(params.block_mn)),
      floormod(physical_offset, Integer(params.block_mn))};
  Array<PrimExpr> input_shape = {Integer(params.block_k),
                                 Integer(params.block_mn)};
  return Layout(input_shape, physical);
}

void ValidateSameLayout(const Layout &actual, const Layout &expected, int block_k,
                        int block_mn, const char *kind) {
  ICHECK(actual.defined()) << kind << " layout is undefined";
  ICHECK(expected.defined()) << "Expected " << kind << " layout is undefined";
  ICHECK_EQ(actual->InputDim(), expected->InputDim())
      << kind << " layout input rank mismatch: actual=" << actual->DebugOutput()
      << ", expected=" << expected->DebugOutput();

  arith::Analyzer analyzer;
  for (int k = 0; k < block_k; ++k) {
    for (int n = 0; n < block_mn; ++n) {
      Array<PrimExpr> logical = {Integer(k), Integer(n)};
      Array<PrimExpr> actual_index = actual->Forward(logical);
      Array<PrimExpr> expected_index = expected->Forward(logical);
      ICHECK_EQ(actual_index.size(), expected_index.size());
      for (size_t dim = 0; dim < actual_index.size(); ++dim) {
        if (!is_zero(
                analyzer.Simplify(actual_index[dim] - expected_index[dim]))) {
          LOG(FATAL) << "HCU GEMM AN/BT " << kind
                     << " layout mismatch at logical coordinate (" << k << ", "
                     << n << "), dimension " << dim
                     << ": actual=" << actual_index[dim]
                     << ", expected=" << expected_index[dim];
        }
      }
    }
  }
}

void ValidateStrategyParameters(const AnBtStrategyParams &params) {
  ICHECK_GT(params.block_k, 0);
  ICHECK_GT(params.block_mn, 0);
  ICHECK_GT(params.block_threads, 0);
  ICHECK_GT(params.warp_size, 0);
  ICHECK_EQ(params.block_threads % params.warp_size, 0);
  ICHECK_EQ(params.block_k % kDsReadTileK, 0);
  ICHECK_EQ(params.block_mn % params.panel_mn, 0);
  ICHECK_EQ(params.copy_transaction_bytes % params.element_bytes, 0);
  ICHECK_EQ(params.read_bytes_per_lane % params.element_bytes, 0);
  ICHECK_EQ(params.phase_bytes % params.read_bytes_per_lane, 0);
  ICHECK_EQ(params.block_mn % params.copy_elements_per_lane, 0);
  ICHECK_EQ(params.copy_segments_per_row * params.copy_elements_per_lane,
            params.block_mn);
  ICHECK_EQ(params.copy_bytes_per_lane,
            params.copy_transactions_per_lane *
                params.copy_transaction_bytes);
  ICHECK_GE(params.copy_transactions_per_lane, 1);
  ICHECK_GE(params.wrap_offset, 0);
  ICHECK_GE(params.wrap_idx_mask, 0);
  if (params.wrap_idx_mask == 0) {
    ICHECK_EQ(params.wrap_offset, 0);
  } else {
    ICHECK_GT(params.wrap_offset, 0);
    ICHECK(IsPowerOfTwo(params.wrap_idx_mask + 1));
  }
}

BankConflictScore EvaluateDsReadBankConflicts(
    const Layout &storage_layout, const AnBtStrategyParams &params) {
  ICHECK_EQ(params.read_bytes_per_lane % params.bank_width_bytes, 0);
  const int banks_per_read =
      params.read_bytes_per_lane / params.bank_width_bytes;
  BankConflictScore score;
  arith::Analyzer analyzer;
  for (int k_tile = 0; k_tile < params.block_k / kDsReadTileK; ++k_tile) {
    for (int panel = 0; panel < params.block_mn / params.panel_mn; ++panel) {
      for (int phase = 0; phase < kDsReadPhaseCount; ++phase) {
        std::vector<int> bank_uses(params.bank_num, 0);
        for (int lane : kDsReadPhaseLanes[phase]) {
          int k = k_tile * kDsReadTileK + (lane >> 2);
          int n = panel * params.panel_mn + (lane & 3) * 8;
          Array<PrimExpr> physical =
              storage_layout->Forward({Integer(k), Integer(n)});
          ICHECK_EQ(physical.size(), 2U);
          int64_t physical_k =
              RequireConst(physical[0], &analyzer, "ds-read row", k, n);
          int64_t physical_n =
              RequireConst(physical[1], &analyzer, "ds-read column", k, n);
          int64_t byte_offset =
              (physical_k * params.block_mn + physical_n) * params.element_bytes;
          ICHECK_EQ(byte_offset % params.bank_width_bytes, 0)
              << "HCU GEMM AN/BT ds-read address must be bank aligned";
          int start_bank = static_cast<int>(
              (byte_offset / params.bank_width_bytes) % params.bank_num);
          for (int bank = 0; bank < banks_per_read; ++bank) {
            int current = (start_bank + bank) % params.bank_num;
            int uses = ++bank_uses[current];
            score.max_bank_uses = std::max(score.max_bank_uses, uses);
            if (uses > 1) {
              ++score.total_conflicts;
              if (score.first_panel < 0) {
                score.first_panel = panel;
                score.first_phase = phase;
                score.first_lane = lane;
                score.first_bank = current;
              }
            }
          }
        }
      }
    }
  }
  return score;
}

bool SelectWrapStrategy(AnBtStrategyParams *params, Fragment *copy_layout,
                        Target target, int forced_wrap_count = 0) {
  const int bytes_per_row = params->block_mn * params->element_bytes;
  std::optional<HcuGemmLdsCopyGeometry> geometry =
      DeriveHcuGemmLdsCopyGeometry(bytes_per_row,
                                   params->copy_transaction_bytes,
                                   params->block_threads, target);
  if (!geometry.has_value() || geometry->warp_size != params->warp_size) {
    return false;
  }
  params->wrap_offset = 0;
  params->wrap_idx_mask = 0;
  *copy_layout = MakeCopyLoopLayout(*params, /*permute_rows=*/false);
  const bool linear_has_conflict =
      bytes_per_row >= geometry->bank_ring_bytes;
  if (!linear_has_conflict && forced_wrap_count == 0) {
    return true;
  }

  const int wrap_count =
      forced_wrap_count == 0 ? kDsReadWrapCount : forced_wrap_count;
  if (wrap_count != kDsReadWrapCount) {
    return false;
  }
  const int permute_row_extent =
      forced_wrap_count == 0 ? kDsReadTileK : params->block_k;
  const int permute_group_rows = geometry->rows_per_group * wrap_count;
  if (permute_group_rows > permute_row_extent ||
      permute_row_extent % permute_group_rows != 0) {
    return false;
  }

  // Row permutation is part of the conflict-resolution strategy, not a
  // baseline requirement of ds_read_m32x16_b16.
  *copy_layout = MakeCopyLoopLayout(*params, /*permute_rows=*/true,
                                    geometry->rows_per_group, wrap_count,
                                    /*permute_across_block_k=*/
                                        forced_wrap_count != 0);
  if (!IsLegalHcuGemmLdsWrap(*geometry, kDsReadWrapStepBytes,
                             wrap_count)) {
    return false;
  }
  params->wrap_offset =
      GetHcuGemmLdsWrapOffset(*geometry, kDsReadWrapStepBytes);
  params->wrap_idx_mask = wrap_count - 1;
  if (forced_wrap_count != 0) {
    Layout wrapped_storage = MakeStorageLayout(*params, *copy_layout);
    BankConflictScore wrapped_score =
        EvaluateDsReadBankConflicts(wrapped_storage, *params);
    if (wrapped_score.max_bank_uses > 1) {
      return false;
    }
  }
  return true;
}

} // namespace

void HcuGemmAnBtLdsStrategyNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<HcuGemmAnBtLdsStrategyNode>()
      .def_ro("strategy_version", &HcuGemmAnBtLdsStrategyNode::strategy_version)
      .def_ro("block_k", &HcuGemmAnBtLdsStrategyNode::block_k)
      .def_ro("block_mn", &HcuGemmAnBtLdsStrategyNode::block_mn)
      .def_ro("block_threads", &HcuGemmAnBtLdsStrategyNode::block_threads)
      .def_ro("warp_size", &HcuGemmAnBtLdsStrategyNode::warp_size)
      .def_ro("bank_num", &HcuGemmAnBtLdsStrategyNode::bank_num)
      .def_ro("bank_width_bytes", &HcuGemmAnBtLdsStrategyNode::bank_width_bytes)
      .def_ro("element_bytes", &HcuGemmAnBtLdsStrategyNode::element_bytes)
      .def_ro("copy_bytes_per_lane",
              &HcuGemmAnBtLdsStrategyNode::copy_bytes_per_lane)
      .def_ro("copy_transaction_bytes",
              &HcuGemmAnBtLdsStrategyNode::copy_transaction_bytes)
      .def_ro("copy_transactions_per_lane",
              &HcuGemmAnBtLdsStrategyNode::copy_transactions_per_lane)
      .def_ro("read_bytes_per_lane",
              &HcuGemmAnBtLdsStrategyNode::read_bytes_per_lane)
      .def_ro("phase_bytes", &HcuGemmAnBtLdsStrategyNode::phase_bytes)
      .def_ro("panel_mn", &HcuGemmAnBtLdsStrategyNode::panel_mn)
      .def_ro("wrap_offset", &HcuGemmAnBtLdsStrategyNode::wrap_offset)
      .def_ro("wrap_idx_mask", &HcuGemmAnBtLdsStrategyNode::wrap_idx_mask)
      .def_ro("storage_layout", &HcuGemmAnBtLdsStrategyNode::storage_layout)
      .def_ro("copy_loop_layout", &HcuGemmAnBtLdsStrategyNode::copy_loop_layout);
}

static Optional<HcuGemmAnBtLdsStrategy>
DeriveHcuGemmAnBtLdsStrategyImpl(const CopyNode &copy, const GemmNode &gemm,
                                 bool feeds_a, int block_threads,
                                 Target target, int forced_wrap_count) {
  if (!TargetIsHCU(target) || !TargetHcuHasAsyncCopy(target) ||
      block_threads <= 0 || !IsGlobalBuffer(copy.src) ||
      !IsSharedBuffer(copy.dst) || copy.src->dtype != copy.dst->dtype ||
      !(copy.dst->dtype.is_float16() || copy.dst->dtype.is_bfloat16()) ||
      copy.src_range.size() < 2 || copy.dst_range.size() < 2 ||
      gemm.kPack_ != 1) {
    return std::nullopt;
  }

  // AN (A, transA=true) and BT (B, transB=false) share the same
  // K-leading MMAC fragment layout. Their physical LDS shape is [K, MN].
  const bool has_an_bt_access =
      GetHcuGemmLdsAccessKind(gemm, feeds_a) ==
      HcuGemmLdsAccessKind::kAnBt;
  if (!has_an_bt_access) {
    return std::nullopt;
  }

  size_t src_rank = copy.src_range.size();
  size_t dst_rank = copy.dst_range.size();
  std::optional<int64_t> src_k = StaticExtent(copy.src_range, src_rank - 2);
  std::optional<int64_t> src_n = StaticExtent(copy.src_range, src_rank - 1);
  std::optional<int64_t> dst_k = StaticExtent(copy.dst_range, dst_rank - 2);
  std::optional<int64_t> dst_n = StaticExtent(copy.dst_range, dst_rank - 1);
  if (!src_k || !src_n || !dst_k || !dst_n || *src_k != *dst_k ||
      *src_n != *dst_n || *dst_k <= 0 || *dst_n <= 0 ||
      *dst_k % kDsReadTileK != 0 || *dst_n % kDsReadPanelN != 0 ||
      !IsPowerOfTwo(static_cast<int>(*dst_k)) ||
      !IsPowerOfTwo(static_cast<int>(*dst_n)) || gemm.k_ != *dst_k ||
      GetHcuGemmLdsMnExtent(gemm, feeds_a) != *dst_n ||
      GetHcuGemmOperandDType(gemm, feeds_a) != copy.dst->dtype) {
    return std::nullopt;
  }

  AnBtStrategyParams params;
  params.block_k = static_cast<int>(*dst_k);
  params.block_mn = static_cast<int>(*dst_n);
  params.block_threads = block_threads;
  params.warp_size = TargetHcuGetWarpSize(target);
  params.bank_num = TargetHcuGetLdsBankCount(target);
  params.bank_width_bytes = TargetHcuGetLdsBankWidthBytes(target);
  params.element_bytes = copy.dst->dtype.bits() / 8;
  params.read_bytes_per_lane = kDsReadBytesPerLane;
  params.phase_bytes = kDsReadPhaseBytes;
  params.panel_mn = kDsReadPanelN;
  if (params.warp_size != 64 ||
      params.block_threads % params.warp_size != 0 ||
      copy.dst->dtype.bits() != 16 ||
      params.phase_bytes % params.read_bytes_per_lane != 0 ||
      params.block_mn % params.panel_mn != 0) {
    return std::nullopt;
  }

  params.copy_transaction_bytes = kAsyncCopyBytesPerLane;
  int64_t tile_bytes = static_cast<int64_t>(params.block_k) * params.block_mn *
                       params.element_bytes;
  if (tile_bytes % params.copy_transaction_bytes != 0) {
    return std::nullopt;
  }
  const int64_t total_transactions =
      tile_bytes / params.copy_transaction_bytes;
  params.copy_transactions_per_lane = static_cast<int>(
      (total_transactions + params.block_threads - 1) / params.block_threads);
  params.copy_bytes_per_lane =
      params.copy_transactions_per_lane * params.copy_transaction_bytes;
  if (params.copy_transactions_per_lane <= 0 ||
      params.copy_transaction_bytes % params.element_bytes != 0) {
    return std::nullopt;
  }
  params.copy_elements_per_lane =
      params.copy_transaction_bytes / params.element_bytes;
  params.copy_segments_per_row =
      params.block_mn / params.copy_elements_per_lane;

  Fragment copy_layout;
  if (!SelectWrapStrategy(&params, &copy_layout, target, forced_wrap_count)) {
    return std::nullopt;
  }
  Layout storage_layout = MakeStorageLayout(params, copy_layout);
  auto node = ffi::make_object<HcuGemmAnBtLdsStrategyNode>();
  node->strategy_version = kStrategyVersion;
  node->block_k = params.block_k;
  node->block_mn = params.block_mn;
  node->block_threads = params.block_threads;
  node->warp_size = params.warp_size;
  node->bank_num = params.bank_num;
  node->bank_width_bytes = params.bank_width_bytes;
  node->element_bytes = params.element_bytes;
  node->copy_bytes_per_lane = params.copy_bytes_per_lane;
  node->copy_transaction_bytes = params.copy_transaction_bytes;
  node->copy_transactions_per_lane = params.copy_transactions_per_lane;
  node->read_bytes_per_lane = params.read_bytes_per_lane;
  node->phase_bytes = params.phase_bytes;
  node->panel_mn = params.panel_mn;
  node->wrap_offset = params.wrap_offset;
  node->wrap_idx_mask = params.wrap_idx_mask;
  node->storage_layout = storage_layout;
  node->copy_loop_layout = copy_layout;
  HcuGemmAnBtLdsStrategy strategy(std::move(node));
  ValidateStrategyParameters(params);
  return strategy;
}

Optional<HcuGemmAnBtLdsStrategy>
DeriveHcuGemmAnBtLdsStrategy(const CopyNode &copy, const GemmNode &gemm,
                             bool feeds_a, int block_threads, Target target) {
  return DeriveHcuGemmAnBtLdsStrategyImpl(copy, gemm, feeds_a, block_threads,
                                          target,
                                          /*forced_wrap_count=*/0);
}

Optional<HcuGemmAnBtLdsStrategy>
DeriveHcuGemmAnBtLdsStrategyWith64ByteWrap(
    const CopyNode &copy, const GemmNode &gemm, bool feeds_a, int block_threads,
    Target target, int wrap_count) {
  if (wrap_count != kDsReadWrapCount) {
    return std::nullopt;
  }
  return DeriveHcuGemmAnBtLdsStrategyImpl(
      copy, gemm, feeds_a, block_threads, target, wrap_count);
}

void ValidateHcuGemmAnBtStorageLayout(const Layout &actual,
                                   const HcuGemmAnBtLdsStrategy &strategy) {
  ValidateSameLayout(actual, strategy->storage_layout, strategy->block_k,
                     strategy->block_mn, "storage");
}

void ValidateHcuGemmAnBtCopyLayout(const Fragment &actual,
                                const HcuGemmAnBtLdsStrategy &strategy) {
  ValidateSameLayout(actual, strategy->copy_loop_layout, strategy->block_k,
                     strategy->block_mn, "copy-loop local");
  arith::Analyzer analyzer;
  for (int k = 0; k < strategy->block_k; ++k) {
    for (int n = 0; n < strategy->block_mn; ++n) {
      Array<PrimExpr> logical = {Integer(k), Integer(n)};
      PrimExpr actual_thread =
          actual->ForwardThread(logical, Optional<PrimExpr>());
      PrimExpr expected_thread = strategy->copy_loop_layout->ForwardThread(
          logical, Optional<PrimExpr>());
      if (!is_zero(analyzer.Simplify(actual_thread - expected_thread))) {
        LOG(FATAL)
            << "HCU GEMM AN/BT copy-loop layout thread mapping mismatch at "
                   << "logical coordinate (" << k << ", " << n
                   << "): actual=" << actual_thread
                   << ", expected=" << expected_thread;
      }
    }
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  HcuGemmAnBtLdsStrategyNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
