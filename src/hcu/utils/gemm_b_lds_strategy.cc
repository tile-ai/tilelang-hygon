/*!
 * \file gemm_b_lds_strategy.cc
 * \brief Compiler-derived LDS strategy for HCU GEMM B ds-read copies.
 */

#include "gemm_b_lds_strategy.h"

#include "hcu/target_utils.h"
#include "op/utils.h"

#include <tvm/arith/analyzer.h>

#include <array>
#include <optional>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

constexpr int kStrategyVersion = 1;
constexpr int kDsReadTileK = 16;
constexpr int kDsReadPanelN = 32;
constexpr int kDsReadPhaseBytes = 128;
constexpr int kDsReadPhaseCount = 8;
constexpr int kDsReadRequestsPerPhase = 8;
constexpr int kAsyncCopyBytesPerLane = 16;
constexpr int kDsReadBytesPerLane = 16;

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

struct GemmBStrategyParams {
  int block_k{0};
  int block_n{0};
  int block_threads{0};
  int warp_size{0};
  int bank_num{0};
  int bank_width_bytes{0};
  int element_bytes{0};
  int copy_bytes_per_lane{0};
  int read_bytes_per_lane{0};
  int phase_bytes{0};
  int panel_n{0};
  int copy_elements_per_lane{0};
  int copy_segments_per_row{0};
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

int64_t RequireConst(const PrimExpr &expr, arith::Analyzer *analyzer,
                     const char *kind, int k, int n) {
  PrimExpr simplified = analyzer->Simplify(expr);
  const int64_t *value = as_const_int(simplified);
  ICHECK(value) << "HCU GEMM B " << kind
                << " mapping must be constant at logical coordinate (" << k
                << ", " << n << "), got " << simplified;
  return *value;
}

Fragment MakeCopyLoopLayout(const GemmBStrategyParams &params) {
  PrimExpr k = InputPlaceholder(0);
  PrimExpr n = InputPlaceholder(1);
  PrimExpr segment = floordiv(n, Integer(params.copy_elements_per_lane));
  PrimExpr copy_warp = floordiv(k, Integer(4)) * 2 + floormod(k, Integer(2));
  PrimExpr copy_lane = floordiv(floormod(k, Integer(4)), Integer(2)) *
                           params.copy_segments_per_row +
                       segment;
  PrimExpr thread = copy_warp * params.warp_size + copy_lane;
  PrimExpr local = floormod(n, Integer(params.copy_elements_per_lane));
  return Fragment({Integer(params.block_k), Integer(params.block_n)}, {local},
                  thread, Integer(1), std::nullopt);
}

Layout MakeStorageLayout(const GemmBStrategyParams &params,
                         const Fragment &copy_layout) {
  PrimExpr k = InputPlaceholder(0);
  PrimExpr n = InputPlaceholder(1);
  Array<PrimExpr> logical = {k, n};
  PrimExpr thread = copy_layout->ForwardThread(logical, std::nullopt);
  Array<PrimExpr> local = copy_layout->Forward(logical);
  ICHECK_EQ(local.size(), 1U);

  PrimExpr copy_warp = floordiv(thread, Integer(params.warp_size));
  PrimExpr copy_lane = floormod(thread, Integer(params.warp_size));
  PrimExpr wrap = floormod(copy_warp, Integer(params.wrap_idx_mask + 1)) *
                  params.wrap_offset;
  PrimExpr physical_lane =
      floormod(copy_lane + wrap, Integer(params.warp_size));
  PrimExpr physical_offset =
      (copy_warp * params.warp_size + physical_lane) *
          params.copy_elements_per_lane +
      local[0];
  Array<PrimExpr> physical = {
      floordiv(physical_offset, Integer(params.block_n)),
      floormod(physical_offset, Integer(params.block_n))};
  Array<PrimExpr> input_shape = {Integer(params.block_k),
                                 Integer(params.block_n)};
  return Layout(input_shape, physical);
}

void ValidateSameLayout(const Layout &actual, const Layout &expected, int block_k,
                        int block_n, const char *kind) {
  ICHECK(actual.defined()) << kind << " layout is undefined";
  ICHECK(expected.defined()) << "Expected " << kind << " layout is undefined";
  ICHECK_EQ(actual->InputDim(), expected->InputDim())
      << kind << " layout input rank mismatch: actual=" << actual->DebugOutput()
      << ", expected=" << expected->DebugOutput();

  arith::Analyzer analyzer;
  for (int k = 0; k < block_k; ++k) {
    for (int n = 0; n < block_n; ++n) {
      Array<PrimExpr> logical = {Integer(k), Integer(n)};
      Array<PrimExpr> actual_index = actual->Forward(logical);
      Array<PrimExpr> expected_index = expected->Forward(logical);
      ICHECK_EQ(actual_index.size(), expected_index.size());
      for (size_t dim = 0; dim < actual_index.size(); ++dim) {
        if (!is_zero(
                analyzer.Simplify(actual_index[dim] - expected_index[dim]))) {
          LOG(FATAL) << "HCU GEMM B " << kind
                     << " layout mismatch at logical coordinate (" << k << ", "
                     << n << "), dimension " << dim
                     << ": actual=" << actual_index[dim]
                     << ", expected=" << expected_index[dim];
        }
      }
    }
  }
}

void ValidateCopyStorageConsistency(const HcuGemmBLdsStrategy &strategy) {
  const int copy_elements =
      strategy->copy_bytes_per_lane / strategy->element_bytes;
  arith::Analyzer analyzer;
  for (int k = 0; k < strategy->block_k; ++k) {
    for (int n = 0; n < strategy->block_n; ++n) {
      Array<PrimExpr> logical = {Integer(k), Integer(n)};
      int64_t thread = RequireConst(
          strategy->copy_loop_layout->ForwardThread(logical, std::nullopt),
          &analyzer, "copy thread", k, n);
      Array<PrimExpr> local = strategy->copy_loop_layout->Forward(logical);
      ICHECK_EQ(local.size(), 1U)
          << "HCU GEMM B copy layout must have one local index";
      int64_t local_index =
          RequireConst(local[0], &analyzer, "copy local", k, n);
      ICHECK_GE(thread, 0);
      ICHECK_LT(thread, strategy->block_threads);
      ICHECK_GE(local_index, 0);
      ICHECK_LT(local_index, copy_elements);

      int64_t copy_warp = thread / strategy->warp_size;
      int64_t copy_lane = thread % strategy->warp_size;
      int64_t wrap = (copy_warp & strategy->wrap_idx_mask) *
                     strategy->wrap_offset;
      int64_t physical_lane = (copy_lane + wrap) % strategy->warp_size;
      int64_t expected_offset =
          (copy_warp * strategy->warp_size + physical_lane) * copy_elements +
          local_index;

      Array<PrimExpr> physical = strategy->storage_layout->Forward(logical);
      ICHECK_EQ(physical.size(), 2U);
      int64_t actual_k =
          RequireConst(physical[0], &analyzer, "storage row", k, n);
      int64_t actual_n =
          RequireConst(physical[1], &analyzer, "storage column", k, n);
      int64_t actual_offset = actual_k * strategy->block_n + actual_n;
      ICHECK_EQ(actual_offset, expected_offset)
          << "HCU GEMM B copy/layout/wrap mismatch at logical coordinate (" << k
          << ", " << n << "): actual physical offset=" << actual_offset
          << ", expected=" << expected_offset;
    }
  }
}

void ValidateDsReadBankConflicts(const HcuGemmBLdsStrategy &strategy) {
  ICHECK_EQ(strategy->read_bytes_per_lane % strategy->bank_width_bytes, 0);
  const int banks_per_read =
      strategy->read_bytes_per_lane / strategy->bank_width_bytes;
  arith::Analyzer analyzer;
  for (int panel = 0; panel < strategy->block_n / strategy->panel_n; ++panel) {
    for (int phase = 0; phase < kDsReadPhaseCount; ++phase) {
      std::vector<int> bank_uses(strategy->bank_num, 0);
      for (int lane : kDsReadPhaseLanes[phase]) {
        int k = lane >> 2;
        int n = panel * strategy->panel_n + (lane & 3) * 8;
        Array<PrimExpr> physical =
            strategy->storage_layout->Forward({Integer(k), Integer(n)});
        ICHECK_EQ(physical.size(), 2U);
        int64_t physical_k =
            RequireConst(physical[0], &analyzer, "ds-read row", k, n);
        int64_t physical_n =
            RequireConst(physical[1], &analyzer, "ds-read column", k, n);
        int64_t byte_offset =
            (physical_k * strategy->block_n + physical_n) *
            strategy->element_bytes;
        ICHECK_EQ(byte_offset % strategy->bank_width_bytes, 0)
            << "HCU GEMM B ds-read address must be bank aligned";
        int start_bank = static_cast<int>(
            (byte_offset / strategy->bank_width_bytes) % strategy->bank_num);
        for (int bank = 0; bank < banks_per_read; ++bank) {
          int current = (start_bank + bank) % strategy->bank_num;
          ICHECK_EQ(bank_uses[current]++, 0)
              << "HCU GEMM B ds_read_m32x16_b16 bank conflict at panel "
              << panel << ", phase " << phase << ", lane T" << lane
              << ", bank " << current;
        }
      }
    }
  }
}

} // namespace

void HcuGemmBLdsStrategyNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<HcuGemmBLdsStrategyNode>()
      .def_ro("strategy_version", &HcuGemmBLdsStrategyNode::strategy_version)
      .def_ro("block_k", &HcuGemmBLdsStrategyNode::block_k)
      .def_ro("block_n", &HcuGemmBLdsStrategyNode::block_n)
      .def_ro("block_threads", &HcuGemmBLdsStrategyNode::block_threads)
      .def_ro("warp_size", &HcuGemmBLdsStrategyNode::warp_size)
      .def_ro("bank_num", &HcuGemmBLdsStrategyNode::bank_num)
      .def_ro("bank_width_bytes", &HcuGemmBLdsStrategyNode::bank_width_bytes)
      .def_ro("element_bytes", &HcuGemmBLdsStrategyNode::element_bytes)
      .def_ro("copy_bytes_per_lane",
              &HcuGemmBLdsStrategyNode::copy_bytes_per_lane)
      .def_ro("read_bytes_per_lane",
              &HcuGemmBLdsStrategyNode::read_bytes_per_lane)
      .def_ro("phase_bytes", &HcuGemmBLdsStrategyNode::phase_bytes)
      .def_ro("panel_n", &HcuGemmBLdsStrategyNode::panel_n)
      .def_ro("wrap_offset", &HcuGemmBLdsStrategyNode::wrap_offset)
      .def_ro("wrap_idx_mask", &HcuGemmBLdsStrategyNode::wrap_idx_mask)
      .def_ro("storage_layout", &HcuGemmBLdsStrategyNode::storage_layout)
      .def_ro("copy_loop_layout", &HcuGemmBLdsStrategyNode::copy_loop_layout);
}

Optional<HcuGemmBLdsStrategy>
DeriveHcuGemmBLdsStrategy(const CopyNode &copy, const GemmNode &gemm,
                          int block_threads, Target target) {
  if (!TargetIsHCU(target) || !TargetHcuHasAsyncCopy(target) ||
      block_threads <= 0 || !IsGlobalBuffer(copy.src) ||
      !IsSharedBuffer(copy.dst) || copy.src->dtype != copy.dst->dtype ||
      !(copy.dst->dtype.is_float16() || copy.dst->dtype.is_bfloat16()) ||
      copy.src_range.size() < 2 || copy.dst_range.size() < 2 || gemm.transB_ ||
      gemm.kPack_ != 1) {
    return std::nullopt;
  }

  size_t src_rank = copy.src_range.size();
  size_t dst_rank = copy.dst_range.size();
  std::optional<int64_t> src_k = StaticExtent(copy.src_range, src_rank - 2);
  std::optional<int64_t> src_n = StaticExtent(copy.src_range, src_rank - 1);
  std::optional<int64_t> dst_k = StaticExtent(copy.dst_range, dst_rank - 2);
  std::optional<int64_t> dst_n = StaticExtent(copy.dst_range, dst_rank - 1);
  if (!src_k || !src_n || !dst_k || !dst_n || *src_k != *dst_k ||
      *src_n != *dst_n || *dst_k != kDsReadTileK || *dst_n != 256 ||
      gemm.k_ != *dst_k || gemm.n_ != *dst_n) {
    return std::nullopt;
  }

  GemmBStrategyParams params;
  params.block_k = static_cast<int>(*dst_k);
  params.block_n = static_cast<int>(*dst_n);
  params.block_threads = block_threads;
  params.warp_size = TargetHcuGetWarpSize(target);
  params.bank_num = TargetHcuGetLdsBankCount(target);
  params.bank_width_bytes = TargetHcuGetLdsBankWidthBytes(target);
  params.element_bytes = copy.dst->dtype.bits() / 8;
  params.read_bytes_per_lane = kDsReadBytesPerLane;
  params.phase_bytes = kDsReadPhaseBytes;
  params.panel_n = kDsReadPanelN;
  if (params.warp_size != 64 || params.block_threads != 512 ||
      copy.dst->dtype.bits() != 16 ||
      params.phase_bytes % params.read_bytes_per_lane != 0 ||
      params.block_n % params.panel_n != 0) {
    return std::nullopt;
  }

  int64_t tile_bytes = static_cast<int64_t>(params.block_k) * params.block_n *
                       params.element_bytes;
  if (tile_bytes % params.block_threads != 0) {
    return std::nullopt;
  }
  params.copy_bytes_per_lane =
      static_cast<int>(tile_bytes / params.block_threads);
  if (params.copy_bytes_per_lane != kAsyncCopyBytesPerLane ||
      params.copy_bytes_per_lane % params.element_bytes != 0) {
    return std::nullopt;
  }
  params.copy_elements_per_lane =
      params.copy_bytes_per_lane / params.element_bytes;
  params.copy_segments_per_row =
      params.block_n / params.copy_elements_per_lane;

  // Each phase has two four-request row-parity groups.  Moving the second
  // group by four 16-byte copy segments separates their bank footprints.
  params.wrap_offset =
      params.phase_bytes / params.read_bytes_per_lane / 2;
  params.wrap_idx_mask = 1;
  const int wrap_field_bits = TargetHcuGetLdsWrapFieldBits(target);
  if (wrap_field_bits <= 0 ||
      params.wrap_offset * params.wrap_idx_mask >= (1 << wrap_field_bits)) {
    return std::nullopt;
  }

  Fragment copy_layout = MakeCopyLoopLayout(params);
  Layout storage_layout = MakeStorageLayout(params, copy_layout);
  auto node = ffi::make_object<HcuGemmBLdsStrategyNode>();
  node->strategy_version = kStrategyVersion;
  node->block_k = params.block_k;
  node->block_n = params.block_n;
  node->block_threads = params.block_threads;
  node->warp_size = params.warp_size;
  node->bank_num = params.bank_num;
  node->bank_width_bytes = params.bank_width_bytes;
  node->element_bytes = params.element_bytes;
  node->copy_bytes_per_lane = params.copy_bytes_per_lane;
  node->read_bytes_per_lane = params.read_bytes_per_lane;
  node->phase_bytes = params.phase_bytes;
  node->panel_n = params.panel_n;
  node->wrap_offset = params.wrap_offset;
  node->wrap_idx_mask = params.wrap_idx_mask;
  node->storage_layout = storage_layout;
  node->copy_loop_layout = copy_layout;
  HcuGemmBLdsStrategy strategy(std::move(node));
  ValidateCopyStorageConsistency(strategy);
  ValidateDsReadBankConflicts(strategy);
  return strategy;
}

void ValidateHcuGemmBStorageLayout(const Layout &actual,
                                   const HcuGemmBLdsStrategy &strategy) {
  ValidateSameLayout(actual, strategy->storage_layout, strategy->block_k,
                     strategy->block_n, "storage");
}

void ValidateHcuGemmBCopyLayout(const Fragment &actual,
                                const HcuGemmBLdsStrategy &strategy) {
  ValidateSameLayout(actual, strategy->copy_loop_layout, strategy->block_k,
                     strategy->block_n, "copy-loop local");
  arith::Analyzer analyzer;
  for (int k = 0; k < strategy->block_k; ++k) {
    for (int n = 0; n < strategy->block_n; ++n) {
      Array<PrimExpr> logical = {Integer(k), Integer(n)};
      PrimExpr actual_thread =
          actual->ForwardThread(logical, Optional<PrimExpr>());
      PrimExpr expected_thread = strategy->copy_loop_layout->ForwardThread(
          logical, Optional<PrimExpr>());
      if (!is_zero(analyzer.Simplify(actual_thread - expected_thread))) {
        LOG(FATAL) << "HCU GEMM B copy-loop layout thread mapping mismatch at "
                   << "logical coordinate (" << k << ", " << n
                   << "): actual=" << actual_thread
                   << ", expected=" << expected_thread;
      }
    }
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  HcuGemmBLdsStrategyNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
