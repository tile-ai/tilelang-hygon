// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file gemm_at_bn_lds_strategy.cc
 * \brief Compiler-derived LDS strategy for HCU GEMM AT/BN accesses.
 */

#include "gemm_at_bn_lds_strategy.h"

#include "hcu/op/gemm_partition.h"
#include "hcu/target_utils.h"
#include "hcu/utils/gemm_lds_access.h"
#include "hcu/utils/gemm_lds_strategy_utils.h"
#include "op/utils.h"
#include "support/check.h"

#include <tvm/arith/analyzer.h>

#include <algorithm>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

constexpr int kStrategyVersion = 9;
// These are instruction semantics, not kernel tile parameters.
constexpr int kLdsIssueBytes = 128;
constexpr int kDsReadB128BytesPerLane = 16;
constexpr int kDsReadB128ConflictGroupLanes = 4;
constexpr int kDsReadB128WrapStepBytes = 32;

int GetAtBnIssueGroupLanes(int read_bytes_per_lane) {
  if (read_bytes_per_lane == kDsReadB128BytesPerLane) {
    return kDsReadB128ConflictGroupLanes;
  }
  if (read_bytes_per_lane <= 0 || kLdsIssueBytes % read_bytes_per_lane != 0) {
    return 0;
  }
  return kLdsIssueBytes / read_bytes_per_lane;
}

int GetAtBnRequiredWrapCount(int row_period, int issue_group_lanes) {
  if (row_period <= 0 || issue_group_lanes <= 0) {
    return 0;
  }
  if (row_period >= issue_group_lanes) {
    return 1;
  }
  if (issue_group_lanes % row_period != 0) {
    return 0;
  }
  return issue_group_lanes / row_period;
}

struct AtBnStrategyParams {
  int block_mn{0};
  int block_k{0};
  int block_threads{0};
  int warp_size{0};
  int warp_mn_count{0};
  int bank_num{0};
  int bank_width_bytes{0};
  int element_bytes{0};
  int copy_bytes_per_lane{0};
  int copy_transaction_bytes{0};
  int copy_transactions_per_lane{0};
  int read_bytes_per_lane{0};
  int issue_group_lanes{0};
  int row_bank_stride{0};
  int row_period{0};
  int rows_per_copy_wave{0};
  int copy_elements_per_lane{0};
  int copy_segments_per_row{0};
  int rows_per_copy_warp_transaction{0};
  int rows_per_copy_transaction{0};
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

bool IsPowerOfTwo(int value) { return value > 0 && (value & (value - 1)) == 0; }

int SelectWrapOffset(const AtBnStrategyParams &params,
                     const HcuGemmLdsCopyGeometry &geometry, int wrap_count) {
  if (geometry.max_wrap_offset_dwords <= 0) {
    return 0;
  }
  const int banks_per_read =
      params.read_bytes_per_lane / params.bank_width_bytes;
  std::vector<bool> base_banks(params.bank_num, false);
  for (int row = 0; row < params.row_period; ++row) {
    const int start = (row * params.row_bank_stride) % params.bank_num;
    for (int bank = 0; bank < banks_per_read; ++bank) {
      base_banks[(start + bank) % params.bank_num] = true;
    }
  }
  for (int offset_dwords = geometry.wrap_granularity_dwords;
       offset_dwords * params.wrap_idx_mask <= geometry.max_wrap_offset_dwords;
       offset_dwords += geometry.wrap_granularity_dwords) {
    const int wrap_step_bytes = offset_dwords * 4;
    // Keep the physical wrap shift aligned to one LDS read pack, and enforce
    // the target-specific wrap-field granularity and range.
    if (wrap_step_bytes % params.read_bytes_per_lane != 0 ||
        !IsLegalHcuGemmLdsWrap(geometry, wrap_step_bytes, wrap_count)) {
      continue;
    }
    const int bank_shift = offset_dwords % params.bank_num;
    bool overlaps = false;
    for (int bank = 0; bank < params.bank_num; ++bank) {
      if (base_banks[bank] &&
          base_banks[(bank + bank_shift) % params.bank_num]) {
        overlaps = true;
        break;
      }
    }
    if (!overlaps) {
      return offset_dwords;
    }
  }
  return 0;
}

PrimExpr AtBnRowPermute(const PrimExpr &row, const AtBnStrategyParams &params) {
  if (params.wrap_idx_mask == 0) {
    return row;
  }
  const int copy_warp_count = params.block_threads / params.warp_size;
  const int groups_per_copy_warp =
      params.rows_per_copy_warp_transaction / params.row_period;
  PrimExpr row_group = floordiv(row, Integer(params.row_period));
  PrimExpr inner_row = floormod(row, Integer(params.row_period));
  PrimExpr copy_warp = floormod(row_group, Integer(copy_warp_count));
  PrimExpr group_round = floordiv(row_group, Integer(copy_warp_count));
  // Distribute repeated-bank row groups across copy warps first. Additional
  // groups fill the same warp and then advance to the next transaction.
  return floordiv(group_round, Integer(groups_per_copy_warp)) *
             params.rows_per_copy_transaction +
         copy_warp * params.rows_per_copy_warp_transaction +
         floormod(group_round, Integer(groups_per_copy_warp)) *
             params.row_period +
         inner_row;
}

Layout MakeStorageLayout(const AtBnStrategyParams &params) {
  PrimExpr row = InputPlaceholder(0);
  PrimExpr col = InputPlaceholder(1);
  PrimExpr mapped_row = AtBnRowPermute(row, params);
  PrimExpr segment = mapped_row * params.copy_segments_per_row +
                     floordiv(col, Integer(params.copy_elements_per_lane));
  PrimExpr transaction = floordiv(segment, Integer(params.block_threads));
  PrimExpr thread = floormod(segment, Integer(params.block_threads));
  PrimExpr copy_warp = floordiv(thread, Integer(params.warp_size));
  PrimExpr lane = floormod(thread, Integer(params.warp_size));
  // Reproduce the hardware wrap on each issued copy transaction so the
  // annotation and the physical LDS writes share the same mapping.
  ICHECK_EQ(4 % params.element_bytes, 0);
  const int elements_per_dword = 4 / params.element_bytes;
  PrimExpr wrap_offset_cur_dwords =
      floormod(copy_warp, Integer(params.wrap_idx_mask + 1)) *
      params.wrap_offset;
  PrimExpr wave_element = lane * params.copy_elements_per_lane +
                          floormod(col, Integer(params.copy_elements_per_lane));
  PrimExpr physical_wave_element =
      floormod(wave_element + wrap_offset_cur_dwords * elements_per_dword,
               Integer(params.warp_size * params.copy_elements_per_lane));
  PrimExpr physical_linear =
      (transaction * params.block_threads + copy_warp * params.warp_size) *
          params.copy_elements_per_lane +
      physical_wave_element;
  PrimExpr physical_row = floordiv(physical_linear, Integer(params.block_k));
  PrimExpr physical_col = floormod(physical_linear, Integer(params.block_k));
  Array<PrimExpr> input_shape = {Integer(params.block_mn),
                                 Integer(params.block_k)};
  Array<PrimExpr> forward_index = {physical_row, physical_col};
  return Layout(input_shape, forward_index);
}

Fragment MakeCopyLoopLayout(const AtBnStrategyParams &params) {
  PrimExpr row = InputPlaceholder(0);
  PrimExpr col = InputPlaceholder(1);
  PrimExpr mapped_row = AtBnRowPermute(row, params);
  PrimExpr segment = mapped_row * params.copy_segments_per_row +
                     floordiv(col, Integer(params.copy_elements_per_lane));
  PrimExpr thread = floormod(segment, Integer(params.block_threads));
  PrimExpr transaction = floordiv(segment, Integer(params.block_threads));
  PrimExpr intra = floormod(col, Integer(params.copy_elements_per_lane));
  if (params.copy_transactions_per_lane == 1) {
    return Fragment({Integer(params.block_mn), Integer(params.block_k)},
                    {intra}, thread, Integer(1), std::nullopt);
  }
  return Fragment({Integer(params.block_mn), Integer(params.block_k)},
                  {transaction, intra}, thread, Integer(1), std::nullopt);
}

void ValidateSameLayout(const Layout &actual, const Layout &expected,
                        int block_mn, int block_k, const char *kind) {
  ICHECK(actual.defined()) << kind << " layout is undefined";
  ICHECK(expected.defined()) << "Expected " << kind << " layout is undefined";
  ICHECK_EQ(actual->InputDim(), expected->InputDim())
      << kind << " layout input rank mismatch: actual=" << actual->DebugOutput()
      << ", expected=" << expected->DebugOutput();

  arith::Analyzer analyzer;
  for (int row = 0; row < block_mn; ++row) {
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
          LOG(FATAL) << "HCU GEMM AT/BN " << kind
                     << " layout does not match the compiler-derived strategy"
                     << " at logical coordinate (" << row << ", " << col
                     << "), output dim " << dim << ": actual=" << actual_index
                     << ", expected=" << expected_index
                     << "\nactual layout: " << actual->DebugOutput()
                     << "\nexpected layout: " << expected->DebugOutput();
        }
      }
    }
  }
}

} // namespace

void HcuGemmAtBnLdsStrategyNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<HcuGemmAtBnLdsStrategyNode>()
      .def_ro("strategy_version", &HcuGemmAtBnLdsStrategyNode::strategy_version)
      .def_ro("block_mn", &HcuGemmAtBnLdsStrategyNode::block_mn)
      .def_ro("block_k", &HcuGemmAtBnLdsStrategyNode::block_k)
      .def_ro("block_threads", &HcuGemmAtBnLdsStrategyNode::block_threads)
      .def_ro("warp_size", &HcuGemmAtBnLdsStrategyNode::warp_size)
      .def_ro("warp_mn_count", &HcuGemmAtBnLdsStrategyNode::warp_mn_count)
      .def_ro("bank_num", &HcuGemmAtBnLdsStrategyNode::bank_num)
      .def_ro("bank_width_bytes", &HcuGemmAtBnLdsStrategyNode::bank_width_bytes)
      .def_ro("element_bytes", &HcuGemmAtBnLdsStrategyNode::element_bytes)
      .def_ro("copy_bytes_per_lane",
              &HcuGemmAtBnLdsStrategyNode::copy_bytes_per_lane)
      .def_ro("copy_transaction_bytes",
              &HcuGemmAtBnLdsStrategyNode::copy_transaction_bytes)
      .def_ro("copy_transactions_per_lane",
              &HcuGemmAtBnLdsStrategyNode::copy_transactions_per_lane)
      .def_ro("read_bytes_per_lane",
              &HcuGemmAtBnLdsStrategyNode::read_bytes_per_lane)
      .def_ro("row_period", &HcuGemmAtBnLdsStrategyNode::row_period)
      .def_ro("row_bank_stride", &HcuGemmAtBnLdsStrategyNode::row_bank_stride)
      .def_ro("rows_per_copy_wave",
              &HcuGemmAtBnLdsStrategyNode::rows_per_copy_wave)
      .def_ro("wrap_offset", &HcuGemmAtBnLdsStrategyNode::wrap_offset)
      .def_ro("wrap_idx_mask", &HcuGemmAtBnLdsStrategyNode::wrap_idx_mask)
      .def_ro("storage_layout", &HcuGemmAtBnLdsStrategyNode::storage_layout)
      .def_ro("copy_loop_layout",
              &HcuGemmAtBnLdsStrategyNode::copy_loop_layout);
}

Optional<HcuGemmAtBnLdsStrategy>
DeriveHcuGemmAtBnLdsStrategy(const CopyNode &copy, const GemmNode &gemm,
                             bool feeds_a, int block_threads, Target target) {
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
  const int block_mn = static_cast<int>(*dst_m);
  const int block_k = static_cast<int>(*dst_k);
  const int element_bits = copy.dst->dtype.bits();
  // AT (A, transA=false) and BN (B, transB=true) share the same
  // spatial-leading MMAC load layout. Their physical LDS shape is [MN, K].
  const bool has_at_bn_access =
      GetHcuGemmLdsAccessKind(gemm, feeds_a) == HcuGemmLdsAccessKind::kAtBn;
  const int gemm_mn = GetHcuGemmLdsMnExtent(gemm, feeds_a);
  const DataType operand_dtype = GetHcuGemmOperandDType(gemm, feeds_a);
  if (!has_at_bn_access || gemm_mn <= 0 || gemm_mn > block_mn ||
      block_mn % gemm_mn != 0 || gemm.k_ != block_k || gemm.kPack_ <= 0 ||
      operand_dtype != copy.dst->dtype) {
    return std::nullopt;
  }

  const hcu::HcuMmacModeInfo mmac_mode = hcu::ResolveHcuMmacMode(
      gemm.a_->dtype, gemm.b_->dtype, /*a_is_fragment=*/false,
      /*b_is_fragment=*/false, /*is_blockscaled=*/false, block_k,
      ScaleLdsFormat::kIdentity, ScaleLdsFormat::kIdentity, target);
  if (element_bits <= 0 || element_bits % 8 != 0 ||
      mmac_mode.element_bits != element_bits || mmac_mode.mmac_k <= 0 ||
      block_k % mmac_mode.mmac_k != 0) {
    return std::nullopt;
  }

  GemmWarpPolicy policy(gemm.policy_->policy_type);
  const hcu::HcuMnPerWarp mn_per_warp = hcu::ComputeWarpPartitionHCU(
      *policy.get(), gemm.m_, gemm.n_, gemm.k_, gemm.kPack_, element_bits,
      block_threads, target,
      /*A_from_mls=*/false,
      /*B_from_mls=*/false,
      /*A_mls_trans=*/false,
      /*B_mls_trans=*/false);
  const int warp_size = TargetHcuGetWarpSize(target);
  const int warp_mn_count = feeds_a ? policy->m_warp : policy->n_warp;
  const int operand_mn_atom =
      feeds_a ? mn_per_warp.m_per_warp : mn_per_warp.n_per_warp;
  if (block_threads % warp_size != 0 || warp_mn_count <= 0 ||
      operand_mn_atom <= 0 || !IsPowerOfTwo(warp_mn_count) ||
      policy->k_warp != 1 ||
      policy->m_warp * policy->n_warp * policy->k_warp !=
          block_threads / warp_size) {
    return std::nullopt;
  }

  AtBnStrategyParams params;
  params.block_mn = block_mn;
  params.block_k = block_k;
  params.block_threads = block_threads;
  params.warp_size = warp_size;
  params.warp_mn_count = warp_mn_count;
  params.element_bytes = element_bits / 8;

  // Derive the LDS-to-VGPR conflict requirement independently from the
  // Global-to-LDS copy transaction width.
  const int row_bytes = block_k * params.element_bytes;
  const int64_t packed_mmac_elements =
      static_cast<int64_t>(operand_mn_atom) * mmac_mode.mmac_k * gemm.kPack_;
  if (packed_mmac_elements % params.warp_size != 0) {
    return std::nullopt;
  }
  params.read_bytes_per_lane = static_cast<int>(
      packed_mmac_elements / params.warp_size * params.element_bytes);
  if (params.read_bytes_per_lane % params.element_bytes != 0 ||
      row_bytes % params.read_bytes_per_lane != 0) {
    return std::nullopt;
  }
  params.issue_group_lanes = GetAtBnIssueGroupLanes(params.read_bytes_per_lane);
  if (params.issue_group_lanes <= 0 ||
      params.warp_size % params.issue_group_lanes != 0) {
    return std::nullopt;
  }
  params.bank_num = TargetHcuGetLdsBankCount(target);
  params.bank_width_bytes = TargetHcuGetLdsBankWidthBytes(target);
  if (params.bank_num <= 0 || params.bank_width_bytes <= 0 ||
      row_bytes % params.bank_width_bytes != 0) {
    return std::nullopt;
  }
  params.row_bank_stride = row_bytes / params.bank_width_bytes;
  params.row_period =
      params.bank_num / std::gcd(params.bank_num, params.row_bank_stride);
  const int required_wrap_count =
      GetAtBnRequiredWrapCount(params.row_period, params.issue_group_lanes);
  if (required_wrap_count <= 0) {
    return std::nullopt;
  }

  // Select the widest transaction that remains contiguous within one row and
  // divides the bytes assigned to each copy thread.
  const int64_t tile_bytes =
      static_cast<int64_t>(block_mn) * block_k * params.element_bytes;
  if (tile_bytes % block_threads != 0) {
    return std::nullopt;
  }
  params.copy_bytes_per_lane = static_cast<int>(tile_bytes / block_threads);
  params.copy_transaction_bytes = SelectHcuGemmLdsCopyTransactionBytes(
      row_bytes, params.copy_bytes_per_lane, params.element_bytes);
  if (params.copy_transaction_bytes < 4 ||
      params.copy_bytes_per_lane < params.copy_transaction_bytes ||
      params.copy_bytes_per_lane % params.copy_transaction_bytes != 0 ||
      params.copy_bytes_per_lane % params.element_bytes != 0 ||
      row_bytes % params.copy_transaction_bytes != 0 ||
      (std::max(params.copy_transaction_bytes, params.read_bytes_per_lane) %
       std::min(params.copy_transaction_bytes, params.read_bytes_per_lane)) !=
          0) {
    return std::nullopt;
  }
  params.copy_transactions_per_lane =
      params.copy_bytes_per_lane / params.copy_transaction_bytes;

  std::optional<HcuGemmLdsCopyGeometry> geometry = DeriveHcuGemmLdsCopyGeometry(
      row_bytes, params.copy_transaction_bytes, params.block_threads, target);
  if (!geometry.has_value() || geometry->warp_size != params.warp_size ||
      geometry->bank_count != params.bank_num ||
      geometry->bank_width_bytes != params.bank_width_bytes ||
      geometry->waves_per_group != 1) {
    return std::nullopt;
  }
  params.copy_elements_per_lane =
      params.copy_transaction_bytes / params.element_bytes;
  params.copy_segments_per_row = row_bytes / params.copy_transaction_bytes;
  if (params.copy_segments_per_row <= 0 ||
      params.warp_size % params.copy_segments_per_row != 0 ||
      params.block_threads % params.copy_segments_per_row != 0) {
    return std::nullopt;
  }
  params.rows_per_copy_warp_transaction =
      params.warp_size / params.copy_segments_per_row;
  params.rows_per_copy_transaction =
      params.block_threads / params.copy_segments_per_row;
  const int64_t copy_segment_count =
      static_cast<int64_t>(block_mn) * params.copy_segments_per_row;
  if (copy_segment_count % block_threads != 0 ||
      copy_segment_count / block_threads != params.copy_transactions_per_lane) {
    return std::nullopt;
  }
  params.rows_per_copy_wave = geometry->rows_per_group;
  if (block_mn % params.rows_per_copy_wave != 0 ||
      params.rows_per_copy_wave % params.row_period != 0 ||
      params.rows_per_copy_warp_transaction % params.row_period != 0 ||
      block_mn % params.warp_mn_count != 0 ||
      operand_mn_atom % params.row_period != 0) {
    return std::nullopt;
  }
  if (gemm_mn % params.warp_mn_count != 0) {
    return std::nullopt;
  }
  const bool uses_ds_read_b128 =
      params.read_bytes_per_lane == kDsReadB128BytesPerLane;
  const int wrap_count =
      uses_ds_read_b128 ? required_wrap_count
                        : std::max(params.warp_mn_count, required_wrap_count);
  const int copy_warp_count = params.block_threads / params.warp_size;
  if (!IsPowerOfTwo(wrap_count) || wrap_count > copy_warp_count) {
    return std::nullopt;
  }
  params.wrap_idx_mask = wrap_count - 1;
  if (wrap_count == 1) {
    params.wrap_offset = 0;
  } else if (uses_ds_read_b128) {
    params.wrap_offset =
        GetHcuGemmLdsWrapOffsetDwords(kDsReadB128WrapStepBytes);
  } else {
    // Select the smallest wrap that separates adjacent repeated-bank groups.
    params.wrap_offset = SelectWrapOffset(params, *geometry, wrap_count);
  }
  const int wrap_step_bytes = params.wrap_offset * 4;
  if (!IsLegalHcuGemmLdsWrap(*geometry, wrap_step_bytes, wrap_count)) {
    return std::nullopt;
  }
  auto node = ffi::make_object<HcuGemmAtBnLdsStrategyNode>();
  node->strategy_version = kStrategyVersion;
  node->block_mn = params.block_mn;
  node->block_k = params.block_k;
  node->block_threads = params.block_threads;
  node->warp_size = params.warp_size;
  node->warp_mn_count = params.warp_mn_count;
  node->bank_num = params.bank_num;
  node->bank_width_bytes = params.bank_width_bytes;
  node->element_bytes = params.element_bytes;
  node->copy_bytes_per_lane = params.copy_bytes_per_lane;
  node->copy_transaction_bytes = params.copy_transaction_bytes;
  node->copy_transactions_per_lane = params.copy_transactions_per_lane;
  node->read_bytes_per_lane = params.read_bytes_per_lane;
  node->row_bank_stride = params.row_bank_stride;
  node->row_period = params.row_period;
  node->rows_per_copy_wave = params.rows_per_copy_wave;
  node->wrap_offset = params.wrap_offset;
  node->wrap_idx_mask = params.wrap_idx_mask;
  // Column segment shift/carry completes the conflict-free physical layout.
  node->storage_layout = MakeStorageLayout(params);
  node->copy_loop_layout = MakeCopyLoopLayout(params);
  return HcuGemmAtBnLdsStrategy(std::move(node));
}

void ValidateHcuGemmAtBnStorageLayout(const Layout &actual,
                                      const HcuGemmAtBnLdsStrategy &strategy) {
  ValidateSameLayout(actual, strategy->storage_layout, strategy->block_mn,
                     strategy->block_k, "storage");
}

void ValidateHcuGemmAtBnCopyLayout(const Fragment &actual,
                                   const HcuGemmAtBnLdsStrategy &strategy) {
  ValidateSameLayout(actual, strategy->copy_loop_layout, strategy->block_mn,
                     strategy->block_k, "copy-loop local");
  arith::Analyzer analyzer;
  for (int row = 0; row < strategy->block_mn; ++row) {
    for (int col = 0; col < strategy->block_k; ++col) {
      Array<PrimExpr> logical = {Integer(row), Integer(col)};
      PrimExpr actual_thread =
          actual->ForwardThread(logical, Optional<PrimExpr>());
      PrimExpr expected_thread = strategy->copy_loop_layout->ForwardThread(
          logical, Optional<PrimExpr>());
      if (!is_zero(analyzer.Simplify(actual_thread - expected_thread))) {
        LOG(FATAL)
            << "HCU GEMM AT/BN copy-loop layout thread mapping mismatch at "
            << "logical coordinate (" << row << ", " << col
            << "): actual=" << actual_thread
            << ", expected=" << expected_thread;
      }
    }
  }
}

int GetHcuGemmAtBnRequiredWrapCount(const HcuGemmAtBnLdsStrategy &strategy) {
  const int issue_group_lanes =
      GetAtBnIssueGroupLanes(strategy->read_bytes_per_lane);
  const int required_wrap_count =
      GetAtBnRequiredWrapCount(strategy->row_period, issue_group_lanes);
  ICHECK_GT(required_wrap_count, 0)
      << "Invalid HCU GEMM AT/BN bank-conflict geometry";
  return required_wrap_count;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  HcuGemmAtBnLdsStrategyNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
