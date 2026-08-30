// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file gemm_lds_strategy_utils.cc
 * \brief Shared geometry helpers for compiler-derived HCU GEMM LDS strategies.
 */

#include "gemm_lds_strategy_utils.h"

#include "hcu/target_utils.h"

#include <numeric>

namespace tvm {
namespace tl {

namespace {

bool IsPowerOfTwo(int value) { return value > 0 && (value & (value - 1)) == 0; }

} // namespace

void HcuGemmLdsCopyStrategyNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<HcuGemmLdsCopyStrategyNode>()
      .def_ro("strategy_version", &HcuGemmLdsCopyStrategyNode::strategy_version)
      .def_ro("use_idxen", &HcuGemmLdsCopyStrategyNode::use_idxen)
      .def_ro("copy_bytes_per_lane",
              &HcuGemmLdsCopyStrategyNode::copy_bytes_per_lane)
      .def_ro("copy_transaction_bytes",
              &HcuGemmLdsCopyStrategyNode::copy_transaction_bytes)
      .def_ro("block_threads", &HcuGemmLdsCopyStrategyNode::block_threads)
      .def_ro("inner_extent", &HcuGemmLdsCopyStrategyNode::inner_extent)
      .def_ro("wrap_offset", &HcuGemmLdsCopyStrategyNode::wrap_offset)
      .def_ro("wrap_idx_mask", &HcuGemmLdsCopyStrategyNode::wrap_idx_mask)
      .def_ro("storage_layout", &HcuGemmLdsCopyStrategyNode::storage_layout)
      .def_ro("copy_loop_layout",
              &HcuGemmLdsCopyStrategyNode::copy_loop_layout);
}

std::optional<HcuGemmLdsCopyGeometry>
DeriveHcuGemmLdsCopyGeometry(int bytes_per_row, int copy_transaction_bytes,
                             int block_threads, Target target) {
  if (!TargetIsHCU(target) || bytes_per_row <= 0 ||
      copy_transaction_bytes <= 0 || block_threads <= 0 ||
      bytes_per_row % copy_transaction_bytes != 0) {
    return std::nullopt;
  }

  HcuGemmLdsCopyGeometry geometry;
  geometry.bank_count = TargetHcuGetLdsBankCount(target);
  geometry.bank_width_bytes = TargetHcuGetLdsBankWidthBytes(target);
  geometry.warp_size = TargetHcuGetWarpSize(target);
  if (geometry.bank_count <= 0 || geometry.bank_width_bytes <= 0 ||
      geometry.warp_size <= 0 || block_threads % geometry.warp_size != 0) {
    return std::nullopt;
  }

  geometry.bank_ring_bytes = geometry.bank_count * geometry.bank_width_bytes;
  geometry.block_threads = block_threads;
  geometry.num_copy_waves = block_threads / geometry.warp_size;
  geometry.copy_transaction_bytes = copy_transaction_bytes;
  const HcuLdsWrapConfig wrap_config = TargetHcuGetLdsWrapConfig(target);
  if (wrap_config.encoding == HcuLdsWrapEncoding::kNone) {
    return std::nullopt;
  }
  geometry.wrap_granularity_dwords =
      TargetHcuGetLdsWrapGranularityDwords(target);
  geometry.bytes_per_row = bytes_per_row;
  geometry.segments_per_row = bytes_per_row / copy_transaction_bytes;

  // Reduce row and wave boundaries to the smallest repeating copy group.
  geometry.row_wave_gcd =
      std::gcd(geometry.segments_per_row, geometry.warp_size);
  geometry.rows_per_group = geometry.warp_size / geometry.row_wave_gcd;
  geometry.waves_per_group = geometry.segments_per_row / geometry.row_wave_gcd;
  geometry.max_wrap_offset_dwords = TargetHcuGetLdsWrapMaxOffsetDwords(target);
  return geometry;
}

int SelectHcuGemmLdsCopyTransactionBytes(int bytes_per_row,
                                         int copy_bytes_per_lane,
                                         int element_bytes) {
  if (bytes_per_row <= 0 || copy_bytes_per_lane <= 0 || element_bytes <= 0) {
    return 0;
  }
  constexpr int kTransferBytes[] = {16, 8, 4};
  for (int transfer_bytes : kTransferBytes) {
    if (transfer_bytes % element_bytes == 0 &&
        bytes_per_row % transfer_bytes == 0 &&
        copy_bytes_per_lane % transfer_bytes == 0) {
      return transfer_bytes;
    }
  }
  return 0;
}

int GetHcuGemmLdsWrapOffsetDwords(int wrap_step_bytes) {
  if (wrap_step_bytes <= 0 || wrap_step_bytes % 4 != 0) {
    return 0;
  }
  return wrap_step_bytes / 4;
}

bool IsLegalHcuGemmLdsWrap(const HcuGemmLdsCopyGeometry &geometry,
                           int wrap_step_bytes, int wrap_count) {
  if (!IsPowerOfTwo(wrap_count) || wrap_count > geometry.num_copy_waves ||
      geometry.num_copy_waves % wrap_count != 0) {
    return false;
  }
  if (wrap_count == 1) {
    return wrap_step_bytes == 0;
  }
  const int wrap_offset_dwords = GetHcuGemmLdsWrapOffsetDwords(wrap_step_bytes);
  if (geometry.wrap_granularity_dwords <= 0 || wrap_offset_dwords <= 0 ||
      wrap_offset_dwords % geometry.wrap_granularity_dwords != 0) {
    return false;
  }
  return wrap_offset_dwords * (wrap_count - 1) <=
         geometry.max_wrap_offset_dwords;
}

HcuGemmLdsCopyStrategy
MakeHcuGemmLdsCopyStrategy(bool use_idxen, int copy_bytes_per_lane,
                           int copy_transaction_bytes, int block_threads,
                           int inner_extent, int wrap_offset, int wrap_idx_mask,
                           Layout storage_layout, Fragment copy_loop_layout) {
  auto node = ffi::make_object<HcuGemmLdsCopyStrategyNode>();
  node->use_idxen = use_idxen;
  node->copy_bytes_per_lane = copy_bytes_per_lane;
  node->copy_transaction_bytes = copy_transaction_bytes;
  node->block_threads = block_threads;
  node->inner_extent = inner_extent;
  node->wrap_offset = wrap_offset;
  node->wrap_idx_mask = wrap_idx_mask;
  node->storage_layout = std::move(storage_layout);
  node->copy_loop_layout = std::move(copy_loop_layout);
  return HcuGemmLdsCopyStrategy(std::move(node));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  HcuGemmLdsCopyStrategyNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
