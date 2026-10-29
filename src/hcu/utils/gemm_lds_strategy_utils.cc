// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file gemm_lds_strategy_utils.cc
 * \brief Shared geometry helpers for compiler-derived HCU GEMM LDS strategies.
 */

#include "gemm_lds_strategy_utils.h"

#include "hcu/target_utils.h"
#include "op/copy.h"
#include "transform/loop_vectorize.h"

#include <climits>
#include <numeric>

namespace tvm {
namespace tl {

namespace {

bool IsPowerOfTwo(int value) { return value > 0 && (value & (value - 1)) == 0; }

} // namespace

Optional<Layout> MakeHcuNativeAnBtLdsLayout(
    const Array<PrimExpr> &shape,
    const std::vector<HcuGemmLdsReadPattern> &consumers) {
  if (shape.size() != 2 || consumers.empty())
    return std::nullopt;
  const int64_t *rows = as_const_int(shape[0]);
  const int64_t *width = as_const_int(shape[1]);
  if (!rows || !width || *rows <= 0 || *width <= 0 || *rows > INT_MAX ||
      *width > INT_MAX)
    return std::nullopt;
  const int bits = consumers.front().element_bits;
  if (bits != 8 && bits != 16)
    return std::nullopt;
  const int mmac_k = 256 / bits;
  const int atom_mn = 16;
  if (*rows % mmac_k != 0 || *width % atom_mn != 0)
    return std::nullopt;
  for (const auto &read : consumers) {
    if (read.read_kind != HcuGemmLdsReadKind::kNative ||
        read.consumer_kind != HcuGemmLdsAccessKind::kAnBt ||
        read.consumer_k_pack != 1 || read.element_bits != bits ||
        read.read_tile_k != mmac_k || read.read_tile_mn != 16 ||
        read.elements_per_read != mmac_k / 4 ||
        read.read_bytes_per_lane != bits * mmac_k / 32)
      return std::nullopt;
  }
  PrimExpr k = InputPlaceholder(0);
  PrimExpr n = InputPlaceholder(1);
  // XOR only the low group bits shared by the row width and the four K
  // groups in a native fragment. This remains a permutation for non-power-
  // of-two widths: e.g. six MN groups use two phases, not four.
  const int phases = std::gcd(int(*width / atom_mn), 4);
  PrimExpr phase = floormod(floordiv(k, mmac_k / 4), phases);
  PrimExpr group = floormod(floordiv(n, atom_mn), phases);
  PrimExpr block = floordiv(n, atom_mn * phases);
  Layout layout(shape, {k, (block * phases + (group ^ phase)) * atom_mn +
                               floormod(n, atom_mn)});
  return layout->WithInverse(layout);
}

Optional<Layout>
MakeHcuPlainGemmLdsLayout(const Array<PrimExpr> &shape, int bank_count,
                          int bank_width_bytes,
                          const std::vector<HcuGemmLdsReadPattern> &consumers) {
  if (shape.size() < 2 || (bank_count != 32 && bank_count != 64) ||
      bank_width_bytes != 4 || consumers.empty()) {
    return std::nullopt;
  }
  for (const PrimExpr &extent : shape) {
    const int64_t *value = as_const_int(extent);
    if (!value || *value <= 0)
      return std::nullopt;
  }
  const int64_t rows = *as_const_int(shape[shape.size() - 2]);
  const int64_t width = *as_const_int(shape.back());
  if (rows < 16 || width < 32 || rows > INT_MAX || width > INT_MAX ||
      !IsPowerOfTwo(rows) || !IsPowerOfTwo(width)) {
    return std::nullopt;
  }
  for (const auto &read : consumers) {
    if (read.read_kind != HcuGemmLdsReadKind::kMatrix &&
        read.consumer_kind == HcuGemmLdsAccessKind::kAnBt)
      return std::nullopt;
  }
  bool has_an_bt = false;
  bool has_at_bn = false;
  bool has_at_bn_pack2 = false;
  const int element_bits = consumers.front().element_bits;
  if (element_bits != 8 && element_bits != 16)
    return std::nullopt;
  const int elements_per_read = 128 / element_bits;
  const int read_tile_k = element_bits == 8 ? 32 : 16;
  if (rows < read_tile_k)
    return std::nullopt;
  for (const auto &read : consumers) {
    if (read.element_bits != element_bits || read.read_tile_k != read_tile_k ||
        read.read_tile_mn != 32 ||
        read.elements_per_read != elements_per_read ||
        read.read_bytes_per_lane != 16)
      return std::nullopt;
    if (read.consumer_kind == HcuGemmLdsAccessKind::kAnBt) {
      if (read.consumer_k_pack != 1)
        return std::nullopt;
      has_an_bt = true;
    } else {
      if (read.consumer_k_pack != 1 && read.consumer_k_pack != 2)
        return std::nullopt;
      has_at_bn = true;
      has_at_bn_pack2 |= read.consumer_k_pack == 2;
    }
  }
  // Pure AT/BN keeps the existing makeHCUSwizzledLayout path. This
  // common layout is only needed by AN/BT and its mixed consumers.
  if (!has_an_bt)
    return std::nullopt;
  PrimExpr row = InputPlaceholder(0);
  PrimExpr col = InputPlaceholder(1);
  PrimExpr shift = Integer(0);
  if (element_bits == 8) {
    // b8 preserves a 16-element suffix. Its K32/N32 requests need a
    // distinct row-bit mapping from the b16 K16/N32 contract.
    if (bank_count == 32) {
      if (width == 64)
        shift = (row & 2) << 4;
      if (width >= 128)
        shift = (row & 3) << 5;
      // NMZ b128 phases pair lanes 0..3 with 20..23. At width 128,
      // injecting r1 into column bit 6 cancels the producer's lane bit 2.
      // Use r1 for column bit 4 and r3 for bit 6 instead; the b8 matrix
      // phase's r3/c relation preserves its three independent bank bits.
      if (width == 128 || (has_at_bn && width >= 128))
        shift = ((row & 1) << 5) ^ ((row & 10) << 3);
      if (has_at_bn) {
        // Cover the full b128 phase for AT/BN kp2 as well as the matrix
        // phase. At width 64, r3 is broadcast into both N16-group bits.
        if (width == 32)
          shift = (row & 4) << 2;
        else if (width == 64)
          shift = ((row & 6) << 3) ^ ((row & 8) * 6);
        else
          shift = shift ^ ((row & 4) << 4);
      }
    } else {
      if (width == 128)
        shift = (row & 2) << 5;
      if (width >= 256)
        shift = (row & 3) << 6;
      if (has_at_bn) {
        shift = shift ^ ((row & 4) << 2);
        if (width >= 64)
          shift = shift ^ ((row & 8) << 2);
      }
      if (width == 64 && (!has_at_bn || has_at_bn_pack2)) {
        // SB's 16-lane matrix and b128-read phases need independent row
        // bits in both N16 selectors. Keep r2 off the low selector: the
        // ordinary-read phase already pairs that bit with its N16 column.
        // A pack1-only mixed consumer already has zero conflicts; retain
        // its existing mapping rather than changing its producer addresses.
        shift = ((row & 4) << 3) ^ ((row & 8) << 1);
      }
    }
  } else if (!has_at_bn) {
    const int row_period = bank_count / std::gcd(bank_count, int(width / 2));
    if (row_period == 1)
      shift = (row & 1) << 5;
  } else if (bank_count == 64) {
    shift = (row & 12) << 1;
    if (width == 64)
      shift = shift ^ ((row & 2) << 4);
    if (width >= 128)
      shift = shift ^ ((row & 3) << 5);
  } else {
    shift = ((row & 2) << 3) ^ ((row & 4) << 1);
    // Mix the fourth row bit as well: scalar AnBt reads can access rows
    // eight apart in the same wave. Preserve the N8 vector suffix.
    if (width >= 64)
      shift = ((row & 1) << 4) ^ ((row & 2) << 2) ^ ((row & 4) << 3) ^
              ((row & 8) << 1);
  }
  // Express the preserved vector bits explicitly so vectorization and the
  // symbolic ds-read contiguity check can see the affine vector suffix.
  PrimExpr physical_col =
      ((floordiv(col, elements_per_read) ^ floordiv(shift, elements_per_read)) *
       elements_per_read) +
      floormod(col, elements_per_read);
  Layout layout(Array<PrimExpr>{Integer(rows), Integer(width)},
                {row, physical_col});
  Array<PrimExpr> leading;
  for (size_t i = 0; i + 2 < shape.size(); ++i)
    leading.push_back(shape[i]);
  Layout expanded = layout->Expand(leading);
  return expanded->WithInverse(expanded);
}

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
      .def_ro("thread_offset", &HcuGemmLdsCopyStrategyNode::thread_offset)
      .def_ro("inner_extent", &HcuGemmLdsCopyStrategyNode::inner_extent)
      .def_ro("warp_size", &HcuGemmLdsCopyStrategyNode::warp_size)
      .def_ro("wrap_offset", &HcuGemmLdsCopyStrategyNode::wrap_offset)
      .def_ro("wrap_idx_mask", &HcuGemmLdsCopyStrategyNode::wrap_idx_mask)
      .def_ro("wrap_uses_copy_transaction",
              &HcuGemmLdsCopyStrategyNode::wrap_uses_copy_transaction)
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

int GetHcuGemmLdsCopyVectorBytes(const CopyNode &copy, Target target) {
  With<Target> target_scope(target);
  arith::Analyzer analyzer;
  For loop = copy.par_op_.defined() ? copy.par_op_->GetRoot()
                                    : copy.MakeSIMTLoop(&analyzer);
  return GetVectorizeSize(loop, &analyzer) * copy.src->dtype.bytes();
}

int GetHcuGemmLdsWrapOffsetDwords(int wrap_step_bytes) {
  if (wrap_step_bytes <= 0 || wrap_step_bytes % 4 != 0) {
    return 0;
  }
  return wrap_step_bytes / 4;
}

bool IsLegalHcuGemmLdsWrap(const HcuGemmLdsCopyGeometry &geometry,
                           int wrap_step_bytes, int wrap_count,
                           int available_wrap_phases) {
  if (!IsPowerOfTwo(wrap_count) || available_wrap_phases <= 0 ||
      wrap_count > available_wrap_phases ||
      available_wrap_phases % wrap_count != 0) {
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

HcuGemmLdsCopyStrategy MakeHcuGemmLdsCopyStrategy(
    bool use_idxen, int copy_bytes_per_lane, int copy_transaction_bytes,
    int block_threads, int thread_offset, int inner_extent, int warp_size,
    int wrap_offset, int wrap_idx_mask, bool wrap_uses_copy_transaction,
    Layout storage_layout, Fragment copy_loop_layout) {
  auto node = ffi::make_object<HcuGemmLdsCopyStrategyNode>();
  node->use_idxen = use_idxen;
  node->copy_bytes_per_lane = copy_bytes_per_lane;
  node->copy_transaction_bytes = copy_transaction_bytes;
  node->block_threads = block_threads;
  node->thread_offset = thread_offset;
  node->inner_extent = inner_extent;
  node->warp_size = warp_size;
  node->wrap_offset = wrap_offset;
  node->wrap_idx_mask = wrap_idx_mask;
  node->wrap_uses_copy_transaction = wrap_uses_copy_transaction;
  node->storage_layout = std::move(storage_layout);
  node->copy_loop_layout = std::move(copy_loop_layout);
  return HcuGemmLdsCopyStrategy(std::move(node));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  HcuGemmLdsCopyStrategyNode::RegisterReflection();
  ffi::reflection::GlobalDef().def(
      "tl.hcu.MakePlainGemmLdsLayout",
      [](Array<PrimExpr> shape, int bank_count, int bank_width_bytes,
         int element_bits, int read_tile_k, int read_tile_mn,
         int elements_per_read, int read_bytes_per_lane,
         Array<Integer> consumer_kinds, Array<Integer> consumer_k_packs) {
        ICHECK_EQ(consumer_kinds.size(), consumer_k_packs.size());
        std::vector<HcuGemmLdsReadPattern> consumers;
        for (size_t i = 0; i < consumer_kinds.size(); ++i) {
          ICHECK(consumer_kinds[i]->value == 0 ||
                 consumer_kinds[i]->value == 1);
          consumers.push_back(
              {element_bits, read_tile_k, read_tile_mn, elements_per_read,
               read_bytes_per_lane,
               static_cast<HcuGemmLdsAccessKind>(consumer_kinds[i]->value),
               int(consumer_k_packs[i]->value)});
        }
        return MakeHcuPlainGemmLdsLayout(shape, bank_count, bank_width_bytes,
                                         consumers);
      });
}

} // namespace tl
} // namespace tvm
