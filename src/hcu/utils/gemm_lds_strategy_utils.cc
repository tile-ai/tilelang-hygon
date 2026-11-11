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

#include <tvm/arith/analyzer.h>

#include <climits>
#include <numeric>

namespace tvm {
namespace tl {

namespace {

bool IsPowerOfTwo(int value) { return value > 0 && (value & (value - 1)) == 0; }

} // namespace

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
  bool has_an_bt = false;
  bool has_at_bn = false;
  bool has_at_bn_b64 = false;
  bool has_at_bn_b128 = false;
  const int element_bits = consumers.front().element_bits;
  if (element_bits != 8 && element_bits != 16)
    return std::nullopt;
  // Preserve the existing shared swizzle's 128-bit contiguous groups,
  // independently of each consumer's instruction width.
  const int swizzle_group_elements = 128 / element_bits;
  const int read_tile_k = element_bits == 8 ? 32 : 16;
  if (rows < read_tile_k)
    return std::nullopt;
  for (const auto &read : consumers) {
    if (read.element_bits != element_bits || read.read_tile_k != read_tile_k ||
        read.read_tile_mn != 32 || read.elements_per_read <= 0 ||
        read.elements_per_read > 128 / element_bits ||
        !IsPowerOfTwo(read.elements_per_read))
      return std::nullopt;
    const int read_bytes_per_lane = read.elements_per_read * element_bits / 8;
    if (read.consumer_kind == HcuGemmLdsAccessKind::kAnBt) {
      if (read_bytes_per_lane != 16)
        return std::nullopt;
      has_an_bt = true;
    } else if (read.consumer_kind == HcuGemmLdsAccessKind::kAtBn) {
      has_at_bn = true;
      has_at_bn_b64 |= read_bytes_per_lane == 8;
      has_at_bn_b128 |= read_bytes_per_lane == 16;
    } else {
      return std::nullopt;
    }
  }
  // Pure AT/BN keeps the existing makeHCUSwizzledLayout path. This
  // common layout is only needed by AN/BT and its mixed consumers.
  if (!has_an_bt)
    return std::nullopt;
  PrimExpr row = InputPlaceholder(0);
  PrimExpr col = InputPlaceholder(1);
  // Measure row stride in bank words, not elements or read vectors.
  // A stride of bank_count words returns to the same bank. The read's
  // contiguous 16B suffix remains a separate instruction constraint.
  const int element_bytes = element_bits / 8;
  const int64_t row_bytes = width * element_bytes;
  const int64_t row_bank_words = row_bytes / bank_width_bytes;
  const int panel_bank_words = 32 * element_bytes / bank_width_bytes;
  const int row_bank_period = static_cast<int>(
      bank_count / std::gcd(static_cast<int64_t>(bank_count), row_bank_words));
  // Keep row-major storage only when the pure matrix-read phase is
  // already separated by the row stride. b8 and b16 have different phases.
  const bool keep_identity_layout =
      !has_at_bn &&
      ((element_bits == 16 && row_bank_words == panel_bank_words) ||
       (element_bits == 8 && bank_count == 32 && row_bank_period >= 4));
  // Emit shifts in element coordinates to retain the established TIR
  // form and code generation while selecting them by bank-word geometry.
  const int read_log2_elements = element_bits == 8 ? 4 : 3;
  auto map_row_bits = [&](int mask, int column_shift) {
    return (row & mask) << column_shift;
  };
  PrimExpr shift = Integer(0);
  if (!keep_identity_layout) {
    if (bank_count == 64) {
      if (element_bits == 16 && row_bank_words == panel_bank_words) {
        if (has_at_bn_b64 && has_at_bn_b128) {
          // Both native read widths need a joint row/vector permutation. Keep
          // each 16B vector intact and encode its original column in the
          // physical row, separating all three read phases and the store.
          PrimExpr vector_col = floordiv(col, swizzle_group_elements);
          PrimExpr physical_row =
              floordiv(row, 16) * 16 + vector_col * 4 +
              floormod(floordiv(row, 4), 2) +
              ((floormod(floordiv(row, 8), 2) ^ floormod(vector_col, 2)) * 2);
          PrimExpr physical_col =
              ((floormod(row, 2) ^ floordiv(vector_col, 2)) *
               swizzle_group_elements) +
              floormod(floordiv(row, 2), 2) * (2 * swizzle_group_elements) +
              floormod(col, swizzle_group_elements);

          // This permutation is not self-inverse. Recover the original
          // vector column before reconstructing the four low row bits.
          PrimExpr inverse_vector_col = floormod(floordiv(row, 4), 4);
          PrimExpr stored_row_bits = floordiv(col, swizzle_group_elements);
          PrimExpr inverse_row =
              floordiv(row, 16) * 16 +
              (floormod(stored_row_bits, 2) ^ floordiv(inverse_vector_col, 2)) +
              floordiv(stored_row_bits, 2) * 2 + floormod(row, 2) * 4 +
              ((floormod(floordiv(row, 2), 2) ^
                floormod(inverse_vector_col, 2)) *
               8);
          PrimExpr inverse_col = inverse_vector_col * swizzle_group_elements +
                                 floormod(col, swizzle_group_elements);
          Array<PrimExpr> matrix_shape{Integer(rows), Integer(width)};
          Layout layout(matrix_shape, {physical_row, physical_col});
          Layout inverse(matrix_shape, {inverse_row, inverse_col});
          Array<PrimExpr> leading;
          for (size_t i = 0; i + 2 < shape.size(); ++i)
            leading.push_back(shape[i]);
          return layout->Expand(leading)->WithInverse(inverse->Expand(leading));
        }
        // A single native read width only needs the original column selectors:
        // 8B uses r2/r3; 16B uses r0/r3.
        const int row_selector_mask = has_at_bn_b64 ? 4 : 1;
        shift = ((row & row_selector_mask) *
                 (swizzle_group_elements / row_selector_mask)) ^
                map_row_bits(8, read_log2_elements - 2);
      } else if (element_bits == 16 && has_at_bn && !has_at_bn_b128) {
        // The existing short-read mapping is already conflict-free. Retain its
        // expression to avoid extra register spills on wide b16 tiles.
        shift =
            map_row_bits(12, read_log2_elements - 2) ^
            map_row_bits(4 - row_bank_period,
                         read_log2_elements + (row_bank_period == 1 ? 2 : 1));
      } else {
        // A 256B read phase needs sixteen distinct 16B bank groups.
        // Row stride supplies low row bits. Put r3 in the first column
        // selector to avoid cancelling the native b128 column bit, then
        // map the remaining r0..r2 above it, preserving each lane's 16B
        // vector and row boundary.
        const int remaining_row_mask = 8 - row_bank_period;
        shift = map_row_bits(8, read_log2_elements - 3) ^
                ((row & remaining_row_mask) *
                 (2 * swizzle_group_elements / row_bank_period));
      }
    } else if (element_bits == 8) {
      // b8 / 32 banks: four K classes in each matrix-read phase.
      if (row_bank_period >= 4) {
        // Pure reads kept identity above; mixed readers need r2.
        shift = map_row_bits(4, read_log2_elements - 2);
      } else if (row_bank_period == 2) {
        // r3 fans out to the two available selectors for mixed reads.
        shift = has_at_bn ? (map_row_bits(6, read_log2_elements - 1) ^
                             ((row & 8) * (3 * swizzle_group_elements / 8)))
                          : map_row_bits(2, read_log2_elements);
      } else {
        // A single bank ring couples the b128 producer's lane bit with
        // r1. Use r3 instead; mixed readers need the same assignment.
        shift = row_bank_words == bank_count || has_at_bn
                    ? (map_row_bits(1, read_log2_elements + 1) ^
                       map_row_bits(10, read_log2_elements - 1))
                    : map_row_bits(3, read_log2_elements + 1);
        if (has_at_bn)
          shift = shift ^ map_row_bits(4, read_log2_elements);
      }
    } else {
      // b16 / 32 banks: combine matrix, ordinary-read and store
      // constraints without changing the continuous N8 suffix.
      if (row_bank_period > 1) {
        shift = map_row_bits(2, read_log2_elements) ^
                map_row_bits(4, read_log2_elements - 2);
      } else {
        shift = map_row_bits(1, read_log2_elements + 1) ^
                map_row_bits(2, read_log2_elements - 1) ^
                map_row_bits(4, read_log2_elements) ^
                map_row_bits(8, read_log2_elements - 2);
      }
    }
  }
  // Keep the vector suffix explicit for vectorization and ds-read proofs.
  PrimExpr physical_col = ((floordiv(col, swizzle_group_elements) ^
                            floordiv(shift, swizzle_group_elements)) *
                           swizzle_group_elements) +
                          floormod(col, swizzle_group_elements);
  Layout layout(Array<PrimExpr>{Integer(rows), Integer(width)},
                {row, physical_col});
  Array<PrimExpr> leading;
  for (size_t i = 0; i + 2 < shape.size(); ++i)
    leading.push_back(shape[i]);
  Layout expanded = layout->Expand(leading);
  return expanded->WithInverse(expanded);
}

Optional<Layout>
MakeHcuDsReadVectorLdsLayout(const Array<PrimExpr> &shape, int bank_count,
                             int bank_width_bytes, int warp_size,
                             const HcuGemmLdsReadPattern &read) {
  auto fallback = [&]() {
    return MakeHcuPlainGemmLdsLayout(shape, bank_count, bank_width_bytes,
                                     {read});
  };
  if (shape.size() < 2 || (bank_count != 32 && bank_count != 64) ||
      bank_width_bytes != 4 || warp_size != 64 || read.element_bits != 16 ||
      read.read_tile_k != 16 || read.read_tile_mn != 32 ||
      read.elements_per_read != 8 ||
      read.consumer_kind != HcuGemmLdsAccessKind::kAnBt)
    return fallback();
  // Leading batch/stage axes pass through unchanged, including symbolic axes.
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    if (const int64_t *value = as_const_int(shape[i]); value && *value <= 0)
      return fallback();
  }
  for (size_t i = shape.size() - 2; i < shape.size(); ++i) {
    const PrimExpr &extent = shape[i];
    const int64_t *value = as_const_int(extent);
    if (!value || *value <= 0 || *value > INT_MAX)
      return fallback();
  }
  const int rows = *as_const_int(shape[shape.size() - 2]);
  const int width = *as_const_int(shape.back());
  const int vector_elements = read.elements_per_read;
  const int read_rows = read.read_tile_k;
  // Each read-row tile is permuted independently; its repeat count need not
  // be a power of two. Width only needs complete vectors that divide a wave.
  if (rows % read_rows != 0 || width % vector_elements != 0)
    return fallback();
  const int vectors_per_row = width / vector_elements;
  if (warp_size % vectors_per_row != 0)
    return fallback();
  const int rows_per_wave = warp_size / vectors_per_row;
  if (read_rows % rows_per_wave != 0)
    return fallback();
  const int row_classes = read_rows / rows_per_wave;
  const int element_bytes = read.element_bits / 8;
  const int panel_bytes = read.read_tile_mn * element_bytes;
  const int bank_ring_bytes = bank_count * bank_width_bytes;
  if (bank_ring_bytes % panel_bytes != 0)
    return fallback();
  // Each matrix phase varies two/four row classes on 32/64 banks. Rotate
  // these classes by one MN panel (64B), keeping each 16B vector intact.
  const int rotations = bank_ring_bytes / panel_bytes;
  const int rotation_vectors = read.read_tile_mn / vector_elements;
  if (row_classes % rotations != 0)
    return fallback();

  PrimExpr row = InputPlaceholder(0);
  PrimExpr col = InputPlaceholder(1);
  const int vectors_per_tile = read_rows * vectors_per_row;
  PrimExpr vector = floordiv(row, read_rows) * vectors_per_tile +
                    floormod(row, row_classes) * warp_size +
                    floormod(floordiv(floormod(row, read_rows), row_classes) *
                                     vectors_per_row +
                                 floordiv(col, vector_elements) +
                                 floormod(row, rotations) * rotation_vectors,
                             warp_size);
  PrimExpr flat = vector * vector_elements + floormod(col, vector_elements);
  Layout layout(Array<PrimExpr>{Integer(rows), Integer(width)},
                {floordiv(flat, width), floormod(flat, width)});

  PrimExpr physical = row * width + col;
  PrimExpr point = floordiv(physical, vector_elements);
  PrimExpr row_class = floordiv(floormod(point, vectors_per_tile), warp_size);
  PrimExpr packed =
      floormod(floormod(point, warp_size) -
                   floormod(row_class, rotations) * rotation_vectors,
               warp_size);
  PrimExpr logical_row = floordiv(point, vectors_per_tile) * read_rows +
                         floordiv(packed, vectors_per_row) * row_classes +
                         row_class;
  PrimExpr logical_col = floormod(packed, vectors_per_row) * vector_elements +
                         floormod(physical, vector_elements);
  Layout inverse(Array<PrimExpr>{Integer(rows), Integer(width)},
                 {logical_row, logical_col});
  Array<PrimExpr> leading;
  for (size_t i = 0; i + 2 < shape.size(); ++i)
    leading.push_back(shape[i]);
  return layout->Expand(leading)->WithInverse(inverse->Expand(leading));
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
      "tl.hcu.MakeDsReadVectorLdsLayout",
      [](Array<PrimExpr> shape, int bank_count, int bank_width_bytes,
         int warp_size, int element_bits) {
        ICHECK(element_bits == 8 || element_bits == 16);
        return MakeHcuDsReadVectorLdsLayout(
            shape, bank_count, bank_width_bytes, warp_size,
            {element_bits, 256 / element_bits, 32, 128 / element_bits,
             HcuGemmLdsAccessKind::kAnBt});
      });
  ffi::reflection::GlobalDef().def(
      "tl.hcu.MakePlainGemmLdsLayout",
      [](Array<PrimExpr> shape, int bank_count, int bank_width_bytes,
         int element_bits, int read_tile_k, int read_tile_mn,
         Array<Integer> elements_per_read, Array<Integer> consumer_kinds) {
        ICHECK_EQ(consumer_kinds.size(), elements_per_read.size());
        std::vector<HcuGemmLdsReadPattern> consumers;
        for (size_t i = 0; i < consumer_kinds.size(); ++i) {
          ICHECK(consumer_kinds[i]->value == 0 ||
                 consumer_kinds[i]->value == 1);
          consumers.push_back(
              {element_bits, read_tile_k, read_tile_mn,
               int(elements_per_read[i]->value),
               static_cast<HcuGemmLdsAccessKind>(consumer_kinds[i]->value)});
        }
        return MakeHcuPlainGemmLdsLayout(shape, bank_count, bank_width_bytes,
                                         consumers);
      });
}

} // namespace tl
} // namespace tvm
