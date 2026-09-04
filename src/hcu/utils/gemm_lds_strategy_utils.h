// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file gemm_lds_strategy_utils.h
 * \brief Shared geometry helpers for compiler-derived HCU GEMM LDS strategies.
 */
#ifndef TVM_TL_HCU_UTILS_GEMM_LDS_STRATEGY_UTILS_H_
#define TVM_TL_HCU_UTILS_GEMM_LDS_STRATEGY_UTILS_H_

#include "layout/layout.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>

#include <optional>

namespace tvm {
namespace tl {

namespace attr {
static constexpr const char *kHcuGemmLdsCopyStrategy =
    "tl.hcu_gemm_lds_copy_strategy";
static constexpr const char *kHcuCopyAsyncPromotable =
    "tl.hcu_copy_async_promotable";
} // namespace attr

class HcuGemmLdsCopyStrategyNode : public ffi::Object {
public:
  int strategy_version{1};
  bool use_idxen{false};
  int copy_bytes_per_lane{0};
  int copy_transaction_bytes{0};
  int block_threads{0};
  int inner_extent{0};
  // Physical wrap shift in dwords before target-specific field encoding.
  int wrap_offset{0};
  int wrap_idx_mask{0};
  Layout storage_layout;
  Fragment copy_loop_layout;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.HcuGemmLdsCopyStrategy",
                                    HcuGemmLdsCopyStrategyNode, ffi::Object);

  static void RegisterReflection();
};

class HcuGemmLdsCopyStrategy : public ffi::ObjectRef {
public:
  explicit HcuGemmLdsCopyStrategy(
      ffi::ObjectPtr<HcuGemmLdsCopyStrategyNode> ptr)
      : ObjectRef(std::move(ptr)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(HcuGemmLdsCopyStrategy,
                                                ffi::ObjectRef,
                                                HcuGemmLdsCopyStrategyNode);
};

struct HcuGemmLdsCopyGeometry {
  int bank_count{0};
  int bank_width_bytes{0};
  int bank_ring_bytes{0};
  int warp_size{0};
  int block_threads{0};
  int num_copy_waves{0};
  int copy_transaction_bytes{0};
  int wrap_granularity_dwords{0};
  int bytes_per_row{0};
  int segments_per_row{0};
  int row_wave_gcd{0};
  int rows_per_group{0};
  int waves_per_group{0};
  int max_wrap_offset_dwords{0};
};

std::optional<HcuGemmLdsCopyGeometry>
DeriveHcuGemmLdsCopyGeometry(int bytes_per_row, int copy_transaction_bytes,
                             int block_threads, Target target);

int SelectHcuGemmLdsCopyTransactionBytes(int bytes_per_row,
                                         int copy_bytes_per_lane,
                                         int element_bytes);

int GetHcuGemmLdsWrapOffsetDwords(int wrap_step_bytes);

bool IsLegalHcuGemmLdsWrap(const HcuGemmLdsCopyGeometry &geometry,
                           int wrap_step_bytes, int wrap_count);

HcuGemmLdsCopyStrategy
MakeHcuGemmLdsCopyStrategy(bool use_idxen, int copy_bytes_per_lane,
                           int copy_transaction_bytes, int block_threads,
                           int inner_extent, int wrap_offset, int wrap_idx_mask,
                           Layout storage_layout, Fragment copy_loop_layout);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_UTILS_GEMM_LDS_STRATEGY_UTILS_H_
