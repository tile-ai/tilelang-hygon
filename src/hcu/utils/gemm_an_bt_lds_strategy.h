// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file gemm_an_bt_lds_strategy.h
 * \brief Compiler-derived LDS strategy for HCU GEMM AN/BT ds-read copies.
 */
#ifndef TVM_TL_HCU_UTILS_GEMM_AN_BT_LDS_STRATEGY_H_
#define TVM_TL_HCU_UTILS_GEMM_AN_BT_LDS_STRATEGY_H_

#include "layout/layout.h"
#include "op/copy.h"
#include "op/gemm.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>

namespace tvm {
namespace tl {

namespace attr {
static constexpr const char *kHcuGemmAnBtLdsStrategy =
    "tl.hcu_gemm_an_bt_lds_strategy";
} // namespace attr

class HcuGemmAnBtLdsStrategyNode : public Object {
public:
  int strategy_version{1};
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
  // Physical wrap shift in dwords before target-specific field encoding.
  int wrap_offset{0};
  int wrap_idx_mask{0};
  Layout storage_layout;
  Fragment copy_loop_layout;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.HcuGemmAnBtLdsStrategy",
                                    HcuGemmAnBtLdsStrategyNode, Object);

  static void RegisterReflection();
};

class HcuGemmAnBtLdsStrategy : public ObjectRef {
public:
  explicit HcuGemmAnBtLdsStrategy(ObjectPtr<HcuGemmAnBtLdsStrategyNode> ptr)
      : ObjectRef(std::move(ptr)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(HcuGemmAnBtLdsStrategy,
                                                ObjectRef,
                                                HcuGemmAnBtLdsStrategyNode);
};

Optional<HcuGemmAnBtLdsStrategy>
DeriveHcuGemmAnBtLdsStrategy(const CopyNode &copy, const GemmNode &gemm,
                             bool feeds_a, int block_threads, Target target);

Optional<HcuGemmAnBtLdsStrategy> DeriveHcuGemmAnBtLdsStrategyWith64ByteWrap(
    const CopyNode &copy, const GemmNode &gemm, bool feeds_a, int block_threads,
    Target target, int wrap_count);

void ValidateHcuGemmAnBtStorageLayout(const Layout &actual,
                                      const HcuGemmAnBtLdsStrategy &strategy);

void ValidateHcuGemmAnBtCopyLayout(const Fragment &actual,
                                   const HcuGemmAnBtLdsStrategy &strategy);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_UTILS_GEMM_AN_BT_LDS_STRATEGY_H_
