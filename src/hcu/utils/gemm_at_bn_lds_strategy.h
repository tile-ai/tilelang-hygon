/*!
 * \file gemm_at_bn_lds_strategy.h
 * \brief Compiler-derived LDS strategy for HCU GEMM AT/BN accesses.
 */
#ifndef TVM_TL_HCU_UTILS_GEMM_AT_BN_LDS_STRATEGY_H_
#define TVM_TL_HCU_UTILS_GEMM_AT_BN_LDS_STRATEGY_H_

#include "layout/layout.h"
#include "op/copy.h"
#include "op/gemm.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>

namespace tvm {
namespace tl {

namespace attr {
static constexpr const char *kHcuGemmAtBnLdsStrategy =
    "tl.hcu_gemm_at_bn_lds_strategy";
} // namespace attr

class HcuGemmAtBnLdsStrategyNode : public Object {
public:
  int strategy_version{1};
  int block_mn{0};
  int block_k{0};
  int block_threads{0};
  int warp_size{0};
  int warp_mn_count{0};
  int bank_num{32};
  int bank_width_bytes{4};
  int element_bytes{0};
  int copy_bytes_per_lane{0};
  int copy_transaction_bytes{0};
  int copy_transactions_per_lane{0};
  int read_bytes_per_lane{0};
  int row_period{0};
  int row_bank_stride{0};
  int rows_per_copy_wave{0};
  // Physical wrap shift in dwords before target-specific field encoding.
  int wrap_offset{0};
  int wrap_idx_mask{0};
  Layout storage_layout;
  Fragment copy_loop_layout;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.HcuGemmAtBnLdsStrategy",
                                    HcuGemmAtBnLdsStrategyNode, Object);

  static void RegisterReflection();
};

class HcuGemmAtBnLdsStrategy : public ObjectRef {
public:
  explicit HcuGemmAtBnLdsStrategy(ObjectPtr<HcuGemmAtBnLdsStrategyNode> ptr)
      : ObjectRef(std::move(ptr)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(
      HcuGemmAtBnLdsStrategy, ObjectRef, HcuGemmAtBnLdsStrategyNode);
};

Optional<HcuGemmAtBnLdsStrategy>
DeriveHcuGemmAtBnLdsStrategy(const CopyNode &copy, const GemmNode &gemm,
                             bool feeds_a, int block_threads, Target target);

void ValidateHcuGemmAtBnStorageLayout(
    const Layout &actual, const HcuGemmAtBnLdsStrategy &strategy);

void ValidateHcuGemmAtBnCopyLayout(
    const Fragment &actual, const HcuGemmAtBnLdsStrategy &strategy);

int GetHcuGemmAtBnRequiredWrapCount(
    const HcuGemmAtBnLdsStrategy &strategy);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_UTILS_GEMM_AT_BN_LDS_STRATEGY_H_
