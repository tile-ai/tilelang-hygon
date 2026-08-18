/*!
 * \file gemm_a_lds_strategy.h
 * \brief Compiler-derived LDS strategy for ordinary HCU GEMM A copies.
 */
#ifndef TVM_TL_HCU_UTILS_GEMM_A_LDS_STRATEGY_H_
#define TVM_TL_HCU_UTILS_GEMM_A_LDS_STRATEGY_H_

#include "layout/layout.h"
#include "op/copy.h"
#include "op/gemm.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>

namespace tvm {
namespace tl {

namespace attr {
static constexpr const char *kHcuGemmALdsStrategy =
    "tl.hcu_gemm_a_lds_strategy";
} // namespace attr

class HcuGemmALdsStrategyNode : public Object {
public:
  int strategy_version{1};
  int block_m{0};
  int block_k{0};
  int block_threads{0};
  int warp_size{0};
  int warp_m_count{0};
  int bank_num{32};
  int bank_width_bytes{4};
  int element_bytes{0};
  int copy_bytes_per_lane{0};
  int copy_transaction_bytes{0};
  int copy_transactions_per_lane{0};
  int read_bytes_per_lane{0};
  int row_period{0};
  int row_bank_stride{0};
  int segment_shift{0};
  int rows_per_copy_wave{0};
  int row_slab_count{0};
  int warp_tile_m{0};
  int wrap_offset{0};
  int wrap_idx_mask{0};
  Layout storage_layout;
  Fragment copy_loop_layout;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.HcuGemmALdsStrategy",
                                    HcuGemmALdsStrategyNode, Object);

  static void RegisterReflection();
};

class HcuGemmALdsStrategy : public ObjectRef {
public:
  explicit HcuGemmALdsStrategy(ObjectPtr<HcuGemmALdsStrategyNode> ptr)
      : ObjectRef(std::move(ptr)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(
      HcuGemmALdsStrategy, ObjectRef, HcuGemmALdsStrategyNode);
};

Optional<HcuGemmALdsStrategy>
DeriveHcuGemmALdsStrategy(const CopyNode &copy, const GemmNode &gemm,
                          int block_threads, Target target);

void ValidateHcuGemmAStorageLayout(const Layout &actual,
                                   const HcuGemmALdsStrategy &strategy);

void ValidateHcuGemmACopyLayout(const Fragment &actual,
                                const HcuGemmALdsStrategy &strategy);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_UTILS_GEMM_A_LDS_STRATEGY_H_
