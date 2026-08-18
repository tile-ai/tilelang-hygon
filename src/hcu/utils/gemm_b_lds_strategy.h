/*!
 * \file gemm_b_lds_strategy.h
 * \brief Compiler-derived LDS strategy for HCU GEMM B ds-read copies.
 */
#ifndef TVM_TL_HCU_UTILS_GEMM_B_LDS_STRATEGY_H_
#define TVM_TL_HCU_UTILS_GEMM_B_LDS_STRATEGY_H_

#include "layout/layout.h"
#include "op/copy.h"
#include "op/gemm.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>

namespace tvm {
namespace tl {

namespace attr {
static constexpr const char *kHcuGemmBLdsStrategy =
    "tl.hcu_gemm_b_lds_strategy";
} // namespace attr

class HcuGemmBLdsStrategyNode : public Object {
public:
  int strategy_version{1};
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
  int wrap_offset{0};
  int wrap_idx_mask{0};
  Layout storage_layout;
  Fragment copy_loop_layout;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.HcuGemmBLdsStrategy",
                                    HcuGemmBLdsStrategyNode, Object);

  static void RegisterReflection();
};

class HcuGemmBLdsStrategy : public ObjectRef {
public:
  explicit HcuGemmBLdsStrategy(ObjectPtr<HcuGemmBLdsStrategyNode> ptr)
      : ObjectRef(std::move(ptr)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(
      HcuGemmBLdsStrategy, ObjectRef, HcuGemmBLdsStrategyNode);
};

Optional<HcuGemmBLdsStrategy>
DeriveHcuGemmBLdsStrategy(const CopyNode &copy, const GemmNode &gemm,
                          int block_threads, Target target);

void ValidateHcuGemmBStorageLayout(const Layout &actual,
                                   const HcuGemmBLdsStrategy &strategy);

void ValidateHcuGemmBCopyLayout(const Fragment &actual,
                                const HcuGemmBLdsStrategy &strategy);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_UTILS_GEMM_B_LDS_STRATEGY_H_
