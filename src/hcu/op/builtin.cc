/*! \file hcu/op/builtin.cc \brief Registration of HCU-owned intrinsic Ops. */
#include "builtin.h"
#include <tvm/ir/transform.h>

#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {
using namespace tirx;

TVM_REGISTER_PASS_CONFIG_OPTION(kEnableHcuWdra, Bool);

#define TIR_DEFINE_HCU_BUILTIN(Name, Inputs, Effect)                           \
  const Op &Name() {                                                           \
    static const Op &op = Op::Get("tl." #Name);                                \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl." #Name)                                                 \
      .set_num_inputs(Inputs)                                                  \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #Name)               \
      .set_attr<TCallEffectKind>("TCallEffectKind",                            \
                                 Integer(CallEffectKind::Effect))

TIR_DEFINE_HCU_BUILTIN(hcu_cp_async_idxen, 6, kOpaque);
TIR_DEFINE_HCU_BUILTIN(ds_read_vector, 3, kOpaque);
TIR_DEFINE_HCU_BUILTIN(async_gld_sld_fence, 1, kOpaque);
TIR_DEFINE_HCU_BUILTIN(get_wave_id, 0, kPure);
TIR_DEFINE_HCU_BUILTIN(hcu_wdra_init, -1, kOpaque);
TIR_DEFINE_HCU_BUILTIN(abarrier_init, 2, kOpaque);
TIR_DEFINE_HCU_BUILTIN(abarrier_inv, 1, kOpaque);
TIR_DEFINE_HCU_BUILTIN(abarrier_arrive, 2, kOpaque);
TIR_DEFINE_HCU_BUILTIN(abarrier_try_wait, 2, kOpaque);
TIR_DEFINE_HCU_BUILTIN(abarrier_wait, 2, kOpaque);
TIR_DEFINE_HCU_BUILTIN(abarrier_test_wait, 2, kOpaque);
TIR_DEFINE_HCU_BUILTIN(abarrier_seq, 1, kOpaque);
TIR_DEFINE_HCU_BUILTIN(abarrier_expect_tx, 2, kOpaque);
TIR_DEFINE_HCU_BUILTIN(abarrier_complete_tx, 2, kOpaque);
TIR_DEFINE_HCU_BUILTIN(ebarrier_sync, 1, kOpaque);
TIR_DEFINE_HCU_BUILTIN(ebarrier_sync_cnt, 2, kOpaque);
TIR_DEFINE_HCU_BUILTIN(ebarrier_arrive, 2, kOpaque);
TIR_DEFINE_HCU_BUILTIN(hcu_get_lane_idx, -1, kPure);
TIR_DEFINE_HCU_BUILTIN(hcu_get_wave_idx, -1, kPure);
TIR_DEFINE_HCU_BUILTIN(hcu_get_wave_idx_sync, -1, kPure);
TIR_DEFINE_HCU_BUILTIN(hcu_get_wave_group_idx, -1, kPure);
TIR_DEFINE_HCU_BUILTIN(hcu_set_max_nreg, 2, kOpaque);
TIR_DEFINE_HCU_BUILTIN(hcu_ieee_fmaf, 4, kPure);
TIR_DEFINE_HCU_BUILTIN(hcu_mmac, -1, kOpaque);
TIR_DEFINE_HCU_BUILTIN(hcu_mmac_store, -1, kOpaque);
TIR_DEFINE_HCU_BUILTIN(hcu_wmma, -1, kOpaque);
TIR_DEFINE_HCU_BUILTIN(hcu_wmma_store, -1, kOpaque);

#undef TIR_DEFINE_HCU_BUILTIN
} // namespace tl
} // namespace tvm
