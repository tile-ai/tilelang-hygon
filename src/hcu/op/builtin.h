/*! \file hcu/op/builtin.h \brief HCU-owned intrinsic Ops and attributes. */
#ifndef TVM_TL_HCU_OP_BUILTIN_H_
#define TVM_TL_HCU_OP_BUILTIN_H_

#include "op/builtin.h"

namespace tvm {
namespace tl {

namespace attr {
static constexpr const char *kHcuWdra = "tl.hcu_wdra";
static constexpr const char *kHcuWdraWavesPerTg = "tl.hcu_wdra_waves_per_tg";
static constexpr const char *kHcuScaleBufferSize = "tl.hcu_scale_buffer_size";
static constexpr const char *kHcuBufferCacheSwizzleStride =
    "tl.hcu_buffer_cache_swizzle_stride";
static constexpr const char *kMlsActualSizeBytesMap =
    "tl.hcu.mls_actual_size_bytes_map";
static constexpr const char *kHcuDirectToLds = "tl.hcu.direct_to_lds";
static constexpr const char *kHcuBufferOpsRebaseMap =
    "tl.hcu.buffer_ops_rebase_map";
} // namespace attr

static constexpr const char *kEnableHcuWdra = "tl.enable_hcu_wdra";

TVM_DLL const Op &hcu_cp_async_idxen();
TVM_DLL const Op &ds_read_vector();
TVM_DLL const Op &async_gld_sld_fence();
TVM_DLL const Op &get_wave_id();
TVM_DLL const Op &hcu_wdra_init();
TVM_DLL const Op &abarrier_init();
TVM_DLL const Op &abarrier_inv();
TVM_DLL const Op &abarrier_arrive();
TVM_DLL const Op &abarrier_try_wait();
TVM_DLL const Op &abarrier_wait();
TVM_DLL const Op &abarrier_test_wait();
TVM_DLL const Op &abarrier_seq();
TVM_DLL const Op &abarrier_expect_tx();
TVM_DLL const Op &abarrier_complete_tx();
TVM_DLL const Op &ebarrier_sync();
TVM_DLL const Op &ebarrier_sync_cnt();
TVM_DLL const Op &ebarrier_arrive();
TVM_DLL const Op &hcu_get_lane_idx();
TVM_DLL const Op &hcu_get_wave_idx();
TVM_DLL const Op &hcu_get_wave_idx_sync();
TVM_DLL const Op &hcu_get_wave_group_idx();
TVM_DLL const Op &hcu_set_max_nreg();
TVM_DLL const Op &hcu_ieee_fmaf();
TVM_DLL const Op &hcu_mmac();
TVM_DLL const Op &hcu_mmac_store();
TVM_DLL const Op &hcu_wmma();
TVM_DLL const Op &hcu_wmma_store();

} // namespace tl
} // namespace tvm
#endif
