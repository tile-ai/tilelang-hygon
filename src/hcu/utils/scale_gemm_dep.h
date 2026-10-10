// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file scale_gemm_dep.h
 * \brief Annotation keys for HCU block-scaled GEMM / copy_scale dependency.
 */

#ifndef TVM_TL_HCU_UTILS_SCALE_GEMM_DEP_H_
#define TVM_TL_HCU_UTILS_SCALE_GEMM_DEP_H_

namespace tvm {
namespace tl {

namespace attr {
static constexpr const char *kScaleGemmDep = "tl.hcu.scale_gemm_dep";
static constexpr const char *kScaleRowBase = "tl.hcu.scale_row_base";
static constexpr const char *kScaleARowBase = "tl.hcu.scale_a_row_base";
static constexpr const char *kScaleBRowBase = "tl.hcu.scale_b_row_base";
static constexpr const char *kScaleAFormat = "tl.hcu.scale_a_format";
static constexpr const char *kScaleBFormat = "tl.hcu.scale_b_format";
static constexpr const char *kScaleGranularityMN =
    "tl.hcu.scale_granularity_mn";
static constexpr const char *kScaleGranularityK = "tl.hcu.scale_granularity_k";
static constexpr const char *kScaleKMajor = "tl.hcu.scale_k_major";
static constexpr const char *kScaleRole = "tl.hcu.scale_role";
static constexpr const char *kHcuBlockscaled = "tl.hcu.blockscaled";
// Gemm clues for deferred ComputeWarpPartitionHCU (not precomputed WarpM/N).
static constexpr const char *kScaleGemmM = "tl.hcu.scale_gemm_m";
static constexpr const char *kScaleGemmN = "tl.hcu.scale_gemm_n";
static constexpr const char *kScaleGemmK = "tl.hcu.scale_gemm_k";
static constexpr const char *kScaleGemmPolicy = "tl.hcu.scale_gemm_policy";
static constexpr const char *kScaleGemmKPack = "tl.hcu.scale_gemm_k_pack";
static constexpr const char *kScaleGemmElemBits = "tl.hcu.scale_gemm_elem_bits";
static constexpr const char *kScaleAFromMls = "tl.hcu.scale_a_from_mls";
static constexpr const char *kScaleBFromMls = "tl.hcu.scale_b_from_mls";
static constexpr const char *kScaleAMlsTrans = "tl.hcu.scale_a_mls_trans";
static constexpr const char *kScaleBMlsTrans = "tl.hcu.scale_b_mls_trans";
// Aggregated ScaleFormat MN-atom floors for ComputeWarpPartitionHCU.
static constexpr const char *kScaleMinMPerWarp = "tl.hcu.scale_min_m_per_warp";
static constexpr const char *kScaleMinNPerWarp = "tl.hcu.scale_min_n_per_warp";
} // namespace attr

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_UTILS_SCALE_GEMM_DEP_H_
