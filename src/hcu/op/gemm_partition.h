/*!
 * \file gemm_partition.h
 * \brief HCU GEMM warp partition helper shared across HCU tile ops.
 */
#ifndef TVM_TL_HCU_OP_GEMM_PARTITION_H_
#define TVM_TL_HCU_OP_GEMM_PARTITION_H_

#include "hcu/op/scale_format.h"
#include "op/gemm.h"

namespace tvm {
namespace tl {
namespace hcu {

/// Effective per-wave MN floors after MLS + optional extra (scale) max.
struct HcuMnPerWarp {
  int m_per_warp{16};
  int n_per_warp{16};
};

enum class HcuMmacOperandMode : int {
  kNative = 0,
  kF8F6F4 = 1,
};

/// Consumer operand representation selected from logical A/B dtypes and the
/// physical representation produced by MLS.  This is the single source of
/// truth shared by GEMM lowering and scale-copy dependency annotation.
struct HcuMmacModeInfo {
  HcuMmacOperandMode mode{HcuMmacOperandMode::kNative};
  int element_bits{0};
  int mmac_k{0};
  int real_ab_type{-1};
};

HcuMmacModeInfo ResolveHcuMmacMode(DataType a_dtype, DataType b_dtype,
                                   bool a_is_fragment, bool b_is_fragment,
                                   bool is_blockscaled, int block_k,
                                   ScaleLdsFormat scale_format_a,
                                   ScaleLdsFormat scale_format_b,
                                   Target target);

/// Resolve MN floors used by warp search / fragment / n_seg.
/// ``extra_min_*`` are additional floors (0 = none), e.g. ScaleFormat atoms.
HcuMnPerWarp ResolveHcuMnPerWarp(int element_bits, bool A_from_mls,
                                 bool B_from_mls, bool A_mls_trans,
                                 bool B_mls_trans, int extra_min_m_per_warp = 0,
                                 int extra_min_n_per_warp = 0);

/// Warp partition. Writes ``policy.m/n/k_warp`` and returns effective MN floors
/// (same values ResolveHcuMnPerWarp would produce for the same args).
HcuMnPerWarp ComputeWarpPartitionHCU(const GemmWarpPolicyNode &policy, int M,
                                     int N, int K, int k_pack, int element_bits,
                                     int block_size, Target target,
                                     bool A_from_mls, bool B_from_mls,
                                     bool A_mls_trans, bool B_mls_trans,
                                     int extra_min_m_per_warp = 0,
                                     int extra_min_n_per_warp = 0);

/// Minimum MN extent required to keep one ScaleFormat atom inside a segment.
int ScaleFormatMinMnPerWarp(ScaleLdsFormat scale_format);

/// Non-recompute MN segments for scale_buffer partition.
/// ``m_seg`` = policy.m_warp; ``n_seg`` = min(n_warp, N/n_per_warp).
struct ScaleWarpSeg {
  int m_seg{1};
  int n_seg{1};
  int total_warps{1};
  int k_warp{1};
  int m_per_warp{16};
  int n_per_warp{16};
};

ScaleWarpSeg ComputeScaleWarpSeg(const GemmWarpPolicyNode &policy, int M, int N,
                                 int K, int k_pack, int element_bits,
                                 int block_size, Target target, bool A_from_mls,
                                 bool B_from_mls, bool A_mls_trans,
                                 bool B_mls_trans, int extra_min_m_per_warp = 0,
                                 int extra_min_n_per_warp = 0);

} // namespace hcu
} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_OP_GEMM_PARTITION_H_
