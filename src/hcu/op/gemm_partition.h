/*!
 * \file gemm_partition.h
 * \brief HCU GEMM warp partition helper shared across HCU tile ops.
 */
#ifndef TVM_TL_HCU_OP_GEMM_PARTITION_H_
#define TVM_TL_HCU_OP_GEMM_PARTITION_H_

#include "op/gemm.h"

namespace tvm {
namespace tl {
namespace hcu {

void ComputeWarpPartitionHCU(const GemmWarpPolicyNode &policy, int M, int N,
                             int K, int k_pack, int element_bits,
                             int block_size, Target target, bool A_from_mls,
                             bool B_from_mls, bool A_mls_trans,
                             bool B_mls_trans);

} // namespace hcu
} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_OP_GEMM_PARTITION_H_
