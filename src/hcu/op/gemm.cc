/*!
 * \file hcu/op/gemm.cc
 * \brief HCU implementation for tl.gemm instruction selection and MLS warp
 * partition.
 */

#include "op/gemm.h"
#include "support/check.h"

#include "hcu/target_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace hcu {

constexpr const char *kHCUMMAC = "hcu.mmac";

void ComputeWarpPartitionHCU(const GemmWarpPolicyNode &policy, int M, int N,
                             int K, int k_pack, int element_byte_size,
                             int block_size, Target target, bool A_from_mls,
                             bool B_from_mls, bool A_mls_trans,
                             bool B_mls_trans) {
  bool use_mls = A_from_mls || B_from_mls;
  if (use_mls) {
    ICHECK(k_pack == 1) << "gemm_mls does not support kPack > 1";
  }

  int num_warps = block_size / TargetHcuGetWarpSize(target);
  int m_warp = 1, n_warp = 1, k_warp = 1;
  int kMPerWarp = 16;
  int kNPerWarp = 16;
  if (A_from_mls && !A_mls_trans) {
    kMPerWarp = 32;
  }
  if (B_from_mls && !B_mls_trans) {
    kNPerWarp = 32;
  }
  ICHECK(element_byte_size == 1 || element_byte_size == 2 ||
         element_byte_size == 4)
      << "element byte width=" << element_byte_size;
  int kKPerWarp = k_pack * (32 / element_byte_size);

  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % kNPerWarp == 0)
      << "N must be divisible by " << kNPerWarp << ", but got " << N;
  ICHECK(K % kKPerWarp == 0)
      << "K must be divisible by " << kKPerWarp << ", but got " << K;

  if (policy.IsFullRow()) {
    m_warp = num_warps;
    n_warp = 1;
    if (M % (m_warp * kMPerWarp) != 0) {
      m_warp = M / kMPerWarp;
      n_warp = num_warps / m_warp;
      if (n_warp == 0) {
        n_warp = 1;
      }
    }
  } else if (policy.IsFullCol()) {
    m_warp = 1;
    n_warp = num_warps;
    if (N % (n_warp * kNPerWarp) != 0) {
      int n_warps_no_recompute = N / kNPerWarp;
      m_warp = num_warps / n_warps_no_recompute;
      if (m_warp == 0) {
        m_warp = 1;
      }
      if (M % (m_warp * kMPerWarp) != 0) {
        m_warp = M / kMPerWarp;
      }
      n_warp = num_warps / m_warp;
    }
  } else if (policy.IsSquare()) {
    int max_m_warps = M / kMPerWarp;
    float ideal_ratio = N > 0 ? static_cast<float>(M) / N : 1.0f;

    int best_m = 1;
    int best_n = 1;
    float best_balance = std::numeric_limits<float>::max();
    int max_no_recompute_warps = (M / kMPerWarp) * (N / kNPerWarp);
    max_no_recompute_warps = std::min(max_no_recompute_warps, num_warps);
    for (int m = 1; m <= max_m_warps && m <= max_no_recompute_warps; m++) {
      int n = max_no_recompute_warps / m;

      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * kNPerWarp);
      if (m_per_warp < 1 || n_per_warp < 1) {
        continue;
      }
      if (m * n != max_no_recompute_warps) {
        continue;
      }

      float balance = std::abs(m_per_warp / n_per_warp - ideal_ratio);
      if (balance < best_balance) {
        best_balance = balance;
        best_m = m;
        best_n = n;
      }
    }
    int recompute = num_warps / max_no_recompute_warps;
    m_warp = best_m;
    n_warp = best_n * recompute;
  } else if (policy.IsFullColK()) {
    ICHECK(!use_mls)
        << "gemm_mls does not support warp partitioning on K (FullColK policy)";
    n_warp = num_warps;
    k_warp = 1;
    if (N % (n_warp * kNPerWarp) != 0) {
      n_warp = N / kNPerWarp;
      k_warp = num_warps / n_warp;
      if (k_warp == 0) {
        k_warp = 1;
      }
      ICHECK(K % (k_warp * kKPerWarp) == 0)
          << "K must be divisible by " << k_warp << " * " << kKPerWarp;
    }
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }

  ICHECK(m_warp * n_warp * k_warp == num_warps)
      << "m_warp * n_warp * k_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", k_warp: " << k_warp
      << ", num_warps: " << num_warps;

  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  policy.k_warp = k_warp;
}

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)op;
    (void)block_size;
    ICHECK(TargetIsHCU(target))
        << "HCU gemm implementation requires target=hcu, got " << target;
    return kHCUMMAC;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    (void)gemm_inst;
    int num_warps = block_size / TargetHcuGetWarpSize(target);
    policy.m_warp = 1;
    policy.n_warp = num_warps;
    policy.k_warp = 1;
    return {1, num_warps};
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    (void)gemm_inst;
    return false;
  }
};

} // namespace hcu

namespace {

bool MatchHCUGemmTarget(Target target) { return TargetIsHCU(target); }

bool RegisterHCUGemm() {
  RegisterGemmImpl(GemmImpl{
      "hcu.Gemm",
      MatchHCUGemmTarget,
      hcu::Gemm::SelectInst,
      hcu::Gemm::ComputeWarpPartition,
      hcu::Gemm::ReuseExistingSharedLayout,
  });
  return true;
}

const bool hcu_gemm_registered = RegisterHCUGemm();

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  refl::GlobalDef().def(
      "tl.GemmWarpPolicyComputeWarpPartitionHCU",
      [](GemmWarpPolicy policy, int M, int N, int K, int k_pack,
         int element_byte_size, int block_size, Target target, int gemm_inst,
         bool A_from_mls, bool B_from_mls, bool A_mls_trans, bool B_mls_trans) {
        (void)gemm_inst;
        ICHECK(policy.defined());
        hcu::ComputeWarpPartitionHCU(
            *policy.get(), M, N, K, k_pack, element_byte_size, block_size,
            target, A_from_mls, B_from_mls, A_mls_trans, B_mls_trans);
      });
}

} // namespace tl
} // namespace tvm
