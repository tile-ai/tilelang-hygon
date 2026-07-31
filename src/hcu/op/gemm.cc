/*!
 * \file hcu/op/gemm.cc
 * \brief HCU implementation for tl.gemm instruction selection and MLS warp
 * partition.
 */

#include "op/gemm.h"
#include "hcu/op/gemm_partition.h"
#include "hcu/op/mls.h"
#include "support/check.h"

#include "hcu/target_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
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

HcuMmacModeInfo ResolveHcuMmacMode(DataType a_dtype, DataType b_dtype,
                                   bool a_is_fragment, bool b_is_fragment,
                                   bool is_blockscaled, int block_k,
                                   ScaleLdsFormat scale_format_a,
                                   ScaleLdsFormat scale_format_b,
                                   Target target) {
  HcuMmacModeInfo info;
  const bool a_fp4 = a_dtype.is_float4();
  const bool b_fp4 = b_dtype.is_float4();
  const bool has_fp4 = a_fp4 || b_fp4;
  const std::string arch = GetHcuArchString(target);
  const bool a_f8_family = a_fp4 || a_dtype.is_float8() || a_dtype.is_float6();
  const bool b_f8_family = b_fp4 || b_dtype.is_float8() || b_dtype.is_float6();
  const bool pure_fp8 = a_dtype.is_float8() && b_dtype.is_float8();
  const bool consider_f8f6f4 =
      ((a_f8_family || b_f8_family) && !pure_fp8) || is_blockscaled;
  const bool a_requires_b8 = a_dtype.is_float4_e2m1_unpacked() ||
                             a_dtype.is_float8() || a_dtype.is_float6();
  const bool b_requires_b8 = b_dtype.is_float4_e2m1_unpacked() ||
                             b_dtype.is_float8() || b_dtype.is_float6();

  bool use_f8f6f4 = false;
  if (consider_f8f6f4) {
    ICHECK(a_f8_family && b_f8_family)
        << "HCU f8f6f4 GEMM requires f8/f6/f4 operands, got A=" << a_dtype
        << " B=" << b_dtype;
    if (a_is_fragment && b_is_fragment) {
      ICHECK_EQ(a_requires_b8, b_requires_b8)
          << "Explicit GEMM fragments have incompatible packed/unpacked "
             "representations: A="
          << a_dtype << " B=" << b_dtype;
      use_f8f6f4 = a_requires_b8;
    } else if (a_is_fragment || b_is_fragment) {
      const bool fragment_b8 = a_is_fragment ? a_requires_b8 : b_requires_b8;
      const bool shared_b8 = a_is_fragment ? b_requires_b8 : a_requires_b8;
      ICHECK(fragment_b8 || !shared_b8)
          << "Packed FP4 fragment cannot consume an unpacked-b8 shared peer "
             "because b8->b4 compression is unsupported";
      use_f8f6f4 = fragment_b8;
    } else {
      const bool mixed_logical_width =
          a_dtype.is_float8() || b_dtype.is_float8() || a_dtype.is_float6() ||
          b_dtype.is_float6();
      const bool packed_fp4_pair =
          a_dtype.is_float4_e2m1fn() && b_dtype.is_float4_e2m1fn();
      const bool scale_requires_k32 =
          is_blockscaled && (ScaleFormatRequiresMmacK32(scale_format_a) ||
                             ScaleFormatRequiresMmacK32(scale_format_b));
      const bool expand_packed_fp4_for_k =
          packed_fp4_pair && (block_k % 64 != 0 || scale_requires_k32);
      ICHECK(!expand_packed_fp4_for_k || block_k % 32 == 0)
          << "Packed FP4 shared GEMM block_K=" << block_k
          << " satisfies neither native FP4 K64 nor f8f6f4 K32";
      // A direct shared/shared GEMM may expand only the packed FP4 side via
      // ds_read_format.  The MMAC-facing register representations still
      // converge to b8 on both sides; b8->b4 compression is never inferred.
      use_f8f6f4 = mixed_logical_width || a_requires_b8 || b_requires_b8 ||
                   arch == "gfx92a" || expand_packed_fp4_for_k;
    }
  }

  info.mode =
      use_f8f6f4 ? HcuMmacOperandMode::kF8F6F4 : HcuMmacOperandMode::kNative;
  info.element_bits = use_f8f6f4 ? 8 : a_dtype.bits();
  info.mmac_k = use_f8f6f4 ? 32 : (has_fp4 ? 64 : 256 / info.element_bits);
  if (use_f8f6f4) {
    auto type_id = [](DataType dtype) {
      if (dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn() ||
          dtype.is_float8_e4m3fnuz())
        return 0;
      if (dtype.is_float8_e5m2() || dtype.is_float8_e5m2fnuz())
        return 1;
      if (dtype.is_float6_e2m3fn())
        return 2;
      if (dtype.is_float6_e3m2fn())
        return 3;
      ICHECK(dtype.is_float4()) << "Unsupported f8f6f4 dtype " << dtype;
      return 4;
    };
    info.real_ab_type = type_id(a_dtype) * 5 + type_id(b_dtype);
  }
  return info;
}

HcuMnPerWarp ResolveHcuMnPerWarp(int element_bits, bool A_from_mls,
                                 bool B_from_mls, bool A_mls_trans,
                                 bool B_mls_trans, int extra_min_m_per_warp,
                                 int extra_min_n_per_warp) {
  // MLS non-trans ds_read_format only has MN=32 tiles; non-b4 trans can use 16.
  // gfx946 b4 no-pad: B MLS always needs N floor 32 (trans and non-trans).
  HcuMnPerWarp floors;
  floors.m_per_warp = (A_from_mls && !A_mls_trans) ? 32 : 16;
  floors.n_per_warp =
      (B_from_mls && (element_bits == 4 || !B_mls_trans)) ? 32 : 16;
  if (extra_min_m_per_warp > 0) {
    floors.m_per_warp = std::max(floors.m_per_warp, extra_min_m_per_warp);
  }
  if (extra_min_n_per_warp > 0) {
    floors.n_per_warp = std::max(floors.n_per_warp, extra_min_n_per_warp);
  }
  return floors;
}

HcuMnPerWarp ComputeWarpPartitionHCU(const GemmWarpPolicyNode &policy, int M,
                                     int N, int K, int k_pack, int element_bits,
                                     int block_size, Target target,
                                     bool A_from_mls, bool B_from_mls,
                                     bool A_mls_trans, bool B_mls_trans,
                                     int extra_min_m_per_warp,
                                     int extra_min_n_per_warp) {
  bool use_mls = A_from_mls || B_from_mls;
  if (use_mls) {
    ICHECK(k_pack == 1) << "gemm_mls does not support kPack > 1";
  }

  int num_warps = block_size / TargetHcuGetWarpSize(target);
  int m_warp = 1, n_warp = 1, k_warp = 1;
  HcuMnPerWarp floors = ResolveHcuMnPerWarp(
      element_bits, A_from_mls, B_from_mls, A_mls_trans, B_mls_trans,
      extra_min_m_per_warp, extra_min_n_per_warp);
  const int kMPerWarp = floors.m_per_warp;
  const int kNPerWarp = floors.n_per_warp;
  ICHECK(element_bits == 4 || element_bits == 8 || element_bits == 16 ||
         element_bits == 32)
      << "element bitwidth=" << element_bits;
  int kKPerWarp = k_pack * (256 / element_bits);

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
  return floors;
}

int ScaleFormatMinMnPerWarp(ScaleLdsFormat scale_format) {
  if (scale_format == ScaleLdsFormat::kK2MN2Interleave ||
      scale_format == ScaleLdsFormat::kMN2Interleave) {
    return 32; // K2MN2 / MN2
  }
  if (scale_format == ScaleLdsFormat::kMN4Interleave) {
    return 64; // MN4
  }
  return 0; // identity / K2 / K4
}

ScaleWarpSeg ComputeScaleWarpSeg(const GemmWarpPolicyNode &policy, int M, int N,
                                 int K, int k_pack, int element_bits,
                                 int block_size, Target target, bool A_from_mls,
                                 bool B_from_mls, bool A_mls_trans,
                                 bool B_mls_trans, int extra_min_m_per_warp,
                                 int extra_min_n_per_warp) {
  HcuMnPerWarp floors = ComputeWarpPartitionHCU(
      policy, M, N, K, k_pack, element_bits, block_size, target, A_from_mls,
      B_from_mls, A_mls_trans, B_mls_trans, extra_min_m_per_warp,
      extra_min_n_per_warp);
  ScaleWarpSeg seg;
  seg.total_warps = block_size / TargetHcuGetWarpSize(target);
  seg.k_warp = policy.k_warp;
  seg.m_seg = policy.m_warp;
  seg.m_per_warp = floors.m_per_warp;
  seg.n_per_warp = floors.n_per_warp;
  // Match emitter ``block_col_warps_no_recompute``.
  seg.n_seg = std::min(policy.n_warp, N / floors.n_per_warp);
  if (seg.n_seg < 1) {
    seg.n_seg = 1;
  }
  return seg;
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
      "tl.ResolveHcuMmacMode",
      [](DataType a_dtype, DataType b_dtype, bool a_is_fragment,
         bool b_is_fragment, bool is_blockscaled, int block_k,
         int scale_format_a, int scale_format_b, Target target) {
        hcu::HcuMmacModeInfo info = hcu::ResolveHcuMmacMode(
            a_dtype, b_dtype, a_is_fragment, b_is_fragment, is_blockscaled,
            block_k, static_cast<ScaleLdsFormat>(scale_format_a),
            static_cast<ScaleLdsFormat>(scale_format_b), target);
        return Array<Integer>{Integer(static_cast<int>(info.mode)),
                              Integer(info.element_bits), Integer(info.mmac_k),
                              Integer(info.real_ab_type)};
      });
  refl::GlobalDef().def(
      "tl.GemmWarpPolicyComputeWarpPartitionHCU",
      [](GemmWarpPolicy policy, int M, int N, int K, int k_pack,
         int element_bits, int block_size, Target target, int gemm_inst,
         bool A_from_mls, bool B_from_mls, bool A_mls_trans, bool B_mls_trans,
         int extra_min_m_per_warp, int extra_min_n_per_warp) {
        (void)gemm_inst;
        ICHECK(policy.defined());
        hcu::HcuMnPerWarp floors = hcu::ComputeWarpPartitionHCU(
            *policy.get(), M, N, K, k_pack, element_bits, block_size, target,
            A_from_mls, B_from_mls, A_mls_trans, B_mls_trans,
            extra_min_m_per_warp, extra_min_n_per_warp);
        return Array<Integer>{Integer(floors.m_per_warp),
                              Integer(floors.n_per_warp)};
      });
}

} // namespace tl
} // namespace tvm
