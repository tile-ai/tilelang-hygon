/*!
 * \file tl/op/gemm.cc
 * \brief Implementation of General Matrix Multiplication (GEMM) operators
 */

#include "gemm.h"
#include "propagation_util.h"

#include "builtin.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/transform.h>
#include <tuple>

#include "../target/utils.h"
#include "tcgen5_meta.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

// HCU gemm_mls template dtype strings 
static inline bool IsFloat8E4Family(DataType dtype) {
  return dtype.is_float8_e4m3fn() || dtype.is_float8_e4m3();
}

static inline bool IsFloat8E5Family(DataType dtype) {
  return dtype.is_float8_e5m2() || dtype.is_float8_e5m2fnuz();
}

/**
 * @brief Construct a Gemm operator from serialized TL arguments and a buffer
 * map.
 *
 * This constructor deserializes operator parameters from `args` and resolves
 * buffer references via `vmap`, populating an internal GemmNode with:
 * - device pointers for A, B, C and their corresponding Buffer objects,
 * - transpose flags for A and B,
 * - matrix dimensions M, N, K,
 * - warp allocation policy and clear_accum flag,
 * - strides and memory offsets for A and B,
 * - optional kPack (must be 1 or 2) and optional wg_wait.
 *
 * The populated GemmNode is stored into the wrapper's internal `data_`.
 *
 * @param args Positional serialized arguments produced by the TL frontend:
 *   expected layout is:
 *     [Aptr, Bptr, Cptr, trans_A (Bool), trans_B (Bool),
 *      M (Int), N (Int), K (Int), policy (Int), clear_accum (Bool),
 *      stride_A (Int), stride_B (Int), offset_A (Int), offset_B (Int),
 *      (optional) kPack (Int), (optional) wg_wait (Int)]
 *
 * @note If `kPack` is provided it must be 1; otherwise the constructor
 *       fails with an ICHECK (runtime assertion). No other validation is
 *       performed here.
 */
// NormalizeToBufferRegion moved to src/op/utils.{h,cc}

// MakeAccessPtrFromRegion moved to src/op/utils.{h,cc}

Gemm::Gemm(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<GemmNode> node = tvm::ffi::make_object<GemmNode>();

  node->aRegion_ = NormalizeToBufferRegion(args[0]);
  node->bRegion_ = NormalizeToBufferRegion(args[1]);
  node->cRegion_ = NormalizeToBufferRegion(args[2]);

  node->a_ = node->aRegion_->buffer;
  node->b_ = node->bRegion_->buffer;
  node->c_ = node->cRegion_->buffer;
  node->transA_ = args[3].as<Bool>().value();
  node->transB_ = args[4].as<Bool>().value();
  node->m_ = args[5].as<IntImm>().value()->value;
  node->n_ = args[6].as<IntImm>().value()->value;
  node->k_ = args[7].as<IntImm>().value()->value;
  node->policy_ = GemmWarpPolicy(args[8].as<IntImm>().value()->value);
  node->clearAccum_ = args[9].as<PrimExpr>().value();
  node->strideA_ = args[10].as<IntImm>().value()->value;
  node->strideB_ = args[11].as<IntImm>().value()->value;
  node->offsetA_ = args[12].as<IntImm>().value()->value;
  node->offsetB_ = args[13].as<IntImm>().value()->value;
  if (args.size() > 14) {
    node->kPack_ = args[14].as<IntImm>().value()->value;
    if (node->kPack_ != 1 && node->kPack_ != 2) {
      ICHECK(false) << "kPack must be 1 or 2";
    }
  }
  if (args.size() > 15) {
    node->wgWait_ = args[15].as<IntImm>().value()->value;
  }
  if (args.size() > 16) {
    if (const auto *load = args[16].as<BufferLoadNode>()) {
      node->mbarRegion_ =
          NormalizeToBufferRegion(Downcast<BufferLoad>(args[16]));
      node->mbar_ = node->mbarRegion_->buffer;
    } else {
      node->mbar_ = std::nullopt;
    }
  }
  node->cCoords_ = Array<PrimExpr>(
      {args[17].as<PrimExpr>().value(), args[18].as<PrimExpr>().value()});
  data_ = std::move(node);
}

/**
 * @brief Create a copy of this GemmNode as a TileOperator.
 *
 * Constructs a new GemmNode by copying the current node state and returns it
 * wrapped in a Gemm TileOperator.
 *
 * @return TileOperator A Gemm operator that owns a copy of this node.
 */
TileOperator GemmNode::Clone() const {
  auto op = tvm::ffi::make_object<GemmNode>(*this);
  return Gemm(op);
}

bool GemmNode::allowTcgen5Mma(Target target) const {
  return TargetIsSm100(target) &&
         ((a_.scope() == "shared.dyn" || a_.scope() == "shared" ||
           a_.scope() == "shared.tmem") &&
          (b_.scope() == "shared.dyn" || b_.scope() == "shared") &&
          c_.scope() == "shared.tmem") &&
         GetTCGEN5MMAMeta(m_, n_, k_, a_->dtype, c_->dtype).first;
}

bool GemmNode::allowWgmma(int block_size, Target target) const {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();

  int warp_size = TargetGetWarpSize(target);
  int num_warps = block_size / warp_size;
  return !ctxt->GetConfig(kDisableWGMMA, Optional<Bool>()).value_or(false) &&
         TargetIsHopper(target) && (this->m_ >= 64) && (num_warps % 4 == 0) &&
         checkWgmma();
}

GemmInst GemmNode::getGemmInst(int block_size, Target target) const {
  if (allowTcgen5Mma(target)) {
    return GemmInst::kTCGEN5MMA;
  } else if (allowWgmma(block_size, target)) {
    return GemmInst::kWGMMA;
  } else if (TargetIsCDNA(target)) {
    return GemmInst::kMFMA;
  } else if (TargetIsCuda(target)) {
    return GemmInst::kMMA;
  } else {
    ICHECK(0) << "Unsupported target for gemm: " << target;
    return GemmInst::kMMA;
  }
}

std::tuple<int, int, int> GemmWarpPolicyNode::computeWarpPartitionHCU(
    int M, int N, int K, int k_pack, int element_byte_size, int block_size,
    Target target, GemmInst gemm_inst, bool A_from_mls, bool B_from_mls,
    bool A_mls_trans, bool B_mls_trans) const {
  bool use_mls = A_from_mls || B_from_mls;
  if (use_mls) {
    ICHECK(k_pack == 1) << "gemm_mls does not support kPack > 1";
  }

  int num_warps = block_size / TargetGetWarpSize(target);
  int m_warp = 1, n_warp = 1, k_warp = 1;
  int kMPerWarp = 16;  // Rows processed by a single warp
  int kNPerWarp = 16;   // Columns processed by a single warp
  if (A_from_mls && !A_mls_trans) {
    kMPerWarp = 32;  // min ds_read_format tilesize
  }
  if (B_from_mls && !B_mls_trans) {
    kNPerWarp = 32;  // min ds_read_format tilesize
  }
  ICHECK(element_byte_size == 1 || element_byte_size == 2 || element_byte_size == 4)
      << "element bitwidth=" << element_byte_size;
  int kKPerWarp = k_pack * (32 / element_byte_size);

  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % kNPerWarp == 0)
      << "N must be divisible by " << kNPerWarp << ", but got " << N;
  ICHECK(K % kKPerWarp == 0)
      << "K must be divisible by " << kKPerWarp << ", but got " << K;

  if (this->isFullRow()) {
    // Try to partition M first
    m_warp = num_warps;
    n_warp = 1;

    // If M cannot be evenly divided by m_warp*16, try to split remaining warps
    // to N
    if (M % (m_warp * kMPerWarp) != 0) {
      // Calculate how many warps we can use for M
      m_warp = M / kMPerWarp;
      // Use remaining warps for N, warps for n may has recompute
      n_warp = num_warps / m_warp;
      if (n_warp == 0)
        n_warp = 1;
    }
  } else if (this->isFullCol()) {
    // Try to partition N first
    m_warp = 1;
    n_warp = num_warps;

    // If N cannot be evenly divided by n_warp, try to split remaining warps
    // to M
    if (N % (n_warp * kNPerWarp) != 0) {
      // Calculate how many warps we can use for N
      int n_warps_no_recompute = N / kNPerWarp;
      // Use remaining warps for M
      m_warp = num_warps / n_warps_no_recompute;
      if (m_warp == 0)
          m_warp = 1;
      if (M % (m_warp * kMPerWarp) != 0) {
        // let warps is just enough for M
        m_warp = M / kMPerWarp;
      }
      // warps for n may has recompute
      n_warp = num_warps / m_warp;
    }
  } else if (this->isSquare()) {
    // First calculate the maximum possible warps for each dimension
    int max_m_warps =
        M / kMPerWarp; // Each warp needs at least 16 elements in M

    // Calculate the ideal ratio of M/N warps based on the matrix dimensions
    float ideal_ratio = 1.0f;
    if (N > 0) {
      ideal_ratio = static_cast<float>(M) / N;
    }

    // Try to find the best balanced partition
    int best_m = 1;
    int best_n = 1;
    float best_balance = std::numeric_limits<float>::max();
    int max_no_recompute_warps = (M / kMPerWarp) * (N / kNPerWarp);
    max_no_recompute_warps = std::min(max_no_recompute_warps, num_warps);
    // Try all possible combinations that satisfy the constraints
    for (int m = 1; m <= max_m_warps && m <= max_no_recompute_warps; m++) {
      int n = max_no_recompute_warps / m;

      // Calculate how balanced this partition is
      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * kNPerWarp);
      // m_per_warp and n_per_warp must be greater than 1
      if (m_per_warp < 1 || n_per_warp < 1)
        continue;
      // m * n must equal num_warps
      if (m * n != max_no_recompute_warps)
        continue;

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
  } else if (this->isFullColK()) {
    ICHECK(!use_mls) << "gemm_mls does not support warp partitioning on K (FullColK policy)";
    // Try to partition N first
    n_warp = num_warps;
    k_warp = 1;

    // if num_warps is larger than n_warp needed, try to split remaining warps to N
    if (N % (n_warp * kNPerWarp) != 0) {
      // Calculate how many warps we can use for K
      n_warp = N / kNPerWarp;
      // Use remaining warps for N
      k_warp = num_warps / n_warp;
      if (k_warp == 0)
          k_warp = 1;
      // if the remaining warps is too large for K, not support recompute for now.
      ICHECK(K % (k_warp * kKPerWarp) == 0) << "K must be divisible by " << k_warp << " * " << kKPerWarp;
    }
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }
  ICHECK(m_warp * n_warp * k_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", k_warp: " << k_warp << ", num_warps: " << num_warps;

  // Store the computed values in the object's member variables
  this->m_warp = m_warp;
  this->n_warp = n_warp;
  this->k_warp = k_warp;

  return std::make_tuple(m_warp, n_warp, k_warp);
}

std::pair<int, int> GemmWarpPolicyNode::computeWarpPartition(
    int M, int N, int block_size, Target target, GemmInst gemm_inst) const {
  int num_warps = block_size / TargetGetWarpSize(target);
  if (gemm_inst == GemmInst::kTCGEN5MMA) {
    return {1, num_warps}; // TCGEN5MMA doesn't care about warp partitioning
  }

  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16; // Rows processed by a single warp
  int kNPerWarp = 8;            // Columns processed by a single warp
  if (TargetIsVolta(target)) {
    kNPerWarp = 16;
  } else if (TargetIsCDNA(target)) {
    kNPerWarp = 16;
  }
  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % kNPerWarp == 0)
      << "N must be divisible by " << kNPerWarp << ", but got " << N;

  if (gemm_inst == GemmInst::kWGMMA) {
    ICHECK(num_warps % 4 == 0) << "Warp-Group MMA requires 128×k threads.";

    constexpr int kGroup = 4; // Number of warps in a warp-group

    m_warp = kGroup; // Initially, only one warp-group on M dimension
    n_warp = num_warps / m_warp; // Rest all on N dimension

    if (this->isFullRow()) {
      // Try to put as many warp-groups as possible on M dimension
      // (decreasing multiples of 4, ensuring divisibility by M)
      for (int cand = num_warps; cand >= kGroup; cand -= kGroup) {
        if (M % (cand * kMPerWarp) == 0) {
          m_warp = cand;
          n_warp = num_warps / m_warp;
          break;
        }
      }
    } else if (this->isFullCol()) {
      // Try to use warps on N dimension; if N is not divisible, split excess
      // groups to M
      int cand_n = n_warp;                 // Initially assume all on N
      if (N % (cand_n * kNPerWarp) != 0) { // N direction division fails
        int max_n = N / kNPerWarp;
        // Find a feasible n_warp from max possible downwards, ensuring
        // num_warps/n_warp is multiple of 4
        for (int n = std::min(cand_n, max_n); n >= 1; --n) {
          if (num_warps % n == 0 && (num_warps / n) % kGroup == 0) {
            n_warp = n;
            m_warp = num_warps / n_warp;
            break;
          }
        }
      }
    } else if (this->isSquare()) {
      // Exhaustive search, but m must be multiple of 4
      int max_m = M / kMPerWarp;
      int max_n = N / kNPerWarp;

      float ideal = N > 0 ? static_cast<float>(M) / N : 1.f;

      float best_score = std::numeric_limits<float>::max();
      int best_m = kGroup, best_n = n_warp;

      for (int m = kGroup; m <= num_warps && m <= max_m; m += kGroup) {
        if (num_warps % m)
          continue;
        int n = num_warps / m;
        if (n > max_n)
          continue;

        float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
        float n_per_warp = static_cast<float>(N) / (n * kNPerWarp);
        float score = std::abs(m_per_warp / n_per_warp - ideal);

        if (score < best_score) {
          best_score = score;
          best_m = m;
          best_n = n;
        }
      }
      m_warp = best_m;
      n_warp = best_n;
    } else {
      ICHECK(0) << "Unknown GemmWarpPolicy";
    }

    ICHECK(m_warp * n_warp == num_warps)
        << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
        << ", n_warp: " << n_warp << ", num_warps: " << num_warps;

    // Store the computed values in the object's member variables
    this->m_warp = m_warp;
    this->n_warp = n_warp;

    return {m_warp, n_warp};
  }

  if (this->isFullRow()) {
    // Try to partition M first
    m_warp = num_warps;
    n_warp = 1;

    // If M cannot be evenly divided by m_warp*16, try to split remaining warps
    // to N
    if (M % (m_warp * kMPerWarp) != 0) {
      // Calculate how many warps we can use for M
      int max_m_warps = M / kMPerWarp;
      m_warp = max_m_warps;
      // Use remaining warps for N
      n_warp = num_warps / m_warp;
      if (n_warp == 0)
        n_warp = 1;
    }
  } else if (this->isFullCol()) {
    // Try to partition N first
    m_warp = 1;
    n_warp = num_warps;

    // If N cannot be evenly divided by n_warp*8, try to split remaining warps
    // to M
    if (N % (n_warp * kNPerWarp) != 0) {
      // Calculate how many warps we can use for N
      int max_n_warps = N / kNPerWarp;
      n_warp = max_n_warps;
      // Use remaining warps for M
      m_warp = num_warps / n_warp;
      if (m_warp == 0)
        m_warp = 1;
    }
  } else if (this->isSquare()) {
    // First calculate the maximum possible warps for each dimension
    int max_m_warps =
        M / kMPerWarp; // Each warp needs at least 16 elements in M

    // Calculate the ideal ratio of M/N warps based on the matrix dimensions
    float ideal_ratio = 1.0f;
    if (N > 0) {
      ideal_ratio = static_cast<float>(M) / N;
    }

    // Try to find the best balanced partition
    int best_m = 1;
    int best_n = 1;
    float best_balance = std::numeric_limits<float>::max();
    // Try all possible combinations that satisfy the constraints
    for (int m = 1; m <= max_m_warps && m <= num_warps; m++) {
      int n = num_warps / m;

      // Calculate how balanced this partition is
      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * kNPerWarp);
      // m_per_warp and n_per_warp must be greater than 1
      if (m_per_warp < 1 || n_per_warp < 1)
        continue;
      // m * n must equal num_warps
      if (m * n != num_warps)
        continue;

      float balance = std::abs(m_per_warp / n_per_warp - ideal_ratio);

      if (balance < best_balance) {
        best_balance = balance;
        best_m = m;
        best_n = n;
      }
    }

    m_warp = best_m;
    n_warp = best_n;
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }
  ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;

  // Store the computed values in the object's member variables
  this->m_warp = m_warp;
  this->n_warp = n_warp;

  return {m_warp, n_warp};
}

/**
 * @brief Checks whether WGMMA (warp-group MMA) can be used for this GEMM.
 *
 * Evaluates device-memory placement, data-type combinations, transpose flags,
 * and K divisibility constraints required for the Hopper WGMMA code path.
 *
 * The check returns true only when:
 * - B resides in shared memory ("shared" or "shared.dyn"); and
 * - (C, A, B) dtypes match one of the supported combinations below and K
 *   satisfies the required alignment; and
 * - for combinations that require specific orientations, A is not transposed
 *   and B is transposed.
 *
 * Supported combinations and constraints:
 * - C=float16:
 *   - A=float16, B=float16: K % 16 == 0
 *   - Various float8 mixes (e4m3/e5m2): require (!trans_A && trans_B) and K %
 * 32 == 0
 * - C=float32:
 *   - A=float16, B=float16: K % 16 == 0
 *   - A=bfloat16, B=bfloat16: K % 16 == 0
 *   - A=float32, B=float32: require (!trans_A && trans_B) and K % 8 == 0
 *   - Various float8 mixes: require (!trans_A && trans_B) and K % 32 == 0
 * - C=int32:
 *   - 8-bit integer combinations (Int8/UInt8): require (!trans_A && trans_B)
 * and K % 32 == 0
 *
 * @return true if WGMMA is supported for the current buffers, dtypes, and
 *         transpose/shape constraints; false otherwise.
 */
bool GemmNode::checkWgmma() const {
  if (b_.scope() != "shared.dyn" && b_.scope() != "shared") {
    return false;
  }

  if (c_->dtype == DataType::Float(16)) {
    if (a_->dtype == DataType::Float(16) && b_->dtype == DataType::Float(16))
      return k_ % 16 == 0;
    else if (a_->dtype.is_float8() && b_->dtype.is_float8())
      return (!transA_) && transB_ && k_ % 32 == 0;
    else
      return false;
  } else if (c_->dtype == DataType::Float(32)) {
    if (a_->dtype == DataType::Float(16) && b_->dtype == DataType::Float(16))
      return k_ % 16 == 0;
    else if (a_->dtype == DataType::BFloat(16) &&
             b_->dtype == DataType::BFloat(16))
      return k_ % 16 == 0;
    else if (a_->dtype == DataType::Float(32) &&
             b_->dtype == DataType::Float(32))
      return (!transA_) && transB_ && k_ % 8 == 0;
    else if (a_->dtype.is_float8() && b_->dtype.is_float8())
      return (!transA_) && transB_ && k_ % 32 == 0;
    else
      return false;
  } else if (c_->dtype == DataType::Int(32)) {
    if (a_->dtype == DataType::Int(8) && b_->dtype == DataType::Int(8))
      return (!transA_) && transB_ && k_ % 32 == 0;
    else if (a_->dtype == DataType::Int(8) && b_->dtype == DataType::UInt(8))
      return (!transA_) && transB_ && k_ % 32 == 0;
    else if (a_->dtype == DataType::UInt(8) && b_->dtype == DataType::Int(8))
      return (!transA_) && transB_ && k_ % 32 == 0;
    else if (a_->dtype == DataType::UInt(8) && b_->dtype == DataType::UInt(8))
      return (!transA_) && transB_ && k_ % 32 == 0;
    else
      return false;
  } else {
    return false;
  }
}

/**
 * @brief Parse and return the numeric GPU architecture from a Target's "arch"
 * attribute.
 *
 * Examines the target's "arch" string and, if it matches the pattern
 * "sm_<num>", returns <num> as an int. If the attribute is present but does not
 * match that pattern, returns 0.
 *
 * Preconditions: the target must have an "arch" attribute (this is checked via
 * ICHECK).
 *
 * @return int The parsed architecture number (e.g., 80 for "sm_80"), or 0 if
 * the arch string does not match "sm_<num>".
 */
static int GetArchInt(Target target) {
  int arch_int = 0;
  auto s = target->GetAttr<tvm::ffi::String>("arch");
  ICHECK(s.has_value());
  std::string arch = s.value();
  if (arch.rfind("sm_", 0) == 0) {
    arch_int = std::stoi(arch.substr(3));
  } else {
    arch_int = 0;
  }
  return arch_int;
}

/**
 * @brief Lower the GEMM operator to a TL TIR call expression.
 *
 * Constructs a tl::gemm call string parameterized by M, N, K, warp partition,
 * transpose flags, accumulation clearing, target-specific stride/offset/kPack
 * and optional workgroup wait value, then returns an Evaluate(call) node
 * invoking tl::tl_gemm with the composed string and the A/B/C buffer handles.
 *
 * @param T Contains lowering context including thread bounds and target.
 * @param analyzer Optional arithmetic analyzer used by lowering (may be
 * nullptr).
 * @return Stmt A TIR statement representing the evaluated TL GEMM call.
 */
Stmt GemmNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  auto block_size = *as_const_int(T.thread_bounds->extent);
  GemmInst gemm_inst = getGemmInst(block_size, T.target);

  auto warp_m = 1, warp_n = 1, warp_k = 1;
  if (TargetIsHCU(T.target)) {
    int element_byte_size = a_->dtype.bits() / 8;
    bool A_from_mls = IsFromMls(a_, T.tir_collector);
    bool B_from_mls = IsFromMls(b_, T.tir_collector);
    bool A_mls_trans = !transA_;
    bool B_mls_trans = transB_;
    auto [warp_m_tmp, warp_n_tmp, warp_k_tmp] =
        policy_->computeWarpPartitionHCU(m_, n_, k_, kPack_, element_byte_size,
                                         block_size, T.target, gemm_inst,
                                         A_from_mls, B_from_mls,
                                         A_mls_trans, B_mls_trans);
    warp_m = warp_m_tmp;
    warp_n = warp_n_tmp;
    warp_k = warp_k_tmp;
  } else {
    auto [warp_m_tmp, warp_n_tmp] =
        policy_->computeWarpPartition(m_, n_, block_size, T.target, gemm_inst);
    warp_m = warp_m_tmp;
    warp_n = warp_n_tmp;
  }

  PrimExpr Aptr =
      MakeAccessPtrFromRegion(aRegion_, /*r*/ 1, /*require_2d*/ true);
  PrimExpr Bptr =
      MakeAccessPtrFromRegion(bRegion_, /*r*/ 1, /*require_2d*/ true);
  PrimExpr Cptr =
      MakeAccessPtrFromRegion(cRegion_, /*rw*/ 3, /*require_2d*/ true);

  std::stringstream ss;
  std::string op_name;

  if (gemm_inst == GemmInst::kTCGEN5MMA) {
    auto [can_use_tcgen5mma, meta] =
        GetTCGEN5MMAMeta(m_, n_, k_, a_->dtype, c_->dtype);
    ICHECK(can_use_tcgen5mma);
    ICHECK(b_.scope() == "shared.dyn" || b_.scope() == "shared");
    ICHECK(c_.scope() == "shared.tmem");
    ICHECK(mbar_.has_value()) << "mbar must be provided for TCGEN5MMA";
    if (a_.scope() == "shared.tmem") {
      op_name = "tl::tcgen5mma_gemm_ts";
    } else if (a_.scope() == "shared.dyn" || a_.scope() == "shared") {
      op_name = "tl::tcgen5mma_gemm_ss";
    } else {
      ICHECK(0)
          << "Unsupported A scope for TCGEN5MMA: "
          << a_.scope(); // If this is triggered, it means Tilelang has bugs.
    }
    ICHECK(wgWait_ == -1)
        << "Currently only wg_wait == -1 is supported for TCGEN5MMA. Please "
           "use "
           "wg_wait = -1 and manually synchronize with mbarrier.";

    std::string accum_dtype = "";
    if (c_->dtype.is_float()) {
      if (c_->dtype.bits() == 32) {
        accum_dtype = "float";
      }
    }
    ICHECK(!accum_dtype.empty())
        << "Unsupported C dtype for TCGEN5MMA: " << c_->dtype;
    ss << op_name << "<" << m_ << ", " << n_ << ", " << k_ << ", ";
    ss << meta.atom_m << ", " << meta.atom_n << ", " << meta.atom_k << ", ";
    ss << transA_ << ", " << transB_ << ", ";
    ss << accum_dtype;
    ss << ">";

    auto C_buffer = T.buffer_remap.count(c_) ? T.buffer_remap[c_] : c_;
    Array<PrimExpr> new_args;
    auto mbarPtr =
        MakeAccessPtrFromRegion(mbarRegion_, /*rw*/ 3, /*require_2d*/ true);
    new_args.push_back(StringImm(ss.str()));
    new_args.push_back(Aptr);
    new_args.push_back(Bptr);
    new_args.push_back(BufferLoad(C_buffer, cCoords_));
    new_args.push_back(mbarPtr);
    new_args.push_back(clearAccum_);
    auto new_call = Call(DataType::Handle(), builtin::call_extern(), new_args);

    // Since TCGEN5MMA atoms provided by CUTLASS always have an internal
    // `elect_one_sync()`, we check if we are calling it using full warps
    constexpr int warp_size = 32;
    ICHECK(
        analyzer->CanProveEqual(FloorMod(T.thread_bounds->min, warp_size), 0) &&
        analyzer->CanProveEqual(FloorMod(T.thread_bounds->extent, warp_size),
                                0))
        << "TCGEN5MMA requires thread bounds to be multiples of warp size (32) "
           "and aligned to warps.";
    if (analyzer->CanProveEqual(T.thread_bounds->extent, warp_size)) {
      // If the thread bounds is exactly one warp, we can use the original call
      return Evaluate(new_call);
    } else {
      // Add an if-else clause
      auto tcgen5mma_call =
          IfThenElse(EQ(FloorDiv(T.thread_var, warp_size),
                        FloorDiv(T.thread_bounds->min, warp_size)),
                     Evaluate(new_call));
      return tcgen5mma_call;
    }
  }

  bool A_from_mls = IsFromMls(a_, T.tir_collector);
  bool B_from_mls = IsFromMls(b_, T.tir_collector);
  bool A_mls_trans = !transA_;
  bool B_mls_trans = transB_;
  bool use_gemm_mls =
      TargetIsHCU(T.target) &&
      ((A_from_mls && !IsFragmentBuffer(a_)) ||
       (B_from_mls && !IsFragmentBuffer(b_)));

  if (use_gemm_mls) {
    ICHECK(kPack_ == 1) << "gemm_mls does not support kPack > 1";
    ICHECK(warp_k == 1) << "gemm_mls does not support warp on K";
    if (A_from_mls && B_from_mls) {
      op_name = "tl::gemm_mls_mls";
    } else if (B_from_mls) {
      op_name = IsFragmentBuffer(a_) ? "tl::gemm_r_mls" : "tl::gemm_s_mls";
    } else {
      ICHECK(A_from_mls);
      LOG(FATAL) << "gemm_mls_s (A mls, B r or s) not implemented";
    }
  } else if (IsFragmentBuffer(a_)) {
    if (IsFragmentBuffer(b_)) {
      op_name = "tl::gemm_rr";
    } else {
      op_name = "tl::gemm_rs";
    }
  } else if (IsFragmentBuffer(b_)) {
    op_name = "tl::gemm_sr";
  } else {
    op_name = "tl::gemm_ss";
  }
  ICHECK(IsFragmentBuffer(c_));

  ss << op_name << "<" << m_ << ", " << n_ << ", " << k_ << ", ";
  ss << warp_m << ", " << warp_n << ", ";
  bool has_warp_k = (warp_k > 1);
  if (has_warp_k) {
    ss << "tl::WarpKParam<" << warp_k << ">, ";
  }
  ss << transA_ << ", " << transB_;

  if (use_gemm_mls) {
    int mls_tile_m = 0, mls_tile_ka = 0, mls_tile_n = 0, mls_tile_kb = 0;
    if (A_from_mls) {
      auto tile_a = GetMlsTileFromProducerChain(a_, T.tir_collector);
      ICHECK(tile_a) << "MatrixLoad/DsReadFormat A tile must be set in InferLayout";
      mls_tile_m = tile_a->first;
      mls_tile_ka = tile_a->second;
    }
    if (B_from_mls) {
      auto tile_b = GetMlsTileFromProducerChain(b_, T.tir_collector);
      ICHECK(tile_b) << "MatrixLoad/DsReadFormat B tile must be set in InferLayout";
      mls_tile_n = tile_b->first;
      mls_tile_kb = tile_b->second;
    }
    ss << ", " << kPack_;
    if (A_from_mls && B_from_mls) {
      ss << ", ck_tile::sequence<" << mls_tile_m << ", " << mls_tile_ka
         << ">, ck_tile::sequence<" << mls_tile_n << ", " << mls_tile_kb
         << ">, 1, 1";
    } else if (B_from_mls) {
      ss << ", ck_tile::sequence<" << mls_tile_n << ", " << mls_tile_kb
         << ">, 1";
    } else {
      ICHECK(A_from_mls);
      ss << ", ck_tile::sequence<" << mls_tile_m << ", " << mls_tile_ka
         << ">, ck_tile::sequence<0, 0>, 1";
    }
    ss << ", ";
    if (a_->dtype.is_bfloat16())
      ss << "ck_tile::bfloat16_t";
    else if (a_->dtype.is_float16())
      ss << "half_t";
    else if (IsFloat8E4Family(a_->dtype))
      ss << "fp8_e4_t";
    else if (IsFloat8E5Family(a_->dtype))
      ss << "fp8_e5_t";
    else
      LOG(FATAL) << "gemm_mls unsupported A dtype: " << a_->dtype;
    ss << ", ";
    if (b_->dtype.is_bfloat16())
      ss << "ck_tile::bfloat16_t";
    else if (b_->dtype.is_float16())
      ss << "half_t";
    else if (IsFloat8E4Family(b_->dtype))
      ss << "fp8_e4_t";
    else if (IsFloat8E5Family(b_->dtype))
      ss << "fp8_e5_t";
    else
      LOG(FATAL) << "gemm_mls unsupported B dtype: " << b_->dtype;
    ss << ", float, float, ck_tile::hcu_target_enum::" << GetHcuArchString(T.target);
  } else {
    auto clear_accum_bool = clearAccum_.as<Bool>();
    ICHECK(clear_accum_bool.has_value())
        << "clear_accum must be a constant Bool type, got " << clearAccum_;
    ss << ", " << bool(clear_accum_bool.value());
    if (TargetIsCuda(T.target) && (GetArchInt(T.target) >= 75)) {
      ss << ", " << strideA_ << ", " << strideB_;
      ss << ", " << offsetA_ << ", " << offsetB_;
    }
    if (TargetIsHCU(T.target)) {
      ss << ", " << kPack_;
      if (B_from_mls && !B_mls_trans) {
        ss << ", 32";
      }
    } else if (TargetIsCDNA(T.target)) {
      ss << ", " << kPack_;
    } else if (TargetIsHopper(T.target)) {
      ss << ", " << (gemm_inst == GemmInst::kWGMMA ? "true" : "false");
    }
    if (TargetIsHopper(T.target)) {
      if (wgWait_ != 0) {
        ss << ", " << wgWait_;
      }
    } else if (TargetIsSm100(T.target)) {
      ICHECK(wgWait_ == 0 || wgWait_ == -1)
          << "wg_wait must be 0 or -1 for Sm100";
    } else {
      ICHECK(wgWait_ == 0)
          << "wg_wait must be 0 for non-Hopper and non-Sm100 targets";
    }
  }
  ss << ">";

  auto new_call = Call(DataType::Handle(), tl::tl_gemm(),
                       Array<PrimExpr>{StringImm(ss.str()), Aptr, Bptr, Cptr});
  return Evaluate(new_call);
}

/**
 * @brief Infer and bind target-specific memory/layout mappings for A, B, and C.
 *
 * Infers per-buffer layouts (fragment or shared-memory layouts) for this GEMM
 * operator according to the target architecture, thread bounds, warp
 * partitioning, data types, and transpose flags, then binds fragment layouts
 * to the thread range when required.
 *
 * Preconditions:
 * - C.scope() == "local.fragment"
 *
 * Side effects:
 * - Marks layout inference as completed (sets completed_ = true).
 * - May abort via ICHECK on unsupported targets, invalid buffer scopes, or
 *   incompatible shape constraints.
 *
 * @param T Input layout-inference context (provides thread bounds and target).
 * @return LayoutMap mapping A, B, and C to their inferred layouts.
 */
LayoutMap GemmNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results;
  auto thread_range = T.thread_bounds;
  auto block_size = *as_const_int(thread_range->extent);
  GemmInst gemm_inst = getGemmInst(block_size, T.target);

  auto warp_m = 1, warp_n = 1, warp_k = 1;
  bool B_from_mls = false;
  bool B_mls_trans = false;
  if (TargetIsHCU(T.target)) {
    int element_byte_size = a_->dtype.bits() / 8;
    bool A_from_mls = IsFromMls(a_, T.tir_collector);
    B_from_mls = IsFromMls(b_, T.tir_collector);
    bool A_mls_trans = !transA_;
    B_mls_trans = transB_;
    auto [warp_m_tmp, warp_n_tmp, warp_k_tmp] =
        policy_->computeWarpPartitionHCU(m_, n_, k_, kPack_, element_byte_size,
                                         block_size, T.target, gemm_inst,
                                         A_from_mls, B_from_mls,
                                         A_mls_trans, B_mls_trans);
    warp_m = warp_m_tmp;
    warp_n = warp_n_tmp;
    warp_k = warp_k_tmp;
  } else {
    auto [warp_m_tmp, warp_n_tmp] =
        policy_->computeWarpPartition(m_, n_, block_size, T.target, gemm_inst);
    warp_m = warp_m_tmp;
    warp_n = warp_n_tmp;
  }

  if (TargetIsVolta(T.target)) {
    ICHECK(IsFragmentBuffer(c_))
        << "Volta gemm only supports C in local.fragment scope, got "
        << c_.scope();
    auto fragment = makeGemmVoltaFragmentC(m_, n_, m_ / warp_m, n_ / warp_n,
                                           c_->dtype.bits());
    results.Set(c_, fragment->BindThreadRange(thread_range));
    if (a_.scope() == "shared" || a_.scope() == "shared.dyn") {
      int dim_A = a_->shape.size();
      results.Set(a_, makeGemmVoltaABLayout(*as_const_int(a_->shape[dim_A - 2]),
                                            *as_const_int(a_->shape[dim_A - 1]),
                                            true, !transA_));
    } else if (IsFragmentBuffer(a_)) {
      ICHECK(transA_ == false);
      auto fragment =
          makeGemmVoltaFragmentA(m_, n_, k_, m_ / warp_m, n_ / warp_n);
      results.Set(a_, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }

    ICHECK(b_.scope() == "shared" || b_.scope() == "shared.dyn");
    int dim_B = b_->shape.size();
    results.Set(b_, makeGemmVoltaABLayout(*as_const_int(b_->shape[dim_B - 2]),
                                          *as_const_int(b_->shape[dim_B - 1]),
                                          false, transB_));
  } else if (TargetIsAmpere(T.target) || TargetIsTuring(T.target) ||
             TargetIsSM120(T.target) ||
             (TargetIsSm100(T.target) && gemm_inst == GemmInst::kMMA)) {
    ICHECK(IsFragmentBuffer(c_))
        << "MMA only supports C in local.fragment scope, got " << c_.scope();

    auto fragment =
        makeGemmFragmentC(m_, n_, m_ / warp_m, n_ / warp_n, c_->dtype.bits());
    results.Set(c_, fragment->BindThreadRange(thread_range));

    if (a_.scope() == "shared" || a_.scope() == "shared.dyn") {
      int dim_A = a_->shape.size();
      const int64_t mat_stride = *as_const_int(a_->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(a_->shape[dim_A - 1]);
      results.Set(a_,
                  makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                   a_->dtype.bits(), !transA_));
    } else if (IsFragmentBuffer(a_)) {
      auto fragment = makeGemmFragmentA(m_, n_, k_, m_ / warp_m, n_ / warp_n,
                                        a_->dtype.bits(), transA_);
      results.Set(a_, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
    if (b_.scope() == "shared" || b_.scope() == "shared.dyn") {
      int dim_B = b_->shape.size();
      const int64_t mat_stride = *as_const_int(b_->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(b_->shape[dim_B - 1]);
      results.Set(b_,
                  makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                   b_->dtype.bits(), transB_));
    } else if (IsFragmentBuffer(b_)) {
      auto fragment =
          makeGemmFragmentB(m_, n_, k_, m_ / warp_m, n_ / warp_n, transB_);
      results.Set(b_, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
  } else if (TargetIsHopper(T.target)) {
    ICHECK(IsFragmentBuffer(c_))
        << (gemm_inst == GemmInst::kWGMMA ? "WGMMA " : "MMA ")
        << "only supports C in local.fragment scope, got " << c_.scope();
    auto fragment = gemm_inst == GemmInst::kWGMMA
                        ? makeGemmFragmentCHopper(m_, n_, m_ / warp_m,
                                                  n_ / warp_n, c_->dtype.bits())
                        : makeGemmFragmentC(m_, n_, m_ / warp_m, n_ / warp_n,
                                            c_->dtype.bits());
    results.Set(c_, fragment->BindThreadRange(thread_range));
    if (a_.scope() == "shared" || a_.scope() == "shared.dyn") {
      int dim_A = a_->shape.size();
      const int64_t mat_stride = *as_const_int(a_->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(a_->shape[dim_A - 1]);
      const int64_t continuity =
          transA_ ? 4 * mat_continuous / warp_m : mat_continuous;
      auto ABLayout =
          gemm_inst == GemmInst::kWGMMA
              ? makeGemmABLayoutHopper(mat_stride, mat_continuous, continuity,
                                       a_->dtype.bits(), !transA_)
              : makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                 a_->dtype.bits(), !transA_);
      results.Set(a_, ABLayout);
    } else {
      auto fragment = makeGemmFragmentA(m_, n_, k_, m_ / warp_m, n_ / warp_n,
                                        a_->dtype.bits(), transA_);
      results.Set(a_, fragment->BindThreadRange(thread_range));
    }
    if (b_.scope() == "shared" || b_.scope() == "shared.dyn") {
      int dim_B = b_->shape.size();
      const int64_t mat_stride = *as_const_int(b_->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(b_->shape[dim_B - 1]);
      const int64_t continuity =
          transB_ ? mat_continuous : mat_continuous / warp_n;

      auto ABLayout =
          gemm_inst == GemmInst::kWGMMA
              ? makeGemmABLayoutHopper(mat_stride, mat_continuous, continuity,
                                       b_->dtype.bits(), transB_)
              : makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                 b_->dtype.bits(), transB_);
      results.Set(b_, ABLayout);
    } else {
      auto fragment =
          makeGemmFragmentB(m_, n_, k_, m_ / warp_m, n_ / warp_n, transB_);
      results.Set(b_, fragment->BindThreadRange(thread_range));
    }
  } else if (gemm_inst == GemmInst::kTCGEN5MMA) {
    ICHECK(c_.scope() == "shared.tmem")
        << "TCGEN5MMA only supports C in shared.tmem scope, got " << c_.scope();
    ICHECK(a_.scope() == "shared.dyn" || a_.scope() == "shared")
        << "Current TCGEN5MMA only supports A in shared.dyn scope";
    auto [can_use_tcgen5mma, meta] =
        GetTCGEN5MMAMeta(m_, n_, k_, a_->dtype, c_->dtype);
    ICHECK(can_use_tcgen5mma);
    {
      int dim_A = a_->shape.size();
      const int64_t mat_stride = *as_const_int(a_->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(a_->shape[dim_A - 1]);
      results.Set(a_, makeGemmABLayoutSm100(mat_stride, mat_continuous,
                                            mat_continuous, a_->dtype.bits(),
                                            transA_ ? 1 : 2));
    }
    {
      int dim_B = b_->shape.size();
      const int64_t mat_stride = *as_const_int(b_->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(b_->shape[dim_B - 1]);
      const int64_t continuity = mat_continuous;
      results.Set(b_,
                  makeGemmABLayoutSm100(mat_stride, mat_continuous, continuity,
                                        b_->dtype.bits(), transB_ ? 2 : 1));
    }
    {
      Layout res;
      IterVar i = make_itervar("i", m_);
      IterVar j = make_itervar("j", n_);
      ICHECK(m_ % meta.atom_m == 0);
      PrimExpr atom_idx = FloorDiv(i, meta.atom_m) +
                          FloorDiv(j, meta.atom_n) * (m_ / meta.atom_m);
      PrimExpr ai = FloorMod(i, meta.atom_m); // "ai" means "atom_i"
      PrimExpr aj = FloorMod(j, meta.atom_n);
      if (meta.atom_m == 128) {
        // Layout D
        // (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d)
        res = Layout(Array{i, j}, {ai, aj + atom_idx * meta.atom_n});
      } else if (meta.atom_m == 64) {
        // Layout E
        // (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-e)
        // since .ws variant is used About why we use .ws variant here, please
        // refer to gemm_sm100.h
        res = Layout(Array{i, j}, {FloorDiv(ai, 32) * 32 + FloorMod(ai, 32) +
                                       FloorDiv(aj, meta.atom_n / 2) * 64,
                                   FloorMod(aj, meta.atom_n / 2) +
                                       atom_idx * (meta.atom_n / 2)});
      } else if (meta.atom_m == 32) {
        // Layout G
        // (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-g)
        res = Layout(
            Array{i, j},
            {FloorMod(ai, 32) + FloorDiv(aj, meta.atom_n / 4) * 32,
             FloorMod(aj, meta.atom_n / 4) + atom_idx * (meta.atom_n / 4)});
      } else {
        ICHECK(0);
      }
      results.Set(c_, res);
    }
  } else if (TargetIsHCU(T.target)) {
    ICHECK(c_.scope() == "local.fragment")
        << "hcu gemm (mmac) only supports C in local.fragment scope, got "
        << c_.scope();
    const int min_n_per_warp = (B_from_mls && !B_mls_trans) ? 32 : 16;
    auto fragment =
        makeGemmFragmentHCU(m_, n_, warp_m, warp_n, warp_k, c_->dtype.bits(),
                            min_n_per_warp);
    if (TargetHasMmacLitLts(T.target)) {
      fragment = makeGemmFragmentHCULit(m_, n_, warp_m, warp_n, warp_k,
                                        c_->dtype.bits(), min_n_per_warp);
    }
    results.Set(c_, fragment->BindThreadRange(thread_range));

    if (a_.scope() == "shared" || a_.scope() == "shared.dyn") {
      int dim_A = a_->shape.size();
      auto shared_layout = makeGemmABLayoutCDNA(
          *as_const_int(a_->shape[dim_A - 2]),
          *as_const_int(a_->shape[dim_A - 1]), a_->dtype.bits(), kPack_);
      results.Set(a_, shared_layout);
    } else if (a_.scope() == "local.fragment") {
      auto fragment = makeGemmFragmentAHCU(m_, n_, k_, warp_m, warp_n, warp_k,
                                         a_->dtype.bits(), kPack_, transA_);
      results.Set(a_, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
    if (b_.scope() == "shared" || b_.scope() == "shared.dyn") {
      int dim_B = b_->shape.size();
      auto shared_layout = makeGemmABLayoutCDNA(
          *as_const_int(b_->shape[dim_B - 2]),
          *as_const_int(b_->shape[dim_B - 1]), b_->dtype.bits(), kPack_);

      results.Set(b_, shared_layout);
    } else if (b_.scope() == "local.fragment") {
      auto fragment = makeGemmFragmentBHCU(m_, n_, k_, warp_m, warp_n, warp_k,
                                           b_->dtype.bits(), kPack_, transB_,
                                           min_n_per_warp);
      results.Set(b_, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
  } else if (TargetIsCDNA(T.target)) {
    ICHECK(IsFragmentBuffer(c_))
        << "CDNA gemm (FMMA) only supports C in local.fragment scope, got "
        << c_.scope();
    auto fragment = makeGemmFragmentCCDNA(m_, n_, m_ / warp_m, n_ / warp_n,
                                          c_->dtype.bits());
    results.Set(c_, fragment->BindThreadRange(thread_range));

    if (a_.scope() == "shared" || a_.scope() == "shared.dyn") {
      int dim_A = a_->shape.size();
      auto shared_layout = makeGemmABLayoutCDNA(
          *as_const_int(a_->shape[dim_A - 2]),
          *as_const_int(a_->shape[dim_A - 1]), a_->dtype.bits(), kPack_);
      results.Set(a_, shared_layout);
    } else if (IsFragmentBuffer(a_)) {
      auto fragment =
          makeGemmFragmentACDNA(m_, n_, k_, m_ / warp_m, n_ / warp_n,
                                a_->dtype.bits(), kPack_, transA_);
      results.Set(a_, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
    if (b_.scope() == "shared" || b_.scope() == "shared.dyn") {
      int dim_B = b_->shape.size();
      auto shared_layout = makeGemmABLayoutCDNA(
          *as_const_int(b_->shape[dim_B - 2]),
          *as_const_int(b_->shape[dim_B - 1]), b_->dtype.bits(), kPack_);

      results.Set(b_, shared_layout);
    } else if (IsFragmentBuffer(b_)) {
      auto fragment =
          makeGemmFragmentB(m_, n_, k_, m_ / warp_m, n_ / warp_n, transB_);
      results.Set(b_, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
  } else {
    ICHECK(0) << "Not supported " << T.target->str();
  }
  completed_ = true;
  return results;
}

TIR_REGISTER_TL_TILE_OP(Gemm, gemm)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.GemmWarpPolicy")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "GemmWarpPolicy");

TVM_FFI_STATIC_INIT_BLOCK() {
  GemmNode::RegisterReflection();
  GemmWarpPolicyNode::RegisterReflection();
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.GemmWarpPolicyComputeWarpPartition",
                        [](GemmWarpPolicy policy, int M, int N, int block_size,
                           Target target, GemmInst gemm_inst) {
                          policy->computeWarpPartition(M, N, block_size, target,
                                                       gemm_inst);
                        });
  refl::GlobalDef().def(
      "tl.GemmWarpPolicyComputeWarpPartitionHCU",
      [](GemmWarpPolicy policy, int M, int N, int K, int k_pack,
         int element_byte_size, int block_size, Target target, int gemm_inst,
         bool A_from_mls, bool B_from_mls, bool A_mls_trans, bool B_mls_trans) {
        policy->computeWarpPartitionHCU(
            M, N, K, k_pack, element_byte_size, block_size, target,
            static_cast<GemmInst>(gemm_inst), A_from_mls, B_from_mls,
            A_mls_trans, B_mls_trans);
      });
}

} // namespace tl
} // namespace tvm
