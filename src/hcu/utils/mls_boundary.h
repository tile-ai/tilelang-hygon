/*!
 * \file hcu/utils/mls_boundary.h
 * \brief Per-axis MLS boundary policy packed as int8 in TIR / template args.
 *
 * -1 = analyze (compiler may prove in-range, else refresh)
 *  0 = skip filter (caller contract)
 *  1 = always refresh (caller contract)
 *
 * Template args after optional DstBits are (k_mode, mn_mode).
 */

#ifndef TVM_TL_HCU_UTILS_MLS_BOUNDARY_H_
#define TVM_TL_HCU_UTILS_MLS_BOUNDARY_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/op.h>

#include <string>
#include <vector>

#include "support/check.h"

namespace tvm {
namespace tl {

enum class MlsBoundaryMode : int {
  kAnalyze = -1,
  kSkip = 0,
  kRefresh = 1,
};

inline int MlsResolveBoundary(int mode, const PrimExpr &base,
                              const PrimExpr &length, int64_t block,
                              arith::Analyzer *analyzer,
                              bool retry_with_fresh_solver = true) {
  // Explicit boundary policies remain caller contracts. Only analyze mode
  // may be specialized, using the caller's scoped assumptions and ranges.
  if (mode != static_cast<int>(MlsBoundaryMode::kAnalyze)) {
    return mode;
  }
  auto prove_bounds = [&](arith::Analyzer *proof_analyzer) {
    return proof_analyzer->CanProve(base >= 0,
                                    arith::ProofStrength::kSymbolicBound) &&
           proof_analyzer->CanProve(
               base + tirx::make_const(base.dtype(), block) <= length,
               arith::ProofStrength::kSymbolicBound);
  };
  if (prove_bounds(analyzer)) {
    return static_cast<int>(MlsBoundaryMode::kSkip);
  }
  // Keep the fast path and its incremental solver. On failure, retry once
  // with copied constraints and rlimit but a freshly constructed solver.
  if (retry_with_fresh_solver) {
    auto fresh = analyzer->Clone();
    if (prove_bounds(fresh.get())) {
      return static_cast<int>(MlsBoundaryMode::kSkip);
    }
  }
  return mode;
}

inline bool MlsIsModeToken(const std::string &s) {
  return s == "-1" || s == "0" || s == "1";
}

inline size_t MlsLoadTileBoundaryIndex(const std::vector<std::string> &args) {
  if (args.size() > 8 && !MlsIsModeToken(args[8]))
    return 9;
  return 8;
}

inline bool MlsLoadTileHasDstBits(const std::vector<std::string> &args) {
  return MlsLoadTileBoundaryIndex(args) == 9;
}

inline MlsBoundaryMode MlsParseModeToken(const std::string &s) {
  if (s == "-1")
    return MlsBoundaryMode::kAnalyze;
  if (s == "0")
    return MlsBoundaryMode::kSkip;
  if (s == "1")
    return MlsBoundaryMode::kRefresh;
  ICHECK(false) << "invalid MLS boundary mode: " << s;
  return MlsBoundaryMode::kAnalyze;
}

inline const char *MlsModeLiteral(MlsBoundaryMode m) {
  switch (m) {
  case MlsBoundaryMode::kAnalyze:
    return "-1";
  case MlsBoundaryMode::kSkip:
    return "0";
  case MlsBoundaryMode::kRefresh:
    return "1";
  }
  return "-1";
}

inline bool MlsOneShotRefresh(MlsBoundaryMode m) {
  return m != MlsBoundaryMode::kSkip;
}

inline const char *MlsRefreshLiteral(MlsBoundaryMode m) {
  return MlsOneShotRefresh(m) ? "true" : "false";
}

struct MlsBoundaryModes {
  MlsBoundaryMode k{MlsBoundaryMode::kAnalyze};
  MlsBoundaryMode mn{MlsBoundaryMode::kAnalyze};
};

inline MlsBoundaryModes
MlsParseBoundaryArgs(const std::vector<std::string> &args) {
  MlsBoundaryModes out;
  const size_t idx = MlsLoadTileBoundaryIndex(args);
  if (args.size() <= idx)
    return out;
  out.k = MlsParseModeToken(args[idx]);
  if (args.size() > idx + 1)
    out.mn = MlsParseModeToken(args[idx + 1]);
  return out;
}

inline MlsBoundaryMode MlsModeFromInt(int64_t value) {
  ICHECK(value == -1 || value == 0 || value == 1)
      << "MLS boundary mode must be -1, 0, or 1, got " << value;
  return static_cast<MlsBoundaryMode>(static_cast<int>(value));
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_UTILS_MLS_BOUNDARY_H_
