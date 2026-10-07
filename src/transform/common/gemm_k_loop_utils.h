#ifndef TVM_TL_TRANSFORM_COMMON_GEMM_K_LOOP_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_GEMM_K_LOOP_UTILS_H_

#include <tvm/tirx/stmt.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace tvm {
namespace tl {

bool IsAsyncCopyCall(const tirx::CallNode *call);
bool IsMmaCall(const tirx::CallNode *call);
bool StmtContainsMma(const tirx::Stmt &stmt);
bool StmtContainsAsyncCopy(const tirx::Stmt &stmt);
bool IsMmaCluster(const tirx::Stmt &stmt);
bool IsSchedBarrierStmt(const tirx::Stmt &stmt);

std::optional<int64_t> GetConstIntValue(const PrimExpr &expr);

struct GemmKLoopFeatures {
  bool has_mma{false};
  bool has_memory_access{false};
  bool has_global_src{false};
};

GemmKLoopFeatures AnalyzeGemmKLoopBody(const tirx::Stmt &body);
bool IsGemmKLoop(const tirx::ForNode *loop);
std::vector<const tirx::ForNode *> CollectGemmKLoops(const tirx::Stmt &stmt);

} // namespace tl
} // namespace tvm

#endif
