#ifndef TVM_TL_TRANSFORM_COMMON_GEMM_K_LOOP_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_GEMM_K_LOOP_UTILS_H_

#include <tvm/tirx/stmt.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace tvm {
namespace tl {

/*!
 * \brief Check whether a call issues an asynchronous global->shared copy.
 * \param call The call node to inspect.
 * \return True for HCU idxen cp.async and ptx cp.async calls.
 */
bool IsAsyncCopyCall(const tirx::CallNode *call);

/*!
 * \brief Check whether a call issues a matrix multiply-accumulate.
 * \param call The call node to inspect.
 * \return True for the MMA calls this backend recognizes.
 */
bool IsMmaCall(const tirx::CallNode *call);

/*!
 * \brief Check whether a statement contains an MMA call anywhere.
 * \param stmt The statement to search.
 * \return True if at least one MMA call is found.
 */
bool StmtContainsMma(const tirx::Stmt &stmt);

/*!
 * \brief Check whether a statement contains an async copy anywhere.
 * \param stmt The statement to search.
 * \return True if at least one async copy call is found.
 */
bool StmtContainsAsyncCopy(const tirx::Stmt &stmt);

/*!
 * \brief Check whether a statement is an MMA cluster.
 *
 * A cluster is a compute-only statement that contains an MMA and no async
 * copy; GEMM K loops are producer/consumer containers, not clusters.
 *
 * \param stmt The statement to inspect.
 * \return True if the statement is an MMA cluster.
 */
bool IsMmaCluster(const tirx::Stmt &stmt);

/*!
 * \brief Check whether a statement is an sched_barrier call.
 * \param stmt The statement to inspect.
 * \return True for a call_extern of __builtin_amdgcn_sched_barrier.
 */
bool IsSchedBarrierStmt(const tirx::Stmt &stmt);

/*!
 * \brief Read the value of a constant integer expression.
 * \param expr The expression to inspect.
 * \return The value of an IntImm, otherwise nullopt.
 */
std::optional<int64_t> GetConstIntValue(const PrimExpr &expr);

/*!
 * \brief Features of a GEMM K-loop body used by the pipeline passes.
 */
struct GemmKLoopFeatures {
  bool has_mma{false};           //!< The body issues at least one MMA call.
  bool has_memory_access{false}; //!< The body accesses a buffer.
  bool has_global_src{false};    //!< The body loads from a global buffer.
};

/*!
 * \brief Analyze a statement for the GEMM K-loop features above.
 * \param body The statement to inspect, usually a loop body.
 * \return The features found in the statement.
 */
GemmKLoopFeatures AnalyzeGemmKLoopBody(const tirx::Stmt &body);

/*!
 * \brief Check whether a loop is a GEMM K loop.
 * \param loop The loop to inspect.
 * \return True if the loop body has an MMA, a memory access and a global
 * source.
 */
bool IsGemmKLoop(const tirx::ForNode *loop);

/*!
 * \brief Collect every GEMM K loop in a statement tree.
 * \param stmt The statement to search.
 * \return The GEMM K loops in traversal order.
 */
std::vector<const tirx::ForNode *> CollectGemmKLoops(const tirx::Stmt &stmt);

/*!
 * \brief Check whether InjectAsyncGlobalLoadFence owns the wait plan of `stmt`.
 *
 * The fence planner replaces a pipeline's commit/wait pairs -- barriers
 * included -- when the first GEMM K loop is a register pipeline or a compiler
 * pipeline with at least two stages.
 *
 * \param stmt The statement to inspect.
 * \return True if the fence planner owns the wait plan.
 */
bool FencePlannerOwnsWaitPlan(const tirx::Stmt &stmt);

} // namespace tl
} // namespace tvm

#endif
