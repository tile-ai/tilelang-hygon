/*!
 * \file propagation_util.h
 * \brief Utilities for buffer producer/consumer propagation (matrix_load,
 * ds_read_format, gemm).
 *
 * Register-side propagation continues through `local.fragment` / `local` and
 * stops at shared-like scopes (`shared` / `shared.dyn` / `shared.tmem`).
 */

#ifndef TVM_TL_OP_PROPAGATION_UTIL_H_
#define TVM_TL_OP_PROPAGATION_UTIL_H_

#include <optional>
#include <utility>

#include "gemm.h"
#include "operator.h"
#include "propagation_tir_collector.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Propagate forward through consumer chain to find Gemm (TIR only).
 * \return True if some consumer chain reaches Gemm.
 */
bool PropagateToFindGemmConsumer(Buffer buffer,
                                 const PropagationTirCollector *tir_collector);

/*!
 * \brief Propagate forward to find first Gemm consumer (TIR only).
 * \return The Gemm op if found, else nullopt.
 */
Optional<TileOperator>
PropagateToFindGemmConsumerOp(Buffer buffer,
                              const PropagationTirCollector *tir_collector);

struct GemmWithInput {
  Gemm gemm;
  Buffer input;
};

/*!
 * \brief Like PropagateToFindGemmConsumerOp, but also returns the buffer that
 * connects to Gemm (may differ from start buffer when there are intermediate
 * ops).
 */
std::optional<GemmWithInput> PropagateToFindGemmConsumerOpWithInput(
    Buffer buffer, const PropagationTirCollector *tir_collector,
    int after_stmt_order = -1);

/*!
 * \brief Site-aware pairing: resolve stmt_order from `after_site_call` via the
 * collector, then find the first downstream GEMM after that call.
 */
std::optional<GemmWithInput> PropagateToFindGemmConsumerOpWithInputAfterCall(
    Buffer buffer, const PropagationTirCollector *tir_collector,
    const CallNode *after_site_call);

/*!
 * \brief Tile-op consumers that read `buffer`, in program order.
 * Prefer this over GetConsumerOpsFromTir when stmt_order pairing matters.
 */
std::vector<ReaderCallRecord>
GetReaderCallsFromTir(Buffer buffer,
                      const PropagationTirCollector *tir_collector);

/*!
 * \brief Check if buffer's producer chain reaches MatrixLoad (via TIR).
 * Returns false when tir_collector is null.
 */
bool IsFromMls(Buffer buffer, const PropagationTirCollector *tir_collector);

/*!
 * \brief Get consumer TileOperators from TIR collector.
 * Returns empty when tir_collector is null.
 */
Array<TileOperator>
GetConsumerOpsFromTir(Buffer buffer,
                      const PropagationTirCollector *tir_collector);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_PROPAGATION_UTIL_H_
