/*!
 * \file propagation_util.h
 * \brief Utilities for buffer producer/consumer propagation (matrix_load,
 * ds_read_format, gemm).
 */

#ifndef TVM_TL_OP_PROPAGATION_UTIL_H_
#define TVM_TL_OP_PROPAGATION_UTIL_H_

#include <optional>
#include <utility>

#include "gemm.h"
#include "operator.h"

namespace tvm {
namespace tl {

using namespace tir;

class PropagationTirCollector;

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
Optional<TileOperator> PropagateToFindGemmConsumerOp(
    Buffer buffer, const PropagationTirCollector *tir_collector);

struct GemmWithInput {
  Gemm gemm;
  Buffer input;
};

/*!
 * \brief Like PropagateToFindGemmConsumerOp, but also returns the buffer that
 * connects to Gemm (may differ from start buffer when there are intermediate ops).
 */
std::optional<GemmWithInput> PropagateToFindGemmConsumerOpWithInput(
    Buffer buffer, const PropagationTirCollector *tir_collector);

/*!
 * \brief Check if buffer's producer chain reaches MatrixLoad (via TIR).
 * Returns false when tir_collector is null.
 */
bool IsFromMls(Buffer buffer, const PropagationTirCollector *tir_collector);

/*!
 * \brief Get consumer TileOperators from TIR collector.
 * Returns empty when tir_collector is null.
 */
Array<TileOperator> GetConsumerOpsFromTir(Buffer buffer,
                                          const PropagationTirCollector *tir_collector);

/*!
 * \brief Get mls_tile (mn, k) from MatrixLoad or DsReadFormat in producer chain.
 * Returns nullopt when not found or mls_tile not set.
 */
std::optional<std::pair<int, int>> GetMlsTileFromProducerChain(
    Buffer buffer, const PropagationTirCollector *tir_collector);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_OP_PROPAGATION_UTIL_H_
