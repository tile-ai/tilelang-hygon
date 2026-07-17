/*!
 * \file propagation_util.cc
 * \brief Implementation of propagation utilities.
 *
 * Design: Use PropagationTirCollector for node lookup (covers non-TileOperator
 * like ieee_mul). mls_tile is derived locally by MatrixLoad/DsReadFormat/Gemm
 * via ComputeMlsWarpPartition, no propagation to query MatrixLoad needed.
 */

#include "propagation_util.h"
#include "op/builtin.h"
#include "op/gemm.h"
#include "op/operator.h"
#include "propagation_tir_collector.h"

#include <tvm/tirx/builtin.h>
#include <unordered_set>

namespace tvm {
namespace tl {

using namespace tirx;

static bool IsGemm(const TileOperator &op) {
  return op->GetTypeKey() == std::string("tl.Gemm");
}

static bool GemmUsesBuffer(const GemmNode *gemm, const Buffer &buffer) {
  return gemm->a_.same_as(buffer) || gemm->b_.same_as(buffer);
}

static std::optional<GemmWithInput> GemmWithInputFromCall(const CallNode *call,
                                                          const Buffer &input) {
  if (call == nullptr || !IsGemmTileOpCall(call))
    return std::nullopt;
  auto op = ParseOperator(tvm::ffi::GetRef<Call>(call));
  if (!op.defined() || !IsGemm(op))
    return std::nullopt;
  auto gemm = Downcast<Gemm>(op);
  if (!GemmUsesBuffer(gemm.get(), input))
    return std::nullopt;
  GemmWithInput r;
  r.gemm = gemm;
  r.input = input;
  return r;
}

static std::optional<ProducerRecord>
FindGemmProducerReading(const PropagationTirCollector *tir,
                        const Buffer &write_buf, const Buffer &read_buf,
                        int after_stmt_order) {
  if (after_stmt_order < 0) {
    return tir->FindLastGemmProducerReading(write_buf, read_buf);
  }
  return tir->FindFirstGemmProducerReading(write_buf, read_buf,
                                           after_stmt_order);
}

static Optional<TileOperator> PropagateToFindGemmConsumerOpTir(
    Buffer buffer, const PropagationTirCollector *tir, int after_stmt_order) {
  if (!tir)
    return Optional<TileOperator>();
  for (const Buffer &out_buf : tir->GetConsumerOutputs(buffer)) {
    if (out_buf.scope() != "local.fragment")
      continue;
    if (auto rec =
            FindGemmProducerReading(tir, out_buf, buffer, after_stmt_order)) {
      if (auto found = GemmWithInputFromCall(rec->call, buffer)) {
        return found->gemm;
      }
    }
    auto found =
        PropagateToFindGemmConsumerOpTir(out_buf, tir, after_stmt_order);
    if (found.defined())
      return found;
  }
  return Optional<TileOperator>();
}

static std::optional<GemmWithInput> PropagateToFindGemmConsumerOpWithInputTir(
    Buffer buffer, const PropagationTirCollector *tir, int after_stmt_order) {
  if (!tir)
    return std::nullopt;
  for (const Buffer &out_buf : tir->GetConsumerOutputs(buffer)) {
    if (out_buf.scope() != "local.fragment")
      continue;
    if (auto rec =
            FindGemmProducerReading(tir, out_buf, buffer, after_stmt_order)) {
      if (auto found = GemmWithInputFromCall(rec->call, buffer)) {
        return found;
      }
    }
    auto found = PropagateToFindGemmConsumerOpWithInputTir(out_buf, tir,
                                                           after_stmt_order);
    if (found)
      return found;
  }
  return std::nullopt;
}

bool PropagateToFindGemmConsumer(Buffer buffer,
                                 const PropagationTirCollector *tir_collector) {
  return PropagateToFindGemmConsumerOp(buffer, tir_collector).defined();
}

Optional<TileOperator>
PropagateToFindGemmConsumerOp(Buffer buffer,
                              const PropagationTirCollector *tir_collector) {
  return PropagateToFindGemmConsumerOpTir(buffer, tir_collector, -1);
}

std::optional<GemmWithInput> PropagateToFindGemmConsumerOpWithInput(
    Buffer buffer, const PropagationTirCollector *tir_collector,
    int after_stmt_order) {
  return PropagateToFindGemmConsumerOpWithInputTir(buffer, tir_collector,
                                                   after_stmt_order);
}

std::optional<GemmWithInput> PropagateToFindGemmConsumerOpWithInputAfterCall(
    Buffer buffer, const PropagationTirCollector *tir_collector,
    const CallNode *after_site_call) {
  int after_order = -1;
  if (tir_collector != nullptr && after_site_call != nullptr) {
    after_order = tir_collector->GetCallStmtOrder(after_site_call);
  }
  return PropagateToFindGemmConsumerOpWithInput(buffer, tir_collector,
                                                after_order);
}

std::vector<ReaderCallRecord>
GetReaderCallsFromTir(Buffer buffer,
                      const PropagationTirCollector *tir_collector) {
  if (!tir_collector)
    return {};
  return tir_collector->GetReaderCalls(buffer);
}

static bool
PropagateToFindProducerMatrixLoadFoundTir(Buffer buffer,
                                          const PropagationTirCollector *tir) {
  if (tir->ProducerIsMatrixLoad(buffer))
    return true;
  if (tir->ProducerIsDsReadFormat(buffer)) {
    auto inputs = tir->GetProducerInputs(buffer);
    if (!inputs.empty()) {
      return PropagateToFindProducerMatrixLoadFoundTir(inputs[0], tir);
    }
  }
  for (const Buffer &in_buf : tir->GetProducerInputs(buffer)) {
    if (in_buf.scope() != "local.fragment")
      continue;
    if (PropagateToFindProducerMatrixLoadFoundTir(in_buf, tir))
      return true;
  }
  return false;
}

bool IsFromMls(Buffer buffer, const PropagationTirCollector *tir_collector) {
  return tir_collector &&
         PropagateToFindProducerMatrixLoadFoundTir(buffer, tir_collector);
}

Array<TileOperator>
GetConsumerOpsFromTir(Buffer buffer,
                      const PropagationTirCollector *tir_collector) {
  Array<TileOperator> result;
  if (!tir_collector)
    return result;
  std::unordered_set<const CallNode *, CallNodePtrHash, CallNodePtrEqual> seen;
  for (const ReaderCallRecord &reader : tir_collector->GetReaderCalls(buffer)) {
    if (reader.call == nullptr || seen.count(reader.call))
      continue;
    seen.insert(reader.call);
    auto op = ParseOperator(tvm::ffi::GetRef<Call>(reader.call));
    if (op.defined())
      result.push_back(op);
  }
  return result;
}

} // namespace tl
} // namespace tvm
