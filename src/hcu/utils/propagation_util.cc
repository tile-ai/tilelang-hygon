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

static bool IsSharedLikeScope(const Buffer &buffer) {
  const String &scope = buffer.scope();
  return scope == "shared" || scope == "shared.dyn" || scope == "shared.tmem";
}

using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;

static Optional<TileOperator> PropagateToFindGemmConsumerOpTirImpl(
    const Buffer &buffer, const PropagationTirCollector *tir,
    int after_stmt_order, BufferSet *visited_buffers) {
  if (!tir || !visited_buffers->insert(buffer).second)
    return Optional<TileOperator>();
  for (const Buffer &out_buf : tir->GetConsumerOutputs(buffer)) {
    // Stop at shared: register-side chains (fragment/local) keep propagating.
    if (IsSharedLikeScope(out_buf))
      continue;
    if (auto rec =
            FindGemmProducerReading(tir, out_buf, buffer, after_stmt_order)) {
      if (auto found = GemmWithInputFromCall(rec->call, buffer)) {
        return found->gemm;
      }
    }
    auto found = PropagateToFindGemmConsumerOpTirImpl(
        out_buf, tir, after_stmt_order, visited_buffers);
    if (found.defined())
      return found;
  }
  return Optional<TileOperator>();
}

static Optional<TileOperator> PropagateToFindGemmConsumerOpTir(
    const Buffer &buffer, const PropagationTirCollector *tir,
    int after_stmt_order) {
  BufferSet visited_buffers;
  return PropagateToFindGemmConsumerOpTirImpl(buffer, tir, after_stmt_order,
                                              &visited_buffers);
}

static std::optional<GemmWithInput>
PropagateToFindGemmConsumerOpWithInputTirImpl(
    const Buffer &buffer, const PropagationTirCollector *tir,
    int after_stmt_order, BufferSet *visited_buffers) {
  if (!tir || !visited_buffers->insert(buffer).second)
    return std::nullopt;
  for (const Buffer &out_buf : tir->GetConsumerOutputs(buffer)) {
    if (IsSharedLikeScope(out_buf))
      continue;
    if (auto rec =
            FindGemmProducerReading(tir, out_buf, buffer, after_stmt_order)) {
      if (auto found = GemmWithInputFromCall(rec->call, buffer)) {
        return found;
      }
    }
    auto found = PropagateToFindGemmConsumerOpWithInputTirImpl(
        out_buf, tir, after_stmt_order, visited_buffers);
    if (found)
      return found;
  }
  return std::nullopt;
}

static std::optional<GemmWithInput> PropagateToFindGemmConsumerOpWithInputTir(
    const Buffer &buffer, const PropagationTirCollector *tir,
    int after_stmt_order) {
  BufferSet visited_buffers;
  return PropagateToFindGemmConsumerOpWithInputTirImpl(
      buffer, tir, after_stmt_order, &visited_buffers);
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

namespace {

void CollectAllGemmConsumers(
    const Buffer &buffer, const PropagationTirCollector *tir_collector,
    int after_order,
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> *visited_buffers,
    std::unordered_set<const CallNode *, CallNodePtrHash, CallNodePtrEqual>
        *seen_calls,
    std::vector<GemmWithInput> *result) {
  if (!visited_buffers->insert(buffer).second) {
    return;
  }
  for (const ReaderCallRecord &reader :
       tir_collector->GetReaderCalls(buffer)) {
    if (reader.stmt_order <= after_order || reader.call == nullptr) {
      continue;
    }
    if (auto gemm = GemmWithInputFromCall(reader.call, buffer)) {
      if (seen_calls->insert(reader.call).second) {
        result->push_back(*gemm);
      }
      continue;
    }
    if (!IsSharedLikeScope(reader.write)) {
      CollectAllGemmConsumers(reader.write, tir_collector, after_order,
                              visited_buffers, seen_calls, result);
    }
  }
}

} // namespace

std::vector<GemmWithInput> PropagateToFindAllGemmConsumersAfterCall(
    Buffer buffer, const PropagationTirCollector *tir_collector,
    const CallNode *after_site_call) {
  std::vector<GemmWithInput> result;
  if (tir_collector == nullptr) {
    return result;
  }
  int after_order = after_site_call == nullptr
                        ? -1
                        : tir_collector->GetCallStmtOrder(after_site_call);
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> visited_buffers;
  std::unordered_set<const CallNode *, CallNodePtrHash, CallNodePtrEqual>
      seen_calls;
  CollectAllGemmConsumers(buffer, tir_collector, after_order, &visited_buffers,
                          &seen_calls, &result);
  return result;
}

std::vector<ReaderCallRecord>
GetReaderCallsFromTir(Buffer buffer,
                      const PropagationTirCollector *tir_collector) {
  if (!tir_collector)
    return {};
  return tir_collector->GetReaderCalls(buffer);
}

static bool PropagateToFindProducerMatrixLoadFoundTirImpl(
    const Buffer &buffer, const PropagationTirCollector *tir,
    BufferSet *visited_buffers) {
  if (!tir || !visited_buffers->insert(buffer).second)
    return false;
  if (tir->ProducerIsMatrixLoad(buffer))
    return true;
  if (tir->ProducerIsDsReadFormat(buffer)) {
    auto inputs = tir->GetProducerInputs(buffer);
    if (!inputs.empty()) {
      return PropagateToFindProducerMatrixLoadFoundTirImpl(
          inputs[0], tir, visited_buffers);
    }
  }
  for (const Buffer &in_buf : tir->GetProducerInputs(buffer)) {
    // Shared ends the register-side reverse walk; matrix_load on shared is
    // already handled by ProducerIsMatrixLoad above.
    if (IsSharedLikeScope(in_buf))
      continue;
    if (PropagateToFindProducerMatrixLoadFoundTirImpl(in_buf, tir,
                                                       visited_buffers))
      return true;
  }
  return false;
}

static bool PropagateToFindProducerMatrixLoadFoundTir(
    const Buffer &buffer, const PropagationTirCollector *tir) {
  BufferSet visited_buffers;
  return PropagateToFindProducerMatrixLoadFoundTirImpl(buffer, tir,
                                                        &visited_buffers);
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
