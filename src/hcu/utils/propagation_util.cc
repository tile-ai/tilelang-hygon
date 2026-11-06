/*!
 * \file propagation_util.cc
 * \brief Implementation of propagation utilities.
 *
 * Design: Use PropagationTirCollector for node lookup (covers non-TileOperator
 * like ieee_mul). mls_tile is derived locally by MatrixLoad/DsReadFormat/Gemm
 * via ComputeMlsWarpPartition, no propagation to query MatrixLoad needed.
 */

#include "propagation_util.h"
#include "hcu/op/builtin.h"
#include "op/gemm.h"
#include "op/operator.h"
#include "propagation_tir_collector.h"

#include <tvm/tirx/builtin.h>
#include <unordered_set>

namespace tvm {
namespace tl {

using namespace tirx;

static bool IsGemm(const TileOperator &op) {
  const std::string type_key = op->GetTypeKey();
  return type_key == "tl.Gemm" || type_key == "tl.GemmBlockScaled";
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
  r.site = GetRef<Call>(call);
  return r;
}

static bool IsSharedLikeScope(const Buffer &buffer) {
  const String &scope = buffer.scope();
  return scope == "shared" || scope == "shared.dyn" || scope == "shared.tmem";
}

using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;

static Optional<TileOperator>
PropagateToFindGemmConsumerOpTir(const Buffer &buffer,
                                 const PropagationTirCollector *tir,
                                 int after_stmt_order) {
  auto result = QueryGemmConsumers(buffer, tir, after_stmt_order, false);
  if (result && !result->empty())
    return result->front().gemm;
  return std::nullopt;
}

static std::optional<GemmWithInput>
PropagateToFindGemmConsumerOpWithInputTir(const Buffer &buffer,
                                          const PropagationTirCollector *tir,
                                          int after_stmt_order) {
  auto result = QueryGemmConsumers(buffer, tir, after_stmt_order, false);
  if (result && !result->empty())
    return result->front();
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

std::optional<std::vector<GemmWithInput>>
QueryGemmConsumers(Buffer buffer, const PropagationTirCollector *collector,
                   int after_order, bool exclusive,
                   const GemmRelayPredicate &accept) {
  if (!collector)
    return std::nullopt;
  BufferSet visited;
  std::unordered_set<const CallNode *, CallNodePtrHash, CallNodePtrEqual> seen;
  std::vector<GemmWithInput> result;
  std::function<bool(const Buffer &)> visit = [&](const Buffer &input) {
    if (!visited.insert(input).second)
      return true;
    bool has_reader = false;
    // Preserve the existing consumer-output traversal order. In particular,
    // an expression relay preceding a direct GEMM must not be reordered after
    // that GEMM just because the latter has a ReaderCallRecord.
    const auto readers = collector->GetReaderCalls(input);
    for (const auto &output : collector->GetConsumerOutputs(input)) {
      if (IsSharedLikeScope(output)) {
        if (exclusive)
          return false;
        continue;
      }
      const auto preferred =
          after_order < 0
              ? collector->FindLastGemmProducerReading(output, input)
              : collector->FindFirstGemmProducerReading(output, input,
                                                        after_order);
      if (preferred) {
        if (auto gemm = GemmWithInputFromCall(preferred->call, input)) {
          has_reader = true;
          if (seen.insert(preferred->call).second)
            result.push_back(*gemm);
        }
      }
      bool call_output = false;
      bool relay = false;
      for (const auto &reader : readers) {
        if (!reader.call || !reader.write.same_as(output))
          continue;
        call_output = true;
        if (reader.stmt_order <= after_order)
          continue;
        has_reader = true;
        if (auto gemm = GemmWithInputFromCall(reader.call, input)) {
          if (seen.insert(reader.call).second)
            result.push_back(*gemm);
        } else if (!accept || accept(input, output, reader.call)) {
          relay = true;
        } else if (exclusive) {
          return false;
        }
      }
      if (!call_output) {
        has_reader = true;
        if (accept && !accept(input, output, nullptr)) {
          if (exclusive)
            return false;
          continue;
        }
        relay = true;
      }
      if (relay && !visit(output) && exclusive)
        return false;
    }
    return has_reader || !exclusive;
  };
  if (!visit(buffer))
    return std::nullopt;
  return result;
}

std::vector<GemmWithInput> PropagateToFindAllGemmConsumersAfterCall(
    Buffer buffer, const PropagationTirCollector *collector,
    const CallNode *after_site_call) {
  int order = collector && after_site_call
                  ? collector->GetCallStmtOrder(after_site_call)
                  : -1;
  auto result = QueryGemmConsumers(buffer, collector, order, false);
  return result.value_or(std::vector<GemmWithInput>{});
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
      return PropagateToFindProducerMatrixLoadFoundTirImpl(inputs[0], tir,
                                                           visited_buffers);
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

static bool
PropagateToFindProducerMatrixLoadFoundTir(const Buffer &buffer,
                                          const PropagationTirCollector *tir) {
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
