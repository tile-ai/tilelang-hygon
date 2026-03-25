/*!
 * \file propagation_util.cc
 * \brief Implementation of propagation utilities.
 *
 * Design: Use PropagationTirCollector for node lookup (covers non-TileOperator
 * like ieee_mul). mls_tile is derived locally by MatrixLoad/DsReadFormat/Gemm
 * via ComputeMlsWarpPartition, no propagation to query MatrixLoad needed.
 */

#include "propagation_util.h"
#include "gemm.h"
#include "mls.h"
#include "operator.h"
#include "propagation_tir_collector.h"

namespace tvm {
namespace tl {

using namespace tir;

static bool IsGemm(const TileOperator &op) {
  return op->GetTypeKey() == std::string("tl.Gemm");
}

static Optional<TileOperator> PropagateToFindGemmConsumerOpTir(
    Buffer buffer, const PropagationTirCollector *tir) {
  if (!tir) return Optional<TileOperator>();
  for (const Buffer &out_buf : tir->GetConsumerOutputs(buffer)) {
    if (out_buf.scope() != "local.fragment") continue;
    auto call_opt = tir->GetProducerCall(out_buf);
    if (call_opt.defined()) {
      auto op = ParseOperator(call_opt.value(), tir->GetBufferDataToBuffer());
      if (op.defined() && IsGemm(op)) return op;
    }
    // Recurse even when no producer call (e.g. BufferStore from mul/add)
    auto found = PropagateToFindGemmConsumerOpTir(out_buf, tir);
    if (found.defined()) return found;
  }
  return Optional<TileOperator>();
}

static std::optional<GemmWithInput> PropagateToFindGemmConsumerOpWithInputTir(
    Buffer buffer, const PropagationTirCollector *tir) {
  if (!tir) return std::nullopt;
  for (const Buffer &out_buf : tir->GetConsumerOutputs(buffer)) {
    if (out_buf.scope() != "local.fragment") continue;
    auto call_opt = tir->GetProducerCall(out_buf);
    if (call_opt.defined()) {
      auto op = ParseOperator(call_opt.value(), tir->GetBufferDataToBuffer());
      if (op.defined() && IsGemm(op)) {
        GemmWithInput r;
        r.gemm = Downcast<Gemm>(op);
        r.input = buffer;
        return r;
      }
    }
    // Recurse even when no producer call (e.g. BufferStore from mul/add)
    auto found = PropagateToFindGemmConsumerOpWithInputTir(out_buf, tir);
    if (found) return found;
  }
  return std::nullopt;
}

bool PropagateToFindGemmConsumer(Buffer buffer,
                                 const PropagationTirCollector *tir_collector) {
  return PropagateToFindGemmConsumerOp(buffer, tir_collector).defined();
}

Optional<TileOperator> PropagateToFindGemmConsumerOp(
    Buffer buffer, const PropagationTirCollector *tir_collector) {
  return PropagateToFindGemmConsumerOpTir(buffer, tir_collector);
}

std::optional<GemmWithInput> PropagateToFindGemmConsumerOpWithInput(
    Buffer buffer, const PropagationTirCollector *tir_collector) {
  return PropagateToFindGemmConsumerOpWithInputTir(buffer, tir_collector);
}

static bool PropagateToFindProducerMatrixLoadFoundTir(
    Buffer buffer, const PropagationTirCollector *tir) {
  if (tir->ProducerIsMatrixLoad(buffer)) return true;
  if (tir->ProducerIsDsReadFormat(buffer)) {
    auto inputs = tir->GetProducerInputs(buffer);
    if (!inputs.empty()) {
      return PropagateToFindProducerMatrixLoadFoundTir(inputs[0], tir);
    }
  }
  for (const Buffer &in_buf : tir->GetProducerInputs(buffer)) {
    if (in_buf.scope() != "local.fragment") continue;
    if (PropagateToFindProducerMatrixLoadFoundTir(in_buf, tir)) return true;
  }
  return false;
}

bool IsFromMls(Buffer buffer, const PropagationTirCollector *tir_collector) {
  return tir_collector && PropagateToFindProducerMatrixLoadFoundTir(buffer, tir_collector);
}

Array<TileOperator> GetConsumerOpsFromTir(Buffer buffer,
                                         const PropagationTirCollector *tir_collector) {
  Array<TileOperator> result;
  if (!tir_collector) return result;
  for (const Buffer &out_buf : tir_collector->GetConsumerOutputs(buffer)) {
    auto call_opt = tir_collector->GetProducerCall(out_buf);
    if (!call_opt.defined()) continue;
    auto op = ParseOperator(call_opt.value(), tir_collector->GetBufferDataToBuffer());
    if (op.defined()) result.push_back(op);
  }
  return result;
}

static std::optional<std::pair<int, int>> GetMlsTileFromProducerChainTir(
    Buffer buffer, const PropagationTirCollector *tir) {
  if (!tir) return std::nullopt;
  if (tir->ProducerIsMatrixLoad(buffer)) {
    auto call_opt = tir->GetProducerCall(buffer);
    if (!call_opt.defined()) return std::nullopt;
    auto op = ParseOperator(call_opt.value(), tir->GetBufferDataToBuffer());
    if (auto mls = op.as<MatrixLoadNode>()) {
      if (mls->mls_tile_mn > 0 && mls->mls_tile_k > 0) {
        return std::make_pair(mls->mls_tile_mn, mls->mls_tile_k);
      }
    }
    return std::nullopt;
  }
  if (tir->ProducerIsDsReadFormat(buffer)) {
    auto inputs = tir->GetProducerInputs(buffer);
    if (!inputs.empty()) {
      return GetMlsTileFromProducerChainTir(inputs[0], tir);
    }
  }
  for (const Buffer &in_buf : tir->GetProducerInputs(buffer)) {
    if (in_buf.scope() != "local.fragment") continue;
    if (auto found = GetMlsTileFromProducerChainTir(in_buf, tir)) return found;
  }
  return std::nullopt;
}

std::optional<std::pair<int, int>> GetMlsTileFromProducerChain(
    Buffer buffer, const PropagationTirCollector *tir_collector) {
  return GetMlsTileFromProducerChainTir(buffer, tir_collector);
}

}  // namespace tl
}  // namespace tvm
