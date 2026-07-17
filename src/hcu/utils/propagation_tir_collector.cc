/*!
 * \file propagation_tir_collector.cc
 * \brief TIR-based buffer producer/consumer collection.
 */

#include "propagation_tir_collector.h"
#include "hcu/utils/extern_call_checker.h"
#include "op/builtin.h"
#include "op/region.h"

#include <algorithm>

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

static bool IsMatrixLoadCall(const CallNode *call) {
  if (!call->op.as<OpNode>())
    return false;
  std::string name = call->op.as<OpNode>()->name;
  return name == "tl.tileop.matrix_load";
}

static bool IsDsReadFormatCall(const CallNode *call) {
  if (!call->op.as<OpNode>())
    return false;
  std::string name = call->op.as<OpNode>()->name;
  return name == "tl.tileop.ds_read_format";
}

static bool BufferInInputs(const Buffer &buf,
                           const std::vector<Buffer> &inputs) {
  for (const Buffer &in : inputs) {
    if (in.same_as(buf))
      return true;
  }
  return false;
}

struct BufferAndMask {
  Buffer buffer;
  int mask{0}; // 1=read, 2=write
};

static Optional<BufferAndMask> GetBufferAndMaskFromExpr(const PrimExpr &expr,
                                                        Map<Var, Buffer> vmap) {
  if (const auto *call = expr.as<CallNode>()) {
    if (call->op.same_as(builtin::tvm_access_ptr()) && call->args.size() >= 5) {
      if (const auto *var = call->args[1].as<VarNode>()) {
        auto it = vmap.find(tvm::ffi::GetRef<Var>(var));
        if (it != vmap.end()) {
          BufferAndMask r;
          r.buffer = (*it).second;
          if (const auto *m = call->args[4].as<IntImmNode>()) {
            r.mask = static_cast<int>(m->value);
          }
          return r;
        }
      }
    } else if (call->op.same_as(RegionOp::Get()) && call->args.size() >= 2) {
      if (const auto *load = call->args[0].as<BufferLoadNode>()) {
        BufferAndMask r;
        r.buffer = load->buffer;
        if (const auto *m = call->args[1].as<IntImmNode>()) {
          r.mask = static_cast<int>(m->value);
        }
        return r;
      }
    }
  } else if (const auto *load = expr.as<BufferLoadNode>()) {
    BufferAndMask r;
    r.buffer = load->buffer;
    r.mask = 1;
    return r;
  }
  return Optional<BufferAndMask>();
}

} // namespace

void PropagationTirCollector::AppendProducerRecord(Buffer write,
                                                   std::vector<Buffer> inputs,
                                                   const CallNode *call) {
  ProducerRecord rec;
  rec.call = call;
  rec.inputs = std::move(inputs);
  rec.stmt_order = next_stmt_order_++;
  if (call != nullptr) {
    call_stmt_order_[call] = rec.stmt_order;
  }
  producer_records_map_[write].push_back(std::move(rec));
}

class PropagationTirCollector::Visitor : public StmtExprVisitor {
public:
  explicit Visitor(PropagationTirCollector *collector)
      : collector_(collector) {}

  void VisitStmt_(const BufferStoreNode *op) final {
    Buffer out_buf = op->buffer;
    buffers_read_.clear();
    current_write_ = out_buf;
    StmtExprVisitor::VisitStmt_(op);
    // Don't overwrite producer_inputs_map_ when buffer was already produced by
    // a Call (e.g. ds_read_format). In-place BufferStore would overwrite with
    // [out_buf] causing self-loop in propagation. Keep the original producer
    // chain.
    if (collector_->producer_records_map_.count(out_buf) == 0) {
      std::vector<Buffer> filtered;
      for (const Buffer &b : buffers_read_) {
        if (!b.same_as(out_buf))
          filtered.push_back(b);
      }
      collector_->producer_inputs_map_[out_buf] = filtered;
      collector_->AppendProducerRecord(out_buf, filtered, nullptr);
    }
    current_write_ = Buffer();
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      VisitCallForBuffers(call);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Buffer b = op->buffer;
    buffers_read_.push_back(b);
    if (current_write_.defined() && !b.same_as(current_write_)) {
      collector_->consumer_outputs_map_[b].push_back(current_write_);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr()) && op->args.size() >= 5) {
      if (const auto *var = op->args[1].as<VarNode>()) {
        auto it =
            collector_->buffer_data_to_buffer_.find(tvm::ffi::GetRef<Var>(var));
        if (it != collector_->buffer_data_to_buffer_.end()) {
          Buffer b = (*it).second;
          int mask = 0;
          if (const auto *m = op->args[4].as<IntImmNode>()) {
            mask = static_cast<int>(m->value);
          }

          if (mask & 1) {
            buffers_read_.push_back(b);
            if (current_write_.defined() && !b.same_as(current_write_)) {
              collector_->consumer_outputs_map_[b].push_back(current_write_);
            }
          }
        }
      }
    } else if (op->op.same_as(builtin::address_of()) && op->args.size() >= 1) {
      if (const auto *load = op->args[0].as<BufferLoadNode>()) {
        Buffer b = load->buffer;
        buffers_read_.push_back(b);
        if (current_write_.defined() && !b.same_as(current_write_)) {
          collector_->consumer_outputs_map_[b].push_back(current_write_);
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

private:
  void VisitCallForBuffers(const CallNode *call) {
    Array<Buffer> read_bufs;
    Array<Buffer> write_bufs;
    int arg_start = 0;
    if (call->op.same_as(builtin::call_extern()) && !call->args.empty()) {
      arg_start = 1;
    }
    for (size_t i = arg_start; i < call->args.size(); i++) {
      auto opt = GetBufferAndMaskFromExpr(call->args[i],
                                          collector_->buffer_data_to_buffer_);
      if (!opt)
        continue;
      const auto &val = opt.value();
      if (val.mask & 2)
        write_bufs.push_back(val.buffer);
      if (val.mask & 1)
        read_bufs.push_back(val.buffer);
    }
    for (const Buffer &w : write_bufs) {
      std::vector<Buffer> read_filtered;
      for (const Buffer &r : read_bufs) {
        if (!r.same_as(w))
          read_filtered.push_back(r);
      }
      collector_->producer_inputs_map_[w] = read_filtered;
      collector_->AppendProducerRecord(w, read_filtered, call);
      for (const Buffer &r : read_bufs) {
        if (!r.same_as(w)) {
          collector_->consumer_outputs_map_[r].push_back(w);
        }
        if (IsGemmTileOpCall(call)) {
          collector_->has_gemm_consumer_.insert(r);
        }
      }
    }
  }

  PropagationTirCollector *collector_;
  std::vector<Buffer> buffers_read_;
  Buffer current_write_;
};

PropagationTirCollector::PropagationTirCollector(
    Map<Var, Buffer> buffer_data_to_buffer)
    : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)) {}

void PropagationTirCollector::Collect(const Stmt &body) {
  Visitor v(this);
  v(body);
}

Array<Buffer>
PropagationTirCollector::GetProducerInputs(const Buffer &buffer) const {
  auto it = producer_inputs_map_.find(buffer);
  if (it == producer_inputs_map_.end())
    return Array<Buffer>();
  Array<Buffer> ret;
  for (const Buffer &b : it->second) {
    ret.push_back(b);
  }
  return ret;
}

Array<Buffer>
PropagationTirCollector::GetConsumerOutputs(const Buffer &buffer) const {
  auto it = consumer_outputs_map_.find(buffer);
  if (it == consumer_outputs_map_.end())
    return Array<Buffer>();
  Array<Buffer> ret;
  for (const Buffer &b : it->second) {
    ret.push_back(b);
  }
  return ret;
}

std::vector<ProducerRecord>
PropagationTirCollector::GetProducerRecords(const Buffer &buffer) const {
  auto it = producer_records_map_.find(buffer);
  if (it == producer_records_map_.end())
    return {};
  return it->second;
}

std::vector<ReaderCallRecord>
PropagationTirCollector::GetReaderCalls(const Buffer &buffer) const {
  std::vector<ReaderCallRecord> result;
  for (const auto &kv : producer_records_map_) {
    const Buffer &write = kv.first;
    for (const ProducerRecord &rec : kv.second) {
      if (rec.call == nullptr)
        continue;
      if (!BufferInInputs(buffer, rec.inputs))
        continue;
      ReaderCallRecord r;
      r.call = rec.call;
      r.write = write;
      r.stmt_order = rec.stmt_order;
      result.push_back(r);
    }
  }
  std::sort(result.begin(), result.end(),
            [](const ReaderCallRecord &a, const ReaderCallRecord &b) {
              return a.stmt_order < b.stmt_order;
            });
  return result;
}

static bool IsGemmProducerRecord(const ProducerRecord &rec) {
  return rec.call != nullptr && IsGemmTileOpCall(rec.call);
}

std::optional<ProducerRecord>
PropagationTirCollector::FindFirstGemmProducerReading(const Buffer &write_buf,
                                                      const Buffer &read_buf,
                                                      int after_order) const {
  auto it = producer_records_map_.find(write_buf);
  if (it == producer_records_map_.end())
    return std::nullopt;
  const ProducerRecord *best = nullptr;
  for (const ProducerRecord &rec : it->second) {
    if (rec.stmt_order <= after_order)
      continue;
    if (!IsGemmProducerRecord(rec))
      continue;
    if (!BufferInInputs(read_buf, rec.inputs))
      continue;
    if (best == nullptr || rec.stmt_order < best->stmt_order) {
      best = &rec;
    }
  }
  if (best == nullptr)
    return std::nullopt;
  return *best;
}

std::optional<ProducerRecord>
PropagationTirCollector::FindLastGemmProducerReading(
    const Buffer &write_buf, const Buffer &read_buf) const {
  auto it = producer_records_map_.find(write_buf);
  if (it == producer_records_map_.end())
    return std::nullopt;
  const ProducerRecord *best = nullptr;
  for (const ProducerRecord &rec : it->second) {
    if (!IsGemmProducerRecord(rec))
      continue;
    if (!BufferInInputs(read_buf, rec.inputs))
      continue;
    if (best == nullptr || rec.stmt_order > best->stmt_order) {
      best = &rec;
    }
  }
  if (best == nullptr)
    return std::nullopt;
  return *best;
}

int PropagationTirCollector::GetCallStmtOrder(const CallNode *call) const {
  if (call == nullptr)
    return -1;
  auto it = call_stmt_order_.find(call);
  if (it == call_stmt_order_.end())
    return -1;
  return it->second;
}

bool PropagationTirCollector::ProducerIsMatrixLoad(const Buffer &buffer) const {
  for (const ProducerRecord &rec : GetProducerRecords(buffer)) {
    if (rec.call != nullptr &&
        (IsMatrixLoadCall(rec.call) || IsMlsLoadTileExternCall(rec.call))) {
      return true;
    }
  }
  return false;
}

bool PropagationTirCollector::ProducerIsDsReadFormat(
    const Buffer &buffer) const {
  for (const ProducerRecord &rec : GetProducerRecords(buffer)) {
    if (rec.call != nullptr &&
        (IsDsReadFormatCall(rec.call) || IsDsReadFormatExternCall(rec.call))) {
      return true;
    }
  }
  return false;
}

bool PropagationTirCollector::ConsumerIsGemm(const Buffer &buffer) const {
  return has_gemm_consumer_.count(buffer) != 0;
}

Optional<Call>
PropagationTirCollector::GetProducerCall(const Buffer &buffer) const {
  const auto records = GetProducerRecords(buffer);
  for (auto it = records.rbegin(); it != records.rend(); ++it) {
    if (it->call != nullptr) {
      return tvm::ffi::GetRef<Call>(it->call);
    }
  }
  return Optional<Call>();
}

} // namespace tl
} // namespace tvm
