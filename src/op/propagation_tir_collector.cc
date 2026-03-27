/*!
 * \file propagation_tir_collector.cc
 * \brief TIR-based buffer producer/consumer collection.
 */

#include "propagation_tir_collector.h"
#include "builtin.h"
#include "region.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

static bool IsMatrixLoadCall(const CallNode *call) {
  if (!call->op.as<OpNode>()) return false;
  std::string name = call->op.as<OpNode>()->name;
  return name == "tl.tileop.matrix_load";
}

static bool IsDsReadFormatCall(const CallNode *call) {
  if (!call->op.as<OpNode>()) return false;
  std::string name = call->op.as<OpNode>()->name;
  return name == "tl.tileop.ds_read_format";
}

static bool IsGemmCall(const CallNode *call) {
  return call->op.same_as(tl::tl_gemm());
}

static bool IsCallExternGemm(const CallNode *call) {
  if (!call->op.same_as(builtin::call_extern()) || call->args.empty())
    return false;
  if (const auto *name = call->args[0].as<StringImmNode>()) {
    std::string s = name->value;
    return s.find("tl::gemm") == 0 || s.find("tl::tcgen5mma_gemm") == 0;
  }
  return false;
}

struct BufferAndMask {
  Buffer buffer;
  int mask{0};  // 1=read, 2=write
};

static Optional<BufferAndMask> GetBufferAndMaskFromExpr(
    const PrimExpr &expr, Map<Var, Buffer> vmap) {
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

}  // namespace

class PropagationTirCollector::Visitor : public StmtExprVisitor {
 public:
  explicit Visitor(PropagationTirCollector *collector)
      : collector_(collector) {}

  void VisitStmt_(const BufferStoreNode *op) final {
    Buffer out_buf = op->buffer;
    buffers_read_.clear();
    current_write_ = out_buf;
    StmtExprVisitor::VisitStmt_(op);
    // Don't overwrite producer_inputs_map_ when buffer was already produced by a Call
    // (e.g. ds_read_format). In-place BufferStore would overwrite with [out_buf] causing
    // self-loop in propagation. Keep the original producer chain.
    if (collector_->producer_call_map_.count(out_buf) == 0) {
      // Filter self-reference: in-place store reads from out_buf, exclude it to avoid loop
      std::vector<Buffer> filtered;
      for (const Buffer &b : buffers_read_) {
        if (!b.same_as(out_buf)) filtered.push_back(b);
      }
      collector_->producer_inputs_map_[out_buf] = std::move(filtered);
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
      if (!opt) continue;
      const auto &val = opt.value();
      if (val.mask & 2) write_bufs.push_back(val.buffer);
      if (val.mask & 1) read_bufs.push_back(val.buffer);
    }
    for (const Buffer &w : write_bufs) {
      std::vector<Buffer> read_filtered;
      for (const Buffer &r : read_bufs) {
        if (!r.same_as(w)) read_filtered.push_back(r);
      }
      collector_->producer_inputs_map_[w] = std::move(read_filtered);
      collector_->producer_call_map_[w] = call;
      for (const Buffer &r : read_bufs) {
        if (!r.same_as(w)) {
          collector_->consumer_outputs_map_[r].push_back(w);
        }
        if (IsGemmCall(call) || IsCallExternGemm(call)) {
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

Array<Buffer> PropagationTirCollector::GetProducerInputs(const Buffer &buffer) const {
  auto it = producer_inputs_map_.find(buffer);
  if (it == producer_inputs_map_.end()) return Array<Buffer>();
  Array<Buffer> ret;
  for (const Buffer &b : it->second) {
    ret.push_back(b);
  }
  return ret;
}

Array<Buffer> PropagationTirCollector::GetConsumerOutputs(const Buffer &buffer) const {
  auto it = consumer_outputs_map_.find(buffer);
  if (it == consumer_outputs_map_.end()) return Array<Buffer>();
  Array<Buffer> ret;
  for (const Buffer &b : it->second) {
    ret.push_back(b);
  }
  return ret;
}

bool PropagationTirCollector::ProducerIsMatrixLoad(const Buffer &buffer) const {
  auto it = producer_call_map_.find(buffer);
  if (it == producer_call_map_.end()) return false;
  const CallNode *call = it->second;
  return IsMatrixLoadCall(call) || IsMlsLoadTileExternCall(call);
}

bool PropagationTirCollector::ProducerIsDsReadFormat(const Buffer &buffer) const {
  auto it = producer_call_map_.find(buffer);
  if (it == producer_call_map_.end()) return false;
  const CallNode *call = it->second;
  return IsDsReadFormatCall(call) || IsDsReadFormatExternCall(call);
}

bool PropagationTirCollector::ConsumerIsGemm(const Buffer &buffer) const {
  return has_gemm_consumer_.count(buffer) != 0;
}

Optional<Call> PropagationTirCollector::GetProducerCall(const Buffer &buffer) const {
  auto it = producer_call_map_.find(buffer);
  if (it == producer_call_map_.end()) return Optional<Call>();
  return tvm::ffi::GetRef<Call>(it->second);
}

}  // namespace tl
}  // namespace tvm
