/*!
 * \file propagation_tir_collector.h
 * \brief TIR-based buffer producer/consumer collection for propagation.
 * Covers all ops (including non-TileOperator like ieee_mul) by traversing TIR.
 */

#ifndef TVM_TL_OP_PROPAGATION_TIR_COLLECTOR_H_
#define TVM_TL_OP_PROPAGATION_TIR_COLLECTOR_H_

#include <tvm/tir/buffer.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include "../support/ffi_aliases.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Collect buffer producer/consumer from TIR by traversing BufferStore,
 * BufferLoad, and call_extern (tvm_access_ptr). Covers all ops including
 * non-TileOperator (e.g. ieee_mul).
 */
class PropagationTirCollector {
 public:
  explicit PropagationTirCollector(Map<Var, Buffer> buffer_data_to_buffer);

  void Collect(const Stmt &body);

  Array<Buffer> GetProducerInputs(const Buffer &buffer) const;
  Array<Buffer> GetConsumerOutputs(const Buffer &buffer) const;
  bool ProducerIsMatrixLoad(const Buffer &buffer) const;
  bool ProducerIsDsReadFormat(const Buffer &buffer) const;
  bool ConsumerIsGemm(const Buffer &buffer) const;
  /*! \brief Get the Call that produces the buffer, or nullopt if unknown. */
  Optional<Call> GetProducerCall(const Buffer &buffer) const;
  /*! \brief Get Var->Buffer map for ParseOperator. */
  Map<Var, Buffer> GetBufferDataToBuffer() const { return buffer_data_to_buffer_; }

  class Visitor;

 private:
  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_map<Buffer, std::vector<Buffer>, ObjectPtrHash, ObjectPtrEqual>
      producer_inputs_map_;
  std::unordered_map<Buffer, std::vector<Buffer>, ObjectPtrHash, ObjectPtrEqual>
      consumer_outputs_map_;
  std::unordered_map<Buffer, const CallNode *, ObjectPtrHash, ObjectPtrEqual>
      producer_call_map_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> has_gemm_consumer_;
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_OP_PROPAGATION_TIR_COLLECTOR_H_
