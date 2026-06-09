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

#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

struct CallNodePtrHash {
  size_t operator()(const CallNode *ptr) const {
    return std::hash<const void *>()(static_cast<const void *>(ptr));
  }
};

struct CallNodePtrEqual {
  bool operator()(const CallNode *a, const CallNode *b) const { return a == b; }
};

/*!
 * \brief One producer event: a tile-op call (or BufferStore) that writes `write`
 * after reading `inputs`, in program order `stmt_order`.
 *
 * `stmt_order` is monotonic over the kernel body and is used to pair a
 * ds_read_format (or other reader) with the **first** downstream GEMM that
 * actually consumes the same fragment after that site — even when multiple GEMMs
 * inplace-accumulate into the same `C_local`.
 */
struct ProducerRecord {
  const CallNode *call{nullptr};  // null for pure BufferStore producers
  std::vector<Buffer> inputs;
  int stmt_order{0};
};

/*!
 * \brief A tile-op call that reads `read` (among its inputs) and writes `write`.
 */
struct ReaderCallRecord {
  const CallNode *call{nullptr};
  Buffer write;
  int stmt_order{0};
};

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

  /*! \brief All producer events for `buffer` (append order = program order). */
  std::vector<ProducerRecord> GetProducerRecords(const Buffer &buffer) const;

  /*!
   * \brief Tile-op calls that list `buffer` among their inputs.
   * One entry per producer event (duplicates kept when the same op pattern repeats).
   */
  std::vector<ReaderCallRecord> GetReaderCalls(const Buffer &buffer) const;

  /*!
   * \brief First GEMM producer of `write_buf` (stmt_order > after_order) whose
   * inputs include `read_buf`. Returns nullopt if none.
   */
  std::optional<ProducerRecord> FindFirstGemmProducerReading(const Buffer &write_buf,
                                                             const Buffer &read_buf,
                                                             int after_order = -1) const;

  /*!
   * \brief Last GEMM producer of `write_buf` whose inputs include `read_buf`.
   * Matches the legacy single-producer (last-writer) default when no stmt site
   * is specified.
   */
  std::optional<ProducerRecord> FindLastGemmProducerReading(const Buffer &write_buf,
                                                            const Buffer &read_buf) const;

  int GetCallStmtOrder(const CallNode *call) const;

  bool ProducerIsMatrixLoad(const Buffer &buffer) const;
  bool ProducerIsDsReadFormat(const Buffer &buffer) const;
  bool ConsumerIsGemm(const Buffer &buffer) const;

  /*!
   * \brief Last tile-op producer of `buffer` (backward compatible with the old
   * single-producer map). Prefer GetProducerRecords for multi-writer buffers.
   */
  Optional<Call> GetProducerCall(const Buffer &buffer) const;

  Map<Var, Buffer> GetBufferDataToBuffer() const { return buffer_data_to_buffer_; }

  class Visitor;

 private:
  friend class Visitor;
  void AppendProducerRecord(Buffer write, std::vector<Buffer> inputs,
                            const CallNode *call);

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::unordered_map<Buffer, std::vector<Buffer>, ObjectPtrHash, ObjectPtrEqual>
      producer_inputs_map_;
  std::unordered_map<Buffer, std::vector<Buffer>, ObjectPtrHash, ObjectPtrEqual>
      consumer_outputs_map_;
  std::unordered_map<Buffer, std::vector<ProducerRecord>, ObjectPtrHash,
                     ObjectPtrEqual>
      producer_records_map_;
  std::unordered_map<const CallNode *, int, CallNodePtrHash, CallNodePtrEqual>
      call_stmt_order_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> has_gemm_consumer_;
  int next_stmt_order_{0};
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_OP_PROPAGATION_TIR_COLLECTOR_H_
