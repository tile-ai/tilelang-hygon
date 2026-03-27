/*!
 * \file target/codegen.h
 * \brief Utility to generate code
 */
#ifndef TVM_TL_TARGET_CODEGEN_HCU_H_
#define TVM_TL_TARGET_CODEGEN_HCU_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangHCU final : public CodeGenC {
public:
  CodeGenTileLangHCU();
  std::string Finish();
  // override behavior
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f, std::ostream &os) final;
  void VisitStmt_(const ForNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) final; // NOLINT(*)
  void PrintVecBinaryOp(const std::string &op, DataType t, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) final;      // NOLINT(*)
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  std::string GetVecLoad(DataType t, const BufferNode *buffer,
                         PrimExpr base) final;
  void PrintVecStore(const BufferNode* buffer, DataType t, PrimExpr base,
                     const std::string& value) final;
  void PrintVecElemLoad(const std::string &vec, DataType t, int i,
                        std::ostream &os) final; // NOLINT(*)
  void PrintVecElemStore(const std::string &vec, DataType t, int i,
                         const std::string &value) final;
  void BindThreadIndex(const IterVar &iv) final; // NOLINT(*)
  void PrintVecElemLoadExpr(DataType t, int i, const std::string &value,
                            std::ostream &os) final;
  std::string CastFromTo(std::string value, DataType from,
                         DataType target) final;
  // Override IfThenElse to fold amd_buffer_load/store with predicate when
  // if-else contains only buffer load/store and zeros.
  void VisitStmt_(const IfThenElseNode *op) final;

  // overload visitor
  void VisitExpr_(const RampNode *op, std::ostream &os) final;      // NOLINT(*)
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitStmt_(const BlockNode *op) final;

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const PrimFunc &f);

protected:
  // Override BufferStore lowering so we can emit CK buffer store ops for
  // global memory using amd_buffer_store.
  void VisitStmt_(const BufferStoreNode *op) final;

  virtual std::string GetBufferRef(DataType t, const BufferNode *buffer,
                                   PrimExpr index) final;
  void PrintCallExtern(Type ret_type, ffi::String global_symbol,
                       const ffi::Array<PrimExpr> &args, bool skip_first_arg,
                       std::ostream &os) final; // NOLINT(*)

private:
  // Structure to store buffer descriptor for HCU buffer load/store optimization
  struct BufferDesc {
    std::string wave_ptr;           // wavewise base pointer
    std::string offset;
    std::string element_space_size;
    std::string data_type;
    std::string scope;
    int num_elements;
  };

  // Handle volatile loads
  void HandleVolatileLoads(const std::string &value, const BufferLoadNode *op,
                           std::ostream &os) final;

  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  BufferDesc GetBufferDesc(DataType t, const BufferNode *buffer, PrimExpr base);
  std::string GetVecLoadWithPredicate(DataType t, const BufferNode *buffer,
                                     PrimExpr base, const std::string &pred);
  void PrintVecStoreWithPredicate(const BufferNode *buffer, DataType t,
                                 PrimExpr base, const std::string &value,
                                 const std::string &pred);
  std::string GetCurrentPredicate() const;
  bool IsFoldableIfThenElse(const IfThenElseNode *op) const;
  bool IsCollapsibleRedundantIfElse(const IfThenElseNode *op) const;
  bool CanUseVMBufferOps(const BufferNode *buffer, int num_elements) const {
    // Match by param name (robust across Var renaming in later passes)
    for (const auto &p : buffer_ops_disable_param_names_) {
      if (buffer->data->name_hint == p)
        return false;
    }
    auto value = std::getenv("HCU_USE_BUFFER_OPS");
    auto scope = GetPtrStorageScope(buffer->data);
    return (value == nullptr || std::atoi(value) != 0) && scope == "global" &&
           (num_elements * buffer->dtype.bits() <= 128);
  }
  bool CanUseLDSBufferOps(const BufferStoreNode *buffer_store) {
    auto value = std::getenv("HCU_DIRECT_TO_LDS");
    if (value == nullptr || std::atoi(value) == 0)
      return false;

    if (const auto *buffer_load = buffer_store->value.as<BufferLoadNode>()) {
      auto src_scope = GetPtrStorageScope(buffer_load->buffer->data);
      auto dst_scope = GetPtrStorageScope(buffer_store->buffer->data);

      return src_scope == "global" && (dst_scope == "shared" || dst_scope == "shared.dyn") &&
             (buffer_load->dtype.bits() * buffer_load->dtype.lanes() <= 128);
    }

    return false;
  }

  bool TryToEmitLDSBufferOp(const BufferStoreNode *op);

  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangHCU *p);

  // whether need math_constants.h
  bool need_math_constants_h_{false};
  // whether need mfma.h
  bool need_wmma_h_{false};
  // whether need fp8.h
  bool enable_fp8_{false};
  // whether need gemm_mls.h (when gemm uses MLS or matrix_load exists)
  bool enable_gemm_mls_{false};
  // The size of the barrier array in shared memory
  int barrier_count_ = -1;
  // whether need mma.h
  bool need_mma_h_{false};
  // whether need cast_smem_ptr_to_int helper function
  bool need_cast_smem_ptr_to_int_{false};
  // The name of the barrier array in shared memory
  const std::string barrier_name_ = "barrier";
  // The alignment of the barrier array in shared memory
  // Set to 16 to maintain minimum alignment requirements for async bulk copy
  const int barrier_alignment_bytes_ = 16;
  std::unordered_map<const VarNode*, bool> direct_to_lds_map_;
  std::unordered_set<std::string> buffer_ops_disable_param_names_;
  std::vector<std::string> predicate_stack_;
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_HCU_H_
