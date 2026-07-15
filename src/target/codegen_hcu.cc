/*!
 * \file target/codegen_hcu.cc
 */

#include "codegen_hcu.h"
#include "arith/pattern_match.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/node/structural_equal.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../op/builtin.h"

namespace tvm {
namespace codegen {

namespace {

bool IsValidCPAsyncTransferBytes(int64_t bytes) {
  return bytes == 4 || bytes == 8 || bytes == 16;
}

std::optional<DataType> GetAccessPtrElementType(const PrimExpr &expr) {
  const auto *ptr_call = expr.as<CallNode>();
  if (ptr_call == nullptr) {
    return std::nullopt;
  }
  if (ptr_call->op.same_as(builtin::address_of())) {
    const auto *buffer_load = ptr_call->args[0].as<BufferLoadNode>();
    ICHECK(buffer_load) << "address_of arg must be BufferLoad";
    return buffer_load->buffer->dtype;
  }
  if (ptr_call->op.same_as(builtin::tvm_access_ptr())) {
    ICHECK(!ptr_call->args.empty());
    return ptr_call->args[0].dtype();
  }
  if (ptr_call->op.same_as(tl::access_ptr())) {
    ICHECK_EQ(ptr_call->args.size(), 3U)
        << "tl.access_ptr expects 3 args: (BufferLoad, extent, rw_mask)";
    const auto *buffer_load = ptr_call->args[0].as<BufferLoadNode>();
    ICHECK(buffer_load) << "tl.access_ptr arg0 must be BufferLoad";
    return buffer_load->buffer->dtype;
  }
  return std::nullopt;
}

int GetTileLangCPAsyncTransferBytes(const CallNode *op) {
  ICHECK(op->args.size() == 3 || op->args.size() == 4)
      << "tl::ptx_cp_async expects 3 or 4 arguments (dst_access_ptr, "
         "src_access_ptr, num_elems, [predicate])";
  const auto *num_elems_imm = op->args[2].as<IntImmNode>();
  ICHECK(num_elems_imm) << "tl::ptx_cp_async num_elems must be IntImm, but got "
                        << op->args[2];
  int64_t num_elems = num_elems_imm->value;
  ICHECK_GT(num_elems, 0);

  auto dst_elem_type = GetAccessPtrElementType(op->args[0]);
  auto src_elem_type = GetAccessPtrElementType(op->args[1]);
  ICHECK(dst_elem_type.has_value() && src_elem_type.has_value())
      << "tl::ptx_cp_async expects address_of, tl.access_ptr, or "
         "tvm_access_ptr operands";

  int64_t dst_total_bits =
      num_elems * dst_elem_type.value().bits() * dst_elem_type.value().lanes();
  int64_t src_total_bits =
      num_elems * src_elem_type.value().bits() * src_elem_type.value().lanes();
  ICHECK_EQ(dst_total_bits, src_total_bits)
      << "tl::ptx_cp_async requires src/dst transfer widths to match, but got "
      << dst_total_bits << " vs " << src_total_bits << " bits";
  ICHECK_EQ(dst_total_bits % 8, 0)
      << "tl::ptx_cp_async requires byte-aligned transfers, but got "
      << dst_total_bits << " bits";

  int64_t total_bytes = dst_total_bits / 8;
  ICHECK(IsValidCPAsyncTransferBytes(total_bytes))
      << "tl::ptx_cp_async requires a final PTX byte width in {4, 8, 16}, but "
         "got "
      << total_bytes;
  return static_cast<int>(total_bytes);
}

/// CK buffer templates use raw ``unsigned short`` for bf16; LDS/MMAC use
/// ``bfloat16_t`` (__bf16).
std::string HcuCkTemplateElemType(DataType dtype) {
  DataType elem_type = dtype.element_of();
  int elem_bits = elem_type.bits();
  if (elem_bits == 8) {
    if (elem_type.is_float8_e4m3fn() || elem_type.is_float8_e4m3fnuz() ||
        elem_type.is_float8_e4m3() ||
        elem_type.code() == DataType::kFloat8_e4m3b11fnuz) {
      return "tl::fp8_t";
    }
    if (elem_type.is_float8_e5m2() || elem_type.is_float8_e5m2fnuz() ||
        elem_type.code() == DataType::kFloat8_e5m2) {
      return "tl::bf8_t";
    }
    return "int8_t";
  }
  if (elem_bits == 16) {
    return elem_type.is_bfloat16() ? "unsigned short" : "half_t";
  }
  if (elem_bits == 32) {
    return "float";
  }
  if (elem_bits == 64) {
    return "double";
  }
  LOG(FATAL) << "Unsupported HCU CK element bits: " << elem_bits;
  return "";
}

/// CK **src** pointers (`const T*`): special-case bf16/i64 mappings, and always
/// cast generic pointers to `const T*` (some HCU paths materialize tensor
/// handles as `void*`).
std::string HcuCkBufferSrcPtrExpr(DataType dtype,
                                  const std::string &wave_ptr_expr) {
  return "reinterpret_cast<const " + HcuCkTemplateElemType(dtype) + "*>(" +
         wave_ptr_expr + ")";
}

/// CK **dst** pointers (`T*`): `reinterpret_cast<unsigned short*>` for
/// `bfloat16_t`, else C-style cast to element pointer, e.g. `(float*)buf`.
std::string HcuCkBufferDstPtrExpr(DataType dtype,
                                  const std::string &wave_ptr_expr) {
  return "reinterpret_cast<" + HcuCkTemplateElemType(dtype) + "*>(" +
         wave_ptr_expr + ")";
}

} // namespace

static std::string GetFP8Type(DataType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.is_scalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "_2";
  } else if (lanes == 4) {
    vec = "_4";
  } else if (lanes == 8) {
    vec = "_8";
  } else if (lanes == 16) {
    vec = "_16";
  } else {
    LOG(FATAL) << "Only support scalar and vector types of width (2, 4, 8, 16) "
                  "for FP8";
  }
  if (type.is_float8_e4m3fn() || type.is_float8_e4m3fnuz() ||
      type.is_float8_e4m3() || type.code() == DataType::kFloat8_e4m3b11fnuz) {
    stream << "fp8_e4" << vec << "_t";
  } else if (type.is_float8_e5m2() || type.is_float8_e5m2fnuz() ||
             type.code() == DataType::kFloat8_e5m2) {
    stream << "fp8_e5" << vec << "_t";
  } else if (type.code() == DataType::kFloat8_e8m0fnu) {
    stream << "fp8_e8" << vec << "_t";
  } else {
    LOG(FATAL) << "Unsupported FP8 type in HCU codegen: " << type;
  }
  return stream.str();
}

/*!
 * \brief Replace patterns with replacement strings.
 * \note should use std::format instead when codebase is ported to C++20.
 */
class Replacer {
public:
  void register_rule(const std::string &pattern,
                     const std::string &replacement) {
    _rules.emplace_back(pattern, replacement);
  }
  std::string rewrite(std::string str) {
    for (auto &&rule : _rules) {
      auto [pattern, replacement] = rule;
      size_t len = pattern.size();
      size_t new_len = replacement.size();
      size_t pos = str.find(pattern);
      while (pos != std::string::npos) {
        str = str.replace(pos, len, replacement);
        pos = str.find(pattern, pos + new_len);
      }
    }
    return str;
  }
  void empty_rules() { _rules.clear(); }

private:
  std::vector<std::pair<std::string, std::string>> _rules;
};

CodeGenTileLangHCU::CodeGenTileLangHCU() { restrict_keyword_ = "__restrict__"; }

void CodeGenTileLangHCU::PrintFuncPrefix(std::ostream &os) {
  os << "extern \"C\" __global__ ";
}

class LaunchConfigExtractor : public tir::StmtVisitor {
private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->var->name_hint == "threadIdx.x" ||
          iv->thread_tag == "threadIdx.x") {
        threadIdx_x_ext = op->value;
      } else if (iv->var->name_hint == "threadIdx.y" ||
                 iv->thread_tag == "threadIdx.y") {
        threadIdx_y_ext = op->value;
      } else if (iv->var->name_hint == "threadIdx.z" ||
                 iv->thread_tag == "threadIdx.z") {
        threadIdx_z_ext = op->value;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

public:
  PrimExpr threadIdx_x_ext = Integer(1);
  PrimExpr threadIdx_y_ext = Integer(1);
  PrimExpr threadIdx_z_ext = Integer(1);
};

// TODO: Add a optimization pass to identify eligible direct_to_lds buffer store
//       patterns instead of performing check here.
bool CodeGenTileLangHCU::StoreWouldEmitLDSBufferOp(
    const BufferStoreNode *buffer_store) const {
  const auto *buffer_load = buffer_store->value.as<BufferLoadNode>();
  const auto *dst_var = buffer_store->buffer->data.get();
  // Same as TryToEmitLDSBufferOp: count + lookup. Use at() (const map has no
  // []).
  if (buffer_load == nullptr || !direct_to_lds_map_.count(dst_var) ||
      !direct_to_lds_map_.at(dst_var))
    return false;
  auto src_scope = GetPtrStorageScope(buffer_load->buffer->data);
  auto dst_scope = GetPtrStorageScope(buffer_store->buffer->data);
  int read_bytes = buffer_load->dtype.bits() * buffer_load->dtype.lanes() / 8;
  return src_scope == "global" &&
         (dst_scope == "shared" || dst_scope == "shared.dyn") &&
         (read_bytes == 4 || read_bytes == 8 || read_bytes == 16);
}

bool CodeGenTileLangHCU::TryToEmitLDSBufferOp(
    const BufferStoreNode *buffer_store) {
  if (!StoreWouldEmitLDSBufferOp(buffer_store))
    return false;
  const auto *buffer_load = buffer_store->value.as<BufferLoadNode>();
  ICHECK(buffer_load != nullptr);

  DataType value_dtype = buffer_store->value.dtype();
  auto store_desc = GetBufferDesc(value_dtype, buffer_store->buffer.get(),
                                  buffer_store->indices[0]);
  auto load_desc = GetBufferDesc(buffer_load->dtype, buffer_load->buffer.get(),
                                 buffer_load->indices[0]);
  std::string lds_base =
      HcuCkBufferDstPtrExpr(value_dtype, store_desc.wave_ptr);
  std::string global_base =
      HcuCkBufferSrcPtrExpr(buffer_load->dtype, load_desc.wave_ptr);
  PrintIndent();
  stream << "ck::hcu_direct_load_global_to_lds<"
         << HcuCkTemplateElemType(buffer_load->dtype) << ", "
         << value_dtype.lanes() << ", false>(" << global_base
         << ", "                      // global_base_ptr
         << load_desc.offset << ", "  // global_offset (in elements)
         << lds_base << ", "          // lds_base_ptr
         << store_desc.offset << ", " // lds_offset (in elements)
         << "true, "                  // is_valid
         << load_desc.element_space_size
         << ", "     // src_element_space_size (in elements)
         << "0);\n"; // wave_lds_wrap_offset

  return true;
}

void CodeGenTileLangHCU::PrintExtraAttrs(const PrimFunc &f, std::ostream &os) {
  if (auto waves = f->GetAttr<Integer>(tl::attr::kHcuWdraWavesPerTg)) {
    stream << " __attribute__((hcu_wdra_waves_per_tg(" << waves.value()->value
           << ")))";
  }
  LaunchConfigExtractor extractor;
  extractor(f->body);
  arith::Analyzer analyzer;
  PrimExpr threadIdx_ext =
      analyzer.Simplify(extractor.threadIdx_x_ext * extractor.threadIdx_y_ext *
                        extractor.threadIdx_z_ext);
  if (const IntImmNode *const threadIdx_ext_int =
          threadIdx_ext.as<IntImmNode>()) {
    if (threadIdx_ext_int->value == 1) {
      // unable to extract the number of threads per block, hence directly
      // return
      return;
    }
    stream << " __launch_bounds__(" << threadIdx_ext_int->value << ")";
  }
}

std::string CodeGenTileLangHCU::Finish() {
  decl_stream << "#include <hip/hip_runtime.h>\n";

  if (enable_fp8_) {
    decl_stream << "#include <tl_templates/hcu/hcu_fp8.h>\n";
  }

  decl_stream << "#include <tl_templates/hcu/atomic.h>\n";
  if (enable_gemm_mls_) {
    decl_stream << "#include <tl_templates/hcu/mls/tilelang_mls_base.hpp>\n";
    decl_stream << "#include <tl_templates/hcu/gemm_mls.h>\n";
  } else {
    decl_stream << "#include <tl_templates/hcu/gemm.h>\n";
  }
  decl_stream << "#include <tl_templates/hcu/copy.h>\n";
  decl_stream << "#include <tl_templates/hcu/reduce.h>\n";
  decl_stream << "#include <tl_templates/hcu/ldsm.h>\n";
  decl_stream << "#include <tl_templates/hcu/threadblock_swizzle.h>\n";
  decl_stream << "#include <tl_templates/hcu/debug.h>\n";
  decl_stream << "#include <tl_templates/hcu/barrier.h>\n";
  decl_stream << "\n";
  return CodeGenC::Finish();
}

namespace {

bool IsZeroValue(const PrimExpr &expr) {
  if (const auto *int_imm = expr.as<IntImmNode>()) {
    return int_imm->value == 0;
  }
  if (const auto *float_imm = expr.as<FloatImmNode>()) {
    return float_imm->value == 0.0;
  }
  if (const auto *broadcast = expr.as<BroadcastNode>()) {
    return IsZeroValue(broadcast->value);
  }
  if (const auto *cast = expr.as<CastNode>()) {
    return IsZeroValue(cast->value);
  }
  return tir::is_zero(expr);
}

const BufferLoadNode *ExtractBufferLoad(const PrimExpr &value) {
  if (const auto *load = value.as<BufferLoadNode>()) {
    return load;
  }
  if (const auto *cast = value.as<CastNode>()) {
    return cast->value.as<BufferLoadNode>();
  }
  return nullptr;
}

// Only allow: BufferLoad, Cast(BufferLoad), Broadcast(const), IntImm, FloatImm.
bool IsSafeValueExpr(const PrimExpr &expr) {
  if (expr.as<tir::BufferLoadNode>()) {
    return true;
  }
  if (expr.as<tir::IntImmNode>() || expr.as<tir::FloatImmNode>()) {
    return true;
  }
  if (const auto *cast = expr.as<tir::CastNode>()) {
    return IsSafeValueExpr(cast->value);
  }
  if (const auto *broadcast = expr.as<tir::BroadcastNode>()) {
    return broadcast->value.as<tir::IntImmNode>() ||
           broadcast->value.as<tir::FloatImmNode>();
  }
  return false;
}

// Extract conditions from a chain of if(cond){body} with no else, ending in
// store. Returns (conditions, store) or (empty, nullptr) if not matching.
struct NestedFoldableResult {
  Array<PrimExpr> conditions;
  const BufferStoreNode *store{nullptr};
  bool ok{false};
};

NestedFoldableResult ExtractNestedFoldableConditions(const Stmt &stmt) {
  if (const auto *store = stmt.as<BufferStoreNode>()) {
    return {{}, store, true};
  }
  if (const auto *iff = stmt.as<IfThenElseNode>()) {
    if (iff->else_case) {
      return {{}, nullptr, false};
    }
    auto inner = ExtractNestedFoldableConditions(iff->then_case);
    if (!inner.ok) {
      return {{}, nullptr, false};
    }
    Array<PrimExpr> conds = {iff->condition};
    for (const auto &c : inner.conditions) {
      conds.push_back(c);
    }
    return {conds, inner.store, true};
  }
  return {{}, nullptr, false};
}

} // namespace

std::string CodeGenTileLangHCU::GetCurrentPredicate() const {
  if (predicate_stack_.empty())
    return "true";
  std::string result = predicate_stack_[0];
  for (size_t i = 1; i < predicate_stack_.size(); ++i) {
    result += " && (" + predicate_stack_[i] + ")";
  }
  return result;
}

bool CodeGenTileLangHCU::IsProvablyZeroOrZeroBroadcast(
    const PrimExpr &expr) const {
  if (IsZeroValue(expr)) {
    return true;
  }
  if (const auto *cast = expr.as<CastNode>()) {
    return IsProvablyZeroOrZeroBroadcast(cast->value);
  }
  if (const auto *broadcast = expr.as<BroadcastNode>()) {
    if (IsZeroValue(broadcast->value)) {
      return true;
    }
    if (const auto *v = broadcast->value.as<VarNode>()) {
      auto it = let_initializer_expr_for_predicate_.find(v);
      if (it != let_initializer_expr_for_predicate_.end() &&
          IsZeroValue(it->second)) {
        return true;
      }
    }
  }
  return false;
}

bool CodeGenTileLangHCU::LoadWillUseAmdBufferOpsWithPredicate(
    const PrimExpr &value_expr) const {
  const BufferLoadNode *load = ExtractBufferLoad(value_expr);
  if (!load) {
    return false;
  }
  if (!CanUseVMBufferOps(load->buffer.get(), load->dtype.lanes())) {
    return false;
  }
  DataType outer_dtype = value_expr.dtype();
  DataType element_dtype = load->buffer->dtype;
  PrimExpr index = load->indices[0];
  int lanes = load->dtype.lanes();

  // Align with CodeGenC::VisitExpr_(BufferLoad): GetBufferRef when the
  // *emitted* value lanes match buffer element lanes. Use outer dtype (e.g.
  // Cast to scalar) so Cast(vector load)->scalar is not mistaken for
  // GetVecLoad.
  if (outer_dtype.lanes() == element_dtype.lanes()) {
    return false;
  }

  bool can_vector_load = false;
  arith::PVar<PrimExpr> base;
  if (arith::ramp(base, 1, lanes).Match(index)) {
    const RampNode *ramp = index.as<RampNode>();
    ICHECK(ramp);
    arith::ModularSet me = arith::Analyzer().modular_set(ramp->base);
    if (me->coeff % lanes == 0 && me->base % lanes == 0) {
      can_vector_load = true;
    }
  }
  if (load->dtype.is_float4_e2m1fn() && lanes != 1) {
    can_vector_load = false;
  }
  return can_vector_load;
}

bool CodeGenTileLangHCU::StoreWillUseAmdBufferOpsWithPredicate(
    const BufferStoreNode *op) const {
  // Predicate is only threaded into tl::amd_buffer_store when
  // VisitStmt_(BufferStore) takes the direct path, or into
  // PrintVecStoreWithPredicate when CodeGenC::VisitStmt_ (BufferStore) calls
  // PrintVecStore (unequal lanes + contiguous ramp, not float4).
  // TryToEmitLDSBufferOp bypasses predicate_stack_ — gate with
  // StoreWouldEmitLDSBufferOp.
  if (StoreWouldEmitLDSBufferOp(op))
    return false;

  DataType value_dtype = op->value.dtype();
  DataType element_dtype = op->buffer->dtype;
  PrimExpr index_expr = op->indices[0];

  bool lanes_ok = (element_dtype.lanes() == 1)
                      ? (value_dtype.element_of() == element_dtype.element_of())
                      : (value_dtype.lanes() == element_dtype.lanes());
  if (!lanes_ok)
    return false;
  if (!CanUseVMBufferOps(op->buffer.get(), value_dtype.lanes()))
    return false;

  if (value_dtype.lanes() == element_dtype.lanes()) {
    // Matches VisitStmt_(BufferStore) before CodeGenC fallback: GetBufferDesc
    // requires stride-1 ramp when index is a Ramp (else codegen ICHECK).
    if (index_expr.as<RampNode>()) {
      arith::PVar<PrimExpr> base;
      if (!arith::ramp(base, 1, value_dtype.lanes()).Match(index_expr))
        return false;
    }
    return true;
  }

  // Unequal lanes: CodeGenC uses per-lane loop unless
  // ramp(base,1,value_lanes).Match.
  if (value_dtype.is_float4_e2m1fn())
    return false;
  arith::PVar<PrimExpr> base;
  if (!arith::ramp(base, 1, value_dtype.lanes()).Match(index_expr))
    return false;
  return true;
}

bool CodeGenTileLangHCU::IsFoldableIfThenElse(const IfThenElseNode *op) const {
  Stmt then_body = op->then_case;
  if (const auto *seq = then_body.as<tir::SeqStmtNode>()) {
    if (seq->size() == 1) {
      then_body = seq->seq[0];
    }
  }
  const auto *then_store = then_body.as<BufferStoreNode>();
  if (!then_store) {
    if (const auto *inner = then_body.as<IfThenElseNode>()) {
      return !op->else_case && IsFoldableIfThenElse(inner);
    }
    return false;
  }

  if (!IsSafeValueExpr(then_store->value))
    return false;

  if (!op->else_case) {
    return StoreWillUseAmdBufferOpsWithPredicate(then_store);
  }

  const auto *else_store = op->else_case.value().as<BufferStoreNode>();
  if (!else_store || !then_store->buffer.same_as(else_store->buffer) ||
      then_store->indices.size() != else_store->indices.size())
    return false;
  for (size_t i = 0; i < then_store->indices.size(); i++) {
    if (!StructuralEqual()(then_store->indices[i], else_store->indices[i]))
      return false;
  }
  if (!IsProvablyZeroOrZeroBroadcast(else_store->value))
    return false;
  if (!LoadWillUseAmdBufferOpsWithPredicate(then_store->value))
    return false;
  return true;
}

// Check if if(cond){store} else {if(c0){if(c1){store}}} with cond == c0 && c1.
// When true, we can collapse to a single store with predicate cond (else is
// dead).
bool CodeGenTileLangHCU::IsCollapsibleRedundantIfElse(
    const IfThenElseNode *op) const {
  if (!op->else_case)
    return false;
  const auto *then_store = op->then_case.as<BufferStoreNode>();
  if (!then_store || !IsSafeValueExpr(then_store->value))
    return false;

  auto else_result = ExtractNestedFoldableConditions(op->else_case.value());
  if (!else_result.ok || else_result.conditions.empty())
    return false;

  if (!then_store->buffer.same_as(else_result.store->buffer) ||
      then_store->indices.size() != else_result.store->indices.size())
    return false;
  for (size_t i = 0; i < then_store->indices.size(); i++) {
    if (!StructuralEqual()(then_store->indices[i],
                           else_result.store->indices[i]))
      return false;
  }
  if (!StructuralEqual()(then_store->value, else_result.store->value))
    return false;

  if (!StoreWillUseAmdBufferOpsWithPredicate(then_store))
    return false;

  PrimExpr combined = else_result.conditions[0];
  for (size_t i = 1; i < else_result.conditions.size(); i++) {
    combined = tir::And(combined, else_result.conditions[i]);
  }
  return arith::Analyzer().CanProveEqual(op->condition, combined);
}

void CodeGenTileLangHCU::VisitStmt_(const IfThenElseNode *op) {
  if (IsCollapsibleRedundantIfElse(op)) {
    std::string cond_str = PrintExpr(op->condition);
    std::string pred_var = name_supply_->FreshName("pred");
    PrintIndent();
    stream << "bool " << pred_var << " = " << cond_str << ";\n";
    predicate_stack_.push_back(pred_var);
    PrintStmt(op->then_case);
    predicate_stack_.pop_back();
    return;
  }
  if (IsFoldableIfThenElse(op)) {
    std::string cond_str = PrintExpr(op->condition);
    std::string pred_var = name_supply_->FreshName("pred");
    PrintIndent();
    stream << "bool " << pred_var << " = " << cond_str << ";\n";
    predicate_stack_.push_back(pred_var);
    PrintStmt(op->then_case);
    predicate_stack_.pop_back();
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangHCU::VisitStmt_(const LetStmtNode *op) {
  // Record Let RHS for predicate folding (IsZeroValue); do not touch
  // var_idmap_.
  let_initializer_expr_for_predicate_[op->var.get()] = op->value;
  CodeGenC::VisitStmt_(op);
}

namespace {

bool IsMlsLoadTileCallExtern(const CallNode *call) {
  if (!call->op.same_as(tir::builtin::call_extern()) || call->args.empty()) {
    return false;
  }
  const auto *name = call->args[0].as<StringImmNode>();
  return name && static_cast<std::string>(name->value)
                         .find("tl::mls::mls_load_tile<") == 0;
}

std::vector<std::string> SplitTopLevelTemplateArgs(const std::string &text) {
  std::vector<std::string> args;
  int depth = 0;
  size_t start = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    char c = text[i];
    if (c == '<') {
      ++depth;
    } else if (c == '>') {
      --depth;
    } else if (c == ',' && depth == 0) {
      size_t end = i;
      while (start < end && text[start] == ' ')
        ++start;
      while (end > start && text[end - 1] == ' ')
        --end;
      args.push_back(text.substr(start, end - start));
      start = i + 1;
    }
  }
  size_t end = text.size();
  while (start < end && text[start] == ' ')
    ++start;
  while (end > start && text[end - 1] == ' ')
    --end;
  args.push_back(text.substr(start, end - start));
  return args;
}

std::string MlsBaseTemplateFromLoadTile(const std::string &sym) {
  const std::string prefix = "tl::mls::mls_load_tile<";
  ICHECK(sym.find(prefix) == 0) << "Unexpected MLS symbol: " << sym;
  ICHECK_EQ(sym.back(), '>') << "Malformed MLS template symbol: " << sym;
  auto args = SplitTopLevelTemplateArgs(
      sym.substr(prefix.size(), sym.size() - prefix.size() - 1));
  ICHECK_GE(args.size(), 8U)
      << "mls_load_tile expects at least 8 template args";
  std::ostringstream os;
  os << "tl::mls::tilelang_mls_base<";
  for (size_t i = 0; i < 8; ++i) {
    if (i != 0)
      os << ", ";
    os << args[i];
  }
  os << ">";
  return os.str();
}

std::string MlsDataTypeFromLoadTile(const std::string &sym) {
  const std::string prefix = "tl::mls::mls_load_tile<";
  auto args = SplitTopLevelTemplateArgs(
      sym.substr(prefix.size(), sym.size() - prefix.size() - 1));
  ICHECK_GE(args.size(), 5U)
      << "mls_load_tile expects DataType as template arg 4";
  return args[4];
}

std::pair<std::string, std::string>
MlsLastLoadTemplateArgs(const std::string &sym) {
  const std::string prefix = "tl::mls::mls_load_tile<";
  auto args = SplitTopLevelTemplateArgs(
      sym.substr(prefix.size(), sym.size() - prefix.size() - 1));
  std::string check_last_load = args.size() > 8 ? args[8] : "true";
  std::string last_load = args.size() > 9 ? args[9] : "false";
  return {check_last_load, last_load};
}

} // namespace

Optional<Array<PrimExpr>> FindHcuWdraInitArgs(const Stmt &body) {
  struct Finder : public StmtExprVisitor {
    Optional<Array<PrimExpr>> args;

    void VisitStmt_(const EvaluateNode *op) final {
      if (args.defined()) {
        return;
      }
      if (const auto *call = op->value.as<CallNode>()) {
        if (call->op.same_as(tl::hcu_wdra_init())) {
          args = call->args;
          return;
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }
  } finder;
  finder(body);
  return finder.args;
}

void PrintHcuWdraInit(std::ostream &os, const Array<PrimExpr> &args) {
  os << "__builtin_hcu_wdra_init(";
  for (size_t i = 0; i < args.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << Downcast<IntImm>(args[i])->value;
  }
  os << ")";
}

void CodeGenTileLangHCU::PreFunctionBody(const PrimFunc &f) {
  wdra_init_emitted_ = false;
  if (auto args = FindHcuWdraInitArgs(f->body)) {
    PrintIndent();
    PrintHcuWdraInit(stream, args.value());
    stream << ";\n";
    wdra_init_emitted_ = true;
  }
  CodeGenC::PreFunctionBody(f);
}

void CodeGenTileLangHCU::VisitStmt_(const EvaluateNode *op) {
  const auto *call = op->value.as<CallNode>();
  if (call && call->op.same_as(tir::builtin::call_extern()) &&
      !call->args.empty()) {
    const auto *extern_sym = call->args[0].as<StringImmNode>();
    if (extern_sym) {
      const std::string sym = extern_sym->value;
      const std::string resource_init_prefix = "tl::mls::resource_init<";
      const std::string async_load_prefix = "tl::mls::async_load<";
      const std::string async_load_mn_prefix = "tl::mls::async_load_mn<";

      if (sym.find(resource_init_prefix) == 0) {
        ICHECK(call->args.size() == 6U || call->args.size() == 7U)
            << "MLS resource_init expects symbol, name, src, stride, mn_len, "
               "k_len[, warp_id_offset]";
        enable_gemm_mls_ = true;
        const auto *name = call->args[1].as<StringImmNode>();
        ICHECK(name) << "MLS resource_init expects a string resource name";
        const std::string obj_name = name->value;
        const std::string base_template =
            sym.substr(resource_init_prefix.size(),
                       sym.size() - resource_init_prefix.size() - 1);
        const std::string warp_id_offset =
            call->args.size() == 7U ? PrintExpr(call->args[6]) : "0";
        PrintIndent();
        stream << "using " << obj_name << "_t = " << base_template << ";\n";
        PrintIndent();
        stream << obj_name << "_t " << obj_name << "("
               << PrintExpr(call->args[2]) << ", " << PrintExpr(call->args[3])
               << ", " << PrintExpr(call->args[4]) << ", "
               << PrintExpr(call->args[5]) << ", " << warp_id_offset << ");\n";
        return;
      }

      if (sym == "tl::mls::set_window_origin") {
        ICHECK_EQ(call->args.size(), 4U)
            << "MLS set_window_origin expects symbol, name, mn_base, k_base";
        enable_gemm_mls_ = true;
        const auto *name = call->args[1].as<StringImmNode>();
        ICHECK(name) << "MLS set_window_origin expects a string resource name";
        PrintIndent();
        stream << name->value
               << ".set_window_origin(tl::make_array<tl::index_t>("
               << PrintExpr(call->args[2]) << ", " << PrintExpr(call->args[3])
               << "));\n";
        return;
      }

      if (sym == "tl::mls::update_base") {
        ICHECK_EQ(call->args.size(), 3U)
            << "MLS update_base expects symbol, name, k_base";
        enable_gemm_mls_ = true;
        const auto *name = call->args[1].as<StringImmNode>();
        ICHECK(name) << "MLS update_base expects a string resource name";
        PrintIndent();
        stream << name->value << ".update_base(" << PrintExpr(call->args[2])
               << ");\n";
        return;
      }

      if (sym == "tl::mls::update_mn_base") {
        ICHECK_EQ(call->args.size(), 3U)
            << "MLS update_mn_base expects symbol, name, mn_base";
        enable_gemm_mls_ = true;
        const auto *name = call->args[1].as<StringImmNode>();
        ICHECK(name) << "MLS update_mn_base expects a string resource name";
        PrintIndent();
        stream << name->value << ".update_mn_base(" << PrintExpr(call->args[2])
               << ");\n";
        return;
      }

      if (sym.find(async_load_prefix) == 0) {
        ICHECK_EQ(call->args.size(), 4U)
            << "MLS async_load expects symbol, name, dst, k_base";
        enable_gemm_mls_ = true;
        const auto *name = call->args[1].as<StringImmNode>();
        ICHECK(name) << "MLS async_load expects a string resource name";
        const std::string template_args =
            sym.substr(async_load_prefix.size(),
                       sym.size() - async_load_prefix.size() - 1);
        PrintIndent();
        stream << name->value << ".template async_mls_load_asm<"
               << template_args << ">(" << PrintExpr(call->args[2]) << ", "
               << PrintExpr(call->args[3]) << ");\n";
        return;
      }

      if (sym.find(async_load_mn_prefix) == 0) {
        ICHECK_EQ(call->args.size(), 4U)
            << "MLS async_load_mn expects symbol, name, dst, mn_base";
        enable_gemm_mls_ = true;
        const auto *name = call->args[1].as<StringImmNode>();
        ICHECK(name) << "MLS async_load_mn expects a string resource name";
        const std::string template_args =
            sym.substr(async_load_mn_prefix.size(),
                       sym.size() - async_load_mn_prefix.size() - 1);
        PrintIndent();
        stream << name->value << ".template async_mls_load_asm_mn<"
               << template_args << ">(" << PrintExpr(call->args[2]) << ", "
               << PrintExpr(call->args[3]) << ");\n";
        return;
      }
    }
  }
  if (call && IsMlsLoadTileCallExtern(call)) {
    ICHECK(call->args.size() == 8U || call->args.size() == 9U)
        << "mls_load_tile extern expects symbol, src, stride, mn_len, k_len, "
           "mn_base, k_base, dst[, warp_id_offset]";
    enable_gemm_mls_ = true;
    const auto *sym_node = call->args[0].as<StringImmNode>();
    const std::string sym = sym_node->value;

    const std::string base_template = MlsBaseTemplateFromLoadTile(sym);
    const std::string data_type = MlsDataTypeFromLoadTile(sym);
    const auto [check_last_load, last_load] = MlsLastLoadTemplateArgs(sym);
    const std::string src_ptr = PrintExpr(call->args[1]);
    const std::string stride = PrintExpr(call->args[2]);
    const std::string mn_len = PrintExpr(call->args[3]);
    const std::string k_len = PrintExpr(call->args[4]);
    const std::string mn_base = PrintExpr(call->args[5]);
    const std::string k_base = PrintExpr(call->args[6]);
    const std::string dst_ptr = PrintExpr(call->args[7]);
    const std::string warp_id_offset =
        call->args.size() == 9U ? PrintExpr(call->args[8]) : "0";

    const std::string obj_name =
        "_tl_mls_" + std::to_string(mls_resource_object_counter_++);
    PrintIndent();
    stream << "using " << obj_name << "_t = " << base_template << ";\n";
    PrintIndent();
    stream << obj_name << "_t " << obj_name << "(" << src_ptr << ", " << stride
           << ", " << mn_len << ", " << k_len << ", " << warp_id_offset
           << ");\n";
    PrintIndent();
    stream << obj_name << ".set_window_origin(tl::make_array<tl::index_t>("
           << mn_base << ", 0));\n";
    PrintIndent();
    stream << obj_name << ".update_base(" << k_base << ");\n";
    PrintIndent();
    stream << obj_name << ".template async_mls_load_asm<" << data_type << ", "
           << check_last_load << ", " << last_load << ">(" << dst_ptr << ", "
           << k_base << ");\n";
    return;
  }

  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangHCU::VisitStmt_(const tir::ForNode *op) {
  if (op->kind == tir::ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  std::string extent =
      PrintExpr(arith::Analyzer().Simplify(op->extent + op->min));
  std::string vid = AllocVarID(op->loop_var.get());
  std::string start = PrintExpr(op->min);
  PrintIndent();
  stream << "for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = " << start << "; " << vid << " < " << extent
         << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenTileLangHCU::VisitStmt_(const BufferStoreNode *op) {
  // Try to use CK buffer store for global memory when enabled.
  DataType value_dtype = op->value.dtype();
  DataType element_dtype = op->buffer->dtype;

  if (TryToEmitLDSBufferOp(op))
    return;

  // Only handle the common case where lanes match; otherwise, fall back to the
  // default CodeGenC implementation (which may invoke PrintVecStore to emit
  // buffer/vectorized store instructions).
  if (value_dtype.lanes() == element_dtype.lanes()) {
    BufferDesc desc =
        GetBufferDesc(value_dtype, op->buffer.get(), op->indices[0]);
    if (CanUseVMBufferOps(op->buffer.get(), value_dtype.lanes())) {
      std::string value = PrintExpr(op->value);

      // Convert the value expression to a thread_buffer using bit_cast.
      // For lanes==1 this becomes thread_buffer<T,1>.
      std::string src_thread_buffer = "tl::bit_cast<tl::thread_buffer<" +
                                      HcuCkTemplateElemType(value_dtype) +
                                      ", " + std::to_string(desc.num_elements) +
                                      ">>(" + value + ")";

      std::string pred = GetCurrentPredicate();
      PrintIndent();
      stream << "tl::amd_buffer_store<" << HcuCkTemplateElemType(value_dtype)
             << ", " << desc.num_elements << ", "
             << (pred == "true" ? "false" : "true") << ">(" << src_thread_buffer
             << ", " << HcuCkBufferDstPtrExpr(value_dtype, desc.wave_ptr)
             << ", " << desc.offset << ", " << pred << ", "
             << desc.element_space_size << ");\n";

      return;
    }
  }

  // Fallback to the base implementation which emits vectorized stores.
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangHCU::BindThreadIndex(const IterVar &iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] =
      CastFromTo(iv->thread_tag, DataType::UInt(32), iv->var.dtype());
}

void CodeGenTileLangHCU::PrintType(DataType t, std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK(t.is_scalar()) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }

  if (t == tl::cuTensorMapType()) {
    os << "CUtensorMap";
    return;
  }

  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
    case 16:
      if (t.is_scalar()) {
        os << "half_t";
      } else if (lanes <= 8) {
        // Emit CUDA code to access fp16 vector elements.
        //
        // half4 is stored as uint2
        //
        // h4.x is emitted as *(half2*)(&(u2.x)).x
        // h4.y is emitted as *(half2*)(&(u2.x)).y
        // h4.z is emitted as *(half2*)(&(u2.y)).x
        // h4.w is emitted as *(half2*)(&(u2.y)).y
        //
        ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
        os << "uint" << lanes / 2;
      } else {
        fail = true;
      }
      break;
    case 32:
      if (lanes <= 4) {
        os << "float";
      } else if (lanes <= 8) {
        // Emit CUDA code to access fp32 vector elements for 4 < lanes <= 8.
        //
        // float8 is stored as ulonglong4
        //
        // f8.v1 is emitted as *(float2*)(&(ul4.x)).x
        // f8.v2 is emitted as *(float2*)(&(ul4.x)).y
        //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for float type with lanes > 4";
        os << "ulonglong" << lanes / 2;
      } else if (lanes == 16) {
        os << "float32x16";
        return;
      } else if (lanes == 32) {
        os << "float32x32";
        return;
      } else {
        fail = true;
      }
      break;
    case 64:
      os << "double";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && (t.is_scalar() || t.bits() == 16))
      return;
    if (!fail && (lanes > 4 && lanes <= 8 && t.bits() == 32))
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_bfloat16()) {
    if (t.is_scalar()) {
      os << "bfloat16_t";
    } else if (lanes <= 8) {
      ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
      os << "uint" << lanes / 2;
    } else if (lanes == 16) {
      os << "bfloat16x16";
      return;
    } else {
      fail = true;
    }
    if (!fail)
      return;
  } else if (t.is_float8()) {
    enable_fp8_ = true;
    os << GetFP8Type(t);
    return;
  } else if (t == DataType::Bool()) {
    os << "bool";
    return;
  } else if (t.is_vector_bool()) {
    // CUDA does not support bool vectors.
    // Use ushort vectors to represent instead.
    int n = t.lanes();
    if (n <= 4) {
      os << "ushort" << n;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "u";
    }
    switch (t.bits()) {
    case 1: {
      if (t.is_scalar()) {
        os << "int";
        return;
      } else if (t.lanes() == 8) {
        os << "int8_t";
        return;
      } else if (t.lanes() == 16) {
        os << "int16_t";
        return;
      } else if (t.lanes() == 32) {
        os << "int";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
      }
    }
    case 4: {
      if (t.is_scalar()) {
        os << "int";
        return;
      } else if (t.lanes() == 4) {
        os << "int16_t";
        return;
      } else if (t.lanes() == 8) {
        // directly 8 4-bit int in integer.
        os << "int";
        return;
      } else if (t.lanes() == 16) {
        os << "int2";
        return;
      } else if (t.lanes() == 32) {
        os << "int4";
        return;
      } else if (t.lanes() == 64) {
        os << "int8";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
      }
    }
    case 8: {
      if (t.lanes() == 4) {
        // directly 4 8 bit int in integer.

        // We use int for int8x4 instead of char4 because using char4 is
        // likely to produce extra instructions to pack four int8 elements
        // into 32-bit data.
        os << "int";
        return;
      } else if (t.lanes() == 8) {
        os << "int2";
        return;
      } else if (t.lanes() == 16) {
        os << "int4";
        return;
      } else if (!t.is_uint() && t.is_scalar()) {
        os << "signed char";
        break;
      } else {
        os << "char";
        break;
      }
    }
    case 16: {
      if (t.is_scalar()) {
        os << "short";
      } else if (t.lanes() <= 4) {
        os << "short" << lanes;
      } else if (t.lanes() <= 8) {
        // Emit CUDA code to access int16 vector elements.
        //
        // short4 is stored as int2
        //
        // s4.x is emitted as *(short2*)(&(i2.x)).x
        // s4.y is emitted as *(short2*)(&(i2.x)).y
        // s4.z is emitted as *(short2*)(&(i2.y)).x
        // s4.w is emitted as *(short2*)(&(i2.y)).y
        //
        ICHECK_EQ(t.lanes() % 2, 0)
            << "only support even lane for shorT type with lanes > 4";
        os << "int" << t.lanes() / 2;
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 32: {
      if (t.is_scalar()) {
        os << "int";
      } else if (t.lanes() <= 4) {
        os << "int" << t.lanes();
      } else if (t.lanes() <= 8) {
        // Emit CUDA code to access int32 vector elements for 4 < lanes <= 8.
        //
        // int8 is stored as longlong4
        //
        // i8.v1 is emitted as *(int2*)(&(l4.x)).x
        // i8.v2 is emitted as *(int2*)(&(l4.x)).y
        //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for int32 type with lanes > 4";
        os << "longlong" << lanes / 2;
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 64: {
      if (t.is_scalar()) {
        os << "int64_t";
      } else if (t.lanes() == 2) {
        os << "longlong2";
      } else if (t.lanes() == 3) {
        os << "longlong3";
      } else if (t.lanes() == 4) {
        os << "longlong4";
      }
      return;
    }
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1) {
      return;
    }
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to CUDA type";
}

void CodeGenTileLangHCU::PrintVecBinaryOp(const std::string &op, DataType t,
                                          PrimExpr lhs, PrimExpr rhs,
                                          std::ostream &os) { // NOLINT(*)
  // Declare the result.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  int ssa_scope = BeginScope();
  {
    // Unpack into individual ops.
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.dtype());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.dtype());

    for (int i = 0, lanes = t.lanes(); i < lanes; ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
  }
  EndScope(ssa_scope);
  os << sret;
}

CodeGenTileLangHCU::BufferDesc
CodeGenTileLangHCU::GetBufferDesc(DataType t, const BufferNode *buffer,
                                  PrimExpr offset) {
  const VarNode *buffer_var = buffer->data.get();
  std::string scope;

  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  if (scope.empty()) {
    scope = GetPtrStorageScope(buffer->data);
  }

  ICHECK_NE(buffer->shape.size(), 0) << "Buffer shape is empty";
  PrimExpr total_size = IntImm(DataType::Int(32), 1);
  for (const auto &dim : buffer->shape) {
    total_size = total_size * dim;
  }
  std::string element_space_size = PrintExpr(total_size);
  std::ostringstream stream;
  PrintType(buffer->dtype.element_of(), stream);

  if (offset.as<RampNode>()) {
    arith::PVar<PrimExpr> base;
    arith::ramp(base, 1, t.lanes()).Match(offset);
    offset = base.Eval();
    ICHECK(offset.defined()) << "Non-contiguous ramp offset is not supported.";
  }

  BufferDesc desc;
  desc.wave_ptr = GetVarID(buffer_var);
  desc.offset = PrintExpr(offset);
  desc.element_space_size = element_space_size;
  desc.data_type = stream.str();
  desc.scope = scope;
  desc.num_elements = t.lanes();

  return std::move(desc);
}

std::string CodeGenTileLangHCU::GetVecLoad(DataType t, const BufferNode *buffer,
                                           PrimExpr base) {
  // Generate CK buffer load for global memory accesses
  // FIXME: Should we check if the offset is not negative?
  return GetVecLoadWithPredicate(t, buffer, base, GetCurrentPredicate());
}

std::string CodeGenTileLangHCU::GetVecLoadWithPredicate(
    DataType t, const BufferNode *buffer, PrimExpr base,
    const std::string &pred) {
  if (CanUseVMBufferOps(buffer, t.lanes())) {
    auto desc = GetBufferDesc(t, buffer, base);
    std::ostringstream os;
    os << "*(";
    PrintType(t, os);
    os << "*)&(tl::amd_buffer_load<" << HcuCkTemplateElemType(t) << ", "
       << desc.num_elements << ", " << (pred == "true" ? "false" : "true")
       << ">(" << HcuCkBufferSrcPtrExpr(t, desc.wave_ptr) << ", " << desc.offset
       << ", " << pred << ", " << desc.element_space_size << ").get())";

    return os.str();
  }

  // For other cases, use the global load.
  return CodeGenC::GetVecLoad(t, buffer, base);
}

void CodeGenTileLangHCU::PrintVecStore(const BufferNode *buffer, DataType t,
                                       PrimExpr base,
                                       const std::string &value) {
  PrintVecStoreWithPredicate(buffer, t, base, value, GetCurrentPredicate());
}

void CodeGenTileLangHCU::PrintVecStoreWithPredicate(const BufferNode *buffer,
                                                    DataType t, PrimExpr base,
                                                    const std::string &value,
                                                    const std::string &pred) {
  if (!CanUseVMBufferOps(buffer, t.lanes())) {
    CodeGenC::PrintVecStore(buffer, t, base, value);
    return;
  }

  auto desc = GetBufferDesc(t, buffer, base);
  // Convert value to thread_buffer and use amd_buffer_store
  // amd_buffer_store signature:
  //   amd_buffer_store<type, num_elements>(src_thread_data, dst_ptr,
  //   dst_offset,
  //                                        is_valid, element_space_size)
  // Convert the value expression to a thread_buffer using bit_cast
  std::string src_thread_buffer =
      "tl::bit_cast<tl::thread_buffer<" + HcuCkTemplateElemType(t) + ", " +
      std::to_string(desc.num_elements) + ">>(" + value + ")";

  this->PrintIndent();
  this->stream << "tl::amd_buffer_store<" << HcuCkTemplateElemType(t) << ", "
               << desc.num_elements << ", "
               << (pred == "true" ? "false" : "true") << ">("
               << src_thread_buffer << ", "
               << HcuCkBufferDstPtrExpr(t, desc.wave_ptr) << ", " << desc.offset
               << ", " << pred << ", " << desc.element_space_size << ");\n";
}

void CodeGenTileLangHCU::PrintVecElemLoad(const std::string &vec, DataType t,
                                          int i,
                                          std::ostream &os) { // NOLINT(*)
  if (t.is_scalar()) {
    os << vec;
    return;
  }

  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < (t.bits() == 8                        ? 16
                        : (t.lanes() == 16)                  ? 16
                        : (t.lanes() == 32)                  ? 32
                        : (t.bits() == 16 || t.bits() == 32) ? 8
                                                             : 4));
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    std::string type_name = t.is_int() ? "char" : "unsigned char";
    if (t.lanes() == 2 || t.lanes() == 3) {
      os << vec << "." << access[i % t.lanes()];
    } else {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      os << "((" << type_name << ")(" << ac << " >> " << i % 4 * 8 << "))";
    }
  } else if ((t.lanes() == 16 || t.lanes() == 32) && t.bits() == 32 &&
             t.is_float()) {
    os << vec << "[" << i << "]";
  } else if (t.lanes() == 16 && t.is_bfloat16()) {
    // bfloat16x16: aggregate with inner data[N]; element access uses .data[i].
    os << vec << ".data[" << i << "]";
  } else if (t.is_float16()) {
    os << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->"
       << access[i % 2];
  } else if (t.is_bfloat16()) {
    os << "((bfloat16x2*)(&(" << vec << "." << access[i / 2] << ")))->"
       << access[i % 2];
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    os << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2]
       << ")))->" << access[i % 2];
  } else {
    os << vec << "." << access[i];
  }
}

void CodeGenTileLangHCU::PrintVecElemStore(const std::string &vec, DataType t,
                                           int i, const std::string &value) {
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < (t.bits() == 8                        ? 16
                        : (t.lanes() == 16)                  ? 16
                        : (t.lanes() == 32)                  ? 32
                        : (t.bits() == 16 || t.bits() == 32) ? 8
                                                             : 4));
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (t.lanes() == 2 || t.lanes() == 3) {
      stream << vec << '.' << access[i % t.lanes()] << "="
             << "(" << value << ");\n";
    } else {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      stream << ac << "=";
      // Do not read the first undef lane.
      if (i != 0) {
        stream << ac << " & ~(0x000000ff << " << i % 4 * 8 << ") |";
      }
      stream << "(" << value << " << " << i % 4 * 8 << ");\n";
    }
  } else if ((t.lanes() == 16 || t.lanes() == 32) && t.bits() == 32 &&
             t.is_float()) {
    stream << vec << "[" << i << "] = " << value << ";\n";
  } else if (t.lanes() == 16 && t.is_bfloat16()) {
    stream << vec << ".data[" << i << "] = " << value << ";\n";
  } else if (t.is_float16()) {
    stream << "*((half_t*)(&(((half2*)(&(" << vec << "." << access[i / 2]
           << ")))->" << access[i % 2] << "))) = " << value << ";\n";
  } else if (t.is_bfloat16()) {
    stream << "((bfloat16_t*)(&(" << vec << "." << access[i / 2] << ")))["
           << (i % 2) << "] = " << value << ";\n";
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    stream << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2]
           << ")))->" << access[i % 2] << " = " << value << ";\n";
  } else {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  }
}

void CodeGenTileLangHCU::PrintStorageSync(const CallNode *op) {
  const std::string &sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    // DO nothing.
  } else if (sync == "shared" || sync == "shared.dyn") {
    this->PrintIndent();
    this->stream << "__syncthreads();\n";
  }
}

void CodeGenTileLangHCU::PrintStorageScope(const std::string &scope,
                                           std::ostream &os) { // NOLINT(*)
  ICHECK_NE(scope, "global")
      << "Cannot allocate global memory when targeting CUDA. You must pass "
         "all global arrays as input instead";
  if (scope == "shared") {
    os << "__shared__ ";
  } else if (scope == "shared.dyn") {
    os << "extern __shared__ __align__(1024) ";
  }
}

std::string CodeGenTileLangHCU::CastFromTo(std::string value, DataType from,
                                           DataType target) {
  if (from == target)
    return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")";
  if (from.is_float16() && (target.is_int() || target.is_uint()) &&
      target.bits() == 8) {
    os << "(";
    if (target.is_uint()) {
      os << "u";
    }
    os << "int)";
  } else if (from.is_float8_e4m3fn() || from.is_float8_e4m3() ||
             target.is_float8_e4m3fn() || target.is_float8_e4m3()) {
    os << "(fp8_cvt_t)";
  } else if (from.is_float8_e5m2() || target.is_float8_e5m2()) {
    os << "(bf8_cvt_t)";
  }
  os << value << ")";
  return os.str();
}

void CodeGenTileLangHCU::VisitExpr_(const CastNode *op, std::ostream &os) {
  DataType from_ty = op->value.dtype();
  DataType target_ty = op->dtype;
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  // Emit simple C-style type conversion.
  if (from_ty.is_scalar())
    return CodeGenC::VisitExpr_(op, os);

  auto type_cvt = "";
  if (from_ty.is_float8_e4m3fn() || from_ty.is_float8_e4m3() ||
      target_ty.is_float8_e4m3fn() || target_ty.is_float8_e4m3()) {
    type_cvt = "(fp8_cvt_t)";
  } else if (from_ty.is_float8_e5m2() || target_ty.is_float8_e5m2()) {
    type_cvt = "(bf8_cvt_t)";
  }
  // We could emit make_float4 like calls, but the emitted code looks
  // too compact to read. Emit this as vectorized unary ops.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(target_ty, stream);
  stream << ' ' << sret << ";\n";
  {
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    for (int i = 0, lanes = from_ty.lanes(); i < lanes; ++i) {
      std::ostringstream val;
      val << "(";
      PrintType(target_ty.element_of(), val);
      val << ")" << type_cvt << "(";
      PrintVecElemLoad(src, from_ty, i, val);
      val << ")";
      PrintVecElemStore(sret, target_ty, i, val.str());
    }
  }
  os << sret;
}

void CodeGenTileLangHCU::VisitExpr_(const FloorDivNode *op,
                                    std::ostream &os) { // NOLINT(*)
  // Match CUDA codegen behavior: lower FloorDiv to plain Div before printing.
  PrintExpr(tir::Div(op->a, op->b), os);
}

void CodeGenTileLangHCU::VisitExpr_(const FloorModNode *op,
                                    std::ostream &os) { // NOLINT(*)
  // Match CUDA codegen behavior: lower FloorMod to plain Mod before printing.
  PrintExpr(tir::Mod(op->a, op->b), os);
}

void CodeGenTileLangHCU::PrintCallExtern(Type ret_type,
                                         ffi::String global_symbol,
                                         const ffi::Array<PrimExpr> &args,
                                         bool skip_first_arg,
                                         std::ostream &os) { // NOLINT(*)
  std::string sym = global_symbol.operator std::string();
  if (sym.find("tl::mls::") != std::string::npos ||
      sym.find("tl::gemm_mls") != std::string::npos ||
      sym.find("tl::gemm_r_mls") != std::string::npos) {
    enable_gemm_mls_ = true;
  }
  DataType ret_dtype = GetRuntimeDataType(ret_type);
  if (ret_dtype.is_vector()) {
    //
    // Emit an unsupported vector call
    //
    // v = intrin_f((float4*)A[0], (float4*)B[0])
    //
    // as
    //
    // float4 __ret;
    // {
    //   float4 __arg0 = ((float4*)A)[0];
    //   float4 __arg1 = ((float4*)B)[0];
    //   __ret.x = intrin_f(__arg0.x, __arg1.x);
    //   __ret.y = intrin_f(__arg0.y, __arg1.y);
    //   __ret.z = intrin_f(__arg0.z, __arg1.z);
    //   __ret.w = intrin_f(__arg0.w, __arg1.w);
    // }
    // v = __ret;
    //
    // Declare the result vector.
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(ret_dtype, stream);
    stream << ' ' << sret << ";\n";
    {
      // Load arguments.
      std::vector<std::string> sargs;
      size_t arg_begin = static_cast<size_t>(skip_first_arg);
      for (size_t i = arg_begin; i < args.size(); ++i) {
        std::string val = SSAGetID(PrintExpr(args[i]), args[i].dtype());
        sargs.push_back(std::move(val));
      }

      // Emit a scalar call for each lane.
      for (int i = 0; i < ret_dtype.lanes(); ++i) {
        std::ostringstream scall;
        scall << global_symbol << "(";
        for (size_t j = 0; j < sargs.size(); ++j) {
          if (j > 0)
            scall << ", ";
          PrintVecElemLoad(sargs[j], args[arg_begin + j].dtype(), i, scall);
        }
        scall << ")";
        PrintVecElemStore(sret, ret_dtype, i, scall.str());
      }
    }
    os << sret;
  } else {
    CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg,
                              os);
  }
}

// Print a reference expression to a buffer.
std::string CodeGenTileLangHCU::GetBufferRef(DataType t,
                                             const BufferNode *buffer,
                                             PrimExpr index) {
  const VarNode *buffer_var = buffer->data.get();
  std::ostringstream os;
  std::string vid = GetVarID(buffer_var);
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  // bool is_vol = IsVolatile(buffer_var);
  // always false for tl cutlass backend.
  bool is_vol = false;

  auto ptr_cast = [this, is_vol, scope](DataType pointed_to) {
    std::ostringstream ptr_os;
    ptr_os << "(";
    if (is_vol) {
      ptr_os << "volatile ";
    }
    if (!scope.empty() && IsScopePartOfType()) {
      PrintStorageScope(scope, ptr_os);
    }
    PrintType(pointed_to, ptr_os);
    ptr_os << "*)";
    return ptr_os.str();
  };

  DataType buffer_element_dtype = buffer->dtype;

  std::string buffer_str = vid;
  if (!HandleTypeMatch(buffer_var, buffer_element_dtype) || is_vol) {
    std::stringstream temp;
    temp << "(" << ptr_cast(buffer_element_dtype) << vid << ")";
    buffer_str = temp.str();
  }

  std::string index_str = PrintExpr(index);
  if (t.bits() == 4 || (t.bits() == 1 && t.is_int())) {
    // This is a special case, because CodegenCUDA::PrintType()
    // returns "int" for bool and for 4-bit integers. In most cases,
    // we divide by the number of lanes to determine the index.
    // However, the backing type for scalar int4 and scalar bool is
    // int32.  Therefore, we need to divide by the ratio of their
    // sizes in that case.
    int div_factor = (t.lanes() == 1) ? (32 / t.bits()) : t.lanes();

    os << "*("
       << "(" << ptr_cast(t) << vid << ")"
       << " + " << index_str << " / " << div_factor << ")";
  } else if (t == buffer_element_dtype) {
    os << buffer_str << "[" << index_str << "]";
  } else {
    os << "*" << ptr_cast(t) << "(" << buffer_str << " + " << index_str << ")";
  }

  return os.str();
}

void CodeGenTileLangHCU::VisitExpr_(const CallNode *op, std::ostream &os) {
  // Optimize if_then_else(cond, BufferLoad/Cast(BufferLoad), zeros) to use
  // amd_buffer_load with predicate instead of if-else.
  if (op->op.same_as(builtin::if_then_else()) && op->args.size() == 3) {
    const BufferLoadNode *load = ExtractBufferLoad(op->args[1]);
    if (load && IsProvablyZeroOrZeroBroadcast(op->args[2]) &&
        LoadWillUseAmdBufferOpsWithPredicate(op->args[1])) {
      std::string cond_str = PrintExpr(op->args[0]);
      std::string pred_var = name_supply_->FreshName("pred");
      PrintIndent();
      stream << "bool " << pred_var << " = " << cond_str << ";\n";
      predicate_stack_.push_back(pred_var);
      os << PrintExpr(op->args[1]);
      predicate_stack_.pop_back();
      return;
    }
  }

  auto print_extern_call_stmt = [&](std::string name, size_t offset = 0) {
    this->PrintIndent();
    this->stream << name << "(";
    for (size_t i = offset; i < op->args.size(); i++) {
      if (i > offset)
        this->stream << ", ";
      this->stream << this->PrintExpr(op->args[i]);
    }
    this->stream << ");\n";
  };
  if (op->op.same_as(builtin::ptx_cp_async())) {
    // args[0] = dst_access_ptr, args[1] = src_access_ptr, args[2] = bytes,
    // args[3] = predicate (optional)
    ICHECK(op->args.size() == 3 || op->args.size() == 4)
        << "ptx_cp_async expects 3 or 4 arguments (dst_access_ptr, "
           "src_access_ptr, bytes, [predicate])";
    std::string dst = this->PrintExpr(op->args[0]);
    std::string src = this->PrintExpr(op->args[1]);
    std::string size = this->PrintExpr(op->args[2]);
    this->PrintIndent();
    if (op->args.size() == 3) {
      this->stream << "tl::cp_async_gs<" << size << ">(" << dst << ", " << src
                   << ");\n";
    } else {
      std::string condition = this->PrintExpr(op->args[3]);
      this->stream << "tl::cp_async_gs_conditional<" << size << ">(" << dst
                   << ", " << src << ", " << condition << ");\n";
    }
  } else if (op->op.same_as(tl::ptx_cp_async())) {
    int total_bytes = GetTileLangCPAsyncTransferBytes(op);
    std::string dst = this->PrintExpr(op->args[0]);
    std::string src = this->PrintExpr(op->args[1]);
    std::string size = std::to_string(total_bytes);
    this->PrintIndent();
    if (op->args.size() == 3) {
      this->stream << "tl::cp_async_gs<" << size << ">(" << dst << ", " << src
                   << ");\n";
    } else {
      std::string condition = this->PrintExpr(op->args[3]);
      this->stream << "tl::cp_async_gs_conditional<" << size << ">(" << dst
                   << ", " << src << ", " << condition << ");\n";
    }
  } else if (op->op.same_as(builtin::ptx_commit_group())) {
    print_extern_call_stmt("tl::cp_async_commit");
  } else if (op->op.same_as(builtin::ptx_wait_group())) {
    int n = Downcast<IntImm>(op->args[0])->value;
    std::string func_name = "tl::cp_async_wait<" + std::to_string(n) + ">";
    print_extern_call_stmt(func_name, 1);
  } else if (op->op.same_as(tl::async_gld_sld_fence())) {
    print_extern_call_stmt("tl::async_gld_sld_fence");
  } else if (op->op.same_as(builtin::create_barriers())) {
    this->PrintIndent();
    int barrier_count = Downcast<IntImm>(op->args[0])->value;
    std::string barrier_name = "_mbarrier";
    this->stream << "__shared__ uint64_t " << barrier_name << "["
                 << barrier_count << "];\n";
  } else if (op->op.same_as(builtin::ptx_arrive_barrier())) {
    print_extern_call_stmt("tl::mbarrier_arrive");
  } else if (op->op.same_as(builtin::ptx_init_barrier_thread_count())) {
    print_extern_call_stmt("tl::mbarrier_init");
  } else if (op->op.same_as(builtin::ptx_arrive_barrier_expect_tx())) {
    print_extern_call_stmt("tl::mbarrier_arrive_expect_tx");
  } else if (op->op.same_as(builtin::ptx_cp_async_barrier())) {
    print_extern_call_stmt("tl::mbarrier_cp_async_arrive");
  } else if (op->op.same_as(tl::mbarrier_expect_tx())) {
    print_extern_call_stmt("tl::mbarrier_expect_tx");
  } else if (op->op.same_as(tl::mbarrier_wait_parity())) {
    print_extern_call_stmt("tl::mbarrier_wait");
  } else if (op->op.same_as(tl::abarrier_init())) {
    print_extern_call_stmt("tl::abarrier_init");
  } else if (op->op.same_as(tl::abarrier_inv())) {
    print_extern_call_stmt("tl::abarrier_inv");
  } else if (op->op.same_as(tl::abarrier_arrive())) {
    std::string abar_id = this->PrintExpr(op->args[0]);
    if (const auto *imm = op->args[1].as<IntImmNode>()) {
      if (imm->value == 1) {
        os << "tl::abarrier_arrive(" << abar_id << ")";
      } else {
        os << "tl::abarrier_arrive_cnt(" << abar_id << ", " << imm->value
           << ")";
      }
    } else {
      os << "tl::abarrier_arrive_cnt(" << abar_id << ", "
         << this->PrintExpr(op->args[1]) << ")";
    }
  } else if (op->op.same_as(tl::abarrier_try_wait())) {
    os << "tl::abarrier_try_wait(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::abarrier_wait())) {
    print_extern_call_stmt("tl::abarrier_wait");
  } else if (op->op.same_as(tl::abarrier_test_wait())) {
    os << "tl::abarrier_test_wait(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::abarrier_seq())) {
    print_extern_call_stmt("tl::abarrier_seq");
  } else if (op->op.same_as(tl::abarrier_expect_tx())) {
    print_extern_call_stmt("tl::abarrier_expect_tx");
  } else if (op->op.same_as(tl::abarrier_complete_tx())) {
    print_extern_call_stmt("tl::abarrier_complete_tx");
  } else if (op->op.same_as(tl::ebarrier_sync())) {
    print_extern_call_stmt("tl::ebarrier_sync");
  } else if (op->op.same_as(tl::ebarrier_sync_cnt())) {
    print_extern_call_stmt("tl::ebarrier_sync_cnt");
  } else if (op->op.same_as(tl::ebarrier_arrive())) {
    print_extern_call_stmt("tl::ebarrier_arrive");
  } else if (op->op.same_as(tl::ptx_stmatrix())) {
    int trans = Downcast<IntImm>(op->args[0])->value;
    int num = Downcast<IntImm>(op->args[1])->value;
    std::string func_name = "tl::ptx_stmatrix_x" + std::to_string(num);
    if (trans == 1)
      func_name += "_trans";
    print_extern_call_stmt(func_name, 2);
  } else if (op->op.same_as(tl::wait_wgmma())) {
    this->PrintIndent();
    int num_mma = Downcast<IntImm>(op->args[0])->value;
    this->stream << "tl::wait_wgmma<" << std::to_string(num_mma) << ">();\n";
  } else if (op->op.same_as(tl::sync_warp())) {
    this->PrintIndent();
    this->stream << "tl::sync_warp(";
    if (!op->args.empty()) {
      this->stream << this->PrintExpr(op->args[0]);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::sync_grid())) {
    LOG(FATAL) << "tl.sync_grid is not supported on HCU";
  } else if (op->op.same_as(tl::pack_b16())) {
    os << "__pack_half2(" << this->PrintExpr(op->args[0]) << ", "
       << this->PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::sync_grid())) {
    LOG(FATAL) << "tl.sync_grid is not supported on HCU";
  } else if (op->op.same_as(tl::any_sync())) {
    ICHECK_EQ(op->args.size(), 2U) << "tl.any_sync expects <mask, predicate>.";
    os << "__any(" << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::all_sync())) {
    ICHECK_EQ(op->args.size(), 2U) << "tl.all_sync expects <mask, predicate>.";
    os << "__all(" << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::ballot_sync())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.ballot_sync expects <mask, predicate>.";
    os << "((unsigned long long)__ballot(" << PrintExpr(op->args[1]) << "))";
  } else if (op->op.same_as(tl::ballot())) {
    ICHECK_EQ(op->args.size(), 1U) << "tl.ballot expects <predicate>.";
    os << "((unsigned long long)__ballot(" << PrintExpr(op->args[0]) << "))";
  } else if (op->op.same_as(tl::activemask())) {
    ICHECK(op->args.empty()) << "tl.activemask takes no arguments.";
    os << "((unsigned long long)__ballot(1))";
  } else if (op->op.same_as(tl::syncthreads_count())) {
    ICHECK_EQ(op->args.size(), 1U)
        << "tl.syncthreads_count expects <predicate>.";
    os << "__syncthreads_count(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::syncthreads_and())) {
    ICHECK_EQ(op->args.size(), 1U) << "tl.syncthreads_and expects <predicate>.";
    os << "__syncthreads_and(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::syncthreads_or())) {
    ICHECK_EQ(op->args.size(), 1U) << "tl.syncthreads_or expects <predicate>.";
    os << "__syncthreads_or(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::shfl_sync())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_sync expects <mask, value, src_lane, width>.";
    os << "__shfl(" << PrintExpr(op->args[1]) << ", " << PrintExpr(op->args[2])
       << ", " << PrintExpr(op->args[3]) << ")";
  } else if (op->op.same_as(tl::shfl_xor_sync())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_xor_sync expects <mask, value, lane_mask, width>.";
    os << "__shfl_xor(" << PrintExpr(op->args[1]) << ", "
       << PrintExpr(op->args[2]) << ", " << PrintExpr(op->args[3]) << ")";
  } else if (op->op.same_as(tl::shfl_down_sync())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_down_sync expects <mask, value, delta, width>.";
    os << "__shfl_down(" << PrintExpr(op->args[1]) << ", "
       << PrintExpr(op->args[2]) << ", " << PrintExpr(op->args[3]) << ")";
  } else if (op->op.same_as(tl::shfl_up_sync())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_up_sync expects <mask, value, delta, width>.";
    os << "__shfl_up(" << PrintExpr(op->args[1]) << ", "
       << PrintExpr(op->args[2]) << ", " << PrintExpr(op->args[3]) << ")";
  } else if (op->op.same_as(tl::match_any_sync())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.match_any_sync expects <mask, value>.";
    os << "tl::match_any_sync((unsigned long long)(" << PrintExpr(op->args[0])
       << "), " << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::match_all_sync())) {
    LOG(FATAL) << "tl.match_all_sync is not supported on HCU";
  } else if (op->op.same_as(tl::get_lane_idx())) {
    ICHECK_LE(op->args.size(), 1)
        << "tl.get_lane_idx expects at most one argument <warp_size>.";
    os << "tl::get_lane_idx(";
    if (!op->args.empty()) {
      os << PrintExpr(op->args[0]);
    }
    os << ")";
  } else if (op->op.same_as(tl::get_warp_idx())) {
    ICHECK_LE(op->args.size(), 1)
        << "tl.get_warp_idx expects at most one argument <warp_size>.";
    os << "tl::get_warp_idx(";
    if (!op->args.empty()) {
      os << PrintExpr(op->args[0]);
    }
    os << ")";
  } else if (op->op.same_as(tl::get_wave_id())) {
    os << "__builtin_hcu_get_wave_id()";
  } else if (op->op.same_as(tl::ieee_fmaf())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.ieee_fmaf expects <x, y, z, rounding_mode>.";
    std::string rounding_mode = Downcast<StringImm>(op->args[3])->value;
    ICHECK(rounding_mode == "rn")
        << "HCU only supports tl.ieee_fmaf(..., rounding_mode=\"rn\").";
    ICHECK(op->dtype.is_float()) << "tl.ieee_fmaf on HCU is currently only "
                                    "implemented for float32/float64.";
    if (op->dtype.bits() == 32) {
      os << "fmaf(" << PrintExpr(op->args[0]) << ", " << PrintExpr(op->args[1])
         << ", " << PrintExpr(op->args[2]) << ")";
    } else {
      os << "fma(" << PrintExpr(op->args[0]) << ", " << PrintExpr(op->args[1])
         << ", " << PrintExpr(op->args[2]) << ")";
    }
  } else if (op->op.same_as(tl::add2()) || op->op.same_as(tl::sub2()) ||
             op->op.same_as(tl::mul2()) || op->op.same_as(tl::fma2()) ||
             op->op.same_as(tl::max2()) || op->op.same_as(tl::min2()) ||
             op->op.same_as(tl::abs2())) {
    std::string op_name;
    std::vector<PrimExpr> packed_args(op->args.begin(), op->args.end());
    if (op->op.same_as(tl::add2()))
      op_name = "add2";
    else if (op->op.same_as(tl::sub2()))
      op_name = "sub2";
    else if (op->op.same_as(tl::mul2()))
      op_name = "mul2";
    else if (op->op.same_as(tl::fma2()))
      op_name = "fma2";
    else if (op->op.same_as(tl::max2()))
      op_name = "max2";
    else if (op->op.same_as(tl::min2()))
      op_name = "min2";
    else
      op_name = "abs2";

    DataType dtype = op->dtype;
    bool need_cast =
        (dtype.is_bfloat16() || dtype.is_float16()) && dtype.lanes() == 2;
    std::string native_type;
    if (dtype.is_bfloat16()) {
      native_type = "bfloat16x2";
    } else if (dtype.is_float16()) {
      native_type = "float16x2";
    }

    auto print_arg = [&](const PrimExpr &arg) -> std::string {
      std::string arg_str = PrintExpr(arg);
      if (need_cast) {
        return "tl::from_uint1<" + native_type + ">(" + arg_str + ")";
      }
      return arg_str;
    };

    if (need_cast) {
      os << "tl::to_uint1(tl::" << op_name << "(";
    } else {
      os << "tl::" << op_name << "(";
    }
    os << print_arg(packed_args[0]);
    for (size_t i = 1; i < packed_args.size(); ++i) {
      os << ", " << print_arg(packed_args[i]);
    }
    os << ")";
    if (need_cast) {
      os << ")";
    }
  } else if (op->op.same_as(tl::__ldg())) {
    const BufferLoadNode *bl = op->args[0].as<BufferLoadNode>();
    ICHECK(bl) << "T.__ldg expects a BufferLoad as the first argument.";
    ICHECK_EQ(bl->indices.size(), 1)
        << "T.__ldg currently supports flattened 1D buffer accesses.";
    const BufferNode *buffer = bl->buffer.get();
    PrimExpr base = bl->indices[0];
    auto buffer_ref = this->GetBufferRef(op->dtype, buffer, base);
    os << buffer_ref;
  } else if (op->op.same_as(builtin::tvm_fill_fragment())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 6U);
    os << "nvcuda::wmma::fill_fragment(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ")";
  } else if (op->op.same_as(builtin::tvm_load_matrix_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::load_matrix_sync(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[6], os);
    os << ")";
  } else if (op->op.same_as(builtin::tvm_store_matrix_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::store_matrix_sync(";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[6], os);
    if (const StringImmNode *str = op->args[7].as<StringImmNode>()) {
      os << ", nvcuda::wmma::mem_" << str->value;
    } else {
      LOG(FATAL) << "Invalid parameters";
    }
    os << ")";
  } else if (op->op.same_as(builtin::tvm_mma_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::mma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(builtin::tvm_bmma_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::bmma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(tl::tvm_mfma())) {
    // arg 0: prefix: {otype}_{intrM}x{intrN}x{intrK}_{itype}
    // arg 1: A layout: row/col
    // arg 2: B layout: row/col
    // arg 3: A precision: float16, float32, ...
    // arg 4: B precision: float16, float32, ...
    // arg 5: C precision: float32, float64, ...
    // arg 6: A multiplicand
    // arg 7: A multiplicand index
    // arg 8: B multiplicand
    // arg 9: B multiplicand index
    // arg 10: C accumulator
    // arg 11: C accumulator index

    ICHECK(op->args.size() == 12U)
        << "Invalid number of arguments for tvm_mfma";
    std::string prefix = Downcast<StringImm>(op->args[0])->value;
    std::string A_layout = Downcast<StringImm>(op->args[1])->value;
    std::string B_layout = Downcast<StringImm>(op->args[2])->value;
    std::string A_dtype = Downcast<StringImm>(op->args[3])->value;
    std::string B_dtype = Downcast<StringImm>(op->args[4])->value;
    std::string C_dtype = Downcast<StringImm>(op->args[5])->value;
    std::string a_ref = this->PrintExpr(op->args[6]);
    std::string a_bias = this->PrintExpr(op->args[7]);
    std::string b_ref = this->PrintExpr(op->args[8]);
    std::string b_bias = this->PrintExpr(op->args[9]);
    std::string c_ref = this->PrintExpr(op->args[10]);
    std::string c_bias = this->PrintExpr(op->args[11]);
    ICHECK(A_layout == "row" || B_layout == "row")
        << "Matrix core only support row major";
    bool hcu_tf32_ab = false;
    auto ab_hint = op->annotations.find("tl.hcu_tf32_ab");
    if (ab_hint != op->annotations.end()) {
      const auto *imm = (*ab_hint).second.as<IntImmNode>();
      hcu_tf32_ab = imm && imm->value != 0;
    }
    // TVM dtype -> C type for __builtin_hcu_mmac_*; packed A/B must match
    // tl_templates/hcu/gemm.h (int8 / FP8 MMAC pass int32x2 to builtins).
    std::unordered_map<std::string, std::string> dtype_map = {
        {"int8", "char"},
        {"int32", "int"},
        {"int8x4", "int32_t"},
        {"int8x8", "int32x2"},
        {"int32x2", "int32x2"},
        {"int32x4", "int32x4"},
        {"float16", "half"},
        {"float32", "float"},
        {"float64", "double"},
        {"float16x4", "float16x4"},
        {"float16x8", "float16x8"},
        {"bfloat16x4", "bfloat16x4_vec"},
        {"bfloat16x8", "bfloat16x8_vec"},
        {"float32x2", "float32x2"},
        {"float32x4", "float32x4"},
        {"float8_e4m3", "fp8_e4_t"},
        {"float8_e4m3x4", "fp8_e4_4_t"},
        {"float8_e4m3x8", "int32x2"},
        {"float8_e4m3fn", "fp8_e4_t"},
        {"float8_e4m3fnx4", "fp8_e4_4_t"},
        {"float8_e4m3fnx8", "int32x2"},
        {"float8_e4m3fnuz", "fp8_e4_t"},
        {"float8_e4m3fnuzx4", "fp8_e4_4_t"},
        {"float8_e4m3fnuzx8", "int32x2"},
        {"float8_e5m2", "fp8_e5_t"},
        {"float8_e5m2x4", "fp8_e5_4_t"},
        {"float8_e5m2x8", "int32x2"},
        {"float8_e5m2fnuzx4", "fp8_e5_4_t"},
        {"float8_e5m2fnuzx8", "int32x2"},
        {"float32x16", "float32x16"},
        {"float32x32", "float32x32"}};

    std::string lit = "";
    if (prefix.find("_lit") != std::string::npos) {
      lit = ", 1";
    }
    std::string clamp = "";
    if (prefix.find("_clamp") != std::string::npos) {
      clamp = ", 0";
    }
    std::string lts = "";
    if (prefix.find("_lts") != std::string::npos) {
      lts = ", 0";
    }

    std::string call_mfma_code =
        "*((({C_dtype}*){c_ref}) + {c_bias}) = {mfma_buildin}("
        "*((({A_dtype}*){a_ref_cast}) + {a_bias}), "
        "*((({B_dtype}*){b_ref_cast}) + {b_bias}), "
        "*((({C_dtype}*){c_ref}) + "
        "{c_bias}){lit_suffix}{clamp_suffix}{lts_suffix})";
    std::string mfma_buildin = "__builtin_hcu_mmac_" + prefix;
    Replacer replacer;

    replacer.register_rule("{mfma_buildin}", mfma_buildin);
    replacer.register_rule("{A_dtype}", dtype_map[A_dtype]);
    replacer.register_rule("{B_dtype}", dtype_map[B_dtype]);
    replacer.register_rule("{C_dtype}", dtype_map[C_dtype]);
    replacer.register_rule(
        "{a_ref_cast}",
        hcu_tf32_ab ? ("reinterpret_cast<int *>(" + a_ref + ")") : a_ref);
    replacer.register_rule(
        "{b_ref_cast}",
        hcu_tf32_ab ? ("reinterpret_cast<int *>(" + b_ref + ")") : b_ref);
    replacer.register_rule("{a_bias}", a_bias);
    replacer.register_rule("{b_bias}", b_bias);
    replacer.register_rule("{c_ref}", c_ref);
    replacer.register_rule("{c_bias}", c_bias);
    replacer.register_rule("{lit_suffix}", lit);
    replacer.register_rule("{clamp_suffix}", clamp);
    replacer.register_rule("{lts_suffix}", lts);
    os << replacer.rewrite(call_mfma_code);
  } else if (op->op.same_as(tl::tvm_rdna_wmma())) {
    ICHECK(op->args.size() == 12U) << "tvm_rdna_wmma expects 12 arguments";
    std::string shape = Downcast<StringImm>(op->args[0])->value;
    std::string a_ref = this->PrintExpr(op->args[6]);
    std::string a_bias = this->PrintExpr(op->args[7]);
    std::string b_ref = this->PrintExpr(op->args[8]);
    std::string b_bias = this->PrintExpr(op->args[9]);
    std::string c_ref = this->PrintExpr(op->args[10]);
    std::string c_bias = this->PrintExpr(op->args[11]);

    std::string wmma_builtin = "__builtin_amdgcn_wmma_" + shape + "_gfx12";

    std::string call_wmma_code = R"({
      typedef __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16 tl_v8f16;
      typedef __attribute__((__vector_size__(8 * sizeof(float)))) float tl_v8f32;
      *((tl_v8f32*){c_ref} + {c_bias}) = {wmma_builtin}(
          *((tl_v8f16*){a_ref} + {a_bias}),
          *((tl_v8f16*){b_ref} + {b_bias}),
          *((tl_v8f32*){c_ref} + {c_bias}));
    })";
    Replacer wmma_replacer;
    wmma_replacer.register_rule("{wmma_builtin}", wmma_builtin);
    wmma_replacer.register_rule("{a_ref}", a_ref);
    wmma_replacer.register_rule("{a_bias}", a_bias);
    wmma_replacer.register_rule("{b_ref}", b_ref);
    wmma_replacer.register_rule("{b_bias}", b_bias);
    wmma_replacer.register_rule("{c_ref}", c_ref);
    wmma_replacer.register_rule("{c_bias}", c_bias);
    os << wmma_replacer.rewrite(call_wmma_code);
  } else if (op->op.same_as(builtin::thread_return())) {
    os << "return";
  } else if (op->op.same_as(tl::tl_gemm())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl_gemm expects 4 arguments <op_instance, "
           "A_ptr, B_ptr, C_ptr>, but got "
        << op->args.size();
    auto op_instance = Downcast<StringImm>(op->args[0]);
    bool hcu_tf32_ab = false;
    auto ab_hint = op->annotations.find("tl.hcu_tf32_ab");
    if (ab_hint != op->annotations.end()) {
      const auto *imm = (*ab_hint).second.as<IntImmNode>();
      hcu_tf32_ab = imm && imm->value != 0;
    }
    if (hcu_tf32_ab) {
      os << static_cast<std::string>(op_instance->value) << "(";
      for (size_t i = 1; i < op->args.size(); ++i) {
        if (i > 1) {
          os << ", ";
        }
        std::string arg = PrintExpr(op->args[i]);
        if (i == 1 || i == 2) {
          os << "reinterpret_cast<int *>(" << arg << ")";
        } else {
          os << arg;
        }
      }
      os << ")";
    } else {
      this->PrintCallExtern(GetType(tvm::ffi::GetRef<PrimExpr>(op)),
                            op_instance->value, op->args, true, os);
    }
  } else if (op->op.same_as(tl::tl_gemm_sp())) {
    LOG(FATAL) << "tl_gemm_sp is not supported on HCU";
  } else if (op->op.same_as(tl::loop_break())) {
    this->PrintIndent();
    this->stream << "break;\n";
  } else if (op->op.same_as(tl::warp_reduce_sum())) {
    os << "tl::warp_reduce_sum(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_max())) {
    os << "tl::warp_reduce_max(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_min())) {
    os << "tl::warp_reduce_min(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_bitand())) {
    os << "tl::warp_reduce_bitand(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_bitor())) {
    os << "tl::warp_reduce_bitor(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::atomic_add_elem_op())) {
    std::string dst_ptr = PrintExpr(op->args[0]);
    std::string src_value = PrintExpr(op->args[1]);
    this->PrintIndent();
    this->stream << "AtomicAdd(" << dst_ptr << ", " << src_value;
    if (op->args.size() > 2) {
      this->stream << ", " << PrintExpr(op->args[2]);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::atomic_add_ret_elem_op())) {
    os << "AtomicAddRet(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]);
    if (op->args.size() > 2) {
      os << ", " << PrintExpr(op->args[2]);
    }
    os << ")";
  } else if (op->op.same_as(tl::atomic_addx2_elem_op())) {
    std::string dst_ptr = PrintExpr(op->args[0]);
    std::string src_ptr = PrintExpr(op->args[1]);
    this->PrintIndent();
    this->stream << "AtomicAddx2(" << dst_ptr << ", " << src_ptr;
    if (op->args.size() > 2) {
      this->stream << ", " << PrintExpr(op->args[2]);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::atomic_addx4_elem_op())) {
    std::string dst_ptr = PrintExpr(op->args[0]);
    std::string src_ptr = PrintExpr(op->args[1]);
    this->PrintIndent();
    this->stream << "AtomicAddx4(" << dst_ptr << ", " << src_ptr;
    if (op->args.size() > 2) {
      this->stream << ", " << PrintExpr(op->args[2]);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::atomic_load_elem_op())) {
    os << "AtomicLoad(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::atomic_store_elem_op())) {
    std::string dst_ptr = PrintExpr(op->args[0]);
    std::string value = PrintExpr(op->args[1]);
    std::string memory_order = PrintExpr(op->args[2]);
    this->PrintIndent();
    this->stream << "AtomicStore(" << dst_ptr << ", " << value << ", "
                 << memory_order << ");\n";
  } else if (op->op.same_as(tl::atomic_max_elem_op())) {
    std::string dst_ptr = PrintExpr(op->args[0]);
    std::string src_value = PrintExpr(op->args[1]);
    this->PrintIndent();
    this->stream << "AtomicMax(" << dst_ptr << ", " << src_value;
    if (op->args.size() > 2) {
      this->stream << ", " << PrintExpr(op->args[2]);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::atomic_max_ret_elem_op())) {
    os << "AtomicMaxRet(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]);
    if (op->args.size() > 2) {
      os << ", " << PrintExpr(op->args[2]);
    }
    os << ")";
  } else if (op->op.same_as(tl::atomic_min_elem_op())) {
    std::string dst_ptr = PrintExpr(op->args[0]);
    std::string src_value = PrintExpr(op->args[1]);
    this->PrintIndent();
    this->stream << "AtomicMin(" << dst_ptr << ", " << src_value;
    if (op->args.size() > 2) {
      this->stream << ", " << PrintExpr(op->args[2]);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::atomic_min_ret_elem_op())) {
    os << "AtomicMinRet(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]);
    if (op->args.size() > 2) {
      os << ", " << PrintExpr(op->args[2]);
    }
    os << ")";
  } else if (op->op.same_as(tl::set_max_nreg())) {
    this->PrintIndent();
    int nreg = Downcast<IntImm>(op->args[0])->value;
    this->stream << "__builtin_hcu_s_set_vgpr_size(" << nreg << ");\n";
    return;
  } else if (op->op.same_as(tl::hcu_wdra_init())) {
    if (wdra_init_emitted_) {
      return;
    }
    this->PrintIndent();
    PrintHcuWdraInit(this->stream, op->args);
    this->stream << ";\n";
    return;
  } else if (op->op.same_as(tl::no_set_max_nreg())) {
    // HCU doesn't need explicit register management like CUDA
    // This is a no-op for HIP
    return;
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangHCU::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == tl::attr::kLexicalAllocScope) {
    PrintIndent();
    stream << "{\n";
    int scope = BeginScope();
    PrintStmt(op->body);
    EndScope(scope);
    PrintIndent();
    stream << "}\n";
    return;
  }
  if (op->attr_key == tir::attr::async_commit_queue_scope) {
    const IntImmNode *queue_id = op->value.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0)
        << "For CUDA, the index of an async queue must be 0.";
    this->VisitStmt(op->body);
    auto commit_group = Call(DataType::Void(), builtin::ptx_commit_group(), {});
    this->VisitExpr(commit_group, this->stream);
    return;
  } else if (op->attr_key == tir::attr::async_wait_queue_scope) {
    auto wait_attrs = GetAsyncWaitAttributes(op);
    auto queue_id = wait_attrs.first.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0)
        << "For CUDA, the index of an async queue must be 0.";
    auto wait_cnt = wait_attrs.second;
    auto wait_group =
        Call(DataType::Void(), builtin::ptx_wait_group(), {wait_cnt});
    this->VisitExpr(wait_group, this->stream);
    auto inner = op->body.as<AttrStmtNode>();
    ICHECK(inner);
    this->VisitStmt(inner->body);
    return;
  } else if (op->attr_key == "threadblock_swizzle_pattern") {
    this->PrintIndent();
    std::string func_name;
    int panel_size = 0;
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(tir::builtin::tvm_tuple()) &&
          call->args.size() >= 2) {
        const auto *name_node = call->args[0].as<StringImmNode>();
        const auto *size_node = call->args[1].as<IntImmNode>();
        ICHECK(name_node && size_node) << "threadblock_swizzle_pattern expects "
                                          "tvm_tuple(device_func, panel_size)";
        func_name = name_node->value;
        panel_size = static_cast<int>(size_node->value);
      }
    }
    ICHECK(!func_name.empty() && panel_size > 0)
        << "threadblock_swizzle_pattern: failed to extract func_name and "
           "panel_size";
    this->stream << "const dim3 blockIdx = tl::" << func_name << "<"
                 << panel_size << ">();\n";
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == tl::attr::kDisableBufferOpsMap) {
    if (auto map = op->node.as<Map<String, PrimExpr>>()) {
      for (const auto &[var, enabled] : map.value()) {
        if (auto int_val = enabled.as<IntImmNode>()) {
          if (int_val->value != 0) {
            buffer_ops_disable_param_names_.insert(var);
          }
        }
      }
    }
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangHCU::VisitStmt_(const tir::BlockNode *op) {
  if (op->annotations.count(tl::attr::kDirectToLDSMap)) {
    auto map = op->annotations.Get(tl::attr::kDirectToLDSMap)
                   ->as<Map<Var, PrimExpr>>()
                   .value();
    for (const auto &[var, enabled] : map) {
      if (auto int_val = enabled.as<IntImmNode>()) {
        direct_to_lds_map_[var.get()] = (int_val->value != 0);
      }
    }
  }

  this->VisitStmt(op->body);
}

void CodeGenTileLangHCU::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  std::string scope = GetPtrStorageScope(op->buffer_var);
  PrintStorageScope(scope, stream);
  PrintType(op->dtype, stream);

  if (scope == "shared.dyn") {
    stream << ' ' << vid << "[];\n";
  } else {
    size_t constant_size = op->ConstantAllocationSize();
    ICHECK_GT(constant_size, 0)
        << "Can only handle constant size stack allocation for now";

    if ((op->dtype == DataType::Int(4) || op->dtype == DataType::UInt(4) ||
         op->dtype == DataType::Int(1)) &&
        scope == "shared") {
      constant_size = constant_size / (32 / op->dtype.bits());
    }
    stream << ' ' << vid << '[' << constant_size << "];\n";
    if (scope == "local.var") {
      PrimExpr init = tir::make_const(op->dtype, 0);
      auto init_it = op->annotations.find(tl::attr::kLocalVarInit);
      if (init_it != op->annotations.end()) {
        PrimExpr user_init = Downcast<PrimExpr>((*init_it).second);
        if (!user_init.dtype().is_void() && user_init.dtype() != op->dtype) {
          user_init = tir::Cast(op->dtype, user_init);
        }
        init = user_init;
      }
      PrintIndent();
      stream << vid << "[0] = " << PrintExpr(init) << ";\n";
    }
  }

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenTileLangHCU::VisitExpr_(const RampNode *op, std::ostream &os) {
  int lanes = static_cast<int>(Downcast<IntImm>(op->lanes)->value);
  CHECK_LE(lanes, 4) << "ValueError: Ramp of more than 4 lanes is not allowed.";
  os << "(make_";
  PrintType(op->dtype, os);
  os << "(";
  for (int i = 0; i < lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    if (i != lanes - 1)
      os << ", ";
  }
  os << "))";
}

void CodeGenTileLangHCU::VisitExpr_(const ShuffleNode *op,
                                    std::ostream &os) { // NOLINT(*)
  DataType t = op->dtype;
  bool is_bf16x2 = t.is_bfloat16() && t.lanes() == 2;
  bool is_fp16x2 = t.is_float16() && t.lanes() == 2;
  if ((is_bf16x2 || is_fp16x2) && op->vectors.size() == 2 &&
      op->vectors[0].dtype().lanes() == 1 &&
      op->vectors[1].dtype().lanes() == 1) {
    std::string e0 = PrintExpr(op->vectors[0]);
    std::string e1 = PrintExpr(op->vectors[1]);
    if (is_bf16x2) {
      os << "uint1{__pack_bfloat162(" << e0 << ", " << e1 << ")}";
    } else {
      os << "uint1{__pack_half2((half_t)(" << e0 << "), (half_t)(" << e1
         << "))}";
    }
    return;
  }
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangHCU::VisitExpr_(const BroadcastNode *op,
                                    std::ostream &os) { // NOLINT(*)
  int lanes = static_cast<int>(Downcast<IntImm>(op->lanes)->value);
  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 8 &&
      lanes == 4) {
    // make_int8x4
    const int64_t *p = as_const_int(op->value);
    ICHECK(p);
    int64_t v = *p & 0xFF;
    v = (v << 24) | (v << 16) | (v << 8) | v;
    if (op->dtype.is_uint()) {
      os << "(uint)" << v;
    } else {
      os << "(int)" << v;
    }
    return;
  }

  if (op->dtype.is_float16()) {
    std::string v = PrintExpr(op->value);
    os << "make_";
    PrintType(op->dtype, os);
    os << '(';
    for (int i = 0; i < lanes / 2; ++i) {
      if (i != 0)
        os << ", ";
      os << "__pack_half2(" << v << ", " << v << ")";
    }
    os << ')';
    return;
  }

  if (op->dtype.is_bfloat16() && lanes == 16) {
    std::string v = PrintExpr(op->value);
    os << "bfloat16x16{";
    for (int bi = 0; bi < 16; ++bi) {
      if (bi != 0)
        os << ", ";
      os << v;
    }
    os << "}";
    return;
  }

  if (op->dtype.is_bfloat16()) {
    std::string v = PrintExpr(op->value);
    os << "make_";
    PrintType(op->dtype, os);
    os << '(';
    for (int i = 0; i < lanes / 2; ++i) {
      if (i != 0)
        os << ", ";
      os << "__pack_bfloat162(" << v << ", " << v << ")";
    }
    os << ')';
    return;
  }

  if (op->dtype.is_float() && op->dtype.bits() == 32 &&
      op->dtype.lanes() == 8) {
    std::string v = PrintExpr(op->value);
    os << "make_ulonglong4(";
    for (int i = 0; i < 4; ++i) {
      if (i != 0)
        os << ", ";
      os << "*(unsigned long long*)&make_float2(" << v << ", " << v << ")";
    }
    os << ')';
    return;
  }

  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 4) {
    bool fail = false;
    const int64_t *p = as_const_int(op->value);
    ICHECK(p);
    int64_t v = *p & 0xF;

    if (lanes == 4) {
      v = (v << 12) | (v << 8) | (v << 4) | v;
      if (op->dtype.is_uint()) {
        os << "(uint16_t)" << v;
      } else {
        os << "(int16_t)" << v;
      }
    } else {
      v = (v << 28) | (v << 24) | (v << 20) | (v << 16) | (v << 12) | (v << 8) |
          (v << 4) | v;
      if (lanes == 8) {
        if (op->dtype.is_uint()) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
      } else if (lanes == 16 || lanes == 32) {
        os << "make_";
        PrintType(op->dtype, os);
        os << '(';
        for (int i = 0; i < lanes / 8; ++i) {
          if (i != 0)
            os << ", ";
          if (op->dtype.is_uint()) {
            os << "(uint)" << v;
          } else {
            os << "(int)" << v;
          }
        }
        os << ')';
      } else {
        fail = true;
      }
    }

    if (!fail) {
      return;
    }
  }

  std::string v = PrintExpr(op->value);
  os << "make_";
  PrintType(op->dtype, os);
  os << '(';
  for (int i = 0; i < lanes; ++i) {
    if (i != 0)
      os << ", ";
    os << v;
  }
  os << ')';
}

inline void PrintConst(const FloatImmNode *op, std::ostream &os,
                       CodeGenTileLangHCU *p) { // NOLINT(*)
  // Type code is kBFloat
  if (op->dtype.is_bfloat16()) {
    os << "bfloat16_t";
    os << '(' << std::scientific << op->value << 'f' << ')';
    return;
  } else if (op->dtype.is_float8_e4m3fnuz() || op->dtype.is_float8_e4m3() ||
             op->dtype.is_float8_e4m3fn()) {
    os << "fp8_e4_t";
    os << '(' << std::scientific << op->value << 'f' << ')';
    return;
  }
  // Type code is kFloat
  switch (op->dtype.bits()) {
  case 64:
  case 32: {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      if (op->value < 0) {
        temp << "-";
      }
      temp << ((op->dtype.bits() == 32) ? "HUGE_VALF" : "HUGE_VAL");
    } else if (std::isnan(op->value)) {
      temp << ((op->dtype.bits() == 32) ? "NAN" : "NAN");
    } else {
      temp << std::scientific << op->value;
      if (op->dtype.bits() == 32)
        temp << 'f';
    }
    p->MarkConst(temp.str());
    os << temp.str();
    break;
  }
  case 16: {
    os << "half_t" << '(';
    FloatImm const_f32 = FloatImm(DataType::Float(32), op->value);
    PrintConst(const_f32.get(), os, p);
    os << ')';
    break;
  }
  default:
    LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenTileLangHCU::VisitExpr_(const FloatImmNode *op,
                                    std::ostream &os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenTileLangHCU::HandleVolatileLoads(const std::string &value,
                                             const BufferLoadNode *op,
                                             std::ostream &os) {
  // Cast away volatile qualifier for fp16 types. That is, only loads and
  // stores are volatile. The loaded objects are not marked as volatile.
  //
  if ((op->dtype.is_float16() || op->dtype.is_bfloat16()) &&
      IsVolatile(op->buffer->data.get())) {
    os << "(";
    PrintType(op->dtype, os);
    os << ")(" << value << ")";
  } else {
    os << value;
  }
}

void CodeGenTileLangHCU::PrintVecElemLoadExpr(DataType t, int i,
                                              const std::string &value,
                                              std::ostream &os) {
  ICHECK_GT(t.lanes(), 1);
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (!(t.lanes() == 2 || t.lanes() == 3)) {
      if (i != 0) {
        os << "|";
      }
      os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8
         << "))";
      return;
    }
  }

  if (t.is_float16()) {
    if (i == 0) {
      os << "make_";
      PrintType(t, os);
      os << '(';
    }
    if (i % 2 == 0) {
      os << "__pack_half2(" << value;
    } else {
      os << "," << value << ")";
      if (i != t.lanes() - 1) {
        os << ",";
      } else {
        os << ")";
      }
    }
    return;
  }

  if ((t.lanes() == 16 || t.lanes() == 32) && t.bits() == 32 && t.is_float()) {
    // float32x16/x32: compound literal for Clang vector types; no matching
    // make_* in templates.
    if (i == 0)
      os << "(float32x" << t.lanes() << "){";
    os << value;
    if (i != t.lanes() - 1)
      os << ",";
    else
      os << "}";
    return;
  }

  if (t.lanes() == 16 && t.is_bfloat16()) {
    if (i == 0)
      os << "bfloat16x16{";
    os << value;
    if (i != t.lanes() - 1)
      os << ",";
    else
      os << "}";
    return;
  }

  if (t.is_bfloat16()) {
    if (i == 0) {
      os << "make_";
      PrintType(t, os);
      os << '(';
    }
    if (i % 2 == 0) {
      os << "__pack_bfloat162(" << value;
    } else {
      os << "," << value << ")";
      if (i != t.lanes() - 1) {
        os << ",";
      } else {
        os << ")";
      }
    }
    return;
  }

  if (i == 0) {
    os << "make_";
    PrintType(t, os);
    os << "(";
  }
  os << value;
  if (i != t.lanes() - 1) {
    os << ",";
  } else {
    os << ")";
  }
  return;
}

void CodeGenTileLangHCU::AddFunction(const PrimFunc &f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();
  buffer_ops_disable_param_names_.clear();
  mls_resource_object_counter_ = 0;
  wdra_init_emitted_ = false;

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.has_value())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);
  std::unordered_set<const VarNode *> non_restrict;
  if (auto opt =
          f->GetAttr<ffi::Array<tir::Var>>(tl::attr::kNonRestrictParams)) {
    for (const tir::Var &v : opt.value())
      non_restrict.insert(v.get());
  }

  this->PrintFuncPrefix(stream);
  CodeGenC::PrintType(f->ret_type, stream);
  this->PrintExtraAttrs(f, stream);
  this->stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0)
      stream << ", ";
    if (v.dtype().is_handle()) {
      // work around for grid constant parameters.
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (ptr->storage_scope == "grid_constant") {
          stream << "__grid_constant__ const ";
          CodeGenC::PrintType(ptr->element_type, stream);
          stream << ' ' << vid;
          continue;
        }
      }

      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
      }

      CodeGenC::PrintType(GetType(v), stream);
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }

      if (no_alias && !non_restrict.count(v.get())) {
        PrintRestrict(v, stream);
      }
    } else {
      CodeGenC::PrintType(GetType(v), stream);
    }
    stream << ' ' << vid;
  }
  stream << ") {\n";
  int func_scope = this->BeginScope();
  this->PreFunctionBody(f);
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

} // namespace codegen
} // namespace tvm
