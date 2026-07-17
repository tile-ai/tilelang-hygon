/*!
 * \file hoist_mls_resource.cc
 * \brief Hoist HCU MLS resource setup before codegen.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hcu/utils/extern_call_checker.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
using tvm::transform::PassContext;

namespace {

static constexpr const char *kMlsLoadTilePrefix = "tl::mls::mls_load_tile<";
static constexpr const char *kMlsResourceInitPrefix = "tl::mls::resource_init<";

bool IsMlsLoadTileCall(const CallNode *call) {
  return IsMlsLoadTileExternCall(call);
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
  ICHECK(sym.find(kMlsLoadTilePrefix) == 0) << "Unexpected MLS symbol: " << sym;
  ICHECK_EQ(sym.back(), '>') << "Malformed MLS template symbol: " << sym;
  auto args = SplitTopLevelTemplateArgs(
      sym.substr(std::strlen(kMlsLoadTilePrefix),
                 sym.size() - std::strlen(kMlsLoadTilePrefix) - 1));
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
  auto args = SplitTopLevelTemplateArgs(
      sym.substr(std::strlen(kMlsLoadTilePrefix),
                 sym.size() - std::strlen(kMlsLoadTilePrefix) - 1));
  ICHECK_GE(args.size(), 5U)
      << "mls_load_tile expects DataType as template arg 4";
  return args[4];
}

std::pair<std::string, std::string>
MlsLastLoadTemplateArgs(const std::string &sym) {
  auto args = SplitTopLevelTemplateArgs(
      sym.substr(std::strlen(kMlsLoadTilePrefix),
                 sym.size() - std::strlen(kMlsLoadTilePrefix) - 1));
  std::string check_last_load = args.size() > 8 ? args[8] : "true";
  std::string last_load = args.size() > 9 ? args[9] : "false";
  return {check_last_load, last_load};
}

std::optional<std::pair<int64_t, int64_t>>
MlsBlockSizesFromLoadTile(const std::string &sym) {
  auto args = SplitTopLevelTemplateArgs(
      sym.substr(std::strlen(kMlsLoadTilePrefix),
                 sym.size() - std::strlen(kMlsLoadTilePrefix) - 1));
  if (args.empty()) {
    return std::nullopt;
  }
  const std::string &block_size = args[0];
  size_t begin = block_size.find('<');
  size_t end = block_size.rfind('>');
  if (begin == std::string::npos || end == std::string::npos || begin >= end) {
    return std::nullopt;
  }
  auto dims =
      SplitTopLevelTemplateArgs(block_size.substr(begin + 1, end - begin - 1));
  if (dims.size() < 2) {
    return std::nullopt;
  }
  try {
    return std::make_pair(std::stoll(dims[0]), std::stoll(dims[1]));
  } catch (...) {
    return std::nullopt;
  }
}

bool ExprUsesVar(const PrimExpr &expr, const VarNode *target) {
  bool found = false;
  tirx::PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (found)
      return;
    if (const auto *var = node.as<VarNode>()) {
      found = var == target;
    }
  });
  return found;
}

bool ExprUsesAnyVar(const PrimExpr &expr,
                    const std::unordered_set<const VarNode *> &vars) {
  bool found = false;
  tirx::PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (found)
      return;
    if (const auto *var = node.as<VarNode>()) {
      found = vars.count(var) != 0;
    }
  });
  return found;
}

void CollectForLoopVars(const Stmt &stmt,
                        std::unordered_set<const VarNode *> *vars) {
  tirx::PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (const auto *for_node = node.as<ForNode>()) {
      vars->insert(for_node->loop_var.get());
    }
  });
}

void CollectMlsLoadTileCalls(const Stmt &stmt,
                             std::vector<const CallNode *> *calls) {
  tirx::PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (const auto *call = node.as<CallNode>()) {
      if (IsMlsLoadTileCall(call)) {
        calls->push_back(call);
      }
    }
  });
}

std::string ExprKey(const PrimExpr &expr) {
  return std::to_string(StructuralHash()(expr));
}

std::string ResourceKey(const std::string &base_template,
                        const CallNode *call) {
  std::ostringstream os;
  os << base_template;
  for (int i = 1; i <= 5; ++i) {
    os << "|" << ExprKey(call->args[i]);
  }
  if (call->args.size() == 9U) {
    os << "|" << ExprKey(call->args[8]);
  } else {
    os << "|" << ExprKey(IntImm(DataType::Int(32), 0));
  }
  return os.str();
}

Stmt MakeExternStmt(const std::string &symbol,
                    const std::vector<PrimExpr> &args) {
  Array<PrimExpr> call_args;
  call_args.push_back(StringImm(symbol));
  for (const PrimExpr &arg : args) {
    call_args.push_back(arg);
  }
  return Evaluate(Call(DataType::Int(32), builtin::call_extern(), call_args));
}

bool UsesThreadOrWavePartition(
    const PrimExpr &expr,
    const std::unordered_set<const VarNode *> &thread_vars,
    const std::unordered_set<const VarNode *> &thread_alias_vars,
    const std::unordered_set<const VarNode *> &seq_thread_aliases) {
  bool found = false;
  tirx::PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (found) {
      return;
    }
    if (const auto *var = node.as<VarNode>()) {
      if (thread_vars.count(var) || thread_alias_vars.count(var) ||
          seq_thread_aliases.count(var)) {
        found = true;
      }
      return;
    }
    if (const auto *call = node.as<CallNode>()) {
      if (call->op.same_as(get_wave_id()) || call->op.same_as(get_warp_idx()) ||
          call->op.same_as(get_warp_idx_sync()) ||
          call->op.same_as(get_lane_idx()) ||
          call->op.same_as(get_warp_group_idx())) {
        found = true;
      }
    }
  });
  return found;
}

bool IsThreadWavePartitionIf(
    const IfThenElseNode *op,
    const std::unordered_set<const VarNode *> &thread_vars,
    const std::unordered_set<const VarNode *> &thread_alias_vars,
    const std::unordered_set<const VarNode *> &seq_thread_aliases) {
  return UsesThreadOrWavePartition(op->condition, thread_vars,
                                   thread_alias_vars, seq_thread_aliases);
}

void CollectSeqThreadAliases(
    const SeqStmtNode *op,
    const std::unordered_set<const VarNode *> &thread_vars,
    std::unordered_set<const VarNode *> *seq_thread_aliases) {
  for (const Stmt &stmt : op->seq) {
    if (const auto *let = stmt.as<BindNode>()) {
      if (const auto *v = let->value.as<VarNode>()) {
        if (thread_vars.count(v)) {
          seq_thread_aliases->insert(let->var.get());
        }
      }
    }
  }
}

Stmt BuildHoistEligibleSeqPrefix(
    const SeqStmtNode *op,
    const std::unordered_set<const VarNode *> &thread_vars,
    const std::unordered_set<const VarNode *> &thread_alias_vars,
    const std::unordered_set<const VarNode *> &seq_thread_aliases) {
  Array<Stmt> prefix;
  for (const Stmt &stmt : op->seq) {
    if (const auto *if_node = stmt.as<IfThenElseNode>()) {
      if (IsThreadWavePartitionIf(if_node, thread_vars, thread_alias_vars,
                                  seq_thread_aliases)) {
        break;
      }
    }
    prefix.push_back(stmt);
  }
  if (prefix.empty()) {
    return Stmt();
  }
  return prefix.size() == 1 ? prefix[0] : SeqStmt(prefix);
}

} // namespace

class HoistMlsResourceMutator : public StmtMutator {
public:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      const std::string &tag = iv->thread_tag;
      if (tag.rfind("threadIdx", 0) == 0) {
        thread_vars_.insert(iv->var.get());
        Stmt body = StmtMutator::VisitStmt_(op);
        thread_vars_.erase(iv->var.get());
        return body;
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BindNode *op) final {
    const VarNode *alias = nullptr;
    if (const auto *v = op->value.as<VarNode>()) {
      if (thread_vars_.count(v)) {
        alias = op->var.get();
        thread_alias_vars_.insert(alias);
      }
    }
    return GetRef<Stmt>(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (!IsThreadWavePartitionIf(op, thread_vars_, thread_alias_vars_,
                                 seq_thread_aliases_)) {
      return StmtMutator::VisitStmt_(op);
    }
    ++scope_depth_;
    Stmt then_case = VisitStmt(op->then_case);
    ExpireScopes();
    Optional<Stmt> else_case = op->else_case;
    if (else_case.defined()) {
      else_case = VisitStmt(else_case.value());
      ExpireScopes();
    }
    --scope_depth_;
    return IfThenElse(op->condition, then_case, else_case);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    std::vector<Stmt> prelude;
    std::unordered_set<const VarNode *> nested_loop_vars;

    std::unordered_set<const VarNode *> saved_seq_aliases = seq_thread_aliases_;
    seq_thread_aliases_.clear();
    CollectSeqThreadAliases(op, thread_vars_, &seq_thread_aliases_);

    Stmt hoist_region = BuildHoistEligibleSeqPrefix(
        op, thread_vars_, thread_alias_vars_, seq_thread_aliases_);
    if (hoist_region.defined()) {
      CollectForLoopVars(hoist_region, &nested_loop_vars);
    }

    ++scope_depth_;
    if (hoist_region.defined()) {
      PredeclareMlsResources(hoist_region, /*loop_var=*/nullptr, &prelude,
                             &nested_loop_vars);
    }

    Array<Stmt> seq;
    seq.reserve(op->seq.size() + prelude.size());
    for (const Stmt &stmt : prelude) {
      seq.push_back(stmt);
    }
    for (const Stmt &stmt : op->seq) {
      seq.push_back(VisitStmt(stmt));
    }

    ExpireScopes();
    --scope_depth_;
    seq_thread_aliases_ = saved_seq_aliases;
    return SeqStmt(seq);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    std::vector<Stmt> prelude;
    PredeclareMlsResources(op->body, op->loop_var.get(), &prelude);
    PreinitializeMlsWindows(op->body, op->loop_var.get(), &prelude);

    ++scope_depth_;
    active_loop_ranges_.push_back(
        {op->loop_var, Range::FromMinExtent(op->min, op->extent)});
    Stmt new_body = VisitStmt(op->body);
    active_loop_ranges_.pop_back();

    Stmt loop =
        For(op->loop_var, op->min, op->extent, op->kind, std::move(new_body),
            op->thread_binding, op->annotations, op->step, op->span);

    ExpireScopes();
    --scope_depth_;

    if (prelude.empty()) {
      return loop;
    }
    prelude.push_back(loop);
    return SeqStmt(prelude);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr || !IsMlsLoadTileCall(call)) {
      return StmtMutator::VisitStmt_(op);
    }
    ICHECK(call->args.size() == 8U || call->args.size() == 9U)
        << "mls_load_tile extern expects symbol, src, stride, mn_len, k_len, "
           "mn_base, k_base, dst[, warp_id_offset]";

    const auto *sym_node = call->args[0].as<StringImmNode>();
    const std::string sym = sym_node->value;
    const std::string base_template = MlsBaseTemplateFromLoadTile(sym);
    const std::string key = ResourceKey(base_template, call);
    auto it = resource_names_.find(key);
    if (it == resource_names_.end()) {
      return StmtMutator::VisitStmt_(op);
    }

    const std::string &obj_name = it->second;
    std::vector<Stmt> seq;
    if (!initialized_scopes_.count(key)) {
      seq.push_back(MakeExternStmt(
          "tl::mls::set_window_origin",
          {StringImm(obj_name), call->args[5], IntImm(DataType::Int(32), 0)}));
      initialized_scopes_[key] = scope_depth_;
    }

    if (update_mn_scopes_.count(key)) {
      if (!updated_mn_scopes_.count(key)) {
        seq.push_back(MakeExternStmt("tl::mls::update_mn_base",
                                     {StringImm(obj_name), call->args[5]}));
        updated_mn_scopes_[key] = scope_depth_;
      }

      bool check_k_filter = true;
      bool check_mn_filter = true;
      if (auto block_sizes = MlsBlockSizesFromLoadTile(sym)) {
        check_mn_filter = !CanProveNoBoundary(call->args[5], call->args[3],
                                              block_sizes->first);
        check_k_filter = !CanProveNoBoundary(call->args[6], call->args[4],
                                             block_sizes->second);
      }

      const std::string data_type = MlsDataTypeFromLoadTile(sym);
      std::ostringstream async_sym;
      async_sym << "tl::mls::async_load_mn<" << data_type << ", false, "
                << (check_k_filter ? "true" : "false") << ", "
                << (check_mn_filter ? "true" : "false") << ">";
      seq.push_back(
          MakeExternStmt(async_sym.str(),
                         {StringImm(obj_name), call->args[7], call->args[5]}));
    } else {
      seq.push_back(MakeExternStmt("tl::mls::update_base",
                                   {StringImm(obj_name), call->args[6]}));
      const std::string data_type = MlsDataTypeFromLoadTile(sym);
      auto [check_last_load, last_load] = MlsLastLoadTemplateArgs(sym);
      if (auto block_sizes = MlsBlockSizesFromLoadTile(sym)) {
        if (CanProveNoBoundary(call->args[6], call->args[4],
                               block_sizes->second)) {
          check_last_load = "false";
          last_load = "false";
        }
      }
      std::ostringstream async_sym;
      async_sym << "tl::mls::async_load<" << data_type << ", "
                << check_last_load << ", " << last_load << ">";
      seq.push_back(
          MakeExternStmt(async_sym.str(),
                         {StringImm(obj_name), call->args[7], call->args[6]}));
    }
    return SeqStmt(seq);
  }

private:
  void PredeclareMlsResources(
      const Stmt &body, const VarNode *loop_var, std::vector<Stmt> *prelude,
      const std::unordered_set<const VarNode *> *forbidden_vars = nullptr) {
    std::vector<const CallNode *> calls;
    CollectMlsLoadTileCalls(body, &calls);
    for (const CallNode *call : calls) {
      ICHECK(call->args.size() == 8U || call->args.size() == 9U)
          << "mls_load_tile extern expects symbol, src, stride, mn_len, k_len, "
             "mn_base, k_base, dst[, warp_id_offset]";
      if ((loop_var != nullptr && (ExprUsesVar(call->args[1], loop_var) ||
                                   ExprUsesVar(call->args[2], loop_var) ||
                                   ExprUsesVar(call->args[3], loop_var) ||
                                   ExprUsesVar(call->args[4], loop_var))) ||
          (forbidden_vars != nullptr &&
           (ExprUsesAnyVar(call->args[1], *forbidden_vars) ||
            ExprUsesAnyVar(call->args[2], *forbidden_vars) ||
            ExprUsesAnyVar(call->args[3], *forbidden_vars) ||
            ExprUsesAnyVar(call->args[4], *forbidden_vars)))) {
        continue;
      }
      const auto *sym_node = call->args[0].as<StringImmNode>();
      const std::string sym = sym_node->value;
      const std::string base_template = MlsBaseTemplateFromLoadTile(sym);
      const std::string key = ResourceKey(base_template, call);
      if (resource_names_.count(key)) {
        continue;
      }
      const std::string obj_name =
          "_tl_mls_h_" + std::to_string(resource_counter_++);
      resource_names_[key] = obj_name;
      resource_scopes_[key] = scope_depth_;
      prelude->push_back(MakeExternStmt(
          kMlsResourceInitPrefix + base_template.substr(0) + ">",
          {StringImm(obj_name), call->args[1], call->args[2], call->args[3],
           call->args[4],
           call->args.size() == 9U ? call->args[8]
                                   : PrimExpr(IntImm(DataType::Int(32), 0))}));
    }
  }

  void PreinitializeMlsWindows(const Stmt &body, const VarNode *loop_var,
                               std::vector<Stmt> *prelude) {
    std::vector<const CallNode *> calls;
    CollectMlsLoadTileCalls(body, &calls);
    std::unordered_set<const VarNode *> moving_loop_vars;
    CollectForLoopVars(body, &moving_loop_vars);
    moving_loop_vars.insert(loop_var);

    std::unordered_set<std::string> keys_with_moving_k_base;
    for (const CallNode *call : calls) {
      const auto *sym_node = call->args[0].as<StringImmNode>();
      const std::string base_template =
          MlsBaseTemplateFromLoadTile(sym_node->value);
      const std::string key = ResourceKey(base_template, call);
      if (ExprUsesAnyVar(call->args[6], moving_loop_vars)) {
        keys_with_moving_k_base.insert(key);
      }
    }

    for (const CallNode *call : calls) {
      const auto *sym_node = call->args[0].as<StringImmNode>();
      const std::string base_template =
          MlsBaseTemplateFromLoadTile(sym_node->value);
      const std::string key = ResourceKey(base_template, call);
      auto it = resource_names_.find(key);
      if (it == resource_names_.end()) {
        continue;
      }

      const bool mn_depends_on_loop = ExprUsesVar(call->args[5], loop_var);
      if (mn_depends_on_loop && keys_with_moving_k_base.count(key)) {
        continue;
      }

      if (!initialized_scopes_.count(key)) {
        PrimExpr init_mn = mn_depends_on_loop
                               ? PrimExpr(IntImm(DataType::Int(32), 0))
                               : call->args[5];
        PrimExpr init_k = mn_depends_on_loop
                              ? call->args[6]
                              : PrimExpr(IntImm(DataType::Int(32), 0));
        prelude->push_back(
            MakeExternStmt("tl::mls::set_window_origin",
                           {StringImm(it->second), init_mn, init_k}));
        initialized_scopes_[key] = scope_depth_;
        if (mn_depends_on_loop) {
          update_mn_scopes_[key] = scope_depth_;
        }
      }

      if (update_mn_scopes_.count(key) && !mn_depends_on_loop &&
          !updated_mn_scopes_.count(key)) {
        prelude->push_back(MakeExternStmt(
            "tl::mls::update_mn_base", {StringImm(it->second), call->args[5]}));
        updated_mn_scopes_[key] = scope_depth_;
      }
    }
  }

  bool CanProveNoBoundary(const PrimExpr &base, const PrimExpr &length,
                          int64_t block_size) const {
    if (block_size <= 0) {
      return false;
    }
    arith::Analyzer analyzer;
    for (const auto &loop_range : active_loop_ranges_) {
      analyzer.Bind(loop_range.first, loop_range.second, true);
    }
    PrimExpr block = tirx::make_const(base.dtype(), block_size);
    return analyzer.CanProve(analyzer.Simplify(base + block <= length));
  }

  void ExpireScopes() {
    std::vector<std::string> expired;
    for (const auto &kv : resource_scopes_) {
      if (kv.second >= scope_depth_) {
        expired.push_back(kv.first);
      }
    }
    for (const std::string &key : expired) {
      resource_names_.erase(key);
      resource_scopes_.erase(key);
      initialized_scopes_.erase(key);
      update_mn_scopes_.erase(key);
      updated_mn_scopes_.erase(key);
    }

    expired.clear();
    for (const auto &kv : initialized_scopes_) {
      if (kv.second >= scope_depth_) {
        expired.push_back(kv.first);
      }
    }
    for (const std::string &key : expired) {
      initialized_scopes_.erase(key);
      update_mn_scopes_.erase(key);
      updated_mn_scopes_.erase(key);
    }

    expired.clear();
    for (const auto &kv : updated_mn_scopes_) {
      if (kv.second >= scope_depth_) {
        expired.push_back(kv.first);
      }
    }
    for (const std::string &key : expired) {
      updated_mn_scopes_.erase(key);
    }
  }

  std::unordered_map<std::string, std::string> resource_names_;
  std::unordered_map<std::string, int> resource_scopes_;
  std::unordered_map<std::string, int> initialized_scopes_;
  std::unordered_map<std::string, int> update_mn_scopes_;
  std::unordered_map<std::string, int> updated_mn_scopes_;
  std::vector<std::pair<tirx::Var, Range>> active_loop_ranges_;
  std::unordered_set<const VarNode *> thread_vars_;
  std::unordered_set<const VarNode *> thread_alias_vars_;
  std::unordered_set<const VarNode *> seq_thread_aliases_;
  int scope_depth_{0};
  int resource_counter_{0};
};

PrimFunc HoistMlsResource(PrimFunc func) {
  std::vector<const CallNode *> mls_calls;
  CollectMlsLoadTileCalls(func->body, &mls_calls);
  if (mls_calls.empty()) {
    return func;
  }
  auto *n = func.CopyOnWrite();
  n->body = HoistMlsResourceMutator()(std::move(func->body));
  return func;
}

} // namespace tl
} // namespace tvm

namespace tvm {
namespace tl {
namespace transform {

tvm::transform::Pass HoistMlsResource() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return tl::HoistMlsResource(std::move(f));
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.HoistMlsResource", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.HoistMlsResource", HoistMlsResource);
}

} // namespace transform
} // namespace tl
} // namespace tvm
