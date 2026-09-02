/*!
 * \file hoist_mls_resource.cc
 * \brief Hoist HCU MLS resource setup before codegen.
 *
 * resource_init is hoisted within each lexical scope but does not cross
 * threadIdx-partition IfThenElse branches (e.g. WDRA `if tx < N`); for those
 * producer paths it is inserted at the top of the matching branch instead of
 * before enclosing loops. Else arms that continue a tx partition ladder (bare
 * nested if, or SeqStmt prefix then nested partition if) only recurse — they
 * do not Predeclare across the remaining branches.
 *
 * Conditions on get_warp_idx / get_wave_id / get_lane_idx / get_warp_group_idx
 * are not treated as partitions: TileLang does not narrow the in-scope tx
 * range for those predicates, so hoisting resource_init outside is safe.
 *
 * IfThenElse whose condition mentions MN origins of enclosed mls_load (or
 * whose then-arm is loop_break) is a window barrier: set_window stays inside
 * that then-arm and does not run on out-of-domain persistent waves.
 *
 * Absolute window placement: decide the fast/slow axis from every load of a
 * resource, then look at the earliest load.  If that load is already outside
 * the fast-axis loop (e.g. persistent prologue inside `for w`, not `for k`),
 * leave set_window at the load; do not pin it to the slow-axis For header.
 *
 * forward_delta / move_* is analyzed only when PassConfig tl.enable_hcu_wdra
 * is set; otherwise address mode stays absolute_rebase (update_*).
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hcu/utils/extern_call_checker.h"
#include "hcu/utils/mls_boundary.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
using tvm::transform::PassContext;

namespace {

static constexpr const char *kMlsLoadTilePrefix = "tl::mls::mls_load_tile<";
static constexpr const char *kMlsResourceInitPrefix = "tl::mls::resource_init<";

enum class MlsKAddressMode { kAbsoluteRebase, kForwardDelta };
enum class MlsResourceAxis { kMN, kK };

bool IsMlsLoadTileCall(const CallNode *call) {
  return IsMlsLoadTileExternCall(call);
}

// Grid / thread-binding loops are not data-plane axes.  Counting them as
// inner K/MN would emit update_* after set_window and apply the origin twice.
bool IsMlsDataLoop(const ForNode *op) {
  if (op->thread_binding.defined()) {
    return false;
  }
  return op->kind == ForKind::kSerial || op->kind == ForKind::kUnrolled;
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

std::string MlsBaseTemplateFromLoadTile(
    const std::string &sym,
    MlsKAddressMode mode = MlsKAddressMode::kAbsoluteRebase,
    std::optional<MlsResourceAxis> requested_axis = std::nullopt) {
  ICHECK(sym.find(kMlsLoadTilePrefix) == 0) << "Unexpected MLS symbol: " << sym;
  ICHECK_EQ(sym.back(), '>') << "Malformed MLS template symbol: " << sym;
  auto args = SplitTopLevelTemplateArgs(
      sym.substr(std::strlen(kMlsLoadTilePrefix),
                 sym.size() - std::strlen(kMlsLoadTilePrefix) - 1));
  ICHECK_GE(args.size(), 8U)
      << "mls_load_tile expects at least 8 template args";
  std::ostringstream os;
  os << "tl::mls::tilelang_mls_base<";
  size_t base_arg_count = 8;
  if (MlsLoadTileHasDstBits(args))
    base_arg_count = 9;
  for (size_t i = 0; i < base_arg_count; ++i) {
    if (i != 0)
      os << ", ";
    os << args[i];
  }
  if (requested_axis.has_value() || mode == MlsKAddressMode::kForwardDelta) {
    if (base_arg_count == 8)
      os << ", tl::mls::mls_elem_bits_v<" << args[4] << ">";
    if (requested_axis.has_value()) {
      os << ", tl::mls::mls_resource_axis::"
         << (*requested_axis == MlsResourceAxis::kMN ? "mn" : "k");
    } else {
      os << ", tl::mls::mls_resource_axis::auto_select";
    }
  }
  if (mode == MlsKAddressMode::kForwardDelta) {
    os << ", tl::mls::mls_address_mode::forward_delta";
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

MlsBoundaryModes MlsBoundaryFromLoadTile(const std::string &sym) {
  auto args = SplitTopLevelTemplateArgs(
      sym.substr(std::strlen(kMlsLoadTilePrefix),
                 sym.size() - std::strlen(kMlsLoadTilePrefix) - 1));
  return MlsParseBoundaryArgs(args);
}

std::string EncodeEstablishedMlsLoadTile(const std::string &sym,
                                         const std::string &refresh_k,
                                         const std::string &refresh_mn) {
  ICHECK(sym.find(kMlsLoadTilePrefix) == 0) << "Unexpected MLS symbol: " << sym;
  auto args = SplitTopLevelTemplateArgs(
      sym.substr(std::strlen(kMlsLoadTilePrefix),
                 sym.size() - std::strlen(kMlsLoadTilePrefix) - 1));
  size_t idx = MlsLoadTileBoundaryIndex(args);
  while (args.size() < idx + 2)
    args.emplace_back("-1");
  args.resize(idx + 2);
  // Settled 0/1 so one-shot and a later SpecializeAxisFilter do not re-analyze.
  args[idx] = MlsModeLiteral(refresh_k == "true" ? MlsBoundaryMode::kRefresh
                                                 : MlsBoundaryMode::kSkip);
  args[idx + 1] =
      MlsModeLiteral(refresh_mn == "true" ? MlsBoundaryMode::kRefresh
                                          : MlsBoundaryMode::kSkip);
  std::ostringstream os;
  os << kMlsLoadTilePrefix;
  for (size_t i = 0; i < args.size(); ++i) {
    if (i > 0)
      os << ", ";
    os << args[i];
  }
  os << ">";
  return os.str();
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

std::optional<std::pair<int64_t, int64_t>>
MlsWarpAccessCountsFromLoadTile(const std::string &sym) {
  auto args = SplitTopLevelTemplateArgs(
      sym.substr(std::strlen(kMlsLoadTilePrefix),
                 sym.size() - std::strlen(kMlsLoadTilePrefix) - 1));
  auto parse_sequence = [](const std::string &text)
      -> std::optional<std::pair<int64_t, int64_t>> {
    const size_t begin = text.find('<');
    const size_t end = text.rfind('>');
    if (begin == std::string::npos || end == std::string::npos || begin >= end)
      return std::nullopt;
    auto dims =
        SplitTopLevelTemplateArgs(text.substr(begin + 1, end - begin - 1));
    if (dims.size() != 2)
      return std::nullopt;
    try {
      return std::make_pair(std::stoll(dims[0]), std::stoll(dims[1]));
    } catch (...) {
      return std::nullopt;
    }
  };
  if (args.size() < 4)
    return std::nullopt;
  auto block = parse_sequence(args[0]);
  auto tile = parse_sequence(args[1]);
  if (!block || !tile)
    return std::nullopt;
  try {
    const int64_t warp_mn = std::stoll(args[2]);
    const int64_t warp_k = std::stoll(args[3]);
    const int64_t access_mn = block->first / (tile->first * warp_mn);
    const int64_t access_k = block->second / (tile->second * warp_k);
    return std::make_pair(access_mn, access_k);
  } catch (...) {
    return std::nullopt;
  }
}

MlsResourceAxis ResourceAxisFromLoadTile(const std::string &sym) {
  auto counts = MlsWarpAccessCountsFromLoadTile(sym);
  return counts && counts->second < counts->first ? MlsResourceAxis::kK
                                                  : MlsResourceAxis::kMN;
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

PrimExpr MlsLoopStep(const ForNode *op) {
  return op->step.defined()
             ? op->step.value()
             : PrimExpr(tirx::make_const(op->loop_var.dtype(), 1));
}

bool LoopUsesCoord(const ForNode *op, const PrimExpr &coord) {
  return ExprUsesVar(coord, op->loop_var.get());
}

// Walk out from the innermost fast-axis loop.  Loops that appear in neither
// MN nor K are transparent dummy parents: keep walking (window may land
// outside them).  Consecutive fast-only parents tile into one axis when
// finishing the next fast child equals one parent step.  Window sits on
// the first slow-axis loop, or on a related fast parent that does not
// tile.  Dummy layers never host the window; absolute rebase pins
// set_window *before* the outermost dummy (still inside any threadIdx
// partition If).  forward_delta rejects dummy and non-tiling fast parents
// via ForwardDeltaNestIsSafe — inner axis then uses update_* only.
int CollapseFastWindowDepth(const std::vector<const ForNode *> &stack,
                            int inner_depth, const PrimExpr &fast_coord,
                            const PrimExpr &slow_coord) {
  int window_depth = inner_depth - 1;
  arith::Analyzer analyzer;
  while (window_depth >= 0) {
    const ForNode *outer = stack[window_depth];
    if (LoopUsesCoord(outer, slow_coord)) {
      break;
    }
    if (!LoopUsesCoord(outer, fast_coord)) {
      --window_depth;
      continue;
    }
    int child_depth = window_depth + 1;
    while (child_depth <= inner_depth &&
           !LoopUsesCoord(stack[child_depth], slow_coord) &&
           !LoopUsesCoord(stack[child_depth], fast_coord)) {
      ++child_depth;
    }
    if (child_depth > inner_depth) {
      --window_depth;
      continue;
    }
    if (LoopUsesCoord(stack[child_depth], slow_coord)) {
      break;
    }
    const ForNode *child = stack[child_depth];
    PrimExpr child_end =
        analyzer.Simplify(child->min + child->extent * MlsLoopStep(child));
    PrimExpr after = analyzer.Simplify(
        Substitute(fast_coord, {{child->loop_var, child_end}}));
    PrimExpr next = analyzer.Simplify(Substitute(
        fast_coord, {{outer->loop_var, outer->loop_var + MlsLoopStep(outer)},
                     {child->loop_var, child->min}}));
    if (!analyzer.CanProve(after == next)) {
      break;
    }
    --window_depth;
  }
  return window_depth;
}

// forward_delta is only safe when every loop between the window and the
// innermost fast axis is itself a tiled fast parent.  A related parent
// that does not tile, or a dummy that wraps the fast axis, restarts the
// fast cursor (e.g. n wrapping k while A only depends on by).  Those
// nests stay absolute_rebase: window is lifted *through* the dummy,
// inner loads use update_* — never move_* across the dummy.
bool ForwardDeltaNestIsSafe(const std::vector<const ForNode *> &stack,
                            int inner_depth, int window_depth,
                            const PrimExpr &fast_coord,
                            const PrimExpr &slow_coord) {
  if (window_depth >= 0) {
    const ForNode *win = stack[window_depth];
    if (!LoopUsesCoord(win, slow_coord) && LoopUsesCoord(win, fast_coord)) {
      return false;
    }
  }
  for (int d = window_depth + 1; d < inner_depth; ++d) {
    if (!LoopUsesCoord(stack[d], fast_coord) &&
        !LoopUsesCoord(stack[d], slow_coord)) {
      return false;
    }
  }
  return true;
}

bool IsThreadWavePartitionIf(
    const IfThenElseNode *op,
    const std::unordered_set<const VarNode *> &thread_vars,
    const std::unordered_set<const VarNode *> &thread_alias_vars,
    const std::unordered_set<const VarNode *> &seq_thread_aliases);

void CollectForLoopVarsSkipThreadPartition(
    const Stmt &stmt, const std::unordered_set<const VarNode *> &thread_vars,
    const std::unordered_set<const VarNode *> &thread_alias_vars,
    const std::unordered_set<const VarNode *> &seq_thread_aliases,
    std::unordered_set<const VarNode *> *vars) {
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &s : seq->seq) {
      CollectForLoopVarsSkipThreadPartition(s, thread_vars, thread_alias_vars,
                                            seq_thread_aliases, vars);
    }
    return;
  }
  if (const auto *if_node = stmt.as<IfThenElseNode>()) {
    if (IsThreadWavePartitionIf(if_node, thread_vars, thread_alias_vars,
                                seq_thread_aliases)) {
      return;
    }
    CollectForLoopVarsSkipThreadPartition(if_node->then_case, thread_vars,
                                          thread_alias_vars, seq_thread_aliases,
                                          vars);
    if (if_node->else_case.defined()) {
      CollectForLoopVarsSkipThreadPartition(if_node->else_case.value(),
                                            thread_vars, thread_alias_vars,
                                            seq_thread_aliases, vars);
    }
    return;
  }
  if (const auto *for_node = stmt.as<ForNode>()) {
    vars->insert(for_node->loop_var.get());
    CollectForLoopVarsSkipThreadPartition(for_node->body, thread_vars,
                                          thread_alias_vars, seq_thread_aliases,
                                          vars);
    return;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    // Do not walk into thread launch from outside; AttrStmt visitor owns that
    // scope.
    if (attr->attr_key == tirx::attr::thread_extent) {
      return;
    }
    CollectForLoopVarsSkipThreadPartition(
        attr->body, thread_vars, thread_alias_vars, seq_thread_aliases, vars);
  }
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

class MlsCallLoopStackCollector : public StmtVisitor {
public:
  explicit MlsCallLoopStackCollector(
      std::unordered_map<const CallNode *, std::vector<const ForNode *>>
          *loop_stacks,
      std::vector<const CallNode *> *program_order)
      : loop_stacks_(loop_stacks), program_order_(program_order) {}

  void VisitStmt_(const ForNode *op) final {
    const bool track = IsMlsDataLoop(op);
    if (track) {
      loop_stack_.push_back(op);
    }
    StmtVisitor::VisitStmt_(op);
    if (track) {
      loop_stack_.pop_back();
    }
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (IsMlsLoadTileCall(call)) {
        (*loop_stacks_)[call] = loop_stack_;
        program_order_->push_back(call);
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

private:
  std::vector<const ForNode *> loop_stack_;
  std::unordered_map<const CallNode *, std::vector<const ForNode *>>
      *loop_stacks_;
  std::vector<const CallNode *> *program_order_;
};

void CollectMlsLoadTileCallsSkipThreadPartition(
    const Stmt &stmt, const std::unordered_set<const VarNode *> &thread_vars,
    const std::unordered_set<const VarNode *> &thread_alias_vars,
    const std::unordered_set<const VarNode *> &seq_thread_aliases,
    std::vector<const CallNode *> *calls) {
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &s : seq->seq) {
      CollectMlsLoadTileCallsSkipThreadPartition(
          s, thread_vars, thread_alias_vars, seq_thread_aliases, calls);
    }
    return;
  }
  if (const auto *if_node = stmt.as<IfThenElseNode>()) {
    if (IsThreadWavePartitionIf(if_node, thread_vars, thread_alias_vars,
                                seq_thread_aliases)) {
      return;
    }
    CollectMlsLoadTileCallsSkipThreadPartition(if_node->then_case, thread_vars,
                                               thread_alias_vars,
                                               seq_thread_aliases, calls);
    if (if_node->else_case.defined()) {
      CollectMlsLoadTileCallsSkipThreadPartition(if_node->else_case.value(),
                                                 thread_vars, thread_alias_vars,
                                                 seq_thread_aliases, calls);
    }
    return;
  }
  if (const auto *for_node = stmt.as<ForNode>()) {
    CollectMlsLoadTileCallsSkipThreadPartition(for_node->body, thread_vars,
                                               thread_alias_vars,
                                               seq_thread_aliases, calls);
    return;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    // Never collect across thread_extent from the outside; the AttrStmt
    // visitor establishes thread_vars_ and owns nested hoist.
    if (attr->attr_key == tirx::attr::thread_extent) {
      return;
    }
    CollectMlsLoadTileCallsSkipThreadPartition(
        attr->body, thread_vars, thread_alias_vars, seq_thread_aliases, calls);
    return;
  }
  CollectMlsLoadTileCalls(stmt, calls);
}

bool ElseDefersMlsPredeclareToNestedPartition(
    const Stmt &else_case,
    const std::unordered_set<const VarNode *> &thread_vars,
    const std::unordered_set<const VarNode *> &thread_alias_vars,
    const std::unordered_set<const VarNode *> &seq_thread_aliases) {
  // Bare nested partition if: `else: if tx < N: ...`
  if (const auto *if_node = else_case.as<IfThenElseNode>()) {
    return IsThreadWavePartitionIf(if_node, thread_vars, thread_alias_vars,
                                   seq_thread_aliases);
  }
  // SeqStmt prefix before the next ladder step, e.g.
  //   else: { sched_barrier; if tx < N: producer else: consumer }
  // Do not Predeclare on the whole else (would place resource_init before the
  // inner partition). SeqStmt visitor stops hoist prefix at the partition if;
  // the nested If visitor Predeclares inside that branch.
  if (const auto *seq = else_case.as<SeqStmtNode>()) {
    for (const Stmt &stmt : seq->seq) {
      if (const auto *if_node = stmt.as<IfThenElseNode>()) {
        if (IsThreadWavePartitionIf(if_node, thread_vars, thread_alias_vars,
                                    seq_thread_aliases)) {
          return true;
        }
      }
    }
  }
  return false;
}

Stmt PrependStmts(const std::vector<Stmt> &prelude, const Stmt &body) {
  if (prelude.empty()) {
    return body;
  }
  Array<Stmt> seq;
  seq.reserve(prelude.size() + 1);
  for (const Stmt &stmt : prelude) {
    seq.push_back(stmt);
  }
  seq.push_back(body);
  return seq.size() == 1 ? seq[0] : SeqStmt(seq);
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

bool UsesThreadIdxPartition(
    const PrimExpr &expr,
    const std::unordered_set<const VarNode *> &thread_vars,
    const std::unordered_set<const VarNode *> &thread_alias_vars,
    const std::unordered_set<const VarNode *> &seq_thread_aliases) {
  // Only threadIdx / aliases matter for hoist barriers. Warp/wave/lane
  // predicates do not shrink the lowered tx domain in TileLang today.
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
    }
  });
  return found;
}

bool ExprUsesWarpRolePred(const PrimExpr &expr) {
  bool found = false;
  tirx::PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (found) {
      return;
    }
    const auto *call = node.as<CallNode>();
    if (call == nullptr) {
      return;
    }
    if (call->op.same_as(get_warp_idx()) ||
        call->op.same_as(get_warp_idx_sync()) ||
        call->op.same_as(get_wave_id()) ||
        call->op.same_as(get_warp_group_idx()) ||
        call->op.same_as(get_lane_idx())) {
      found = true;
    }
  });
  return found;
}

bool IsThreadWavePartitionIf(
    const IfThenElseNode *op,
    const std::unordered_set<const VarNode *> &thread_vars,
    const std::unordered_set<const VarNode *> &thread_alias_vars,
    const std::unordered_set<const VarNode *> &seq_thread_aliases) {
  return UsesThreadIdxPartition(op->condition, thread_vars, thread_alias_vars,
                                seq_thread_aliases);
}

bool ThenIsLoopBreak(const Stmt &stmt) {
  if (const auto *ev = stmt.as<EvaluateNode>()) {
    if (const auto *call = ev->value.as<CallNode>()) {
      return call->op.same_as(loop_break());
    }
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &s : seq->seq) {
      if (ThenIsLoopBreak(s)) {
        return true;
      }
    }
  }
  return false;
}

bool IsMlsWindowBarrierIf(
    const IfThenElseNode *op,
    const std::unordered_set<const VarNode *> &thread_vars,
    const std::unordered_set<const VarNode *> &thread_alias_vars,
    const std::unordered_set<const VarNode *> &seq_thread_aliases) {
  if (IsThreadWavePartitionIf(op, thread_vars, thread_alias_vars,
                              seq_thread_aliases)) {
    return false;
  }
  if (ThenIsLoopBreak(op->then_case)) {
    return true;
  }
  std::vector<const CallNode *> loads;
  CollectMlsLoadTileCalls(op->then_case, &loads);
  std::unordered_set<const VarNode *> mn_vars;
  for (const CallNode *call : loads) {
    if (call->args.size() < 6) {
      continue;
    }
    tirx::PostOrderVisit(call->args[5], [&](const ObjectRef &node) {
      if (const auto *var = node.as<VarNode>()) {
        mn_vars.insert(var);
      }
    });
  }
  return ExprUsesAnyVar(op->condition, mn_vars);
}

bool StmtContainsMlsWindowBarrier(
    const Stmt &stmt, const std::unordered_set<const VarNode *> &thread_vars,
    const std::unordered_set<const VarNode *> &thread_alias_vars,
    const std::unordered_set<const VarNode *> &seq_thread_aliases) {
  bool found = false;
  tirx::PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (found) {
      return;
    }
    if (const auto *if_node = node.as<IfThenElseNode>()) {
      if (IsMlsWindowBarrierIf(if_node, thread_vars, thread_alias_vars,
                               seq_thread_aliases)) {
        found = true;
      }
    }
  });
  return found;
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
    // Do not pull mls_load out from under thread_extent / tx partitions.
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == tirx::attr::thread_extent) {
        break;
      }
    }
    if (const auto *if_node = stmt.as<IfThenElseNode>()) {
      if (IsThreadWavePartitionIf(if_node, thread_vars, thread_alias_vars,
                                  seq_thread_aliases) ||
          IsMlsWindowBarrierIf(if_node, thread_vars, thread_alias_vars,
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
  struct ForwardWindowReset {
    std::string key;
    PrimExpr mn_base;
    PrimExpr first_k_base;
  };

  struct ForwardLoopMove {
    std::string key;
    PrimExpr delta;
    PrimExpr next_coord;
    std::string filter_args;
  };

public:
  explicit HoistMlsResourceMutator(const Stmt &body) {
    MlsCallLoopStackCollector collector(&call_loop_stacks_,
                                        &call_program_order_);
    collector(body);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      const std::string &tag = iv->thread_tag;
      const bool is_thread_idx = tag.rfind("threadIdx", 0) == 0;
      if (is_thread_idx)
        thread_vars_.insert(iv->var.get());
      active_loop_ranges_.push_back(
          {iv->var, Range::FromMinExtent(0, op->value)});
      Stmt body = StmtMutator::VisitStmt_(op);
      active_loop_ranges_.pop_back();
      if (is_thread_idx)
        thread_vars_.erase(iv->var.get());
      return body;
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BindNode *op) final {
    // tirx::Bind is flat (no body); aliases apply to subsequent SeqStmt peers.
    if (const auto *v = op->value.as<VarNode>()) {
      if (thread_vars_.count(v)) {
        thread_alias_vars_.insert(op->var.get());
      }
    }
    return GetRef<Stmt>(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (IsMlsWindowBarrierIf(op, thread_vars_, thread_alias_vars_,
                             seq_thread_aliases_) &&
        !ThenIsLoopBreak(op->then_case)) {
      std::vector<Stmt> then_prelude;
      EmitDeferredWindows(op->then_case, &then_prelude);
      Stmt then_case = VisitStmt(op->then_case);
      then_case = PrependStmts(then_prelude, then_case);
      Optional<Stmt> else_case = op->else_case;
      if (else_case.defined()) {
        std::vector<Stmt> else_prelude;
        EmitDeferredWindows(else_case.value(), &else_prelude);
        else_case = VisitStmt(else_case.value());
        else_case = PrependStmts(else_prelude, else_case.value());
      }
      return IfThenElse(op->condition, then_case, else_case);
    }
    if (ExprUsesWarpRolePred(op->condition) && op->else_case.defined() &&
        !IsThreadWavePartitionIf(op, thread_vars_, thread_alias_vars_,
                                 seq_thread_aliases_)) {
      // Same warp-role split as main: each arm executes set_window at its
      // first load.  Do not keep initialized_scopes_ across the else.
      auto saved_init = initialized_scopes_;
      Stmt then_case = VisitStmt(op->then_case);
      auto init_after_then = initialized_scopes_;
      initialized_scopes_ = saved_init;
      Stmt else_case = VisitStmt(op->else_case.value());
      for (const auto &kv : init_after_then) {
        initialized_scopes_[kv.first] = kv.second;
      }
      return IfThenElse(op->condition, then_case, else_case);
    }
    if (!IsThreadWavePartitionIf(op, thread_vars_, thread_alias_vars_,
                                 seq_thread_aliases_)) {
      return StmtMutator::VisitStmt_(op);
    }
    ++scope_depth_;
    std::vector<Stmt> then_prelude;
    PredeclareMlsResources(op->then_case, nullptr, &then_prelude);
    Stmt then_case = VisitStmt(op->then_case);
    then_case = PrependStmts(then_prelude, then_case);
    ExpireScopes();
    Optional<Stmt> else_case = op->else_case;
    if (else_case.defined() && ElseDefersMlsPredeclareToNestedPartition(
                                   else_case.value(), thread_vars_,
                                   thread_alias_vars_, seq_thread_aliases_)) {
      // Nested WDRA/thread ladder (optionally under SeqStmt prefix): recurse
      // only; let the nested partition If own resource_init.
      else_case = VisitStmt(else_case.value());
      ExpireScopes();
    } else if (else_case.defined()) {
      std::vector<Stmt> else_prelude;
      PredeclareMlsResources(else_case.value(), nullptr, &else_prelude);
      else_case = VisitStmt(else_case.value());
      else_case = PrependStmts(else_prelude, else_case.value());
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
      CollectForLoopVarsSkipThreadPartition(
          hoist_region, thread_vars_, thread_alias_vars_, seq_thread_aliases_,
          &nested_loop_vars);
    }

    ++scope_depth_;
    // Use partition-aware collection so Alloc/Decl wrappers above an
    // `if tx` cannot smuggle mls_load into the SeqStmt prologue.
    if (hoist_region.defined()) {
      std::vector<const CallNode *> hoist_calls;
      CollectMlsLoadTileCallsSkipThreadPartition(
          hoist_region, thread_vars_, thread_alias_vars_, seq_thread_aliases_,
          &hoist_calls);
      PredeclareMlsResourcesFromCalls(hoist_calls, /*loop_var=*/nullptr,
                                      &prelude, &nested_loop_vars);
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
    ICHECK(!seq.empty());
    return seq.size() == 1 ? seq[0] : Stmt(SeqStmt(seq));
  }

  Stmt VisitStmt_(const ForNode *op) final {
    const bool track = IsMlsDataLoop(op);
    if (track) {
      data_loop_stack_.push_back(op);
    }
    std::vector<Stmt> prelude;
    std::vector<const CallNode *> hoist_calls;
    CollectMlsLoadTileCallsSkipThreadPartition(
        op->body, thread_vars_, thread_alias_vars_, seq_thread_aliases_,
        &hoist_calls);
    std::unordered_set<const VarNode *> nested_loop_vars;
    CollectForLoopVarsSkipThreadPartition(
        op->body, thread_vars_, thread_alias_vars_, seq_thread_aliases_,
        &nested_loop_vars);
    nested_loop_vars.insert(op->loop_var.get());
    PredeclareMlsResourcesFromCalls(hoist_calls, op->loop_var.get(), &prelude);
    const bool defer_windows = StmtContainsMlsWindowBarrier(
        op->body, thread_vars_, thread_alias_vars_, seq_thread_aliases_);
    auto before_it = windows_before_loop_.find(op);
    if (!defer_windows && before_it != windows_before_loop_.end()) {
      for (const ForwardWindowReset &reset : before_it->second) {
        auto name_it = resource_names_.find(reset.key);
        if (name_it == resource_names_.end()) {
          continue;
        }
        prelude.push_back(MakeExternStmt(
            "tl::mls::set_window_origin",
            {StringImm(name_it->second), reset.mn_base, reset.first_k_base}));
        initialized_scopes_[reset.key] = scope_depth_;
      }
    }
    if (!defer_windows) {
      PreinitializeMlsWindowsFromCalls(hoist_calls, op->loop_var.get(),
                                       &prelude, nested_loop_vars);
    }

    ++scope_depth_;
    active_loop_ranges_.push_back(
        {op->loop_var, Range::FromMinExtent(op->min, op->extent)});
    std::vector<Stmt> window_prelude;
    auto window_it = forward_window_resets_.find(op);
    if (!defer_windows && window_it != forward_window_resets_.end()) {
      for (const ForwardWindowReset &reset : window_it->second) {
        auto name_it = resource_names_.find(reset.key);
        if (name_it == resource_names_.end()) {
          continue;
        }
        const std::string set_window_sym = "tl::mls::set_window_origin";
        window_prelude.push_back(MakeExternStmt(
            set_window_sym,
            {StringImm(name_it->second), reset.mn_base, reset.first_k_base}));
        initialized_scopes_[reset.key] = scope_depth_;
      }
    }
    Stmt new_body = VisitStmt(op->body);
    new_body = PrependStmts(window_prelude, new_body);
    auto move_it = forward_moves_after_loop_.find(op);
    if (move_it != forward_moves_after_loop_.end()) {
      std::vector<Stmt> after;
      for (const ForwardLoopMove &move : move_it->second) {
        auto name_it = resource_names_.find(move.key);
        if (name_it == resource_names_.end()) {
          continue;
        }
        const bool fast_is_mn = fast_is_mn_.count(move.key);
        const std::string move_sym =
            (fast_is_mn ? std::string("tl::mls::move_mn_base<")
                        : std::string("tl::mls::move_k_base<")) +
            move.filter_args + ">";
        after.push_back(
            MakeExternStmt(move_sym, {StringImm(name_it->second), move.delta,
                                      move.next_coord}));
      }
      if (!after.empty()) {
        after.insert(after.begin(), new_body);
        new_body = SeqStmt(after);
      }
    }
    active_loop_ranges_.pop_back();

    Stmt loop =
        For(op->loop_var, op->min, op->extent, op->kind, std::move(new_body),
            op->thread_binding, op->annotations, op->step, op->span);

    ExpireScopes();
    --scope_depth_;

    if (track) {
      data_loop_stack_.pop_back();
    }
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
      const std::string specialized = EncodeEstablishedMlsLoadTile(
          sym, SpecializeAxisFilter(call, sym, /*is_mn=*/false),
          SpecializeAxisFilter(call, sym, /*is_mn=*/true));
      if (specialized == sym) {
        return StmtMutator::VisitStmt_(op);
      }
      Array<PrimExpr> args(call->args.begin(), call->args.end());
      args.Set(0, StringImm(specialized));
      return Evaluate(Call(op->value.dtype(), call->op, args));
    }

    const std::string &obj_name = it->second;
    const MlsKAddressMode mode = resource_modes_.at(key);
    std::vector<Stmt> seq;

    if (mode == MlsKAddressMode::kForwardDelta) {
      const bool fast_is_mn = fast_is_mn_.count(key);
      if (!initialized_scopes_.count(key)) {
        PrimExpr mn = fast_is_mn ? forward_first_mn_.at(key) : call->args[5];
        PrimExpr k = fast_is_mn ? call->args[6] : forward_first_k_.at(key);
        seq.push_back(MakeExternStmt("tl::mls::set_window_origin",
                                     {StringImm(obj_name), mn, k}));
        initialized_scopes_[key] = scope_depth_;
      }
      AppendAsyncLoad(call, sym, obj_name, &seq);
      auto delta_it = forward_delta_after_.find(call);
      if (delta_it != forward_delta_after_.end()) {
        std::string filter_args = forward_move_filter_args_.at(call);
        auto next_call_it = forward_move_next_call_.find(call);
        if (next_call_it != forward_move_next_call_.end()) {
          const auto *next_sym =
              next_call_it->second->args[0].as<StringImmNode>();
          filter_args = SpecializeAxisFilter(next_call_it->second,
                                             next_sym->value, fast_is_mn);
        }
        const std::string move_sym =
            (fast_is_mn ? std::string("tl::mls::move_mn_base<")
                        : std::string("tl::mls::move_k_base<")) +
            filter_args + ">";
        seq.push_back(
            MakeExternStmt(move_sym, {StringImm(obj_name), delta_it->second,
                                      forward_next_k_after_.at(call)}));
      }
      return seq.size() == 1 ? seq[0] : Stmt(SeqStmt(seq));
    }

    if (full_window_each_load_.count(key)) {
      seq.push_back(
          MakeExternStmt("tl::mls::set_window_origin",
                         {StringImm(obj_name), call->args[5], call->args[6]}));
      AppendAsyncLoad(call, sym, obj_name, &seq);
      return seq.size() == 1 ? seq[0] : Stmt(SeqStmt(seq));
    }

    // absolute_rebase: set_window_origin lives on the outer window loop
    // (MN for K-inner, K for MN-inner).  Inner loads only update the moving
    // axis.
    if (!initialized_scopes_.count(key)) {
      seq.push_back(
          MakeExternStmt("tl::mls::set_window_origin",
                         {StringImm(obj_name), call->args[5], call->args[6]}));
      initialized_scopes_[key] = scope_depth_;
    }

    AppendAbsoluteAxisUpdates(call, sym, obj_name, key, &seq);
    AppendAsyncLoad(call, sym, obj_name, &seq);
    ICHECK(!seq.empty());
    return seq.size() == 1 ? seq[0] : Stmt(SeqStmt(seq));
  }

private:
  void AppendAbsoluteAxisUpdates(const CallNode *call, const std::string &sym,
                                 const std::string &obj_name,
                                 const std::string &key,
                                 std::vector<Stmt> *seq) {
    if (full_window_each_load_.count(key)) {
      return;
    }
    if (fast_is_mn_.count(key)) {
      auto mn_filt = SpecializeAxisFilter(call, sym, /*is_mn=*/true);
      const std::string update_mn_sym =
          std::string("tl::mls::update_mn_base<") + mn_filt + ">";
      seq->push_back(
          MakeExternStmt(update_mn_sym, {StringImm(obj_name), call->args[5]}));
    } else {
      auto k_filt = SpecializeAxisFilter(call, sym, /*is_mn=*/false);
      const std::string update_k_sym =
          std::string("tl::mls::update_k_base<") + k_filt + ">";
      seq->push_back(
          MakeExternStmt(update_k_sym, {StringImm(obj_name), call->args[6]}));
    }
  }

  void RecordAbsoluteWindow(const std::string &key, const CallNode *rep) {
    if (resource_modes_.count(key) &&
        resource_modes_.at(key) == MlsKAddressMode::kForwardDelta) {
      return;
    }
    if (!call_loop_stacks_.count(rep) || call_loop_stacks_.at(rep).empty()) {
      full_window_each_load_.insert(key);
      return;
    }

    int glob_mn = -1;
    int glob_k = -1;
    for (const CallNode *c : call_program_order_) {
      const auto *csym = c->args[0].as<StringImmNode>();
      if (csym == nullptr ||
          ResourceKey(MlsBaseTemplateFromLoadTile(csym->value), c) != key) {
        continue;
      }
      auto it = call_loop_stacks_.find(c);
      if (it == call_loop_stacks_.end()) {
        continue;
      }
      for (size_t i = 0; i < it->second.size(); ++i) {
        const VarNode *var = it->second[i]->loop_var.get();
        if (ExprUsesVar(c->args[5], var)) {
          glob_mn = std::max(glob_mn, static_cast<int>(i));
        }
        if (ExprUsesVar(c->args[6], var)) {
          glob_k = std::max(glob_k, static_cast<int>(i));
        }
      }
    }
    const bool glob_k_inner = glob_k >= 0 && (glob_mn < 0 || glob_mn < glob_k);
    const bool glob_mn_inner = glob_mn >= 0 && (glob_k < 0 || glob_mn > glob_k);
    if (glob_mn_inner) {
      fast_is_mn_.insert(key);
    }

    const CallNode *earliest = nullptr;
    for (const CallNode *c : call_program_order_) {
      const auto *csym = c->args[0].as<StringImmNode>();
      if (csym != nullptr &&
          ResourceKey(MlsBaseTemplateFromLoadTile(csym->value), c) == key) {
        earliest = c;
        break;
      }
    }
    if (earliest != nullptr && call_loop_stacks_.count(earliest)) {
      const auto &estack = call_loop_stacks_.at(earliest);
      bool earliest_in_fast = false;
      for (const ForNode *loop : estack) {
        if (glob_k_inner &&
            ExprUsesVar(earliest->args[6], loop->loop_var.get())) {
          earliest_in_fast = true;
        }
        if (glob_mn_inner &&
            ExprUsesVar(earliest->args[5], loop->loop_var.get())) {
          earliest_in_fast = true;
        }
      }
      // Prologue load already sits in the slow loop and not in the fast loop:
      // keep set_window at that load (main-style). Do not pin to For w.
      if (!earliest_in_fast) {
        return;
      }
    }

    const auto &stack = call_loop_stacks_.at(rep);
    int mn_depth = -1;
    int k_depth = -1;
    for (size_t i = 0; i < stack.size(); ++i) {
      if (ExprUsesVar(rep->args[5], stack[i]->loop_var.get())) {
        mn_depth = static_cast<int>(i);
      }
      if (ExprUsesVar(rep->args[6], stack[i]->loop_var.get())) {
        k_depth = static_cast<int>(i);
      }
    }
    const bool k_inner = k_depth >= 0 && (mn_depth < 0 || mn_depth < k_depth);
    const bool mn_inner = mn_depth >= 0 && (k_depth < 0 || mn_depth > k_depth);
    if (!k_inner && !mn_inner) {
      full_window_each_load_.insert(key);
      return;
    }
    const bool fast_is_mn = mn_inner;
    if (fast_is_mn) {
      fast_is_mn_.insert(key);
    }
    const int inner_depth = fast_is_mn ? mn_depth : k_depth;
    const PrimExpr &fast_coord = fast_is_mn ? rep->args[5] : rep->args[6];
    const PrimExpr &slow_coord = fast_is_mn ? rep->args[6] : rep->args[5];
    // Same dummy-transparent walk as forward_delta window placement.
    const int win_depth =
        CollapseFastWindowDepth(stack, inner_depth, fast_coord, slow_coord);
    PrimExpr mn =
        fast_is_mn ? PrimExpr(IntImm(rep->args[5].dtype(), 0)) : rep->args[5];
    PrimExpr k =
        fast_is_mn ? rep->args[6] : PrimExpr(IntImm(rep->args[6].dtype(), 0));
    if (win_depth >= 0) {
      forward_window_resets_[stack[win_depth]].push_back({key, mn, k});
      forward_window_for_key_.insert(key);
      return;
    }
    // Lifted through dummy parents (window_depth == -1).  Pin before the
    // outermost data loop on this stack so we do not cross a threadIdx
    // partition If that contains that loop.
    windows_before_loop_[stack.front()].push_back({key, mn, k});
    forward_window_for_key_.insert(key);
  }

  void AppendAsyncLoad(const CallNode *call, const std::string &sym,
                       const std::string &obj_name, std::vector<Stmt> *seq) {
    const std::string data_type = MlsDataTypeFromLoadTile(sym);
    const std::string refresh_k =
        SpecializeAxisFilter(call, sym, /*is_mn=*/false);
    const std::string refresh_mn =
        SpecializeAxisFilter(call, sym, /*is_mn=*/true);
    std::ostringstream async_sym;
    async_sym << "tl::mls::async_load<" << data_type << ", " << refresh_k
              << ", " << refresh_mn << ">";
    seq->push_back(
        MakeExternStmt(async_sym.str(), {StringImm(obj_name), call->args[7],
                                         call->args[6], call->args[5]}));
  }

  std::string SpecializeAxisFilter(const CallNode *call, const std::string &sym,
                                   bool is_mn) const {
    const auto modes = MlsBoundaryFromLoadTile(sym);
    const MlsBoundaryMode mode = is_mn ? modes.mn : modes.k;
    // User boundary is a contract and wins over proof.
    // -1 analyze: prove in-range (false) else refresh (true).
    //  0 skip / 1 refresh: settled.
    if (mode != MlsBoundaryMode::kAnalyze) {
      return mode == MlsBoundaryMode::kRefresh ? "true" : "false";
    }
    auto block_sizes = MlsBlockSizesFromLoadTile(sym);
    if (!block_sizes) {
      return "true";
    }
    const PrimExpr &base = is_mn ? call->args[5] : call->args[6];
    const PrimExpr &length = is_mn ? call->args[3] : call->args[4];
    const int64_t block = is_mn ? block_sizes->first : block_sizes->second;
    if (CanProveNoBoundary(base, length, block)) {
      return "false";
    }
    return "true";
  }

  bool SameOuterNest(const std::vector<const ForNode *> &call_stack,
                     const std::vector<const ForNode *> &stack, int inner_depth,
                     const CallNode *call, int coord_arg) const {
    if (call_stack.size() != static_cast<size_t>(inner_depth)) {
      return false;
    }
    if (inner_depth > 0 && ExprUsesVar(call->args[coord_arg],
                                       stack[inner_depth]->loop_var.get())) {
      return false;
    }
    for (int i = 0; i < inner_depth; ++i) {
      if (call_stack[i] != stack[i]) {
        return false;
      }
    }
    return true;
  }

  bool TryAnalyzeForwardOnInnerAxis(const std::string &key,
                                    const CallNode *representative,
                                    const std::vector<const CallNode *> &calls,
                                    const std::vector<const ForNode *> &stack,
                                    int inner_depth, int coord_arg,
                                    bool inner_is_mn) {
    struct AxisGroup {
      const ForNode *inner{nullptr};
      std::vector<const CallNode *> head;
      std::vector<const CallNode *> loop;
      std::vector<const CallNode *> tail;
    };
    std::vector<AxisGroup> groups;
    std::vector<const CallNode *> pending_outer;
    AxisGroup cur;
    bool have_cur = false;
    auto flush_group = [&]() {
      if (!have_cur) {
        return;
      }
      groups.push_back(std::move(cur));
      cur = AxisGroup();
      have_cur = false;
    };
    for (const CallNode *call : calls) {
      auto stack_it = call_loop_stacks_.find(call);
      if (stack_it == call_loop_stacks_.end()) {
        return false;
      }
      const auto &call_stack = stack_it->second;
      const bool in_some_inner =
          call_stack.size() > static_cast<size_t>(inner_depth) &&
          ExprUsesVar(call->args[coord_arg],
                      call_stack[inner_depth]->loop_var.get());
      if (in_some_inner) {
        const ForNode *I = call_stack[inner_depth];
        for (int i = 0; i < inner_depth; ++i) {
          if (call_stack[i] != stack[i]) {
            return false;
          }
        }
        if (have_cur && cur.inner == I) {
          if (!pending_outer.empty()) {
            return false;
          }
          cur.loop.push_back(call);
        } else {
          flush_group();
          cur.inner = I;
          cur.head = std::move(pending_outer);
          pending_outer.clear();
          cur.loop.push_back(call);
          have_cur = true;
        }
        continue;
      }
      if (!SameOuterNest(call_stack, stack, inner_depth, call, coord_arg)) {
        return false;
      }
      pending_outer.push_back(call);
    }
    if (have_cur) {
      cur.tail = std::move(pending_outer);
      flush_group();
    } else if (!pending_outer.empty()) {
      return false;
    }
    if (groups.empty()) {
      return false;
    }

    auto refresh_for = [&](const CallNode *c) {
      return SpecializeAxisFilter(c, c->args[0].as<StringImmNode>()->value,
                                  inner_is_mn);
    };

    arith::Analyzer analyzer;
    auto coord_at_inner_min = [&](const CallNode *call, const ForNode *inner) {
      return analyzer.Simplify(
          Substitute(call->args[coord_arg], {{inner->loop_var, inner->min}}));
    };
    auto emit_consecutive = [&](const std::vector<const CallNode *> &seq,
                                const ForNode *inner, size_t begin,
                                size_t end_exclusive) -> bool {
      for (size_t i = begin; i + 1 < end_exclusive; ++i) {
        const bool src_in =
            ExprUsesVar(seq[i]->args[coord_arg], inner->loop_var.get());
        const bool dst_in =
            ExprUsesVar(seq[i + 1]->args[coord_arg], inner->loop_var.get());
        PrimExpr src = seq[i]->args[coord_arg];
        PrimExpr dst = seq[i + 1]->args[coord_arg];
        // Head (no inner var) to first loop load: evaluate the loop coord at
        // min.
        if (!src_in && dst_in) {
          dst = coord_at_inner_min(seq[i + 1], inner);
        }
        PrimExpr delta = analyzer.Simplify(dst - src);
        if (!analyzer.CanProve(delta >= tirx::make_const(delta.dtype(), 0))) {
          return false;
        }
        forward_delta_after_[seq[i]] = delta;
        forward_next_k_after_[seq[i]] = dst;
        forward_move_next_call_[seq[i]] = seq[i + 1];
        forward_move_filter_args_[seq[i]] = refresh_for(seq[i + 1]);
      }
      return true;
    };

    for (AxisGroup &g : groups) {
      if (g.loop.empty() || g.inner == nullptr) {
        return false;
      }
      std::vector<const CallNode *> stream;
      stream.insert(stream.end(), g.head.begin(), g.head.end());
      stream.insert(stream.end(), g.loop.begin(), g.loop.end());
      if (!emit_consecutive(stream, g.inner, 0, stream.size())) {
        return false;
      }
      PrimExpr loop_step =
          g.inner->step.defined()
              ? g.inner->step.value()
              : PrimExpr(tirx::make_const(g.inner->loop_var.dtype(), 1));
      PrimExpr next_iteration_first =
          Substitute(g.loop.front()->args[coord_arg],
                     {{g.inner->loop_var, g.inner->loop_var + loop_step}});
      PrimExpr wrap_delta = analyzer.Simplify(next_iteration_first -
                                              g.loop.back()->args[coord_arg]);
      if (!analyzer.CanProve(wrap_delta >=
                             tirx::make_const(wrap_delta.dtype(), 0))) {
        return false;
      }
      const CallNode *loop_last = g.loop.back();
      forward_delta_after_[loop_last] = wrap_delta;
      forward_next_k_after_[loop_last] = next_iteration_first;
      if (g.tail.empty()) {
        forward_move_filter_args_[loop_last] = refresh_for(g.loop.front());
      } else {
        forward_move_filter_args_[loop_last] = std::string("true");
      }
      if (!g.tail.empty()) {
        PrimExpr after_loop_var =
            analyzer.Simplify(g.inner->min + g.inner->extent * loop_step);
        PrimExpr after_loop_first = analyzer.Simplify(
            Substitute(g.loop.front()->args[coord_arg],
                       {{g.inner->loop_var, after_loop_var}}));
        if (!analyzer.CanProve(g.tail.front()->args[coord_arg] ==
                               after_loop_first)) {
          return false;
        }
        if (!emit_consecutive(g.tail, g.inner, 0, g.tail.size())) {
          return false;
        }
      }
    }

    const PrimExpr &fast_coord =
        inner_is_mn ? representative->args[5] : representative->args[6];
    const PrimExpr &slow_coord =
        inner_is_mn ? representative->args[6] : representative->args[5];
    bool transparent_inside_fast = false;
    size_t loop_call_count = 0;
    for (const AxisGroup &g : groups) {
      loop_call_count += g.loop.size();
      for (const CallNode *call : g.loop) {
        const auto &cs = call_loop_stacks_.at(call);
        for (size_t i = static_cast<size_t>(inner_depth) + 1; i < cs.size();
             ++i) {
          if (!LoopUsesCoord(cs[i], fast_coord) &&
              !LoopUsesCoord(cs[i], slow_coord)) {
            transparent_inside_fast = true;
          }
        }
      }
    }
    if (transparent_inside_fast && loop_call_count != 1U) {
      return false;
    }

    const int window_depth =
        CollapseFastWindowDepth(stack, inner_depth, fast_coord, slow_coord);
    if (!ForwardDeltaNestIsSafe(stack, inner_depth, window_depth, fast_coord,
                                slow_coord)) {
      return false;
    }
    const AxisGroup &g0 = groups.front();
    Map<Var, PrimExpr> first_subst;
    for (int i = window_depth + 1; i <= inner_depth; ++i) {
      first_subst.Set(stack[i]->loop_var, stack[i]->min);
    }
    PrimExpr first_coord;
    if (!g0.head.empty()) {
      first_coord = analyzer.Simplify(g0.head.front()->args[coord_arg]);
    } else {
      first_coord = analyzer.Simplify(
          Substitute(g0.loop.front()->args[coord_arg], first_subst));
    }
    resource_modes_[key] = MlsKAddressMode::kForwardDelta;
    if (inner_is_mn) {
      fast_is_mn_.insert(key);
      forward_first_mn_[key] = first_coord;
    } else {
      forward_first_k_[key] = first_coord;
    }
    if (transparent_inside_fast) {
      PrimExpr loop_step =
          g0.inner->step.defined()
              ? g0.inner->step.value()
              : PrimExpr(tirx::make_const(g0.inner->loop_var.dtype(), 1));
      PrimExpr next_iteration_first =
          Substitute(g0.loop.front()->args[coord_arg],
                     {{g0.inner->loop_var, g0.inner->loop_var + loop_step}});
      PrimExpr wrap_delta = analyzer.Simplify(next_iteration_first -
                                              g0.loop.back()->args[coord_arg]);
      forward_moves_after_loop_[g0.inner].push_back(
          {key, wrap_delta, next_iteration_first,
           g0.tail.empty() ? refresh_for(g0.loop.front())
                           : std::string("true")});
    }
    if (window_depth >= 0) {
      PrimExpr mn = inner_is_mn ? first_coord : representative->args[5];
      PrimExpr k = inner_is_mn ? representative->args[6] : first_coord;
      if (!g0.head.empty()) {
        if (inner_is_mn) {
          mn = first_coord;
        } else {
          k = first_coord;
          mn = g0.head.front()->args[5];
        }
      }
      forward_window_resets_[stack[window_depth]].push_back({key, mn, k});
      forward_window_for_key_.insert(key);
    }
    return true;
  }

  MlsKAddressMode
  AnalyzeAddressMode(const std::string &key,
                     const CallNode * /*representative*/,
                     const std::vector<const CallNode *> &all_calls) {
    if (resource_modes_.count(key)) {
      return resource_modes_.at(key);
    }

    // Deepest MN/K data-loop across every load of this resource.  A K=0
    // prefetch before Serial(k) must not define the inner axis as MN.
    std::vector<const CallNode *> calls;
    for (const CallNode *candidate : all_calls) {
      const auto *sym = candidate->args[0].as<StringImmNode>();
      const std::string candidate_template =
          MlsBaseTemplateFromLoadTile(sym->value);
      if (ResourceKey(candidate_template, candidate) == key) {
        calls.push_back(candidate);
      }
    }
    int mn_depth = -1;
    int k_depth = -1;
    for (const CallNode *call : calls) {
      auto it = call_loop_stacks_.find(call);
      if (it == call_loop_stacks_.end()) {
        continue;
      }
      for (size_t i = 0; i < it->second.size(); ++i) {
        const VarNode *var = it->second[i]->loop_var.get();
        if (ExprUsesVar(call->args[5], var)) {
          mn_depth = std::max(mn_depth, static_cast<int>(i));
        }
        if (ExprUsesVar(call->args[6], var)) {
          k_depth = std::max(k_depth, static_cast<int>(i));
        }
      }
    }
    const bool k_inner = k_depth >= 0 && (mn_depth < 0 || mn_depth < k_depth);
    const bool mn_inner = mn_depth >= 0 && (k_depth < 0 || mn_depth > k_depth);
    const CallNode *rep = nullptr;
    for (const CallNode *call : calls) {
      auto it = call_loop_stacks_.find(call);
      if (it == call_loop_stacks_.end()) {
        continue;
      }
      int mn = -1;
      int k = -1;
      for (size_t i = 0; i < it->second.size(); ++i) {
        const VarNode *var = it->second[i]->loop_var.get();
        if (ExprUsesVar(call->args[5], var)) {
          mn = static_cast<int>(i);
        }
        if (ExprUsesVar(call->args[6], var)) {
          k = static_cast<int>(i);
        }
      }
      const int inner = k_inner ? k : (mn_inner ? mn : -1);
      const int want = k_inner ? k_depth : (mn_inner ? mn_depth : -1);
      if (want >= 0 && inner == want) {
        rep = call;
        break;
      }
      if (rep == nullptr) {
        rep = call;
      }
    }
    if (calls.empty() || rep == nullptr || !call_loop_stacks_.count(rep)) {
      resource_modes_[key] = MlsKAddressMode::kAbsoluteRebase;
      return resource_modes_[key];
    }

    const auto &stack = call_loop_stacks_.at(rep);
    const bool wdra_enabled = [&]() {
      auto cfg = tvm::transform::PassContext::Current()->GetConfig(
          kEnableHcuWdra, Optional<Bool>());
      return cfg.defined() && cfg.value()->value;
    }();
    if (wdra_enabled && k_inner &&
        TryAnalyzeForwardOnInnerAxis(key, rep, calls, stack, k_depth,
                                     /*coord_arg=*/6, /*inner_is_mn=*/false)) {
      return resource_modes_[key];
    }
    if (wdra_enabled && mn_inner &&
        TryAnalyzeForwardOnInnerAxis(key, rep, calls, stack, mn_depth,
                                     /*coord_arg=*/5, /*inner_is_mn=*/true)) {
      return resource_modes_[key];
    }

    resource_modes_[key] = MlsKAddressMode::kAbsoluteRebase;
    return resource_modes_[key];
  }

  void PredeclareMlsResources(
      const Stmt &body, const VarNode *loop_var, std::vector<Stmt> *prelude,
      const std::unordered_set<const VarNode *> *forbidden_vars = nullptr) {
    std::vector<const CallNode *> calls;
    CollectMlsLoadTileCalls(body, &calls);
    PredeclareMlsResourcesFromCalls(calls, loop_var, prelude, forbidden_vars);
  }

  MlsResourceAxis
  AnalyzeResourceAxis(const std::string &sym, const std::string &key,
                      const std::vector<const CallNode *> &all_calls) const {
    auto counts = MlsWarpAccessCountsFromLoadTile(sym);
    if (!counts)
      return MlsResourceAxis::kMN;

    constexpr int64_t kMaxOuterResources = 2;
    int mn_depth = -1;
    int k_depth = -1;
    for (const CallNode *call : all_calls) {
      const auto *csym = call->args[0].as<StringImmNode>();
      if (ResourceKey(MlsBaseTemplateFromLoadTile(csym->value), call) != key) {
        continue;
      }
      auto stack_it = call_loop_stacks_.find(call);
      if (stack_it == call_loop_stacks_.end()) {
        continue;
      }
      const auto &stack = stack_it->second;
      for (size_t depth = 0; depth < stack.size(); ++depth) {
        const VarNode *var = stack[depth]->loop_var.get();
        if (ExprUsesVar(call->args[5], var))
          mn_depth = std::max(mn_depth, static_cast<int>(depth));
        if (ExprUsesVar(call->args[6], var))
          k_depth = std::max(k_depth, static_cast<int>(depth));
      }
    }
    const bool k_inner = k_depth > mn_depth;
    const bool mn_inner = mn_depth > k_depth;
    const int64_t access_mn = counts->first;
    const int64_t access_k = counts->second;

    // Outer axis with at most 2 descriptors: keep resources there and
    // moffset on the inner loop, even if the inner axis is cheaper.
    if (k_inner && access_mn >= 1 && access_mn <= kMaxOuterResources)
      return MlsResourceAxis::kMN;
    if (mn_inner && access_k >= 1 && access_k <= kMaxOuterResources)
      return MlsResourceAxis::kK;

    // Otherwise pick the fewer-descriptor axis.  A tie still prefers the
    // outer loop when nesting is known; no loop info keeps MN.
    if (access_k < access_mn)
      return MlsResourceAxis::kK;
    if (access_mn < access_k)
      return MlsResourceAxis::kMN;
    if (mn_inner)
      return MlsResourceAxis::kK;
    return MlsResourceAxis::kMN;
  }

  void PredeclareMlsResourcesFromCalls(
      const std::vector<const CallNode *> &calls, const VarNode *loop_var,
      std::vector<Stmt> *prelude,
      const std::unordered_set<const VarNode *> *forbidden_vars = nullptr) {
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
      const MlsKAddressMode mode = AnalyzeAddressMode(key, call, calls);
      const MlsResourceAxis resource_axis =
          AnalyzeResourceAxis(sym, key, calls);
      resource_names_[key] = obj_name;
      resource_axes_[key] = resource_axis;
      resource_scopes_[key] = scope_depth_;
      if (mode == MlsKAddressMode::kAbsoluteRebase) {
        const CallNode *abs_rep = call;
        int best_inner = -1;
        for (const CallNode *c : calls) {
          const auto *csym = c->args[0].as<StringImmNode>();
          if (ResourceKey(MlsBaseTemplateFromLoadTile(csym->value), c) != key) {
            continue;
          }
          auto it = call_loop_stacks_.find(c);
          if (it == call_loop_stacks_.end()) {
            continue;
          }
          int mn = -1;
          int k = -1;
          for (size_t i = 0; i < it->second.size(); ++i) {
            const VarNode *var = it->second[i]->loop_var.get();
            if (ExprUsesVar(c->args[5], var)) {
              mn = static_cast<int>(i);
            }
            if (ExprUsesVar(c->args[6], var)) {
              k = static_cast<int>(i);
            }
          }
          const int inner = std::max(mn, k);
          if (inner > best_inner) {
            best_inner = inner;
            abs_rep = c;
          }
        }
        RecordAbsoluteWindow(key, abs_rep);
      }
      prelude->push_back(MakeExternStmt(
          kMlsResourceInitPrefix +
              MlsBaseTemplateFromLoadTile(sym, mode, resource_axis).substr(0) +
              ">",
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
    tirx::PostOrderVisit(body, [&](const ObjectRef &node) {
      if (const auto *for_node = node.as<ForNode>()) {
        moving_loop_vars.insert(for_node->loop_var.get());
      }
    });
    if (loop_var != nullptr) {
      moving_loop_vars.insert(loop_var);
    }
    PreinitializeMlsWindowsFromCalls(calls, loop_var, prelude,
                                     moving_loop_vars);
  }

  void PreinitializeMlsWindowsFromCalls(
      const std::vector<const CallNode *> &calls, const VarNode *loop_var,
      std::vector<Stmt> *prelude,
      const std::unordered_set<const VarNode *> &moving_loop_vars) {
    (void)loop_var;
    for (const CallNode *call : calls) {
      const auto *sym_node = call->args[0].as<StringImmNode>();
      const std::string base_template =
          MlsBaseTemplateFromLoadTile(sym_node->value);
      const std::string key = ResourceKey(base_template, call);
      auto it = resource_names_.find(key);
      if (it == resource_names_.end()) {
        continue;
      }
      if (resource_modes_.at(key) == MlsKAddressMode::kForwardDelta) {
        // A forward resource is initialized at its slow-axis window
        // boundary.  When that axis is invariant, initialize here instead.
        if (!forward_window_for_key_.count(key) &&
            !initialized_scopes_.count(key)) {
          const bool fast_is_mn = fast_is_mn_.count(key);
          PrimExpr mn = fast_is_mn ? forward_first_mn_.at(key) : call->args[5];
          PrimExpr k = fast_is_mn ? call->args[6] : forward_first_k_.at(key);
          prelude->push_back(MakeExternStmt("tl::mls::set_window_origin",
                                            {StringImm(it->second), mn, k}));
          initialized_scopes_[key] = scope_depth_;
        }
        continue;
      }

      if (forward_window_for_key_.count(key) ||
          initialized_scopes_.count(key) || loop_var == nullptr) {
        continue;
      }
      const bool k_on_this_loop = ExprUsesVar(call->args[6], loop_var);
      const bool mn_on_this_loop = ExprUsesVar(call->args[5], loop_var);
      const bool fast_is_mn = fast_is_mn_.count(key);
      const bool fast_on_this_loop =
          fast_is_mn ? mn_on_this_loop : k_on_this_loop;
      const bool slow_on_this_loop =
          fast_is_mn ? k_on_this_loop : mn_on_this_loop;
      // Absolute: pin the slow axis to 0 before the fast loop when the
      // slow axis is invariant on this loop.
      if (fast_on_this_loop && !slow_on_this_loop) {
        PrimExpr mn = fast_is_mn ? PrimExpr(IntImm(call->args[5].dtype(), 0))
                                 : call->args[5];
        PrimExpr k = fast_is_mn ? call->args[6]
                                : PrimExpr(IntImm(call->args[6].dtype(), 0));
        prelude->push_back(MakeExternStmt("tl::mls::set_window_origin",
                                          {StringImm(it->second), mn, k}));
        initialized_scopes_[key] = scope_depth_;
      }
    }
  }

  void AppendWindowResets(const std::vector<ForwardWindowReset> &resets,
                          std::vector<Stmt> *prelude) {
    for (const ForwardWindowReset &reset : resets) {
      if (initialized_scopes_.count(reset.key)) {
        continue;
      }
      auto name_it = resource_names_.find(reset.key);
      if (name_it == resource_names_.end()) {
        continue;
      }
      prelude->push_back(MakeExternStmt(
          "tl::mls::set_window_origin",
          {StringImm(name_it->second), reset.mn_base, reset.first_k_base}));
      initialized_scopes_[reset.key] = scope_depth_;
    }
  }

  void EmitDeferredWindows(const Stmt &region, std::vector<Stmt> *prelude) {
    for (const ForNode *loop : data_loop_stack_) {
      auto before_it = windows_before_loop_.find(loop);
      if (before_it != windows_before_loop_.end()) {
        AppendWindowResets(before_it->second, prelude);
      }
      auto window_it = forward_window_resets_.find(loop);
      if (window_it != forward_window_resets_.end()) {
        AppendWindowResets(window_it->second, prelude);
      }
    }
    std::vector<const CallNode *> calls;
    CollectMlsLoadTileCalls(region, &calls);
    std::unordered_set<const VarNode *> nested_loop_vars;
    CollectForLoopVarsSkipThreadPartition(
        region, thread_vars_, thread_alias_vars_, seq_thread_aliases_,
        &nested_loop_vars);
    const VarNode *slow_var = data_loop_stack_.empty()
                                  ? nullptr
                                  : data_loop_stack_.front()->loop_var.get();
    PreinitializeMlsWindowsFromCalls(calls, slow_var, prelude,
                                     nested_loop_vars);
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
      resource_modes_.erase(key);
      resource_axes_.erase(key);
      forward_first_k_.erase(key);
      forward_first_mn_.erase(key);
      fast_is_mn_.erase(key);
      forward_window_for_key_.erase(key);
      initialized_scopes_.erase(key);
      full_window_each_load_.erase(key);
    }

    expired.clear();
    for (const auto &kv : initialized_scopes_) {
      if (kv.second >= scope_depth_) {
        expired.push_back(kv.first);
      }
    }
    for (const std::string &key : expired) {
      initialized_scopes_.erase(key);
    }
  }

  std::unordered_map<std::string, std::string> resource_names_;
  std::unordered_map<std::string, MlsKAddressMode> resource_modes_;
  std::unordered_map<std::string, MlsResourceAxis> resource_axes_;
  std::unordered_map<std::string, PrimExpr> forward_first_k_;
  std::unordered_map<std::string, PrimExpr> forward_first_mn_;
  std::unordered_set<std::string> fast_is_mn_;
  std::unordered_map<const CallNode *, PrimExpr> forward_delta_after_;
  std::unordered_map<const CallNode *, PrimExpr> forward_next_k_after_;
  std::unordered_map<const CallNode *, const CallNode *>
      forward_move_next_call_;
  std::unordered_map<const CallNode *, std::string> forward_move_filter_args_;
  std::unordered_map<const ForNode *, std::vector<ForwardLoopMove>>
      forward_moves_after_loop_;
  std::unordered_map<const CallNode *, std::vector<const ForNode *>>
      call_loop_stacks_;
  std::vector<const CallNode *> call_program_order_;
  std::unordered_map<const ForNode *, std::vector<ForwardWindowReset>>
      forward_window_resets_;
  // set_window immediately before this For (outside the dummy), not in body.
  std::unordered_map<const ForNode *, std::vector<ForwardWindowReset>>
      windows_before_loop_;
  std::unordered_set<std::string> full_window_each_load_;
  std::unordered_set<std::string> forward_window_for_key_;
  std::unordered_map<std::string, int> resource_scopes_;
  std::unordered_map<std::string, int> initialized_scopes_;
  std::vector<std::pair<tirx::Var, Range>> active_loop_ranges_;
  std::vector<const ForNode *> data_loop_stack_;
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
  HoistMlsResourceMutator mutator(func->body);
  n->body = mutator(std::move(func->body));
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
