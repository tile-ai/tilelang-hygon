/*!
 * \file lower_and_validate_hcu_wdra.cc
 * \brief Lower tx-based warp specialization to HCU WDRA form and validate.
 */
#include "../op/builtin.h"
#include "../support/ffi_aliases.h"
#include "../target/utils.h"
#include "hcu_wdra_op_classify.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

constexpr int kWaveSize = 64;
constexpr int kVgprGranularity = 8;
constexpr int kWavesPerWdraBranch = 4;
constexpr int kMaxWdraBranches = 4;

bool IsValidWavesPerTg(int waves) {
  return waves == 8 || waves == 12 || waves == 16;
}

bool PassConfigEnabled(const tvm::transform::PassContext &ctx,
                       const char *key) {
  if (auto cfg = ctx->GetConfig(key, Optional<Bool>())) {
    return cfg.value()->value;
  }
  return false;
}

bool ExprUsesVar(const PrimExpr &expr, const VarNode *var) {
  struct Visitor : public ExprVisitor {
    const VarNode *target{nullptr};
    bool found{false};

    void VisitExpr_(const VarNode *op) final {
      if (op == target) {
        found = true;
      }
      ExprVisitor::VisitExpr_(op);
    }
  } visitor;
  visitor.target = var;
  visitor(expr);
  return visitor.found;
}

bool IsThreadBindingLet(const LetStmtNode *let, const Var &thread_x) {
  if (let->value.same_as(thread_x)) {
    return true;
  }
  if (const auto *var = let->value.as<VarNode>()) {
    return var == thread_x.get();
  }
  return false;
}

PrimExpr MakeWaveIdExpr() { return Call(DataType::Int(32), get_wave_id(), {}); }

Evaluate MakeSharedSync() {
  return Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                       {StringImm("shared")}));
}

bool IsSharedSync(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_storage_sync()) &&
          !call->args.empty()) {
        if (const auto *scope = call->args[0].as<StringImmNode>()) {
          return scope->value == "shared" || scope->value == "shared.dyn";
        }
      }
    }
  }
  return false;
}

bool ContainsSetMaxNreg(const Stmt &stmt) {
  struct Visitor : public StmtExprVisitor {
    bool found{false};

    void VisitStmt_(const EvaluateNode *op) final {
      if (const auto *call = op->value.as<CallNode>()) {
        if (call->op.same_as(set_max_nreg())) {
          found = true;
          return;
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }
  } visitor;
  visitor(stmt);
  return visitor.found;
}

Optional<int> ExtractFirstSetMaxNreg(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      if (call->op.same_as(set_max_nreg())) {
        return call->args[0].as<IntImmNode>()->value;
      }
    }
    return Optional<int>();
  }

  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    if (!seq->seq.empty()) {
      return ExtractFirstSetMaxNreg(seq->seq[0]);
    }
  }
  return Optional<int>();
}

Optional<int> ParseWaveUpperBound(const PrimExpr &condition) {
  if (const auto *lt = condition.as<LTNode>()) {
    if (const auto *call = lt->a.as<CallNode>()) {
      if (call->op.same_as(get_wave_id())) {
        if (const auto *imm = lt->b.as<IntImmNode>()) {
          return imm->value;
        }
      }
    }
  }
  return Optional<int>();
}

void CollectBranchNregsFromLadder(const IfThenElse &node, int start_wave,
                                  int waves_per_tg,
                                  std::vector<int> *branch_nregs) {
  Optional<int> upper = ParseWaveUpperBound(node->condition);
  ICHECK(upper.has_value())
      << "HCU WDRA fork ladder expects `get_wave_id() < N` conditions";
  ICHECK_GT(upper.value(), start_wave)
      << "HCU WDRA fork ladder has non-increasing wave_id bounds";
  ICHECK_EQ(upper.value() - start_wave, kWavesPerWdraBranch)
      << "HCU WDRA branch must cover exactly " << kWavesPerWdraBranch
      << " waves, got [" << start_wave << ", " << upper.value() << ")";

  Optional<int> then_nreg = ExtractFirstSetMaxNreg(node->then_case);
  ICHECK(then_nreg.has_value())
      << "HCU WDRA branch must begin with tl.set_max_nreg";
  branch_nregs->push_back(then_nreg.value());

  if (node->else_case.defined()) {
    if (const auto *nested_if = node->else_case.value().as<IfThenElseNode>()) {
      CollectBranchNregsFromLadder(ffi::GetRef<IfThenElse>(nested_if),
                                   upper.value(), waves_per_tg, branch_nregs);
    } else {
      ICHECK_EQ(waves_per_tg - upper.value(), kWavesPerWdraBranch)
          << "HCU WDRA final branch must cover exactly " << kWavesPerWdraBranch
          << " waves, got [" << upper.value() << ", " << waves_per_tg << ")";
      Optional<int> else_nreg = ExtractFirstSetMaxNreg(node->else_case.value());
      ICHECK(else_nreg.has_value())
          << "HCU WDRA else branch must begin with tl.set_max_nreg";
      branch_nregs->push_back(else_nreg.value());
    }
  }
}

Evaluate MakeWdraInit(const std::vector<int> &branch_nregs) {
  ICHECK_LE(branch_nregs.size(), kMaxWdraBranches)
      << "HCU WDRA supports at most " << kMaxWdraBranches << " branches";
  Array<PrimExpr> args;
  args.reserve(kMaxWdraBranches);
  for (int i = 0; i < kMaxWdraBranches; ++i) {
    int nreg = i < static_cast<int>(branch_nregs.size()) ? branch_nregs[i] : 0;
    args.push_back(IntImm(DataType::Int(32), nreg));
  }
  return Evaluate(Call(DataType::Handle(), hcu_wdra_init(), args));
}

class TxConditionRewriter : public ExprMutator {
public:
  TxConditionRewriter(Optional<Var> tx_var, int wave_size)
      : tx_var_(std::move(tx_var)), wave_size_(wave_size) {}

  PrimExpr Rewrite(const PrimExpr &expr) { return VisitExpr(expr); }

private:
  Optional<Var> tx_var_;
  int wave_size_;

  PrimExpr ConvertBound(int bound) {
    ICHECK(bound % wave_size_ == 0)
        << "HCU WDRA expects tx thresholds divisible by wave size "
        << wave_size_ << ", got " << bound;
    return IntImm(DataType::Int(32), bound / wave_size_);
  }

  PrimExpr VisitExpr_(const LTNode *op) final {
    if (tx_var_.defined() && op->a.same_as(tx_var_.value())) {
      if (const auto *imm = op->b.as<IntImmNode>()) {
        return MakeWaveIdExpr() < ConvertBound(imm->value);
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const GENode *op) final {
    if (tx_var_.defined() && op->a.same_as(tx_var_.value())) {
      if (const auto *imm = op->b.as<IntImmNode>()) {
        return MakeWaveIdExpr() >= ConvertBound(imm->value);
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const GTNode *op) final {
    if (tx_var_.defined() && op->a.same_as(tx_var_.value())) {
      if (const auto *imm = op->b.as<IntImmNode>()) {
        return MakeWaveIdExpr() > ConvertBound(imm->value);
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const LENode *op) final {
    if (tx_var_.defined() && op->a.same_as(tx_var_.value())) {
      if (const auto *imm = op->b.as<IntImmNode>()) {
        return MakeWaveIdExpr() <= ConvertBound(imm->value);
      }
    }
    return ExprMutator::VisitExpr_(op);
  }
};

IfThenElse RewriteIfLadder(const IfThenElse &node, Optional<Var> tx_var,
                           int wave_size) {
  TxConditionRewriter cond_rewriter(tx_var, wave_size);
  PrimExpr condition = cond_rewriter.Rewrite(node->condition);

  Stmt then_case = node->then_case;
  Optional<Stmt> else_case = node->else_case;
  if (else_case.defined()) {
    if (const auto *nested_if = else_case.value().as<IfThenElseNode>()) {
      else_case = RewriteIfLadder(ffi::GetRef<IfThenElse>(nested_if), tx_var,
                                  wave_size);
    }
  }
  return IfThenElse(condition, then_case, else_case);
}

Stmt PrependBranchSetup(Stmt body, Optional<Var> tx_var, const Var &thread_x) {
  Optional<Evaluate> max_nreg_eval;

  if (ContainsSetMaxNreg(body)) {
    struct Hoister : public StmtExprMutator {
      Optional<Evaluate> first;
      bool changed{false};

      Stmt VisitStmt_(const SeqStmtNode *op) final {
        Array<Stmt> seq;
        for (const Stmt &stmt : op->seq) {
          if (!first.defined()) {
            if (const auto *eval = stmt.as<EvaluateNode>()) {
              if (const auto *call = eval->value.as<CallNode>()) {
                if (call->op.same_as(set_max_nreg())) {
                  int nreg = call->args[0].as<IntImmNode>()->value;
                  int is_inc = call->args[1].as<IntImmNode>()->value;
                  first = Evaluate(Call(DataType::Handle(), set_max_nreg(),
                                        {IntImm(DataType::Int(32), nreg),
                                         IntImm(DataType::Int(32), is_inc)}));
                  changed = true;
                  continue;
                }
              }
            }
          }
          seq.push_back(VisitStmt(stmt));
        }
        if (!changed) {
          return ffi::GetRef<Stmt>(op);
        }
        return seq.size() == 1 ? seq[0] : SeqStmt(seq);
      }

      Stmt VisitStmt_(const EvaluateNode *op) final {
        if (!first.defined()) {
          if (const auto *call = op->value.as<CallNode>()) {
            if (call->op.same_as(set_max_nreg())) {
              int nreg = call->args[0].as<IntImmNode>()->value;
              int is_inc = call->args[1].as<IntImmNode>()->value;
              first = Evaluate(Call(DataType::Handle(), set_max_nreg(),
                                    {IntImm(DataType::Int(32), nreg),
                                     IntImm(DataType::Int(32), is_inc)}));
              return Evaluate(0);
            }
          }
        }
        return StmtExprMutator::VisitStmt_(op);
      }
    } hoister;

    body = hoister(body);
    max_nreg_eval = hoister.first;
  }

  if (max_nreg_eval.defined() && tx_var.defined() &&
      !tx_var.value().same_as(thread_x)) {
    return SeqStmt(
        {max_nreg_eval.value(), LetStmt(tx_var.value(), thread_x, body)});
  }
  if (max_nreg_eval.defined()) {
    return SeqStmt({max_nreg_eval.value(), body});
  }
  if (tx_var.defined() && !tx_var.value().same_as(thread_x)) {
    return LetStmt(tx_var.value(), thread_x, body);
  }
  return body;
}

IfThenElse RewriteBranchBodies(const IfThenElse &node, Optional<Var> tx_var,
                               const Var &thread_x) {
  Stmt then_case = PrependBranchSetup(node->then_case, tx_var, thread_x);
  Optional<Stmt> else_case = node->else_case;
  if (else_case.defined()) {
    if (const auto *nested_if = else_case.value().as<IfThenElseNode>()) {
      else_case = RewriteBranchBodies(ffi::GetRef<IfThenElse>(nested_if),
                                      tx_var, thread_x);
    } else {
      else_case = PrependBranchSetup(else_case.value(), tx_var, thread_x);
    }
  }
  return IfThenElse(node->condition, then_case, else_case);
}

struct ForkSite {
  Array<Stmt> prologue;
  IfThenElse fork;
  Array<Stmt> epilogue;
};

bool IsTxForkCondition(const PrimExpr &condition, const Var &thread_x,
                       Optional<Var> *tx_var) {
  if (tx_var->defined() && ExprUsesVar(condition, tx_var->value().get())) {
    return true;
  }
  if (ExprUsesVar(condition, thread_x.get())) {
    if (!tx_var->defined()) {
      *tx_var = thread_x;
    }
    return true;
  }
  return false;
}

Stmt UnwrapThreadExtentBody(const Stmt &stmt) {
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    if (attr->attr_key == tir::attr::thread_extent) {
      return UnwrapThreadExtentBody(attr->body);
    }
  }
  return stmt;
}

bool FindForkSite(const Stmt &stmt, const Var &thread_x, Optional<Var> *tx_var,
                  ForkSite *site) {
  Stmt body = UnwrapThreadExtentBody(stmt);
  const auto *seq = body.as<SeqStmtNode>();
  if (seq == nullptr) {
    if (const auto *if_node = body.as<IfThenElseNode>()) {
      if (IsTxForkCondition(if_node->condition, thread_x, tx_var)) {
        site->fork = ffi::GetRef<IfThenElse>(if_node);
        return true;
      }
    }
    return false;
  }

  bool seen_fork = false;
  site->prologue.clear();
  site->epilogue.clear();
  for (const Stmt &item : seq->seq) {
    if (!seen_fork) {
      if (const auto *let = item.as<LetStmtNode>()) {
        if (IsThreadBindingLet(let, thread_x)) {
          *tx_var = let->var;
          continue;
        }
      }
      if (const auto *if_node = item.as<IfThenElseNode>()) {
        if (IsTxForkCondition(if_node->condition, thread_x, tx_var)) {
          site->fork = ffi::GetRef<IfThenElse>(if_node);
          seen_fork = true;
          continue;
        }
      }
      site->prologue.push_back(item);
      continue;
    }
    site->epilogue.push_back(item);
  }

  return seen_fork;
}

Stmt EnsureFinalSharedSync(Array<Stmt> seq) {
  if (seq.empty()) {
    return MakeSharedSync();
  }
  if (IsSharedSync(seq.back())) {
    return seq.size() == 1 ? seq[0] : SeqStmt(seq);
  }
  seq.push_back(MakeSharedSync());
  return SeqStmt(seq);
}

class LowerHcuWdraRewriter : public StmtExprMutator {
public:
  LowerHcuWdraRewriter(int *waves_per_tg) : waves_per_tg_(waves_per_tg) {}

private:
  int *waves_per_tg_;
  Var thread_x_;
  Optional<Var> tx_var_;

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>())
            << "HCU WDRA requires constant threadIdx.x extent";
        int threads = iv->dom->extent.as<IntImmNode>()->value;
        ICHECK(threads % kWaveSize == 0)
            << "HCU WDRA requires thread block size divisible by " << kWaveSize;
        *waves_per_tg_ = threads / kWaveSize;
        thread_x_ = iv->var;
        ForkSite site;
        tx_var_ = Optional<Var>();
        ICHECK(FindForkSite(op->body, thread_x_, &tx_var_, &site))
            << "HCU WDRA lowering expects `tx = T.get_thread_binding()` and "
               "`if tx ...` ladder inside threadIdx.x scope";

        IfThenElse fork = RewriteIfLadder(site.fork, tx_var_, kWaveSize);
        fork = RewriteBranchBodies(fork, tx_var_, thread_x_);

        std::vector<int> branch_nregs;
        CollectBranchNregsFromLadder(fork, 0, *waves_per_tg_, &branch_nregs);
        ICHECK_EQ(static_cast<int>(branch_nregs.size()),
                  *waves_per_tg_ / kWavesPerWdraBranch)
            << "HCU WDRA expects " << (*waves_per_tg_ / kWavesPerWdraBranch)
            << " branches for " << *waves_per_tg_ << " waves per TG";

        Array<Stmt> merged;
        merged.push_back(MakeWdraInit(branch_nregs));
        merged.insert(merged.end(), site.prologue.begin(), site.prologue.end());
        merged.push_back(fork);
        merged.insert(merged.end(), site.epilogue.begin(), site.epilogue.end());
        Stmt new_body = EnsureFinalSharedSync(merged);
        return AttrStmt(op->node, op->attr_key, op->value, new_body);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

class ValidateHcuWdraVisitor : public StmtExprVisitor {
public:
  explicit ValidateHcuWdraVisitor(int waves_per_tg)
      : waves_per_tg_(waves_per_tg) {}

  void ValidateFunction(const Stmt &body) {
    before_fork_ = true;
    VisitStmt(body);
    ICHECK(has_final_sync_) << "HCU WDRA kernel must end with shared sync";
  }

private:
  int waves_per_tg_;
  Var thread_x_;
  bool before_fork_{true};
  bool in_branch_{false};
  bool branch_seen_set_max_nreg_{false};
  bool has_final_sync_{false};
  int branch_depth_{0};

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        thread_x_ = iv->var;
        ICHECK(IsValidWavesPerTg(waves_per_tg_))
            << "HCU WDRA requires 8/12/16 waves per TG, got " << waves_per_tg_;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const LetStmtNode *op) final {
    if (before_fork_ && !in_branch_ && thread_x_.defined()) {
      if (IsThreadBindingLet(op, thread_x_)) {
        LOG(FATAL) << "HCU WDRA forbids thread binding let outside branches: "
                   << op->var->name_hint;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    if (before_fork_) {
      before_fork_ = false;
    }

    branch_seen_set_max_nreg_ = false;
    in_branch_ = true;
    branch_depth_++;
    VisitStmt(op->then_case);
    branch_depth_--;
    in_branch_ = false;
    ICHECK(branch_seen_set_max_nreg_)
        << "HCU WDRA branch must begin with tl.set_max_nreg";

    if (op->else_case.defined()) {
      if (op->else_case.value().as<IfThenElseNode>()) {
        VisitStmt(op->else_case.value());
      } else {
        branch_seen_set_max_nreg_ = false;
        in_branch_ = true;
        branch_depth_++;
        VisitStmt(op->else_case.value());
        branch_depth_--;
        in_branch_ = false;
        ICHECK(branch_seen_set_max_nreg_)
            << "HCU WDRA else branch must begin with tl.set_max_nreg";
      }
    }
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    if (!before_fork_ && branch_depth_ == 0 && !in_branch_) {
      if (!op->seq.empty() && IsSharedSync(op->seq.back())) {
        has_final_sync_ = true;
      }
    }
    if (before_fork_) {
      bool seen_fork = false;
      for (const Stmt &stmt : op->seq) {
        if (!seen_fork) {
          if (stmt.as<IfThenElseNode>()) {
            seen_fork = true;
            VisitStmt(stmt);
          } else {
            ICHECK(!IsHcuWdraPrologueForbiddenStmt(stmt))
                << "HCU WDRA prologue must not contain VGPR-class operations";
            VisitStmt(stmt);
          }
        } else {
          if (IsSharedSync(stmt)) {
            has_final_sync_ = true;
          }
          VisitStmt(stmt);
        }
      }
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (in_branch_ && branch_depth_ == 1 && !branch_seen_set_max_nreg_) {
      if (const auto *call = op->value.as<CallNode>()) {
        if (call->op.same_as(set_max_nreg())) {
          int nreg = call->args[0].as<IntImmNode>()->value;
          ICHECK(nreg % kVgprGranularity == 0)
              << "HCU WDRA set_max_nreg must be multiple of "
              << kVgprGranularity << ", got " << nreg;
          branch_seen_set_max_nreg_ = true;
        }
      }
    }
    if (!before_fork_ && !in_branch_) {
      if (IsSharedSync(ffi::GetRef<Stmt>(op))) {
        has_final_sync_ = true;
      }
    }
    if (before_fork_) {
      ICHECK(!IsHcuWdraPrologueForbiddenStmt(ffi::GetRef<Stmt>(op)))
          << "HCU WDRA prologue must not contain VGPR-class operations";
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

bool ShouldApplyHcuWdra(const PrimFunc &f, const Target &target,
                        const tvm::transform::PassContext &ctx) {
  if (!TargetSupportsHcuWdra(target)) {
    return false;
  }
  bool enabled = PassConfigEnabled(ctx, kEnableHcuWdra);
  if (ContainsSetMaxNreg(f->body) && !enabled) {
    LOG(FATAL) << "HCU WDRA: kernel uses tl.set_max_nreg but pass config "
               << kEnableHcuWdra << " is not enabled. Set "
               << "PassConfigKey.TL_ENABLE_HCU_WDRA=True in pass_configs.";
  }
  return enabled;
}

PrimFunc ApplyLowerAndValidateHcuWdra(PrimFunc f) {
  auto target = f->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target.defined()) << "LowerAndValidateHcuWdra requires target attr";

  int waves_per_tg = 0;
  LowerHcuWdraRewriter lower(&waves_per_tg);
  f.CopyOnWrite()->body = lower(f->body);

  ValidateHcuWdraVisitor validator(waves_per_tg);
  validator.ValidateFunction(f->body);

  f = WithAttr(std::move(f), attr::kHcuWdra, IntImm(DataType::Int(32), 1));
  f = WithAttr(std::move(f), attr::kHcuWdraWavesPerTg,
               IntImm(DataType::Int(32), waves_per_tg));
  return f;
}

} // namespace

using namespace tir::transform;

Pass LowerAndValidateHcuWdra() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined()) {
      return f;
    }
    // Host wrappers created by SplitHostDevice are not device kernels.
    if (!f->HasNonzeroAttr(tir::attr::kIsGlobalFunc)) {
      return f;
    }
    if (!ShouldApplyHcuWdra(f, target.value(), ctx)) {
      return f;
    }
    return ApplyLowerAndValidateHcuWdra(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerAndValidateHcuWdra", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerAndValidateHcuWdra",
                        LowerAndValidateHcuWdra);
}

} // namespace tl
} // namespace tvm
