/*!
 * \file verify_pipeline_sync.cc
 * \brief Verify the final synchronization structure of HCU-owned scopes.
 *
 * The fence planner takes over every wait in a GEMM K loop it owns and pairs
 * each wait with a barrier. ThreadSync leaves that scope alone through the
 * tl.sync_owned_scope annotation, so nothing after the planner re-checks the
 * result. This pass states the invariants the plan is written against and
 * fails the build when the final IR stops satisfying them:
 *
 *  - at most one owned scope per function (the plan is authored for one K loop
 *    per kernel, and FindOwnedWaitPlanKLoop declines anything else);
 *  - the owned scope is a GEMM K loop carrying commits, waits and at least one
 *    barrier;
 *  - the asynchronous-write effect annotation appears exactly inside owned
 *    scopes, where the plan compensates for the barriers it makes ThreadSync
 *    drop;
 *  - every wait is followed by a barrier before the next commit, so a producer
 *    never overwrites a slot that a consumer may still be reading; and, in a
 *    loop the register pipeline did not rewrite, by a barrier before the next
 *    MMA, so a consumer never reads staged data without a block-wide ordering
 *    point.
 *
 * A register-pipeline loop keeps a different barrier discipline: its waits are
 * issued and consumed inside one warp-group branch, the plan pairs them with
 * the release barrier that precedes the next slot refill, and the MMA reads
 * ordered by the LDS fence do not require an acquire barrier of their own.
 * The verifier therefore checks the release side there, not the acquire side.
 */

#include "hcu/op/builtin.h"
#include "hcu/target_utils.h"
#include "hcu/transform/gemm_k_loop_utils.h"
#include "op/builtin.h"
#include "transform/common/attr.h"
#include "transform/common/pipeline_utils.h"

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
using namespace tirx;
using ffi::GetRef;
using ffi::String;

namespace {

bool IsBarrierCall(const CallNode *call) {
  if (call == nullptr) {
    return false;
  }
  if (call->op.same_as(wave_barrier()) || call->op.same_as(sync_warp())) {
    return true;
  }
  if (!call->op.same_as(builtin::tvm_storage_sync())) {
    return false;
  }
  if (call->args.empty()) {
    return true;
  }
  const auto *scope = call->args[0].as<StringImmNode>();
  return scope == nullptr || std::string(scope->value).rfind("shared", 0) == 0;
}

/*! \brief The executable events the invariants are stated over. */
enum class EventKind {
  kBarrier,
  kCommit,
  kWait,
  kMma,
  kOther,
};

struct Event {
  EventKind kind;
  int wait_count;
};

/*!
 * \brief Flatten a scope body into the lexical event sequence.
 *
 * Loop and if bodies are followed once: every iteration has the same
 * structure, and the invariants are stated per pass through the sequence, so
 * one visit is enough.
 */
class EventCollector : public StmtExprVisitor {
public:
  std::vector<Event> events;

private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::ptx_commit_group())) {
      events.push_back({EventKind::kCommit, -1});
    } else if (op->op.same_as(builtin::ptx_wait_group()) ||
               IsRegisteredAsyncWaitOp(op)) {
      int count = -1;
      if (!op->args.empty()) {
        if (const auto *imm = op->args[0].as<IntImmNode>()) {
          count = static_cast<int>(imm->value);
        }
      }
      events.push_back({EventKind::kWait, count});
    } else if (IsBarrierCall(op)) {
      events.push_back({EventKind::kBarrier, -1});
    } else if (IsMmaCall(op)) {
      events.push_back({EventKind::kMma, -1});
    } else {
      events.push_back({EventKind::kOther, -1});
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

struct ScopeSummary {
  int commits{0};
  int waits{0};
  int barriers{0};
  int mmas{0};
  int idxen_copies{0};
  int annotated_idxen_copies{0};
};

ScopeSummary Summarize(const Stmt &stmt) {
  class SummaryCollector : public StmtExprVisitor {
  public:
    ScopeSummary summary;

  private:
    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(builtin::ptx_commit_group())) {
        summary.commits++;
      } else if (op->op.same_as(builtin::ptx_wait_group()) ||
                 IsRegisteredAsyncWaitOp(op)) {
        summary.waits++;
      } else if (IsBarrierCall(op)) {
        summary.barriers++;
      } else if (IsMmaCall(op)) {
        summary.mmas++;
      } else if (op->op.same_as(hcu_cp_async_idxen())) {
        summary.idxen_copies++;
        if (op->annotations.Get(attr::kAsyncSharedWrite)) {
          summary.annotated_idxen_copies++;
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }
  };
  SummaryCollector collector;
  collector(stmt);
  return collector.summary;
}

/*!
 * \brief Verify that every wait is ordered against the next piece of work.
 *
 * A wait releases a slot only together with a block-wide barrier: the next
 * commit into that slot must be ordered after every read of the previous
 * contents. Walking the lexical sequence, every wait must see a barrier (or
 * another wait, for chained waits) before the next commit. A shared-memory
 * pipeline additionally needs the acquire side, a barrier before the next MMA.
 * Scheduler barriers do not order memory, so they do not count.
 */
void VerifyWaitBarrierPairs(const std::vector<Event> &events,
                            const std::string &where, bool register_pipeline) {
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i].kind != EventKind::kWait) {
      continue;
    }
    ICHECK_GE(events[i].wait_count, 0)
        << where << ": async wait count must be a non-negative immediate";
    bool saw_barrier = false;
    for (size_t j = i + 1; j < events.size(); ++j) {
      if (events[j].kind == EventKind::kBarrier) {
        saw_barrier = true;
        break;
      }
      if (events[j].kind == EventKind::kWait) {
        continue;
      }
      if (events[j].kind == EventKind::kCommit) {
        break;
      }
      if (events[j].kind == EventKind::kMma && !register_pipeline) {
        break;
      }
    }
    ICHECK(saw_barrier)
        << where
        << ": an async wait is not followed by a barrier before the next "
        << (register_pipeline ? "commit" : "commit or MMA")
        << "; the wait plan must pair every wait with a block-wide ordering "
           "point";
  }
}

class SyncOwnershipVerifier : public StmtExprVisitor {
public:
  explicit SyncOwnershipVerifier(std::string func_name)
      : func_name_(std::move(func_name)) {}

  void Verify(const Stmt &stmt) { VisitStmt(stmt); }

private:
  void VisitStmt_(const ForNode *op) final {
    const bool owned = LoopAnnotationIsTruthy(op, attr::kSyncOwnedScope);
    ScopeSummary summary;
    std::vector<Event> events;
    if (owned) {
      ICHECK(owned_scope_depth_ == 0)
          << func_name_
          << ": nested tl.sync_owned_scope loops are not "
             "supported; the fence planner owns one K loop";
      summary = Summarize(op->body);
      EventCollector collector;
      collector(op->body);
      events = std::move(collector.events);
      owned_scope_depth_++;
    }
    StmtExprVisitor::VisitStmt_(op);
    if (owned) {
      owned_scope_depth_--;
    }
    if (!owned) {
      return;
    }

    const std::string where =
        func_name_ + ": sync-owned loop `" + op->loop_var->name_hint + "`";
    ICHECK(IsGemmKLoop(op))
        << where
        << " is marked tl.sync_owned_scope but is not a GEMM K loop; only the "
           "loop the fence planner rewrites may own its synchronization";
    ICHECK(summary.commits > 0)
        << where << " owns its synchronization but issues no async commits";
    ICHECK(summary.waits > 0)
        << where << " owns its synchronization but issues no async waits";
    ICHECK(summary.mmas > 0)
        << where << " owns its synchronization but contains no MMA";
    ICHECK(summary.barriers > 0)
        << where
        << " owns its synchronization but contains no block-wide barrier";
    ICHECK_EQ(summary.idxen_copies, summary.annotated_idxen_copies)
        << where << " contains " << summary.idxen_copies
        << " indexed async copy(ies) but only "
        << summary.annotated_idxen_copies << " declare "
        << attr::kAsyncSharedWrite
        << "; every copy in an owned scope must let ThreadSync see the "
           "asynchronous write it compensates for";
    VerifyWaitBarrierPairs(events, where, LoopHasAppliedRegisterPipeline(op));
  }

  void VisitExpr_(const CallNode *op) final {
    ICHECK(!(op->op.same_as(hcu_cp_async_idxen()) &&
             op->annotations.Get(attr::kAsyncSharedWrite) &&
             owned_scope_depth_ == 0))
        << func_name_ << ": indexed async copy declares "
        << attr::kAsyncSharedWrite
        << " outside a tl.sync_owned_scope; only the scope owner may drop the "
           "barriers ThreadSync would insert";
    StmtExprVisitor::VisitExpr_(op);
  }

  std::string func_name_;
  int owned_scope_depth_{0};
};

} // namespace

namespace transform {
using namespace tirx::transform;

Pass VerifyPipelineSync() {
  auto pass_func = [](PrimFunc f, IRModule, PassContext) {
    Target target = f->GetAttr<Target>(tvm::attr::kTarget).value_or(Target());
    if (!target.defined() || !TargetIsHCU(target)) {
      return f;
    }
    std::string func_name =
        f->GetAttr<String>(tvm::attr::kGlobalSymbol).value_or(String("kernel"));
    SyncOwnershipVerifier verifier(func_name);
    verifier.Verify(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.VerifyPipelineSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VerifyPipelineSync", VerifyPipelineSync);
}
} // namespace transform
} // namespace tl
} // namespace tvm
