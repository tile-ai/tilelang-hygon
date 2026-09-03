#include "common/gemm_k_loop_utils.h"

#include <algorithm>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

class AsyncCountAnalyzer : public StmtExprVisitor {
public:
  static int64_t Analyze(const Stmt &stmt) {
    AsyncCountAnalyzer analyzer;
    analyzer.VisitStmt(stmt);
    return analyzer.count_;
  }

private:
  void VisitStmt_(const ForNode *op) override {
    int64_t sub = Analyze(op->body);
    int64_t extent = 1;
    if (auto e = op->extent.as<IntImmNode>()) {
      extent = e->value;
    }
    count_ += sub * extent;
  }

  void VisitExpr_(const CallNode *op) override {
    if (IsAsyncCopyCall(op)) {
      count_++;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  int64_t count_ = 0;
};

class GlobalMaxAsyncFinder : public StmtVisitor {
public:
  static int64_t FindMax(const Stmt &stmt) {
    GlobalMaxAsyncFinder finder;
    finder.VisitStmt(stmt);
    return std::max(static_cast<int64_t>(1), finder.max_multiplier_);
  }

private:
  void VisitStmt_(const ForNode *op) override {
    int64_t inner_count = AsyncCountAnalyzer::Analyze(op->body);
    if (inner_count > max_multiplier_) {
      max_multiplier_ = inner_count;
    }
    StmtVisitor::VisitStmt_(op);
  }

  int64_t max_multiplier_ = 0;
};

class AsyncWaitCountRewriter : public StmtMutator {
public:
  static Stmt Substitute(const Stmt &stmt) {
    int64_t max_mult = GlobalMaxAsyncFinder::FindMax(stmt);
    AsyncWaitCountRewriter rewriter(max_mult);
    return rewriter(stmt);
  }

private:
  explicit AsyncWaitCountRewriter(int64_t mult) : global_max_mult_(mult) {}

  Stmt VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == s_tir::attr::async_wait_inflight_count ||
        op->attr_key == "async_wait_inflight_count") {
      if (auto int_imm = op->value.as<IntImmNode>()) {
        int64_t new_val = int_imm->value * global_max_mult_;
        return AttrStmt(op->node, op->attr_key,
                        make_const(DataType::Int(32), new_val),
                        this->VisitStmt(op->body), op->span);
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  int64_t global_max_mult_;
};

} // namespace

namespace transform {

tirx::transform::Pass CanonicalizeAsyncWaitCount() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, IRModule, PassContext) {
    auto *n = f.CopyOnWrite();
    n->body = AsyncWaitCountRewriter::Substitute(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.CanonicalizeAsyncWaitCount", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.CanonicalizeAsyncWaitCount",
                        CanonicalizeAsyncWaitCount);
}

} // namespace transform
} // namespace tl
} // namespace tvm
