// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file insert_scale_buffer_sync.cc
 * \brief Insert shared sync before blockscaled gemm when copy_scale may precede
 * it.
 */

#include "hcu/target_utils.h"
#include "op/builtin.h"
#include "op/gemm.h"
#include "op/operator.h"

#include <tvm/ir/transform.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

class WarpSpecializationPresenceChecker : public StmtExprVisitor {
public:
  bool found() const { return found_; }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == attr::kWarpSpecializationScope) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool found_{false};
};

bool ContainsWarpSpecialization(const Stmt &body) {
  WarpSpecializationPresenceChecker checker;
  checker(body);
  return checker.found();
}

bool IsSharedStorageSync(const Stmt &stmt) {
  const auto *eval = stmt.as<EvaluateNode>();
  if (eval == nullptr) {
    return false;
  }
  const auto *call = eval->value.as<CallNode>();
  if (call == nullptr || !call->op.same_as(builtin::tvm_storage_sync()) ||
      call->args.empty()) {
    return false;
  }
  if (const auto *scope = call->args[0].as<StringImmNode>()) {
    return scope->value == "shared" || scope->value == "shared.dyn";
  }
  return false;
}

bool IsBlockscaledGemmEvaluate(const EvaluateNode *op) {
  const auto *call = op->value.as<CallNode>();
  if (call == nullptr || !call->op.as<OpNode>()) {
    return false;
  }
  Op tir_op = Downcast<Op>(call->op);
  if (tir_op != Gemm::Get()) {
    return false;
  }
  auto gemm = Downcast<Gemm>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
  return gemm->sfaRegion_.defined() && gemm->sfbRegion_.defined();
}

Stmt MakeSharedSync() {
  return Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                       {StringImm("shared")}));
}

class InsertScaleBufferSyncMutator : public StmtExprMutator {
public:
  static Stmt Transform(const Stmt &body) {
    InsertScaleBufferSyncMutator mutator;
    return mutator(body);
  }

private:
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> seq;
    bool seq_changed = false;
    for (size_t i = 0; i < op->seq.size(); ++i) {
      Stmt cur = StmtExprMutator::VisitStmt(op->seq[i]);
      if (const auto *eval = cur.as<EvaluateNode>()) {
        if (IsBlockscaledGemmEvaluate(eval)) {
          bool need_sync = true;
          if (!seq.empty() && IsSharedStorageSync(seq.back())) {
            need_sync = false;
          }
          if (need_sync) {
            seq.push_back(MakeSharedSync());
            seq_changed = true;
          }
        }
      }
      seq.push_back(cur);
      if (!cur.same_as(op->seq[i])) {
        seq_changed = true;
      }
    }
    if (!seq_changed) {
      return GetRef<Stmt>(op);
    }
    return SeqStmt(seq);
  }
};

} // namespace

tvm::transform::Pass InsertScaleBufferSync() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    (void)m;
    (void)ctx;
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsHCU(target.value())) {
      return f;
    }
    if (ContainsWarpSpecialization(f->body)) {
      return f;
    }
    PrimFuncNode *fn = f.CopyOnWrite();
    fn->body = InsertScaleBufferSyncMutator::Transform(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InsertScaleBufferSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InsertScaleBufferSync",
                        InsertScaleBufferSync);
}

} // namespace tl
} // namespace tvm
