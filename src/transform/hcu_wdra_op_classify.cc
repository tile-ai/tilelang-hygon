/*!
 * \file hcu_wdra_op_classify.cc
 */
#include "hcu_wdra_op_classify.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

bool IsHcuWdraVgprCall(const CallNode *call) {
  if (call == nullptr) {
    return false;
  }

  // Uniform / WDRA control flow helpers.
  if (call->op.same_as(get_wave_id()) || call->op.same_as(set_max_nreg()) ||
      call->op.same_as(no_set_max_nreg()) ||
      call->op.same_as(hcu_wdra_init())) {
    return false;
  }

  // HCU ABarrier / EBarrier are SGPR-side for POC.
  if (call->op.same_as(abarrier_init()) || call->op.same_as(abarrier_inv()) ||
      call->op.same_as(abarrier_arrive()) ||
      call->op.same_as(abarrier_try_wait()) ||
      call->op.same_as(abarrier_wait()) ||
      call->op.same_as(abarrier_test_wait()) ||
      call->op.same_as(abarrier_seq()) ||
      call->op.same_as(abarrier_expect_tx()) ||
      call->op.same_as(abarrier_complete_tx()) ||
      call->op.same_as(ebarrier_sync()) ||
      call->op.same_as(ebarrier_sync_cnt()) ||
      call->op.same_as(ebarrier_arrive())) {
    return false;
  }

  if (call->op.same_as(builtin::tvm_storage_sync()) ||
      call->op.same_as(sync_warp()) || call->op.same_as(sync_grid()) ||
      call->op.same_as(builtin::address_of()) ||
      call->op.same_as(builtin::reinterpret()) ||
      call->op.same_as(builtin::call_llvm_intrin()) ||
      call->op.same_as(builtin::if_then_else())) {
    return false;
  }

  if (call->op.same_as(tl_gemm()) || call->op.same_as(tl_gemm_sp()) ||
      call->op.same_as(tma_load()) || call->op.same_as(tma_load_im2col()) ||
      call->op.same_as(tma_store()) || call->op.same_as(tma_store_arrive()) ||
      call->op.same_as(tma_store_wait()) || call->op.same_as(ptx_wgmma_ss()) ||
      call->op.same_as(ptx_wgmma_rs()) || call->op.same_as(wait_wgmma()) ||
      call->op.same_as(warpgroup_arrive()) ||
      call->op.same_as(warpgroup_commit_batch()) ||
      call->op.same_as(warpgroup_wait()) ||
      call->op.same_as(warpgroup_fence_operand()) ||
      call->op.same_as(ptx_cp_async()) ||
      call->op.same_as(async_gld_sld_fence()) ||
      call->op.same_as(ptx_ldmatrix()) || call->op.same_as(ptx_stmatrix()) ||
      call->op.same_as(ptx_mma_sm70()) ||
      call->op.same_as(ptx_tcgen05_mma_ss()) ||
      call->op.same_as(ptx_tcgen05_mma_ts()) ||
      call->op.same_as(get_lane_idx()) || call->op.same_as(get_warp_idx()) ||
      call->op.same_as(get_warp_idx_sync()) ||
      call->op.same_as(get_warp_group_idx()) || call->op.same_as(ldg32()) ||
      call->op.same_as(ldg64()) || call->op.same_as(ldg128()) ||
      call->op.same_as(ldg256()) || call->op.same_as(stg32()) ||
      call->op.same_as(stg64()) || call->op.same_as(stg128()) ||
      call->op.same_as(stg256())) {
    return true;
  }

  if (IsMlsLoadTileExternCall(call) || IsMlsAsyncLoadExternCall(call) ||
      IsDsReadFormatExternCall(call)) {
    return true;
  }

  if (call->op.same_as(builtin::call_extern())) {
    if (const auto *name = call->args[0].as<StringImmNode>()) {
      const std::string &value = name->value;
      if (value.find("tl::mls::") == 0 || value.find("tl::gemm") == 0 ||
          value.find("Atomic") == 0) {
        return true;
      }
    }
  }

  // tl.* intrinsics not explicitly whitelisted are treated as VGPR-class.
  if (const auto *op = call->op.as<OpNode>()) {
    const std::string name = op->name;
    if (name.rfind("tl.", 0) == 0) {
      return true;
    }
  }

  return false;
}

bool IsHcuWdraVgprStmt(const Stmt &stmt) {
  struct Visitor : public StmtExprVisitor {
    bool found = false;

    void VisitExpr_(const CallNode *op) final {
      if (IsHcuWdraVgprCall(op)) {
        found = true;
        return;
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const BufferLoadNode *op) final { found = true; }

    void VisitStmt_(const BufferStoreNode *op) final { found = true; }

    void VisitStmt_(const ForNode *op) final {
      if (op->kind == ForKind::kParallel) {
        found = true;
        return;
      }
      StmtExprVisitor::VisitStmt_(op);
    }
  } visitor;

  visitor(stmt);
  return visitor.found;
}

bool IsHcuWdraDeclarativeStmt(const Stmt &stmt) {
  // Allow pure allocate / decl_buffer in WDRA prologue. DeclBuffer may embed
  // BufferLoad in data=buf.data alias forms that would otherwise trip VGPR
  // detection when walking the statement.
  if (stmt->IsInstance<AllocateNode>() || stmt->IsInstance<DeclBufferNode>()) {
    return true;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    if (attr->attr_key == tir::attr::thread_extent ||
        attr->attr_key == tir::attr::device_scope) {
      return true;
    }
  }
  return false;
}

bool IsHcuWdraPrologueForbiddenStmt(const Stmt &stmt) {
  if (IsHcuWdraDeclarativeStmt(stmt)) {
    return false;
  }
  return IsHcuWdraVgprStmt(stmt);
}

} // namespace tl
} // namespace tvm
