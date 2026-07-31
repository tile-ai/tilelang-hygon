// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file annotate_scale_gemm_dep.cc
 * \brief Pre-layout pass: bind copy_scale producers to gemm_blockscaled
 * consumers.
 */

#include "hcu/op/copy_scale.h"
#include "hcu/op/ds_read_format.h"
#include "hcu/op/gemm_partition.h"
#include "hcu/op/mls.h"
#include "hcu/target_utils.h"
#include "hcu/utils/mls_gemm_dep.h"
#include "hcu/utils/scale_gemm_dep.h"
#include "op/gemm.h"
#include "op/operator.h"

#include <tvm/ir/transform.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

Optional<int> GetIntAnn(const Map<String, ObjectRef> &annotations,
                        const char *key) {
  if (auto val = annotations.Get(key)) {
    if (const auto *imm = val->as<IntImmNode>()) {
      return static_cast<int>(imm->value);
    }
    LOG(FATAL) << "Annotation `" << key << "` expects IntImm, got "
               << val->GetTypeKey();
  }
  return Optional<int>();
}

struct ScaleBufferMeta {
  int granularity_mn{1};
  int granularity_k{1};
  int scale_k_major{0};
  int scale_format{0};
  int role{0};
  // Deferred warp partition clues (from consumer gemm).
  int gemm_m{0};
  int gemm_n{0};
  int gemm_k{0};
  int gemm_policy{0};
  int gemm_k_pack{1};
  int gemm_elem_bits{4};
  int a_from_mls{0};
  int b_from_mls{0};
  int a_mls_trans{1};
  int b_mls_trans{1};
  // ScaleFormat atom floor for this buffer's MN side (0 = none).
  int format_min_mn_per_warp{0};
  bool defined{false};
};

void MergeScaleBufferMeta(ScaleBufferMeta *meta, int granularity_mn,
                          int granularity_k, int scale_k_major, int role,
                          int scale_format, int format_min_mn_per_warp,
                          int gemm_elem_bits, const GemmNode *gemm,
                          const Buffer &buf, const char *site) {
  const Map<String, ObjectRef> &ann = gemm->annotations_;
  auto get_flag = [&](const char *key, int def) {
    if (auto v = GetIntAnn(ann, key)) {
      return *v;
    }
    return def;
  };
  int a_from = get_flag(attr::kHcuAFromMls, 0);
  int b_from = get_flag(attr::kHcuBFromMls, 0);
  // Prefer operand-specific mls_trans when annotated; else shared kMlsTrans.
  int a_trans = get_flag(attr::kMlsTrans, 1);
  int b_trans = get_flag(attr::kMlsTrans, 1);
  if (auto v = GetIntAnn(ann, "tl.a_mls_trans")) {
    a_trans = *v;
  }
  if (auto v = GetIntAnn(ann, "tl.b_mls_trans")) {
    b_trans = *v;
  }
  const int gemm_policy = gemm->policy_->policy_type;
  const int gemm_k_pack = gemm->kPack_;

  if (!meta->defined) {
    meta->granularity_mn = granularity_mn;
    meta->granularity_k = granularity_k;
    meta->scale_k_major = scale_k_major;
    meta->scale_format = scale_format;
    meta->role = role;
    meta->gemm_m = gemm->m_;
    meta->gemm_n = gemm->n_;
    meta->gemm_k = gemm->k_;
    meta->gemm_policy = gemm_policy;
    meta->gemm_k_pack = gemm_k_pack;
    meta->gemm_elem_bits = gemm_elem_bits;
    meta->a_from_mls = a_from;
    meta->b_from_mls = b_from;
    meta->a_mls_trans = a_trans;
    meta->b_mls_trans = b_trans;
    meta->format_min_mn_per_warp = format_min_mn_per_warp;
    meta->defined = true;
    return;
  }
  meta->format_min_mn_per_warp =
      std::max(meta->format_min_mn_per_warp, format_min_mn_per_warp);

  // Scale geometry must always match.
  ICHECK_EQ(meta->granularity_mn, granularity_mn)
      << "Inconsistent scale granularity_mn for buffer " << buf << " at "
      << site;
  ICHECK_EQ(meta->granularity_k, granularity_k)
      << "Inconsistent scale granularity_k for buffer " << buf << " at "
      << site;
  ICHECK_EQ(meta->scale_k_major, scale_k_major)
      << "Inconsistent scale_k_major for buffer " << buf << " at " << site;
  ICHECK_EQ(meta->scale_format, scale_format)
      << "Inconsistent ScaleFormat for shared scale buffer " << buf << " at "
      << site;
  ICHECK_EQ(meta->gemm_k, gemm->k_)
      << "Inconsistent gemm K for shared scale buffer " << buf << " at "
      << site;
  // Warp-partition clues: must agree so ComputeScaleWarpSeg is unique.
  ICHECK_EQ(meta->gemm_policy, gemm_policy)
      << "Inconsistent gemm policy for shared scale buffer " << buf << " at "
      << site;
  ICHECK_EQ(meta->gemm_k_pack, gemm_k_pack)
      << "Inconsistent gemm k_pack for shared scale buffer " << buf << " at "
      << site;
  ICHECK_EQ(meta->gemm_elem_bits, gemm_elem_bits)
      << "Inconsistent gemm elem_bits for shared scale buffer " << buf << " at "
      << site;

  if (meta->role == role) {
    // Same side (both ScaleA or both ScaleB): full gemm MN + MLS must match.
    ICHECK_EQ(meta->gemm_m, gemm->m_)
        << "Inconsistent gemm M for shared scale buffer " << buf << " at "
        << site;
    ICHECK_EQ(meta->gemm_n, gemm->n_)
        << "Inconsistent gemm N for shared scale buffer " << buf << " at "
        << site;
    ICHECK_EQ(meta->a_from_mls, a_from)
        << "Inconsistent a_from_mls for shared scale buffer " << buf << " at "
        << site;
    ICHECK_EQ(meta->b_from_mls, b_from)
        << "Inconsistent b_from_mls for shared scale buffer " << buf << " at "
        << site;
    ICHECK_EQ(meta->a_mls_trans, a_trans)
        << "Inconsistent a_mls_trans for shared scale buffer " << buf << " at "
        << site;
    ICHECK_EQ(meta->b_mls_trans, b_trans)
        << "Inconsistent b_mls_trans for shared scale buffer " << buf << " at "
        << site;
    return;
  }

  // Cross-role share: buffer is ScaleA of one gemm and ScaleB of another.
  // Allow when consumers are MN-symmetric so MnWarps / scale MN extent agree:
  //   stored(A): side=M, other=N, mls_a/b
  //   incoming(B): side=N, other=M, mls_b/a  (swapped)
  const int stored_side = (meta->role == 0) ? meta->gemm_m : meta->gemm_n;
  const int stored_other = (meta->role == 0) ? meta->gemm_n : meta->gemm_m;
  const int incoming_side = (role == 0) ? gemm->m_ : gemm->n_;
  const int incoming_other = (role == 0) ? gemm->n_ : gemm->m_;
  ICHECK_EQ(stored_side, incoming_side)
      << "Cross-role scale share requires matching MN-side extent for buffer "
      << buf << " at " << site << " (stored_side=" << stored_side
      << ", incoming_side=" << incoming_side << ")";
  ICHECK_EQ(stored_other, incoming_other)
      << "Cross-role scale share requires MN-symmetric gemm shape for buffer "
      << buf << " at " << site << " (stored_other=" << stored_other
      << ", incoming_other=" << incoming_other << ")";

  const int stored_side_mls =
      (meta->role == 0) ? meta->a_from_mls : meta->b_from_mls;
  const int stored_other_mls =
      (meta->role == 0) ? meta->b_from_mls : meta->a_from_mls;
  const int incoming_side_mls = (role == 0) ? a_from : b_from;
  const int incoming_other_mls = (role == 0) ? b_from : a_from;
  ICHECK_EQ(stored_side_mls, incoming_side_mls)
      << "Cross-role scale share: MLS flag on scale side mismatch for buffer "
      << buf << " at " << site;
  ICHECK_EQ(stored_other_mls, incoming_other_mls)
      << "Cross-role scale share: MLS flag on other side mismatch for buffer "
      << buf << " at " << site;

  const int stored_side_trans =
      (meta->role == 0) ? meta->a_mls_trans : meta->b_mls_trans;
  const int stored_other_trans =
      (meta->role == 0) ? meta->b_mls_trans : meta->a_mls_trans;
  const int incoming_side_trans = (role == 0) ? a_trans : b_trans;
  const int incoming_other_trans = (role == 0) ? b_trans : a_trans;
  ICHECK_EQ(stored_side_trans, incoming_side_trans)
      << "Cross-role scale share: mls_trans on scale side mismatch for buffer "
      << buf << " at " << site;
  ICHECK_EQ(stored_other_trans, incoming_other_trans)
      << "Cross-role scale share: mls_trans on other side mismatch for buffer "
      << buf << " at " << site;
  // Keep first-seen role + gemm clues on the buffer (copy_scale annotations).
}

Var MakeScaleRowBaseVar(const Buffer &buf) {
  return Var("scale_row_base_" + buf->name, DataType::Int(32));
}

class ScaleTileOpPresenceChecker : public StmtExprVisitor {
public:
  bool found() const { return found_; }

private:
  void VisitStmt(const Stmt &stmt) final {
    if (found_) {
      return;
    }
    StmtExprVisitor::VisitStmt(stmt);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (found_) {
      return;
    }
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.as<OpNode>()) {
        Op tir_op = Downcast<Op>(call->op);
        if (tir_op == CopyScale::Get()) {
          found_ = true;
          return;
        }
        if (tir_op == Gemm::Get()) {
          auto gemm =
              Downcast<Gemm>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
          if (gemm->sfaRegion_.defined()) {
            found_ = true;
            return;
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool found_{false};
};

bool ContainsScaleTileOps(const Stmt &body) {
  ScaleTileOpPresenceChecker checker;
  checker(body);
  return checker.found();
}

struct CopyScaleSite {
  Call call;
  CopyScale op;
};

struct GemmSite {
  Call call;
  Gemm op;
};

hcu::HcuMmacModeInfo
ResolveConsumerMmacMode(const GemmNode *gemm, const Map<String, ObjectRef> &ann,
                        int block_size, ScaleLdsFormat scale_format_a,
                        ScaleLdsFormat scale_format_b, Target target) {
  (void)ann;
  (void)block_size;
  const Buffer &a = gemm->aRegion_->buffer;
  const Buffer &b = gemm->bRegion_->buffer;
  return hcu::ResolveHcuMmacMode(
      a->dtype, b->dtype, a.scope() == "local.fragment",
      b.scope() == "local.fragment", /*is_blockscaled=*/true, gemm->k_,
      scale_format_a, scale_format_b, target);
}

class ScaleSiteCollector : public StmtExprVisitor {
public:
  std::vector<CopyScaleSite> copy_sites;
  std::vector<GemmSite> gemm_sites;
  int thread_x_extent{-1};

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>()) {
        if (iv->thread_tag == "threadIdx.x") {
          if (const int64_t *ext = as_const_int(op->value)) {
            thread_x_extent = static_cast<int>(*ext);
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.as<OpNode>()) {
        Op tir_op = Downcast<Op>(call->op);
        if (tir_op == CopyScale::Get()) {
          copy_sites.push_back({tvm::ffi::GetRef<Call>(call),
                                Downcast<CopyScale>(ParseOperator(
                                    tvm::ffi::GetRef<Call>(call)))});
        } else if (tir_op == Gemm::Get()) {
          auto gemm =
              Downcast<Gemm>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
          if (gemm->sfaRegion_.defined() && gemm->sfbRegion_.defined()) {
            gemm_sites.push_back({tvm::ffi::GetRef<Call>(call), gemm});
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

} // namespace

class AnnotateScaleGemmDepMutator : public StmtExprMutator {
public:
  AnnotateScaleGemmDepMutator(
      const std::unordered_map<Buffer, ScaleBufferMeta, ObjectPtrHash,
                               ObjectPtrEqual> &meta_map,
      const std::unordered_map<Buffer, Var, ObjectPtrHash, ObjectPtrEqual>
          &row_base_vars,
      const std::unordered_map<Buffer, std::pair<int, int>, ObjectPtrHash,
                               ObjectPtrEqual> &buf_partition_mins,
      const std::unordered_map<const CallNode *, std::pair<int, int>>
          &gemm_partition_mins)
      : meta_map_(meta_map), row_base_vars_(row_base_vars),
        buf_partition_mins_(buf_partition_mins),
        gemm_partition_mins_(gemm_partition_mins) {}

  static PrimFunc Substitute(PrimFunc f) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsHCU(target.value())) {
      return f;
    }
    if (!ContainsScaleTileOps(f->body)) {
      return f;
    }

    ScaleSiteCollector collector;
    collector(f->body);
    ICHECK_GT(collector.thread_x_extent, 0)
        << "AnnotateScaleGemmDep requires static threadIdx.x extent";

    std::unordered_map<Buffer, ScaleBufferMeta, ObjectPtrHash, ObjectPtrEqual>
        meta_map;
    std::unordered_map<Buffer, Var, ObjectPtrHash, ObjectPtrEqual>
        row_base_vars;

    for (const auto &site : collector.copy_sites) {
      const Buffer &dst = site.op->dst;
      bool matched = false;
      for (const auto &gemm_site : collector.gemm_sites) {
        const GemmNode *gemm = gemm_site.op.get();
        const Map<String, ObjectRef> &ann = gemm->annotations_;
        ScaleLdsFormat scale_format_a = ScaleLdsFormat::kIdentity;
        ScaleLdsFormat scale_format_b = ScaleLdsFormat::kIdentity;
        for (const auto &format_site : collector.copy_sites) {
          if (gemm->sfaRegion_.defined() &&
              format_site.op->dst.same_as(gemm->sfaRegion_->buffer)) {
            scale_format_a =
                static_cast<ScaleLdsFormat>(format_site.op->scale_format_);
          }
          if (gemm->sfbRegion_.defined() &&
              format_site.op->dst.same_as(gemm->sfbRegion_->buffer)) {
            scale_format_b =
                static_cast<ScaleLdsFormat>(format_site.op->scale_format_);
          }
        }
        const hcu::HcuMmacModeInfo mmac_mode = ResolveConsumerMmacMode(
            gemm, ann, collector.thread_x_extent, scale_format_a,
            scale_format_b, target.value());
        const int gemm_elem_bits = mmac_mode.element_bits;
        if (gemm->sfaRegion_.defined() &&
            dst.same_as(gemm->sfaRegion_->buffer)) {
          auto gran_m = GetIntAnn(ann, "sf_a_granularity_m");
          auto gran_k = GetIntAnn(ann, "sf_a_granularity_k");
          auto k_major = GetIntAnn(ann, "a_scale_k_major");
          ICHECK(gran_m && gran_k && k_major)
              << "gemm_blockscaled missing sf_a_granularity_* / "
                 "a_scale_k_major";
          const int format_floor = hcu::ScaleFormatMinMnPerWarp(
              static_cast<ScaleLdsFormat>(site.op->scale_format_));
          MergeScaleBufferMeta(&meta_map[dst], *gran_m, *gran_k, *k_major, 0,
                               site.op->scale_format_, format_floor,
                               gemm_elem_bits, gemm, dst,
                               "copy_scale->gemm ScaleA");
          matched = true;
        }
        if (gemm->sfbRegion_.defined() &&
            dst.same_as(gemm->sfbRegion_->buffer)) {
          auto gran_n = GetIntAnn(ann, "sf_b_granularity_n");
          auto gran_k = GetIntAnn(ann, "sf_b_granularity_k");
          auto k_major = GetIntAnn(ann, "b_scale_k_major");
          ICHECK(gran_n && gran_k && k_major)
              << "gemm_blockscaled missing sf_b_granularity_* / "
                 "b_scale_k_major";
          const int format_floor = hcu::ScaleFormatMinMnPerWarp(
              static_cast<ScaleLdsFormat>(site.op->scale_format_));
          MergeScaleBufferMeta(&meta_map[dst], *gran_n, *gran_k, *k_major, 1,
                               site.op->scale_format_, format_floor,
                               gemm_elem_bits, gemm, dst,
                               "copy_scale->gemm ScaleB");
          matched = true;
        }
      }
      ICHECK(matched) << "copy_scale dst " << dst
                      << " has no matching gemm_blockscaled consumer";
    }

    // Per-buffer: joint (min_m, min_n) of every gemm that consumes this scale
    // buffer, so copy_scale / gemm / ds_read share one partition floor.
    std::unordered_map<Buffer, std::pair<int, int>, ObjectPtrHash,
                       ObjectPtrEqual>
        buf_partition_mins;
    std::unordered_map<const CallNode *, std::pair<int, int>>
        gemm_partition_mins;
    for (const auto &gemm_site : collector.gemm_sites) {
      const GemmNode *gemm = gemm_site.op.get();
      int min_m = 0;
      int min_n = 0;
      if (gemm->sfaRegion_.defined()) {
        auto it = meta_map.find(gemm->sfaRegion_->buffer);
        if (it != meta_map.end()) {
          min_m = std::max(min_m, it->second.format_min_mn_per_warp);
        }
      }
      if (gemm->sfbRegion_.defined()) {
        auto it = meta_map.find(gemm->sfbRegion_->buffer);
        if (it != meta_map.end()) {
          min_n = std::max(min_n, it->second.format_min_mn_per_warp);
        }
      }
      gemm_partition_mins[gemm_site.call.get()] = {min_m, min_n};
      auto bump = [&](const Buffer &buf) {
        auto &p = buf_partition_mins[buf];
        p.first = std::max(p.first, min_m);
        p.second = std::max(p.second, min_n);
      };
      if (gemm->sfaRegion_.defined()) {
        bump(gemm->sfaRegion_->buffer);
      }
      if (gemm->sfbRegion_.defined()) {
        bump(gemm->sfbRegion_->buffer);
      }
    }

    for (const auto &[buf, meta] : meta_map) {
      (void)meta;
      row_base_vars.emplace(buf, MakeScaleRowBaseVar(buf));
    }

    AnnotateScaleGemmDepMutator mutator(
        meta_map, row_base_vars, buf_partition_mins, gemm_partition_mins);
    PrimFuncNode *fn = f.CopyOnWrite();
    fn->body = mutator(f->body);
    return f;
  }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.as<OpNode>()) {
        Op tir_op = Downcast<Op>(call->op);
        if (tir_op == CopyScale::Get()) {
          auto cs =
              Downcast<CopyScale>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
          const Buffer &dst = cs->dst;
          auto meta_it = meta_map_.find(dst);
          ICHECK(meta_it != meta_map_.end())
              << "copy_scale dst missing ScaleBufferMeta: " << dst;
          auto row_it = row_base_vars_.find(dst);
          ICHECK(row_it != row_base_vars_.end());
          const ScaleBufferMeta &meta = meta_it->second;
          int min_m = 0;
          int min_n = 0;
          auto mins_it = buf_partition_mins_.find(dst);
          if (mins_it != buf_partition_mins_.end()) {
            min_m = mins_it->second.first;
            min_n = mins_it->second.second;
          }
          auto annotations = call->annotations;
          annotations.Set(attr::kScaleGranularityMN,
                          IntImm(DataType::Int(32), meta.granularity_mn));
          annotations.Set(attr::kScaleGranularityK,
                          IntImm(DataType::Int(32), meta.granularity_k));
          annotations.Set(attr::kScaleKMajor,
                          IntImm(DataType::Int(32), meta.scale_k_major));
          annotations.Set(attr::kScaleRole,
                          IntImm(DataType::Int(32), meta.role));
          annotations.Set(attr::kScaleRowBase, row_it->second);
          annotations.Set(attr::kScaleGemmM,
                          IntImm(DataType::Int(32), meta.gemm_m));
          annotations.Set(attr::kScaleGemmN,
                          IntImm(DataType::Int(32), meta.gemm_n));
          annotations.Set(attr::kScaleGemmK,
                          IntImm(DataType::Int(32), meta.gemm_k));
          annotations.Set(attr::kScaleGemmPolicy,
                          IntImm(DataType::Int(32), meta.gemm_policy));
          annotations.Set(attr::kScaleGemmKPack,
                          IntImm(DataType::Int(32), meta.gemm_k_pack));
          annotations.Set(attr::kScaleGemmElemBits,
                          IntImm(DataType::Int(32), meta.gemm_elem_bits));
          annotations.Set(attr::kScaleAFromMls,
                          IntImm(DataType::Int(32), meta.a_from_mls));
          annotations.Set(attr::kScaleBFromMls,
                          IntImm(DataType::Int(32), meta.b_from_mls));
          annotations.Set(attr::kScaleAMlsTrans,
                          IntImm(DataType::Int(32), meta.a_mls_trans));
          annotations.Set(attr::kScaleBMlsTrans,
                          IntImm(DataType::Int(32), meta.b_mls_trans));
          annotations.Set(attr::kScaleMinMPerWarp,
                          IntImm(DataType::Int(32), min_m));
          annotations.Set(attr::kScaleMinNPerWarp,
                          IntImm(DataType::Int(32), min_n));
          Call new_call(call->dtype, call->op, call->args, annotations,
                        call->span);
          return Evaluate(new_call);
        }
        if (tir_op == Gemm::Get()) {
          auto gemm =
              Downcast<Gemm>(ParseOperator(tvm::ffi::GetRef<Call>(call)));
          if (!gemm->sfaRegion_.defined() || !gemm->sfbRegion_.defined()) {
            return StmtExprMutator::VisitStmt_(op);
          }
          auto annotations = call->annotations;
          if (gemm->sfaRegion_.defined()) {
            const Buffer &sfa = gemm->sfaRegion_->buffer;
            auto it = row_base_vars_.find(sfa);
            ICHECK(it != row_base_vars_.end())
                << "gemm ScaleA buffer missing row_base var: " << sfa;
            annotations.Set(attr::kScaleARowBase, it->second);
            const auto meta_it = meta_map_.find(sfa);
            ICHECK(meta_it != meta_map_.end());
            annotations.Set(
                attr::kScaleAFormat,
                IntImm(DataType::Int(32), meta_it->second.scale_format));
          }
          if (gemm->sfbRegion_.defined()) {
            const Buffer &sfb = gemm->sfbRegion_->buffer;
            auto it = row_base_vars_.find(sfb);
            ICHECK(it != row_base_vars_.end())
                << "gemm ScaleB buffer missing row_base var: " << sfb;
            annotations.Set(attr::kScaleBRowBase, it->second);
            const auto meta_it = meta_map_.find(sfb);
            ICHECK(meta_it != meta_map_.end());
            annotations.Set(
                attr::kScaleBFormat,
                IntImm(DataType::Int(32), meta_it->second.scale_format));
          }
          int min_m = 0;
          int min_n = 0;
          auto gmin = gemm_partition_mins_.find(call);
          if (gmin != gemm_partition_mins_.end()) {
            min_m = gmin->second.first;
            min_n = gmin->second.second;
          }
          annotations.Set(attr::kScaleMinMPerWarp,
                          IntImm(DataType::Int(32), min_m));
          annotations.Set(attr::kScaleMinNPerWarp,
                          IntImm(DataType::Int(32), min_n));
          Call new_call(call->dtype, call->op, call->args, annotations,
                        call->span);
          return Evaluate(new_call);
        }
        if (tir_op == DsReadFormat::Get()) {
          // Propagate scale floors into MlsGemmDepMeta so InferLayout uses the
          // same ComputeWarpPartitionHCU mins as gemm_blockscaled.
          auto annotations = call->annotations;
          auto dep = GetMlsGemmDepFromAnnotations(annotations);
          if (!dep.defined()) {
            return StmtExprMutator::VisitStmt_(op);
          }
          int min_m = 0;
          int min_n = 0;
          for (const auto &kv : gemm_partition_mins_) {
            const auto *gcall = kv.first;
            auto gemm =
                Downcast<Gemm>(ParseOperator(tvm::ffi::GetRef<Call>(gcall)));
            if (gemm->m_ == dep.value()->gemm_m &&
                gemm->n_ == dep.value()->gemm_n &&
                gemm->k_ == dep.value()->gemm_k &&
                gemm->policy_->policy_type == dep.value()->gemm_policy) {
              min_m = std::max(min_m, kv.second.first);
              min_n = std::max(min_n, kv.second.second);
            }
          }
          if (min_m == 0 && min_n == 0) {
            return StmtExprMutator::VisitStmt_(op);
          }
          ObjectPtr<MlsGemmDepMetaNode> n =
              tvm::ffi::make_object<MlsGemmDepMetaNode>(*dep.value().get());
          n->min_m_per_warp = std::max(n->min_m_per_warp, min_m);
          n->min_n_per_warp = std::max(n->min_n_per_warp, min_n);
          annotations.Set(attr::kMlsGemmDep, MlsGemmDepMeta(n));
          Call new_call(call->dtype, call->op, call->args, annotations,
                        call->span);
          return Evaluate(new_call);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  std::unordered_map<Buffer, ScaleBufferMeta, ObjectPtrHash, ObjectPtrEqual>
      meta_map_;
  std::unordered_map<Buffer, Var, ObjectPtrHash, ObjectPtrEqual> row_base_vars_;
  std::unordered_map<Buffer, std::pair<int, int>, ObjectPtrHash, ObjectPtrEqual>
      buf_partition_mins_;
  std::unordered_map<const CallNode *, std::pair<int, int>>
      gemm_partition_mins_;
};

tvm::transform::Pass AnnotateScaleGemmDep() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    (void)m;
    (void)ctx;
    return AnnotateScaleGemmDepMutator::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateScaleGemmDep", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnnotateScaleGemmDep",
                        AnnotateScaleGemmDep);
}

} // namespace tl
} // namespace tvm
