/*!
 * \file mls_gemm_dep.cc
 */

#include "mls_gemm_dep.h"

#include "ds_read_format.h"
#include "gemm.h"
#include "operator.h"

namespace tvm {
namespace tl {

using namespace tir;

MlsGemmDepMeta BuildMlsGemmDepMeta(const GemmWithInput &gwi,
                                    const PropagationTirCollector *collector) {
  auto gemm = gwi.gemm;
  auto *g = gemm.get();
  auto node = tvm::ffi::make_object<MlsGemmDepMetaNode>();
  if (gwi.input.same_as(g->a_)) {
    node->feeds_slot = 0;
    node->trans = !g->transA_;
  } else {
    ICHECK(gwi.input.same_as(g->b_)) << "ds_read_format dst must feed Gemm A or B";
    node->feeds_slot = 1;
    node->trans = g->transB_;
  }
  node->gemm_m = g->m_;
  node->gemm_n = g->n_;
  node->gemm_k = g->k_;
  node->gemm_k_pack = g->kPack_;
  node->gemm_trans_a = g->transA_;
  node->gemm_trans_b = g->transB_;
  if (g->policy_.defined()) {
    node->gemm_policy = g->policy_->policy_type;
  }
  if (collector != nullptr) {
    node->a_from_mls = IsFromMls(g->a_, collector);
    node->b_from_mls = IsFromMls(g->b_, collector);
  }
  return MlsGemmDepMeta(std::move(node));
}

static std::optional<bool> MlsTransFromGemmAndBuffer(const GemmNode *gemm,
                                                     const Buffer &buf) {
  if (gemm->a_.same_as(buf)) {
    return !gemm->transA_;
  }
  if (gemm->b_.same_as(buf)) {
    return gemm->transB_;
  }
  return std::nullopt;
}

bool ComputeSharedDstMlsTrans(const Buffer &dst,
                              const PropagationTirCollector *collector,
                              bool *out_trans) {
  ICHECK(collector != nullptr);
  ICHECK(out_trans != nullptr);
  bool found = false;
  bool mls_trans = true;
  for (const ReaderCallRecord &reader : collector->GetReaderCalls(dst)) {
    ICHECK(reader.call != nullptr);
    auto op = ParseOperator(tvm::ffi::GetRef<Call>(reader.call));
    if (!op.defined()) {
      continue;
    }
    if (auto gemm = op.as<GemmNode>()) {
      ICHECK(gemm->kPack_ == 1)
          << "MatrixLoad dst Gemm consumer must have kPack=1, got " << gemm->kPack_;
      auto cur = MlsTransFromGemmAndBuffer(gemm, dst);
      if (!cur) {
        continue;
      }
      if (!found) {
        mls_trans = *cur;
        found = true;
      } else if (*cur != mls_trans) {
        LOG(FATAL) << "MatrixLoad dst Gemm consumers must have consistent mls_trans";
      }
      continue;
    }
    if (op->GetTypeKey() == std::string("tl.DsReadFormat")) {
      auto ds = Downcast<DsReadFormat>(op);
      auto gemm_with_input = PropagateToFindGemmConsumerOpWithInput(
          ds->dst, collector, reader.stmt_order);
      if (!gemm_with_input) {
        continue;
      }
      auto cur = MlsTransFromGemmAndBuffer(gemm_with_input->gemm.get(),
                                           gemm_with_input->input);
      ICHECK(cur) << "paired Gemm must consume ds_read_format fragment";
      if (!found) {
        mls_trans = *cur;
        found = true;
      } else if (*cur != mls_trans) {
        LOG(FATAL) << "MatrixLoad dst Gemm consumers must have consistent mls_trans";
      }
    }
  }
  if (!found) {
    *out_trans = true;
    return false;
  }
  *out_trans = mls_trans;
  return true;
}

Optional<MlsGemmDepMeta> GetMlsGemmDepFromAnnotations(
    const Map<String, ObjectRef> &annotations) {
  auto val = annotations.Get(attr::kMlsGemmDep);
  if (!val) {
    return Optional<MlsGemmDepMeta>();
  }
  return Downcast<MlsGemmDepMeta>(val.value());
}

Map<String, ObjectRef> AnnotateGemmHcuMlsFlags(const Map<String, ObjectRef> &annotations,
                                               const GemmNode *gemm,
                                               const PropagationTirCollector *collector) {
  Map<String, ObjectRef> out = annotations;
  if (collector == nullptr || gemm == nullptr) {
    return out;
  }
  if (IsFromMls(gemm->a_, collector)) {
    out.Set(attr::kHcuAFromMls, IntImm(DataType::Int(32), 1));
  }
  if (IsFromMls(gemm->b_, collector)) {
    out.Set(attr::kHcuBFromMls, IntImm(DataType::Int(32), 1));
  }
  return out;
}

Optional<bool> GetMlsTransFromAnnotations(const Map<String, ObjectRef> &annotations) {
  auto val = annotations.Get(attr::kMlsTrans);
  if (!val) {
    return Optional<bool>();
  }
  if (const auto *imm = val->as<IntImmNode>()) {
    return Optional<bool>(imm->value != 0);
  }
  LOG(FATAL) << "Annotation " << attr::kMlsTrans << " must be IntImm";
  return Optional<bool>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  MlsGemmDepMetaNode::RegisterReflection();
}

}  // namespace tl
}  // namespace tvm
