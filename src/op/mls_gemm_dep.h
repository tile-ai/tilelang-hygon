/*!
 * \file mls_gemm_dep.h
 * \brief Annotated consumer-GEMM snapshot for matrix_load / ds_read_format.
 */

#ifndef TVM_TL_OP_MLS_GEMM_DEP_H_
#define TVM_TL_OP_MLS_GEMM_DEP_H_

#include <optional>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>

#include "propagation_tir_collector.h"
#include "propagation_util.h"

namespace tvm {
namespace tl {

namespace attr {
static constexpr const char *kMlsGemmDep = "tl.mls_gemm_dep";
static constexpr const char *kMlsTrans = "tl.mls_trans";
static constexpr const char *kHcuAFromMls = "tl.a_from_mls";
static constexpr const char *kHcuBFromMls = "tl.b_from_mls";
} // namespace attr

/*!
 * \brief Minimal GEMM-side facts needed by ds_read_format InferLayout/Lower.
 *
 * feeds_slot: 0 = feeds Gemm A, 1 = feeds Gemm B.
 * trans: MLS/ds_read_format trans for this site (!gemm_trans_a or
 * gemm_trans_b).
 */
class MlsGemmDepMetaNode : public Object {
public:
  int feeds_slot{0};
  bool trans{true};
  int gemm_m{0};
  int gemm_n{0};
  int gemm_k{0};
  int gemm_k_pack{1};
  bool gemm_trans_a{false};
  bool gemm_trans_b{false};
  bool a_from_mls{false};
  bool b_from_mls{false};
  int gemm_policy{0};

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.MlsGemmDepMeta", MlsGemmDepMetaNode,
                                    Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MlsGemmDepMetaNode>()
        .def_ro("feeds_slot", &MlsGemmDepMetaNode::feeds_slot)
        .def_ro("trans", &MlsGemmDepMetaNode::trans)
        .def_ro("gemm_m", &MlsGemmDepMetaNode::gemm_m)
        .def_ro("gemm_n", &MlsGemmDepMetaNode::gemm_n)
        .def_ro("gemm_k", &MlsGemmDepMetaNode::gemm_k)
        .def_ro("gemm_k_pack", &MlsGemmDepMetaNode::gemm_k_pack)
        .def_ro("gemm_trans_a", &MlsGemmDepMetaNode::gemm_trans_a)
        .def_ro("gemm_trans_b", &MlsGemmDepMetaNode::gemm_trans_b)
        .def_ro("a_from_mls", &MlsGemmDepMetaNode::a_from_mls)
        .def_ro("b_from_mls", &MlsGemmDepMetaNode::b_from_mls)
        .def_ro("gemm_policy", &MlsGemmDepMetaNode::gemm_policy);
  }
};

class MlsGemmDepMeta : public ObjectRef {
public:
  explicit MlsGemmDepMeta(ObjectPtr<MlsGemmDepMetaNode> ptr)
      : ObjectRef(std::move(ptr)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MlsGemmDepMeta, ObjectRef,
                                                MlsGemmDepMetaNode);
};

MlsGemmDepMeta BuildMlsGemmDepMeta(const GemmWithInput &gwi,
                                   const PropagationTirCollector *collector);

bool ComputeSharedDstMlsTrans(const Buffer &dst,
                              const PropagationTirCollector *collector,
                              Target target, bool *out_trans);

Optional<MlsGemmDepMeta>
GetMlsGemmDepFromAnnotations(const Map<String, ObjectRef> &annotations);

Optional<bool>
GetMlsTransFromAnnotations(const Map<String, ObjectRef> &annotations);

class GemmNode;
Map<String, ObjectRef>
AnnotateGemmHcuMlsFlags(const Map<String, ObjectRef> &annotations,
                        const GemmNode *gemm,
                        const PropagationTirCollector *collector);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_MLS_GEMM_DEP_H_
