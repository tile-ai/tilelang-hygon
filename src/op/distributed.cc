/*!
 * \file tl/op/distributed.cc
 * \brief Backend-neutral distributed intrinsics.
 */

#include "distributed.h"

#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {

using namespace tirx;

#define TIR_DEFINE_TL_DISTRIBUTED_BUILTIN(OpName)                              \
  const Op &OpName() {                                                          \
    static const Op &op = Op::Get("tl." #OpName);                              \
    return op;                                                                  \
  }                                                                              \
  TVM_REGISTER_OP("tl." #OpName)                                               \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)

TIR_DEFINE_TL_DISTRIBUTED_BUILTIN(get_rank)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_DISTRIBUTED_BUILTIN(get_num_ranks)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm
