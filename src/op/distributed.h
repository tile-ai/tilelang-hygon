/*!
 * \file tl/op/distributed.h
 * \brief Backend-neutral distributed intrinsics.
 */

#ifndef TVM_TL_OP_DISTRIBUTED_H_
#define TVM_TL_OP_DISTRIBUTED_H_

#include <tvm/ir/op.h>

namespace tvm {
namespace tl {

const Op &get_rank();
const Op &get_num_ranks();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_DISTRIBUTED_H_
