// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file scale_lds_layout.h
 * \brief Automatic bank-aware Scale LDS layout selection.
 */
#ifndef TVM_TL_HCU_UTILS_SCALE_LDS_LAYOUT_H_
#define TVM_TL_HCU_UTILS_SCALE_LDS_LAYOUT_H_

#include "op/operator.h"

namespace tvm {
namespace tl {
class CopyScaleNode;
namespace hcu {

ffi::Optional<Layout> SelectAutoScaleLdsLayout(const CopyScaleNode *op,
                                               const LayoutInferArgs &args);

} // namespace hcu
} // namespace tl
} // namespace tvm
#endif // TVM_TL_HCU_UTILS_SCALE_LDS_LAYOUT_H_
