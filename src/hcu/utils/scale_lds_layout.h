// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file scale_lds_layout.h
 * \brief Automatic bank-aware Scale LDS layout selection.
 */
#ifndef TVM_TL_HCU_UTILS_SCALE_LDS_LAYOUT_H_
#define TVM_TL_HCU_UTILS_SCALE_LDS_LAYOUT_H_

#include "hcu/op/scale_format.h"
#include "op/operator.h"
#include "tl_templates/hcu/scale/scale_row_layout.hpp"
#include <array>
#include <optional>

namespace tvm {
namespace tl {
class CopyScaleNode;
namespace hcu {

std::optional<std::array<int, 3>>
ScoreScaleRowLayout(const CopyScaleNode *op,
                    const ::tl::hcu::ScaleRowLayoutDesc &layout, int block_size,
                    Target target,
                    ffi::Optional<Layout> resolved_layout = std::nullopt);

ffi::Optional<Layout> SelectAutoScaleLdsLayout(const CopyScaleNode *op,
                                               const LayoutInferArgs &args);

ScaleBufferRowOrder SelectScaleBufferRowOrder(const CopyScaleNode *op,
                                              int block_size, Target target);

} // namespace hcu
} // namespace tl
} // namespace tvm
#endif // TVM_TL_HCU_UTILS_SCALE_LDS_LAYOUT_H_
