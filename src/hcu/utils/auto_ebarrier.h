// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file hcu/utils/auto_ebarrier.h
 * \brief Marker used to defer HCU EBarrier allocation until ThreadSync.
 */
#ifndef TVM_TL_HCU_UTILS_AUTO_EBARRIER_H_
#define TVM_TL_HCU_UTILS_AUTO_EBARRIER_H_

namespace tvm {
namespace tl {
namespace hcu {

static constexpr const char *kAutoEBarrierPolicyMarker =
    "tl::__hcu_auto_ebarrier_policy";

} // namespace hcu
} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_UTILS_AUTO_EBARRIER_H_
