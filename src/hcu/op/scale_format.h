// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file scale_format.h
 * \brief Scale LDS format identifiers shared by HCU scale/GEMM helpers.
 */
#ifndef TVM_TL_HCU_OP_SCALE_FORMAT_H_
#define TVM_TL_HCU_OP_SCALE_FORMAT_H_

namespace tvm {
namespace tl {

enum class ScaleLdsFormat : int {
  kIdentity = 0,
  kK2Interleave = 1,
  kK4Interleave = 2,
  kK2MN2Interleave = 3,
  kMN2Interleave = 4,
  kMN4Interleave = 5,
};

constexpr int ScaleLdsFormatId(ScaleLdsFormat format) {
  return static_cast<int>(format);
}

constexpr bool ScaleFormatRequiresMmacK32(ScaleLdsFormat format) {
  return format == ScaleLdsFormat::kMN2Interleave ||
         format == ScaleLdsFormat::kMN4Interleave;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_OP_SCALE_FORMAT_H_
