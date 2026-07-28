// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

/*
 * DsreadmFormatDispatcher: tilelang port of ck_tile
 * WarpDsreadmFormatDispatcherV2. Uses __builtin_hcu_ds_read_matrix_format_*
 * instead of inline asm. Naming: removed "Warp" and "V2".
 */

#include <tl_templates/hcu/core.hpp>

#include <tl_templates/hcu/mls/tl_dsreadm_format_attribute_impl.hpp>

namespace tl {
namespace mls {
namespace impl {

template <::tl::index_t ElemBytes, ::tl::index_t Row, ::tl::index_t Col,
          ::tl::index_t Alt, bool Trans>
struct DsreadmFormatDispatcher;

// M32x16 B16 non-trans Alt1
template <> struct DsreadmFormatDispatcher<2, 32, 16, 1, false> {
  using Type = DsreadmFormatAttribute<DsreadmFormatAttributeImpl_M32x16_B16>;
};

// M32x16 B16 non-trans Alt2
template <> struct DsreadmFormatDispatcher<2, 32, 16, 2, false> {
  using Type =
      DsreadmFormatAttribute<DsreadmFormatAttributeImpl_M32x16_B16_ALT2>;
};

// MT32x16 B16 trans Alt1
template <> struct DsreadmFormatDispatcher<2, 32, 16, 1, true> {
  using Type = DsreadmFormatAttribute<DsreadmFormatAttributeImpl_MT32x16_B16>;
};

// MT16x32 B16 trans Alt2
template <> struct DsreadmFormatDispatcher<2, 16, 32, 2, true> {
  using Type =
      DsreadmFormatAttribute<DsreadmFormatAttributeImpl_MT16x32_B16_ALT2>;
};

// ---------------------------
// DS_READ_MATRIX_FORMAT_B8
// ---------------------------

// DS_S_READ_M32X32_B8
template <> struct DsreadmFormatDispatcher<1, 32, 32, 1, false> {
  using Type = DsreadmFormatAttribute<DsreadmFormatAttributeImpl_M32x32_B8>;
};

// DS_S_READ_M32X32_B8_ALT2
template <> struct DsreadmFormatDispatcher<1, 32, 32, 2, false> {
  using Type =
      DsreadmFormatAttribute<DsreadmFormatAttributeImpl_M32x32_B8_ALT2>;
};

// DS_S_READ_M64X16_B8_ALT4
template <> struct DsreadmFormatDispatcher<1, 64, 16, 4, false> {
  using Type =
      DsreadmFormatAttribute<DsreadmFormatAttributeImpl_M64x16_B8_ALT4>;
};

// --------------------------------
// DS_READ_MATRIX_TRANS_FORMAT_B8
// --------------------------------

// DS_S_READ_MT64X16_B8
template <> struct DsreadmFormatDispatcher<1, 16, 64, 1, true> {
  using Type = DsreadmFormatAttribute<DsreadmFormatAttributeImpl_MT16x64_B8>;
};

// DS_S_READ_MT32X32_B8_ALT2
template <> struct DsreadmFormatDispatcher<1, 32, 32, 2, true> {
  using Type =
      DsreadmFormatAttribute<DsreadmFormatAttributeImpl_MT32x32_B8_ALT2>;
};

} // namespace impl

template <::tl::index_t ElemBytes, ::tl::index_t Row, ::tl::index_t Col,
          ::tl::index_t Alt, bool Trans>
using DsreadmFormatDispatcher =
    typename impl::DsreadmFormatDispatcher<ElemBytes, Row, Col, Alt,
                                           Trans>::Type;

} // namespace mls
} // namespace tl
