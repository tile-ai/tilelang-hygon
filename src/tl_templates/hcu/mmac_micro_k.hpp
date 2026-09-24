// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

namespace tl {

// Per-arch per-dtype K extent for one MMAC along the reduction axis.
// Shared by gemm.h (MmacTraits) and ds_read_format fragment scatter.
template <typename T> struct MmacMicroKDim {
  static constexpr int value = 32 / sizeof(T);
};

template <> struct MmacMicroKDim<float> {
#if defined(__gfx92a__) || defined(__gfx946__)
  static constexpr int value = 4;
#elif defined(__gfx938__)
  static constexpr int value = 8;
#else
  static constexpr int value = 8;
#endif
};

} // namespace tl
