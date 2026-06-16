// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include <stdint.h>
#include <tuple>
#include <type_traits>
#include <tl_templates/hcu/core/config.hpp>
#include <tl_templates/hcu/core/numeric/half.hpp>
#include <tl_templates/hcu/core/numeric/bfloat16.hpp>
#include <tl_templates/hcu/core/numeric/float8.hpp>
#include <tl_templates/hcu/core/numeric/int8.hpp>
#include <tl_templates/hcu/core/numeric/mxfp_convert.hpp>

namespace tl {

#if TL_USE_CUSTOM_DATA_TYPE
template <typename Y, typename X>
TL_HOST_DEVICE constexpr remove_cvref_t<Y> type_convert(const X& x)
{
    return static_cast<Y>(x);
}
#else
// Convert X to Y, both X and Y are non-const data types.
template <typename Y,
          typename X,
          std::enable_if_t<!(std::is_const_v<Y> || std::is_const_v<X>), bool> = false>
TL_HOST_DEVICE constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);
    return static_cast<Y>(x);
}

// Convert X to Y, either X or Y is a const data type.
template <typename Y,
          typename X,
          std::enable_if_t<std::is_const_v<Y> || std::is_const_v<X>, bool> = false>
TL_HOST_DEVICE constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    using non_const_y = std::remove_const_t<Y>;
    using non_const_x = std::remove_const_t<X>;
    return static_cast<Y>(type_convert<non_const_y, non_const_x>(x));
}

#define TL_TYPE_CONVERT(dtype_, dname_, stype_, sname_)                    \
    template <>                                                                 \
    TL_HOST_DEVICE constexpr dtype_ type_convert<dtype_, stype_>(stype_ x) \
    {                                                                           \
        return sname_##_to_##dname_(x);                                         \
    }

TL_TYPE_CONVERT(float, float, fp16_t, fp16)
TL_TYPE_CONVERT(float, float, bf16_t, bf16)
TL_TYPE_CONVERT(float, float, fp8_t, fp8)
TL_TYPE_CONVERT(float, float, bf8_t, bf8)

TL_TYPE_CONVERT(fp16_t, fp16, float, float)
TL_TYPE_CONVERT(bf16_t, bf16, float, float)
TL_TYPE_CONVERT(fp8_t, fp8, float, float)
TL_TYPE_CONVERT(bf8_t, bf8, float, float)

TL_TYPE_CONVERT(float, float, int8_t, int8)
TL_TYPE_CONVERT(int8_t, int8, float, float)
#undef TL_TYPE_CONVERT
#endif

} // namespace tl

/*
#include <tl_templates/hcu/core/numeric/pk_fp4.hpp>

namespace tl {

TL_TYPE_CONVERT(pk_fp4_t, pk_fp4, fp32x2_t, fp32x2)
TL_TYPE_CONVERT(fp32x2_t, fp32x2, pk_fp4_t, pk_fp4)
TL_TYPE_CONVERT(pk_fp4_t, pk_fp4, fp16x2_t, fp16x2)
TL_TYPE_CONVERT(fp16x2_t, fp16x2, pk_fp4_t, pk_fp4)
TL_TYPE_CONVERT(pk_fp4_t, pk_fp4, bf16x2_t, bf16x2)
TL_TYPE_CONVERT(bf16x2_t, bf16x2, pk_fp4_t, pk_fp4)
TL_TYPE_CONVERT(pk_fp4_t, pk_fp4, float, float)
TL_TYPE_CONVERT(pk_fp4_t, pk_fp4, bf16_t, bf16)
TL_TYPE_CONVERT(pk_fp4_t, pk_fp4, fp16_t, fp16)
#undef TL_TYPE_CONVERT
#endif

} // namespace tl
*/
