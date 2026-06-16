// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include <tl_templates/hcu/core/config.hpp>
#include <tl_templates/hcu/core/numeric/integer.hpp>

namespace tl {

template <auto v>
struct constant
{
    using value_type                  = decltype(v);
    using type                        = constant; // using injected-class-name
    static constexpr value_type value = v;
    TL_HOST_DEVICE constexpr operator value_type() const noexcept { return value; }
    TL_HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
    TL_HOST_DEVICE static constexpr bool is_static() { return true; }
};

template <auto v>
TL_HOST_DEVICE static void print(const constant<v>&)
{
    printf("%ld", static_cast<long>(v));
}

template <typename T, T v>
struct integral_constant : constant<v>
{
    using value_type         = T;
    using type               = integral_constant; // using injected-class-name
    static constexpr T value = v;
    // constexpr TL_HOST_DEVICE operator   value_type() const noexcept { return value; }
    // constexpr TL_HOST_DEVICE value_type operator()() const noexcept { return value; } //
};

template <index_t v>
using number = constant<v>;

template <long_index_t v>
using long_number = constant<v>;

template <bool b>
using bool_constant = constant<b>;

#define TL_LEFT_UNARY_OP(OP)                               \
    template <auto x>                                           \
    TL_HOST_DEVICE constexpr auto operator OP(constant<x>) \
    {                                                           \
        return constant<(OP x)>{};                              \
    }

#define TL_BINARY_OP(OP)                                                \
    template <auto x, auto y>                                                \
    TL_HOST_DEVICE constexpr auto operator OP(constant<x>, constant<y>) \
    {                                                                        \
        return constant<(x OP y)>{};                                         \
    }

TL_LEFT_UNARY_OP(+)
TL_LEFT_UNARY_OP(-)
TL_LEFT_UNARY_OP(~)
TL_LEFT_UNARY_OP(!)

TL_BINARY_OP(+)
TL_BINARY_OP(-)
TL_BINARY_OP(*)
TL_BINARY_OP(/)
TL_BINARY_OP(%)
TL_BINARY_OP(&)
TL_BINARY_OP(|)
TL_BINARY_OP(^)
TL_BINARY_OP(<<)
TL_BINARY_OP(>>)
TL_BINARY_OP(&&)
TL_BINARY_OP(||)
TL_BINARY_OP(==)
TL_BINARY_OP(!=)
TL_BINARY_OP(>)
TL_BINARY_OP(<)
TL_BINARY_OP(>=)
TL_BINARY_OP(<=)

#undef TL_LEFT_UNARY_OP
#undef TL_BINARY_OP

template <typename T>
struct is_constant : std::false_type
{
};
template <auto v>
struct is_constant<constant<v>> : std::true_type
{
};
template <typename T>
inline constexpr bool is_constant_v = is_constant<T>::value;
} // namespace tl
