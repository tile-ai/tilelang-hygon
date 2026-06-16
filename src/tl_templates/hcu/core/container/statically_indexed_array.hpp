// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include <tl_templates/hcu/core/config.hpp>
#include <tl_templates/hcu/core/container/array.hpp>
#include <tl_templates/hcu/core/container/tuple.hpp>
#include <tl_templates/hcu/core/numeric/integer.hpp>

namespace tl {

#if TL_STATICALLY_INDEXED_ARRAY_DEFAULT == TL_STATICALLY_INDEXED_ARRAY_USE_TUPLE

template <typename T, index_t N>
using statically_indexed_array = tuple_array<T, N>;

#else

// consider mark this struct as deprecated
template <typename T, index_t N>
using statically_indexed_array = array<T, N>;

#endif

// consider always use tl::array for this purpose
#if 0
template <typename X, typename... Xs>
TL_HOST_DEVICE constexpr auto make_statically_indexed_array(const X& x, const Xs&... xs)
{
    return statically_indexed_array<X, sizeof...(Xs) + 1>(x, static_cast<X>(xs)...);
}

// make empty statically_indexed_array
template <typename X>
TL_HOST_DEVICE constexpr auto make_statically_indexed_array()
{
    return statically_indexed_array<X, 0>();
}
#endif
} // namespace tl
