// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#include <tl_templates/hcu/core/config.hpp>
#include <tl_templates/hcu/core/numeric/half.hpp>
#include <tl_templates/hcu/core/numeric/integral_constant.hpp>
#include <tl_templates/hcu/core/numeric/math.hpp>
#include <tl_templates/hcu/core/numeric/numeric.hpp>
#include <tl_templates/hcu/core/utility/bit_cast.hpp>
#include <tl_templates/hcu/core/utility/random.hpp>
#include <stdint.h>
#include <type_traits>

#pragma once

namespace tl {

// use int8_t directly for int8 arithemetic
// here one can use tl::int8_t to access original int8_t
using int8_t = int8_t;

// limits
template <class T>
struct numeric;

template <>
struct numeric<int8_t>
{
    // minimum finite value, or minimum positive normalized value for float
    TL_HOST_DEVICE static constexpr int8_t min() { return int8_t(-128); }

    // minumum finite value
    TL_HOST_DEVICE static constexpr int8_t lowest() { return int8_t(-128); }

    // maximum finite value
    TL_HOST_DEVICE static constexpr int8_t max() { return int8_t(127); }

    // difference between 1.0 and next value representable by float
    TL_HOST_DEVICE static constexpr int8_t epsilon()
    {
        return 1; // not used
    }

    TL_HOST_DEVICE static constexpr int8_t round_error()
    {
        return 1; // not used
    }

    // positive infinity value
    TL_HOST_DEVICE static constexpr int8_t infinity()
    {
        return 1; // not used
    }

    // quiet NaN
    TL_HOST_DEVICE static constexpr int8_t quiet_NaN()
    {
        return 1; // not used
    }

    // signaling NaN
    TL_HOST_DEVICE static constexpr int8_t signaling_NaN()
    {
        return 1; // not used
    }

    // smallest positive subnormal value
    TL_HOST_DEVICE static constexpr int8_t denorm_min()
    {
        return 1; // not used
    }

    TL_HOST_DEVICE static constexpr int8_t zero() { return 0; }
};

#if 0

template <>
struct numeric_traits<int8_t>
{
    static constexpr int exp            = 5;
    static constexpr int mant           = 10;
    static constexpr int bias           = 15;
    static constexpr uint16_t nan_mask  = 0x7C00;
    static constexpr uint16_t head_mask = 0xFC00;
    static constexpr uint16_t mant_mask = 0x3FF;
    static constexpr uint16_t exp_mask  = 0x1F;
    static constexpr uint32_t Inf       = 0x7C00;
    static constexpr uint32_t NegInf    = 0xFC00;
    static constexpr uint32_t NaN       = 0x7C01;
    static constexpr uint32_t Neg0      = 0x8000;
    static constexpr int PackedSize           = 1;
    using bitwise_type                  = uint16_t;
};
#endif

TL_HOST_DEVICE
constexpr float int8_to_float(const int8_t& x) { return static_cast<float>(x); }

TL_HOST_DEVICE
constexpr int8_t float_to_int8(const float& x) { return static_cast<int8_t>(x); }

} // namespace tl
