#pragma once

#include "common.h"

#define HIP_FP8_ENABLED 1

using fp8_e4_t = ck_tile::fp8_t;
using fp8_e4_2_t = ck_tile::fp8x2_t;
using fp8_e4_4_t = ck_tile::fp8x4_t;
using fp8_e4_8_t = ck_tile::fp8x8_t;
using fp8_e4_16_t = ck_tile::fp8x16_t;

using fp8_e5_t = ck_tile::bf8_t;
using fp8_e5_2_t = ck_tile::bf8x2_t;
using fp8_e5_4_t = ck_tile::bf8x4_t;
using fp8_e5_8_t = ck_tile::bf8x8_t;
using fp8_e5_16_t = ck_tile::bf8x16_t;

// Tilelang's own fp8_e4_t (FP8 E4M3) conversion class
// This provides true conversion with float and supports conversion to ck_tile::fp8_t
struct __align__(1) fp8_cvt_t {
  using raw_type = uint8_t;
  raw_type data;

  // Static method for bit_cast
  TL_DEVICE static constexpr fp8_cvt_t bit_cast(raw_type x) {
    fp8_cvt_t y;
    y.data = x;
    return y;
  }

  // Default constructor
  TL_DEVICE constexpr fp8_cvt_t() : data() {}

  // Construct from float - use ck_tile's conversion function
  TL_DEVICE explicit constexpr fp8_cvt_t(const float& x)
    : data(ck_tile::float_to_fp8_raw(x)) {}

  // Construct from double - convert to float first, then to fp8
  TL_DEVICE explicit constexpr fp8_cvt_t(const double& x)
    : data(ck_tile::float_to_fp8_raw(static_cast<float>(x))) {}

  // Construct from int
  TL_DEVICE explicit constexpr fp8_cvt_t(const int& x)
    : data(ck_tile::float_to_fp8_raw(static_cast<float>(x))) {}

  // Construct from unsigned int
  TL_DEVICE explicit constexpr fp8_cvt_t(const unsigned int& x)
    : data(ck_tile::float_to_fp8_raw(static_cast<float>(x))) {}

  // Construct from ck_tile::fp8_t
  // Direct bit_cast since ck_tile::fp8_t may be uint8_t or _BitInt(8)
  TL_DEVICE constexpr fp8_cvt_t(const ck_tile::fp8_t& v)
    : data(ck_tile::bit_cast<uint8_t>(v)) {}

  // Cast to float - use ck_tile's conversion function
  TL_DEVICE explicit constexpr operator float() const {
    return ck_tile::fp8_to_float_raw(data);
  }

  // Cast to double - convert through float
  TL_DEVICE explicit constexpr operator double() const {
    return static_cast<double>(ck_tile::fp8_to_float_raw(data));
  }

  // Cast to int
  TL_DEVICE explicit constexpr operator int() const {
    return static_cast<int>(ck_tile::fp8_to_float_raw(data));
  }

  // Conversion to ck_tile::fp8_t
  // This allows seamless interoperability with ck_tile
  TL_DEVICE constexpr operator ck_tile::fp8_t() const {
    return ck_tile::bit_cast<ck_tile::fp8_t>(data);
  }

  // Internal access
  TL_DEVICE constexpr raw_type& get() { return data; }
  TL_DEVICE constexpr raw_type get() const { return data; }
};

// Tilelang's own bf8_t (BF8 E5M2) conversion class
// This provides true conversion with float and supports conversion to ck_tile::bf8_t
struct __align__(1) bf8_cvt_t {
  using raw_type = uint8_t;
  raw_type data;

  // Static method for bit_cast
  TL_DEVICE static constexpr bf8_cvt_t bit_cast(raw_type x) {
    bf8_cvt_t y;
    y.data = x;
    return y;
  }

  // Default constructor
  TL_DEVICE constexpr bf8_cvt_t() : data() {}

  // Construct from float - use ck_tile's conversion function
  TL_DEVICE explicit constexpr bf8_cvt_t(const float& x)
    : data(ck_tile::float_to_bf8_raw(x)) {}

  // Construct from double - convert to float first, then to bf8
  TL_DEVICE explicit constexpr bf8_cvt_t(const double& x)
    : data(ck_tile::float_to_bf8_raw(static_cast<float>(x))) {}

  // Construct from int
  TL_DEVICE explicit constexpr bf8_cvt_t(const int& x)
    : data(ck_tile::float_to_bf8_raw(static_cast<float>(x))) {}

  // Construct from unsigned int
  TL_DEVICE explicit constexpr bf8_cvt_t(const unsigned int& x)
    : data(ck_tile::float_to_bf8_raw(static_cast<float>(x))) {}

  // Construct from ck_tile::bf8_t
  // Direct bit_cast since ck_tile::bf8_t may be uint8_t or unsigned _BitInt(8)
  TL_DEVICE constexpr bf8_cvt_t(const ck_tile::bf8_t& v)
    : data(ck_tile::bit_cast<uint8_t>(v)) {}

  // Cast to float - use ck_tile's conversion function
  TL_DEVICE explicit constexpr operator float() const {
    return ck_tile::bf8_to_float_raw(data);
  }

  // Cast to double - convert through float
  TL_DEVICE explicit constexpr operator double() const {
    return static_cast<double>(ck_tile::bf8_to_float_raw(data));
  }

  // Cast to int
  TL_DEVICE explicit constexpr operator int() const {
    return static_cast<int>(ck_tile::bf8_to_float_raw(data));
  }

  // Conversion to ck_tile::bf8_t
  // This allows seamless interoperability with ck_tile
  TL_DEVICE constexpr operator ck_tile::bf8_t() const {
    return ck_tile::bit_cast<ck_tile::bf8_t>(data);
  }

  // Internal access
  TL_DEVICE constexpr raw_type& get() { return data; }
  TL_DEVICE constexpr raw_type get() const { return data; }
};

// Pack four fp8_e4_t values into fp8_e4_4_t
// Similar to HIP version, using int instead of uint32_t for consistency
TL_DEVICE fp8_e4_4_t make_fp8_e4_4_t(fp8_e4_t x, fp8_e4_t y, fp8_e4_t z,
                                      fp8_e4_t w) {
  // Reinterpret the 4 fp8_e4_t values to uint8_t for bit manipulation
  uint8_t x_val = ck_tile::bit_cast<uint8_t>(x);
  uint8_t y_val = ck_tile::bit_cast<uint8_t>(y);
  uint8_t z_val = ck_tile::bit_cast<uint8_t>(z);
  uint8_t w_val = ck_tile::bit_cast<uint8_t>(w);

  // Pack into int (matching HIP version which uses int)
  // Cast to int for bit operations (same as HIP's signed char approach)
  int res = (static_cast<int>(w_val) << 24) |
            (static_cast<int>(z_val) << 16) |
            (static_cast<int>(y_val) << 8) |
            static_cast<int>(x_val);

  // Reinterpret as fp8_e4_4_t (ck_tile::fp8x4_t)
  return ck_tile::bit_cast<fp8_e4_4_t>(res);
}

// Pack eight fp8_e4_t values into fp8_e4_8_t
TL_DEVICE fp8_e4_8_t make_fp8_e4_8_t(fp8_e4_t x, fp8_e4_t y, fp8_e4_t z,
                                      fp8_e4_t w, fp8_e4_t v, fp8_e4_t u,
                                      fp8_e4_t t, fp8_e4_t s) {
  // Reinterpret the 8 fp8_e4_t values to uint8_t for bit manipulation
  uint8_t x_val = ck_tile::bit_cast<uint8_t>(x);
  uint8_t y_val = ck_tile::bit_cast<uint8_t>(y);
  uint8_t z_val = ck_tile::bit_cast<uint8_t>(z);
  uint8_t w_val = ck_tile::bit_cast<uint8_t>(w);
  uint8_t v_val = ck_tile::bit_cast<uint8_t>(v);
  uint8_t u_val = ck_tile::bit_cast<uint8_t>(u);
  uint8_t t_val = ck_tile::bit_cast<uint8_t>(t);
  uint8_t s_val = ck_tile::bit_cast<uint8_t>(s);

  // Pack first 4 values into int (matching HIP version which uses int)
  int a = (static_cast<int>(w_val) << 24) |
          (static_cast<int>(z_val) << 16) |
          (static_cast<int>(y_val) << 8) |
          static_cast<int>(x_val);

  // Pack last 4 values into int
  int b = (static_cast<int>(s_val) << 24) |
          (static_cast<int>(t_val) << 16) |
          (static_cast<int>(u_val) << 8) |
          static_cast<int>(v_val);

  ck_tile::int32x2_t packed = {a, b};

  // Reinterpret as fp8_e4_8_t (ck_tile::fp8x8_t)
  return ck_tile::bit_cast<fp8_e4_8_t>(packed);
}
