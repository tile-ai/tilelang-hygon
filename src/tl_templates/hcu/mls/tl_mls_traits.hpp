#pragma once

/*
 * mls_traits for tilelang MlsAtom types. Ported from ck_tile hcu_mls_traits_gfx938.
 * Used by mls_generic_detail and make_lds_desc_generic.
 */

#include <ck_tile/core.hpp>

#include "tl_mls_atom_gfx938.hpp"

namespace tl {
namespace mls {

template <typename MlsAtom, ck_tile::index_t Alt>
struct mls_traits;

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx938_mls_32x16_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<32>{};
    static constexpr auto kK  = ck_tile::number<16>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK, kMN);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx938_mls_16x32_trans_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<16>{};
    static constexpr auto kK  = ck_tile::number<32>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN, kK);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx938_mls_32x32_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<32>{};
    static constexpr auto kK  = ck_tile::number<32>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK, kMN);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx938_mls_32x32_trans_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<32>{};
    static constexpr auto kK  = ck_tile::number<32>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN, kK);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx938_mls_64x16_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");
    static constexpr auto kMN0 = ck_tile::number<2>{};
    static constexpr auto kMN1 = ck_tile::number<32>{};
    static constexpr auto kK   = ck_tile::number<16>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN0, kK, kMN1);
};

template <>
struct mls_traits<tl::mls::gfx938_mls_16x64_trans_b16, 1>
{
    static constexpr auto kMN = ck_tile::number<16>{};
    static constexpr auto kK0 = ck_tile::number<2>{};
    static constexpr auto kK1 = ck_tile::number<32>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK0, kMN, kK1);
};

template <>
struct mls_traits<tl::mls::gfx938_mls_16x64_trans_b16, 2>
{
    static constexpr auto kMN = ck_tile::number<16>{};
    static constexpr auto kK0 = ck_tile::number<2>{};
    static constexpr auto kK1 = ck_tile::number<32>{};
    static constexpr auto kSlots = ck_tile::number<2>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK0, kSlots, kMN, kK1);
};

template <>
struct mls_traits<tl::mls::gfx938_mls_64x16_b8, 1>
{
    static constexpr auto kMN = ck_tile::number<64>{};
    static constexpr auto kK  = ck_tile::number<16>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK, kMN);
};

template <>
struct mls_traits<tl::mls::gfx938_mls_64x16_b8, 2>
{
    static constexpr auto kMN = ck_tile::number<64>{};
    static constexpr auto kK0 = ck_tile::number<2>{};
    static constexpr auto kK1 = ck_tile::number<8>{};
    static constexpr auto kSlots = ck_tile::number<2>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK0, kSlots, kK1, kMN);
};

template <>
struct mls_traits<tl::mls::gfx938_mls_64x16_b8, 4>
{
    static constexpr auto kMN = ck_tile::number<64>{};
    static constexpr auto kK0 = ck_tile::number<2>{};
    static constexpr auto kK1 = ck_tile::number<8>{};
    static constexpr auto kSlots = ck_tile::number<2>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK0, kSlots, kK1, kMN);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx938_mls_16x64_trans_b8, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<16>{};
    static constexpr auto kK  = ck_tile::number<64>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN, kK);
};

template <>
struct mls_traits<tl::mls::gfx938_mls_16x64_trans_b8, 4>
{
    static constexpr auto kMN0 = ck_tile::number<2>{};
    static constexpr auto kMN1 = ck_tile::number<8>{};
    static constexpr auto kK   = ck_tile::number<64>{};
    static constexpr auto kSlots = ck_tile::number<4>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN0, kSlots, kMN1, kK);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx938_mls_64x32_b8, Alt>
{
    static_assert(Alt == 1 || Alt == 2 || Alt == 4, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<64>{};
    static constexpr auto kK  = ck_tile::number<32>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK, kMN);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx938_mls_32x64_trans_b8, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<32>{};
    static constexpr auto kK  = ck_tile::number<64>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN, kK);
};

template <>
struct mls_traits<tl::mls::gfx938_mls_32x64_trans_b8, 4>
{
    static constexpr auto kMN0 = ck_tile::number<4>{};
    static constexpr auto kMN1 = ck_tile::number<8>{};
    static constexpr auto kK   = ck_tile::number<64>{};
    static constexpr auto kSlots = ck_tile::number<2>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN0, kSlots, kMN1, kK);
};

template <>
struct mls_traits<tl::mls::gfx938_mls_128x16_b8, 1>
{
    static constexpr auto kMN0 = ck_tile::number<2>{};
    static constexpr auto kMN1 = ck_tile::number<64>{};
    static constexpr auto kK   = ck_tile::number<16>{};
    static constexpr auto kSlots = ck_tile::number<2>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN0, kSlots, kK, kMN1);
};

template <>
struct mls_traits<tl::mls::gfx938_mls_128x16_b8, 2>
{
    static constexpr auto kMN0 = ck_tile::number<2>{};
    static constexpr auto kMN1 = ck_tile::number<64>{};
    static constexpr auto kK0  = ck_tile::number<2>{};
    static constexpr auto kK1  = ck_tile::number<8>{};
    static constexpr auto kSlots = ck_tile::number<2>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN0, kK0, kSlots, kK1, kMN1);
};

template <>
struct mls_traits<tl::mls::gfx938_mls_128x16_b8, 4>
{
    static constexpr auto kMN0 = ck_tile::number<2>{};
    static constexpr auto kMN1 = ck_tile::number<64>{};
    static constexpr auto kK0  = ck_tile::number<2>{};
    static constexpr auto kK1  = ck_tile::number<8>{};
    static constexpr auto kSlots = ck_tile::number<2>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN0, kK0, kSlots, kK1, kMN1);
};

template <>
struct mls_traits<tl::mls::gfx938_mls_16x128_trans_b8, 1>
{
    static constexpr auto kMN = ck_tile::number<16>{};
    static constexpr auto kK0 = ck_tile::number<2>{};
    static constexpr auto kK1 = ck_tile::number<64>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK0, kMN, kK1);
};

template <>
struct mls_traits<tl::mls::gfx938_mls_16x128_trans_b8, 2>
{
    static constexpr auto kMN = ck_tile::number<16>{};
    static constexpr auto kK0 = ck_tile::number<2>{};
    static constexpr auto kK1 = ck_tile::number<64>{};
    static constexpr auto kSlots = ck_tile::number<2>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK0, kSlots, kMN, kK1);
};

template <>
struct mls_traits<tl::mls::gfx938_mls_16x128_trans_b8, 4>
{
    static constexpr auto kMN = ck_tile::number<16>{};
    static constexpr auto kK0 = ck_tile::number<2>{};
    static constexpr auto kK1 = ck_tile::number<64>{};
    static constexpr auto kSlots = ck_tile::number<2>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK0, kSlots, kMN, kK1);
};

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx946__)
#include "tl_mls_atom_gfx946.hpp"

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx946_mls_32x16_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<32>{};
    static constexpr auto kK  = ck_tile::number<16>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK, kMN);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx946_mls_16x32_trans_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<16>{};
    static constexpr auto kK  = ck_tile::number<32>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN, kK);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx946_mls_32x32_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<32>{};
    static constexpr auto kK  = ck_tile::number<32>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK, kMN);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx946_mls_32x32_trans_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<32>{};
    static constexpr auto kK  = ck_tile::number<32>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN, kK);
};

// gfx946 64x16_b16: differs from gfx938 - has kSlots=4, PackedShape=(kMN0, kSlots, kK, kMN1)
template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx946_mls_64x16_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");
    static constexpr auto kMN0 = ck_tile::number<2>{};
    static constexpr auto kMN1 = ck_tile::number<32>{};
    static constexpr auto kK   = ck_tile::number<16>{};
    static constexpr auto kSlots = ck_tile::number<4>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN0, kSlots, kK, kMN1);
};

// gfx946 16x64_trans_b16: differs from gfx938 - both Alt 1,2 use kSlots=4, PackedShape=(kK0, kSlots, kMN, kK1)
template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx946_mls_16x64_trans_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<16>{};
    static constexpr auto kK0 = ck_tile::number<2>{};
    static constexpr auto kK1 = ck_tile::number<32>{};
    static constexpr auto kSlots = ck_tile::number<4>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK0, kSlots, kMN, kK1);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx946_mls_64x16_b8, Alt>
{
    static_assert(Alt == 1 || Alt == 2 || Alt == 4, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<64>{};
    static constexpr auto kK  = ck_tile::number<16>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK, kMN);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx946_mls_16x64_trans_b8, Alt>
{
    static_assert(Alt == 1 || Alt == 2 || Alt == 4, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<16>{};
    static constexpr auto kK  = ck_tile::number<64>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN, kK);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx946_mls_64x32_b8, Alt>
{
    static_assert(Alt == 1 || Alt == 2 || Alt == 4, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<64>{};
    static constexpr auto kK  = ck_tile::number<32>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK, kMN);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx946_mls_32x64_trans_b8, Alt>
{
    static_assert(Alt == 1 || Alt == 2 || Alt == 4, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<32>{};
    static constexpr auto kK  = ck_tile::number<64>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN, kK);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx946_mls_128x16_b8, Alt>
{
    static_assert(Alt == 1 || Alt == 2 || Alt == 4, "Unsupported interleave config");
    static constexpr auto kMN0 = ck_tile::number<2>{};
    static constexpr auto kK   = ck_tile::number<16>{};
    static constexpr auto kMN1 = ck_tile::number<64>{};
    static constexpr auto kSlots = ck_tile::number<4>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kMN0, kSlots, kK, kMN1);
};

template <ck_tile::index_t Alt>
struct mls_traits<tl::mls::gfx946_mls_16x128_trans_b8, Alt>
{
    static_assert(Alt == 1 || Alt == 2 || Alt == 4, "Unsupported interleave config");
    static constexpr auto kMN = ck_tile::number<16>{};
    static constexpr auto kK0 = ck_tile::number<2>{};
    static constexpr auto kK1 = ck_tile::number<64>{};
    static constexpr auto kSlots = ck_tile::number<4>{};
    static constexpr auto PackedShape = ck_tile::make_tuple(kK0, kSlots, kMN, kK1);
};
#endif

} // namespace mls
} // namespace tl
