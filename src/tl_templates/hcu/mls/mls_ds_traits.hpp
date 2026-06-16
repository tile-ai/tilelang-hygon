#pragma once

/*
 * mls_ds_traits: maps (MlsAtom, ElemBytes, Alt) -> DsFormatInst (ds_read_matrix_format).
 * Uses tilelang DsreadmFormatDispatcher (builtin-based).
 * MlsAtom: tl::mls::gfx938_* or tl::mls::gfx946_* from mls_atom_for_tile.
 */

#include <tl_templates/hcu/core.hpp>

#include <tl_templates/hcu/mls/tl_dsreadm_format_dispatcher.hpp>

namespace tl {
namespace mls {

template <typename MlsAtom, ::tl::index_t ElemBytes, ::tl::index_t Alt>
struct mls_ds_traits;

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
#include <tl_templates/hcu/mls/tl_mls_atom_gfx938.hpp>
// gfx938 b16 (ElemBytes=2)
template <::tl::index_t Alt>
struct mls_ds_traits<tl::mls::gfx938_mls_32x16_b16, 2, Alt>
{
    using Type = DsreadmFormatDispatcher<2, 32, 16, Alt, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x32_trans_b16, 2, 1>
{
    using Type = DsreadmFormatDispatcher<2, 32, 16, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x32_trans_b16, 2, 2>
{
    using Type = DsreadmFormatDispatcher<2, 16, 32, 2, true>;
};

template <::tl::index_t Alt>
struct mls_ds_traits<tl::mls::gfx938_mls_32x32_b16, 2, Alt>
{
    using Type = DsreadmFormatDispatcher<2, 32, 16, Alt, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_32x32_trans_b16, 2, 1>
{
    using Type = DsreadmFormatDispatcher<2, 32, 16, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_32x32_trans_b16, 2, 2>
{
    using Type = DsreadmFormatDispatcher<2, 16, 32, 2, true>;
};

template <::tl::index_t Alt>
struct mls_ds_traits<tl::mls::gfx938_mls_64x16_b16, 2, Alt>
{
    using Type = DsreadmFormatDispatcher<2, 32, 16, Alt, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x64_trans_b16, 2, 1>
{
    using Type = DsreadmFormatDispatcher<2, 32, 16, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x64_trans_b16, 2, 2>
{
    using Type = DsreadmFormatDispatcher<2, 16, 32, 2, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_64x16_b8, 1, 1>
{
    using Type = DsreadmFormatDispatcher<1, 32, 32, 1, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_64x16_b8, 1, 2>
{
    using Type = DsreadmFormatDispatcher<1, 32, 32, 2, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_64x16_b8, 1, 4>
{
    using Type = DsreadmFormatDispatcher<1, 64, 16, 4, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_128x16_b8, 1, 1>
{
    using Type = DsreadmFormatDispatcher<1, 32, 32, 1, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_128x16_b8, 1, 2>
{
    using Type = DsreadmFormatDispatcher<1, 32, 32, 2, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_128x16_b8, 1, 4>
{
    using Type = DsreadmFormatDispatcher<1, 64, 16, 4, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_64x32_b8, 1, 1>
{
    using Type = DsreadmFormatDispatcher<1, 32, 32, 1, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_64x32_b8, 1, 2>
{
    using Type = DsreadmFormatDispatcher<1, 32, 32, 2, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_64x32_b8, 1, 4>
{
    using Type = DsreadmFormatDispatcher<1, 64, 16, 4, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x64_trans_b8, 1, 1>
{
    using Type = DsreadmFormatDispatcher<1, 16, 64, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x128_trans_b8, 1, 1>
{
    using Type = DsreadmFormatDispatcher<1, 16, 64, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x128_trans_b8, 1, 2>
{
    using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_16x128_trans_b8, 1, 4>
{
    using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_32x64_trans_b8, 1, 1>
{
    using Type = DsreadmFormatDispatcher<1, 16, 64, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_32x64_trans_b8, 1, 2>
{
    using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx938_mls_32x64_trans_b8, 1, 4>
{
    using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx946__)
#include <tl_templates/hcu/mls/tl_mls_atom_gfx946.hpp>

template <::tl::index_t Alt>
struct mls_ds_traits<tl::mls::gfx946_mls_32x16_b16, 2, Alt>
{
    using Type = DsreadmFormatDispatcher<2, 32, 16, Alt, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x32_trans_b16, 2, 1>
{
    using Type = DsreadmFormatDispatcher<2, 32, 16, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x32_trans_b16, 2, 2>
{
    using Type = DsreadmFormatDispatcher<2, 16, 32, 2, true>;
};

template <::tl::index_t Alt>
struct mls_ds_traits<tl::mls::gfx946_mls_32x32_b16, 2, Alt>
{
    using Type = DsreadmFormatDispatcher<2, 32, 16, Alt, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_32x32_trans_b16, 2, 1>
{
    using Type = DsreadmFormatDispatcher<2, 32, 16, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_32x32_trans_b16, 2, 2>
{
    using Type = DsreadmFormatDispatcher<2, 16, 32, 2, true>;
};

template <::tl::index_t Alt>
struct mls_ds_traits<tl::mls::gfx946_mls_64x16_b16, 2, Alt>
{
    using Type = DsreadmFormatDispatcher<2, 32, 16, Alt, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x64_trans_b16, 2, 1>
{
    using Type = DsreadmFormatDispatcher<2, 32, 16, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x64_trans_b16, 2, 2>
{
    using Type = DsreadmFormatDispatcher<2, 16, 32, 2, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_64x16_b8, 1, 1>
{
    using Type = DsreadmFormatDispatcher<1, 32, 32, 1, false>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x64_trans_b8, 1, 1>
{
    using Type = DsreadmFormatDispatcher<1, 16, 64, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x128_trans_b8, 1, 1>
{
    using Type = DsreadmFormatDispatcher<1, 16, 64, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x128_trans_b8, 1, 2>
{
    using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_16x128_trans_b8, 1, 4>
{
    using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_32x64_trans_b8, 1, 1>
{
    using Type = DsreadmFormatDispatcher<1, 16, 64, 1, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_32x64_trans_b8, 1, 2>
{
    using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

template <>
struct mls_ds_traits<tl::mls::gfx946_mls_32x64_trans_b8, 1, 4>
{
    using Type = DsreadmFormatDispatcher<1, 32, 32, 2, true>;
};

#endif

// Primary: no specialization for (MlsAtom, ElemBytes, Alt)
template <typename MlsAtom, ::tl::index_t ElemBytes, ::tl::index_t Alt>
struct mls_ds_traits
{
    static_assert(sizeof(MlsAtom) == 0,
                  "Unsupported (MlsAtom, ElemBytes, Alt). "
                  "Add specialization in mls_ds_traits.hpp. "
                  "Known gfx938 b16: gfx938_mls_32x16_b16, 16x32_trans, 32x32, 32x32_trans, "
                  "64x16, 16x64_trans.");
};

} // namespace mls
} // namespace tl
