#pragma once

/*
 * mls_atom_for_tile: maps (MlsTileMN, MlsTileK, Trans, Bits, HcuArch) -> tilelang MlsAtom type.
 * Returns tl::mls::* types (defined in tl_mls_atom_gfx938.hpp, tl_mls_atom_gfx946.hpp).
 */

#include <ck_tile/core.hpp>

namespace tl {
namespace mls {

template <ck_tile::index_t MlsTileMN,
          ck_tile::index_t MlsTileK,
          bool Trans,
          ck_tile::index_t Bits,
          ck_tile::hcu_target_enum HcuArch>
struct mls_atom_for_tile;

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
#include "tl_mls_atom_gfx938.hpp"

template <>
struct mls_atom_for_tile<16, 64, true, 16, ck_tile::hcu_target_enum::gfx938>
{
    using Type = tl::mls::gfx938_mls_16x64_trans_b16;
};

template <>
struct mls_atom_for_tile<32, 32, true, 16, ck_tile::hcu_target_enum::gfx938>
{
    using Type = tl::mls::gfx938_mls_32x32_trans_b16;
};

template <>
struct mls_atom_for_tile<16, 32, true, 16, ck_tile::hcu_target_enum::gfx938>
{
    using Type = tl::mls::gfx938_mls_16x32_trans_b16;
};

template <>
struct mls_atom_for_tile<32, 32, false, 16, ck_tile::hcu_target_enum::gfx938>
{
    using Type = tl::mls::gfx938_mls_32x32_b16;
};

template <>
struct mls_atom_for_tile<32, 16, false, 16, ck_tile::hcu_target_enum::gfx938>
{
    using Type = tl::mls::gfx938_mls_32x16_b16;
};

template <>
struct mls_atom_for_tile<64, 16, false, 16, ck_tile::hcu_target_enum::gfx938>
{
    using Type = tl::mls::gfx938_mls_64x16_b16;
};

template <>
struct mls_atom_for_tile<64, 16, false, 8, ck_tile::hcu_target_enum::gfx938>
{
    using Type = tl::mls::gfx938_mls_64x16_b8;
};

template <>
struct mls_atom_for_tile<16, 64, true, 8, ck_tile::hcu_target_enum::gfx938>
{
    using Type = tl::mls::gfx938_mls_16x64_trans_b8;
};

template <>
struct mls_atom_for_tile<64, 32, false, 8, ck_tile::hcu_target_enum::gfx938>
{
    using Type = tl::mls::gfx938_mls_64x32_b8;
};

template <>
struct mls_atom_for_tile<32, 64, true, 8, ck_tile::hcu_target_enum::gfx938>
{
    using Type = tl::mls::gfx938_mls_32x64_trans_b8;
};

template <>
struct mls_atom_for_tile<128, 16, false, 8, ck_tile::hcu_target_enum::gfx938>
{
    using Type = tl::mls::gfx938_mls_128x16_b8;
};

template <>
struct mls_atom_for_tile<16, 128, true, 8, ck_tile::hcu_target_enum::gfx938>
{
    using Type = tl::mls::gfx938_mls_16x128_trans_b8;
};
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx946__)
#include "tl_mls_atom_gfx946.hpp"

template <>
struct mls_atom_for_tile<16, 64, true, 16, ck_tile::hcu_target_enum::gfx946>
{
    using Type = tl::mls::gfx946_mls_16x64_trans_b16;
};

template <>
struct mls_atom_for_tile<32, 32, true, 16, ck_tile::hcu_target_enum::gfx946>
{
    using Type = tl::mls::gfx946_mls_32x32_trans_b16;
};

template <>
struct mls_atom_for_tile<16, 32, true, 16, ck_tile::hcu_target_enum::gfx946>
{
    using Type = tl::mls::gfx946_mls_16x32_trans_b16;
};

template <>
struct mls_atom_for_tile<32, 32, false, 16, ck_tile::hcu_target_enum::gfx946>
{
    using Type = tl::mls::gfx946_mls_32x32_b16;
};

template <>
struct mls_atom_for_tile<32, 16, false, 16, ck_tile::hcu_target_enum::gfx946>
{
    using Type = tl::mls::gfx946_mls_32x16_b16;
};

template <>
struct mls_atom_for_tile<64, 16, false, 16, ck_tile::hcu_target_enum::gfx946>
{
    using Type = tl::mls::gfx946_mls_64x16_b16;
};

template <>
struct mls_atom_for_tile<64, 16, false, 8, ck_tile::hcu_target_enum::gfx946>
{
    using Type = tl::mls::gfx946_mls_64x16_b8;
};

template <>
struct mls_atom_for_tile<16, 64, true, 8, ck_tile::hcu_target_enum::gfx946>
{
    using Type = tl::mls::gfx946_mls_16x64_trans_b8;
};

template <>
struct mls_atom_for_tile<64, 32, false, 8, ck_tile::hcu_target_enum::gfx946>
{
    using Type = tl::mls::gfx946_mls_64x32_b8;
};

template <>
struct mls_atom_for_tile<32, 64, true, 8, ck_tile::hcu_target_enum::gfx946>
{
    using Type = tl::mls::gfx946_mls_32x64_trans_b8;
};

template <>
struct mls_atom_for_tile<128, 16, false, 8, ck_tile::hcu_target_enum::gfx946>
{
    using Type = tl::mls::gfx946_mls_128x16_b8;
};

template <>
struct mls_atom_for_tile<16, 128, true, 8, ck_tile::hcu_target_enum::gfx946>
{
    using Type = tl::mls::gfx946_mls_16x128_trans_b8;
};
#endif

// Primary: no specialization
template <ck_tile::index_t MlsTileMN,
          ck_tile::index_t MlsTileK,
          bool Trans,
          ck_tile::index_t Bits,
          ck_tile::hcu_target_enum HcuArch>
struct mls_atom_for_tile
{
    static_assert(MlsTileMN != MlsTileMN,
                  "Unsupported (MlsTileMN, MlsTileK, Trans, Bits, HcuArch). "
                  "Add specialization in tl_mls_atom_dispatcher.hpp.");
};

} // namespace mls
} // namespace tl
