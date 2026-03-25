#pragma once

/*
 * DsreadmFormatAttribute and Impl structs for DsreadmFormatDispatcher.
 * Impl uses __builtin_hcu_ds_read_matrix_format_* instead of inline asm.
 */

#include <ck_tile/core.hpp>

namespace tl {
namespace mls {

template <typename Impl>
struct DsreadmFormatAttribute
{
    using ImplType = ck_tile::remove_cvref_t<Impl>;

    static constexpr ck_tile::index_t kMN = ImplType::kMN;
    static constexpr ck_tile::index_t kK  = ImplType::kK;

    static constexpr ck_tile::index_t kMN0StorePerlane = ImplType::kMN0StorePerLane;
    static constexpr ck_tile::index_t kMNStoreLane     = ImplType::kMNStoreLane;
    static constexpr ck_tile::index_t kMN1StorePerLane = ImplType::kMN1StorePerLane;
    static constexpr ck_tile::index_t kKStoreLane      = ImplType::kKStoreLane;
    static constexpr ck_tile::index_t kKStorePerLane   = ImplType::kKStorePerLane;

    static constexpr ck_tile::index_t kVectorLength = ImplType::kVectorLength;

    using WarpStoreDstrEncoding =
        ck_tile::tile_distribution_encoding<ck_tile::sequence<>,
            ck_tile::tuple<ck_tile::sequence<kMN0StorePerlane, kMNStoreLane, kMN1StorePerLane>,
                           ck_tile::sequence<kKStoreLane, kKStorePerLane>>,
            ck_tile::tuple<ck_tile::sequence<2, 1>>,
            ck_tile::tuple<ck_tile::sequence<0, 1>>,
            ck_tile::sequence<1, 1, 2>,
            ck_tile::sequence<0, 2, 1>>;

    template <typename T, ck_tile::index_t offset>
    CK_TILE_DEVICE auto operator()(CK_TILE_LDS_ADDR T* smem_ptr,
                                  ck_tile::number<offset>) const
    {
        using VectorType = ck_tile::ext_vector_t<T, kVectorLength>;
        using RetType    = ck_tile::thread_buffer<VectorType, 1>;

        RetType ret;
        Impl{}(smem_ptr, ret.template get_as<VectorType>()[ck_tile::number<0>{}],
               ck_tile::number<offset>{});

        return ret;
    }
};

// ========== Impl structs (builtin-based) ==========

struct DsreadmFormatAttributeImpl_M32x16_B16
{
    static constexpr ck_tile::index_t kMN = 32;
    static constexpr ck_tile::index_t kK  = 16;

    static constexpr ck_tile::index_t kMNStoreLane = 16;
    static constexpr ck_tile::index_t kKStoreLane  = 4;

    static constexpr ck_tile::index_t kMN0StorePerLane = 2;
    static constexpr ck_tile::index_t kMN1StorePerLane = 1;
    static constexpr ck_tile::index_t kKStorePerLane   = 4;

    static constexpr ck_tile::index_t kMNInterleave = 1;
    static constexpr ck_tile::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ck_tile::index_t offset>
    CK_TILE_DEVICE void operator()(CK_TILE_LDS_ADDR T* smem,
                                  ck_tile::ext_vector_t<T, kVectorLength>& ret,
                                  ck_tile::number<offset>) const
    {
        // element:0x2, row:0x2, col:0x1, alt:0x0
        CK_TILE_LDS_ADDR short* ptr = reinterpret_cast<CK_TILE_LDS_ADDR short*>(smem);
        ret = ck_tile::bit_cast<ck_tile::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_format_u16(
                ptr, offset,
                static_cast<char>(0x2), static_cast<char>(0x1), static_cast<char>(0x0)));
    }
};

struct DsreadmFormatAttributeImpl_M32x16_B16_ALT2
{
    static constexpr ck_tile::index_t kMN = 32;
    static constexpr ck_tile::index_t kK  = 16;

    static constexpr ck_tile::index_t kMNStoreLane = 16;
    static constexpr ck_tile::index_t kKStoreLane  = 4;

    static constexpr ck_tile::index_t kMN0StorePerLane = 1;
    static constexpr ck_tile::index_t kMN1StorePerLane = 2;
    static constexpr ck_tile::index_t kKStorePerLane   = 4;

    static constexpr ck_tile::index_t kMNInterleave = 2;
    static constexpr ck_tile::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ck_tile::index_t offset>
    CK_TILE_DEVICE void operator()(CK_TILE_LDS_ADDR T* smem,
                                  ck_tile::ext_vector_t<T, kVectorLength>& ret,
                                  ck_tile::number<offset>) const
    {
        // element:0x2, row:0x2, col:0x1, alt:0x1
        CK_TILE_LDS_ADDR short* ptr = reinterpret_cast<CK_TILE_LDS_ADDR short*>(smem);
        ret = ck_tile::bit_cast<ck_tile::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_format_u16(
                ptr, offset,
                static_cast<char>(0x2), static_cast<char>(0x1), static_cast<char>(0x1)));
    }
};

struct DsreadmFormatAttributeImpl_MT32x16_B16
{
    static constexpr ck_tile::index_t kMN = 16;
    static constexpr ck_tile::index_t kK  = 32;

    static constexpr ck_tile::index_t kMNStoreLane = 16;
    static constexpr ck_tile::index_t kKStoreLane  = 4;

    static constexpr ck_tile::index_t kMN0StorePerLane = 1;
    static constexpr ck_tile::index_t kMN1StorePerLane = 1;
    static constexpr ck_tile::index_t kKStorePerLane   = 8;

    static constexpr ck_tile::index_t kMNInterleave = 1;
    static constexpr ck_tile::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ck_tile::index_t offset>
    CK_TILE_DEVICE void operator()(CK_TILE_LDS_ADDR T* smem,
                                  ck_tile::ext_vector_t<T, kVectorLength>& ret,
                                  ck_tile::number<offset>) const
    {
        // element:0x2, row:0x2, col:0x1, alt:0x0
#if 0  // TEMP: inline asm when builtin causes compile error on gfx938
        ck_tile::hcu_ds_read_matrix_trans_format_asm_impl(
            reinterpret_cast<uintptr_t>(smem), ret, ck_tile::number<offset>{},
            ck_tile::number<0x2>{}, ck_tile::number<0x2>{}, ck_tile::number<0x1>{},
            ck_tile::number<0x0>{});
#else
        CK_TILE_LDS_ADDR short* ptr = reinterpret_cast<CK_TILE_LDS_ADDR short*>(smem);
        ret = ck_tile::bit_cast<ck_tile::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_trans_format_u16(
                ptr, offset,
                static_cast<char>(0x2), static_cast<char>(0x1), static_cast<char>(0x0)));
#endif
    }
};

struct DsreadmFormatAttributeImpl_MT16x32_B16_ALT2
{
    static constexpr ck_tile::index_t kMN = 32;
    static constexpr ck_tile::index_t kK  = 16;

    static constexpr ck_tile::index_t kMNStoreLane = 16;
    static constexpr ck_tile::index_t kKStoreLane  = 4;

    static constexpr ck_tile::index_t kMN0StorePerLane = 1;
    static constexpr ck_tile::index_t kMN1StorePerLane = 2;
    static constexpr ck_tile::index_t kKStorePerLane   = 4;

    static constexpr ck_tile::index_t kMNInterleave = 2;
    static constexpr ck_tile::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ck_tile::index_t offset>
    CK_TILE_DEVICE void operator()(CK_TILE_LDS_ADDR T* smem,
                                  ck_tile::ext_vector_t<T, kVectorLength>& ret,
                                  ck_tile::number<offset>) const
    {
        // element:0x2, row:0x1, col:0x2, alt:0x1
        CK_TILE_LDS_ADDR short* ptr = reinterpret_cast<CK_TILE_LDS_ADDR short*>(smem);
        ret = ck_tile::bit_cast<ck_tile::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_trans_format_u16(
                ptr, offset,
                static_cast<char>(0x1), static_cast<char>(0x2), static_cast<char>(0x1)));
    }
};

// b8 related

// DS_S_READ_M32X32_B8
struct DsreadmFormatAttributeImpl_M32x32_B8
{
    static constexpr ck_tile::index_t kMN = 32;
    static constexpr ck_tile::index_t kK  = 32;

    static constexpr ck_tile::index_t kMNStoreLane = 16;
    static constexpr ck_tile::index_t kKStoreLane  = 4;

    static constexpr ck_tile::index_t kMN0StorePerLane = 2;
    static constexpr ck_tile::index_t kMN1StorePerLane = 1;
    static constexpr ck_tile::index_t kKStorePerLane   = 8;

    static constexpr ck_tile::index_t kMNInterleave = 1;
    static constexpr ck_tile::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ck_tile::index_t offset>
    CK_TILE_DEVICE void operator()(CK_TILE_LDS_ADDR T* smem,
                                  ck_tile::ext_vector_t<T, kVectorLength>& ret,
                                  ck_tile::number<offset>) const
    {
        // element:0x1, row:2, col:2, alt:0
        CK_TILE_LDS_ADDR unsigned char* ptr =
            reinterpret_cast<CK_TILE_LDS_ADDR unsigned char*>(smem);
        ret = ck_tile::bit_cast<ck_tile::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_format_u8(
                ptr, offset,
                static_cast<char>(0x2), static_cast<char>(0x2), static_cast<char>(0x0)));
    }
};

// DS_S_READ_M32X32_B8_ALT2
struct DsreadmFormatAttributeImpl_M32x32_B8_ALT2
{
    static constexpr ck_tile::index_t kMN = 32;
    static constexpr ck_tile::index_t kK  = 32;

    static constexpr ck_tile::index_t kMNStoreLane = 16;
    static constexpr ck_tile::index_t kKStoreLane  = 4;

    static constexpr ck_tile::index_t kMN0StorePerLane = 1;
    static constexpr ck_tile::index_t kMN1StorePerLane = 2;
    static constexpr ck_tile::index_t kKStorePerLane   = 8;

    static constexpr ck_tile::index_t kMNInterleave = 2;
    static constexpr ck_tile::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ck_tile::index_t offset>
    CK_TILE_DEVICE void operator()(CK_TILE_LDS_ADDR T* smem,
                                  ck_tile::ext_vector_t<T, kVectorLength>& ret,
                                  ck_tile::number<offset>) const
    {
        // element:0x1, row:2, col:1, alt:1
        CK_TILE_LDS_ADDR unsigned char* ptr =
            reinterpret_cast<CK_TILE_LDS_ADDR unsigned char*>(smem);
        ret = ck_tile::bit_cast<ck_tile::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_format_u8(
                ptr, offset,
                static_cast<char>(0x2), static_cast<char>(0x2), static_cast<char>(0x0)));
    }
};

// DS_S_READ_M64X16_B8_ALT4
struct DsreadmFormatAttributeImpl_M64x16_B8_ALT4
{
    static constexpr ck_tile::index_t kMN = 64;
    static constexpr ck_tile::index_t kK  = 16;

    static constexpr ck_tile::index_t kMNStoreLane = 16;
    static constexpr ck_tile::index_t kKStoreLane  = 8;

    static constexpr ck_tile::index_t kMN0StorePerLane = 1;
    static constexpr ck_tile::index_t kMN1StorePerLane = 4;
    static constexpr ck_tile::index_t kKStorePerLane   = 4;

    static constexpr ck_tile::index_t kMNInterleave = 4;
    static constexpr ck_tile::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ck_tile::index_t offset>
    CK_TILE_DEVICE void operator()(CK_TILE_LDS_ADDR T* smem,
                                  ck_tile::ext_vector_t<T, kVectorLength>& ret,
                                  ck_tile::number<offset>) const
    {
        // element:0x1, row:0x3, col:0x1, alt:0x2
        CK_TILE_LDS_ADDR unsigned char* ptr =
            reinterpret_cast<CK_TILE_LDS_ADDR unsigned char*>(smem);
        ret = ck_tile::bit_cast<ck_tile::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_trans_format_u8(
                ptr, offset,
                static_cast<char>(0x3), static_cast<char>(0x1), static_cast<char>(0x2)));
    }
};

// DS_S_READ_MT64X16_B8
struct DsreadmFormatAttributeImpl_MT16x64_B8
{
    static constexpr ck_tile::index_t kMN = 16;
    static constexpr ck_tile::index_t kK  = 64;

    static constexpr ck_tile::index_t kMNStoreLane = 16;
    static constexpr ck_tile::index_t kKStoreLane  = 4;

    static constexpr ck_tile::index_t kMN0StorePerLane = 1;
    static constexpr ck_tile::index_t kMN1StorePerLane = 1;
    static constexpr ck_tile::index_t kKStorePerLane   = 16;

    static constexpr ck_tile::index_t kMNInterleave = 1;
    static constexpr ck_tile::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ck_tile::index_t offset>
    CK_TILE_DEVICE void operator()(CK_TILE_LDS_ADDR T* smem,
                                  ck_tile::ext_vector_t<T, kVectorLength>& ret,
                                  ck_tile::number<offset>) const
    {
        // element:0x1, row:0x3, col:0x1, alt:0x0
        CK_TILE_LDS_ADDR unsigned char* ptr =
            reinterpret_cast<CK_TILE_LDS_ADDR unsigned char*>(smem);
        ret = ck_tile::bit_cast<ck_tile::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_trans_format_u8(
                ptr, offset,
                static_cast<char>(0x3), static_cast<char>(0x1), static_cast<char>(0x0)));
    }
};

// DS_S_READ_MT32X32_B8_ALT2
struct DsreadmFormatAttributeImpl_MT32x32_B8_ALT2
{
    static constexpr ck_tile::index_t kMN = 32;
    static constexpr ck_tile::index_t kK  = 32;

    static constexpr ck_tile::index_t kMNStoreLane = 16;
    static constexpr ck_tile::index_t kKStoreLane  = 4;

    static constexpr ck_tile::index_t kMN0StorePerLane = 1;
    static constexpr ck_tile::index_t kMN1StorePerLane = 2;
    static constexpr ck_tile::index_t kKStorePerLane   = 8;

    static constexpr ck_tile::index_t kMNInterleave = 2;
    static constexpr ck_tile::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ck_tile::index_t offset>
    CK_TILE_DEVICE void operator()(CK_TILE_LDS_ADDR T* smem,
                                  ck_tile::ext_vector_t<T, kVectorLength>& ret,
                                  ck_tile::number<offset>) const
    {
        // element:0x1, row:0x2, col:0x2, alt:0x1
        CK_TILE_LDS_ADDR unsigned char* ptr =
            reinterpret_cast<CK_TILE_LDS_ADDR unsigned char*>(smem);
        ret = ck_tile::bit_cast<ck_tile::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_trans_format_u8(
                ptr, offset,
                static_cast<char>(0x2), static_cast<char>(0x2), static_cast<char>(0x1)));
    }
};

} // namespace mls
} // namespace tl
