#pragma once

/*
 * DsreadmFormatAttribute and Impl structs for DsreadmFormatDispatcher.
 * Impl uses __builtin_hcu_ds_read_matrix_format_* instead of inline asm.
 */

#include <tl_templates/hcu/core.hpp>

namespace tl {
namespace mls {

template <typename Impl>
struct DsreadmFormatAttribute
{
    using ImplType = ::tl::remove_cvref_t<Impl>;

    static constexpr ::tl::index_t kMN = ImplType::kMN;
    static constexpr ::tl::index_t kK  = ImplType::kK;

    static constexpr ::tl::index_t kMN0StorePerlane = ImplType::kMN0StorePerLane;
    static constexpr ::tl::index_t kMNStoreLane     = ImplType::kMNStoreLane;
    static constexpr ::tl::index_t kMN1StorePerLane = ImplType::kMN1StorePerLane;
    static constexpr ::tl::index_t kKStoreLane      = ImplType::kKStoreLane;
    static constexpr ::tl::index_t kKStorePerLane   = ImplType::kKStorePerLane;

    static constexpr ::tl::index_t kVectorLength = ImplType::kVectorLength;

    using WarpStoreDstrEncoding =
        ::tl::tile_distribution_encoding<::tl::sequence<>,
            ::tl::tuple<::tl::sequence<kMN0StorePerlane, kMNStoreLane, kMN1StorePerLane>,
                           ::tl::sequence<kKStoreLane, kKStorePerLane>>,
            ::tl::tuple<::tl::sequence<2, 1>>,
            ::tl::tuple<::tl::sequence<0, 1>>,
            ::tl::sequence<1, 1, 2>,
            ::tl::sequence<0, 2, 1>>;

    template <typename T, ::tl::index_t offset>
    TL_DEVICE auto operator()(TL_LDS_ADDR T* smem_ptr,
                                  ::tl::number<offset>) const
    {
        using VectorType = ::tl::ext_vector_t<T, kVectorLength>;
        using RetType    = ::tl::thread_buffer<VectorType, 1>;

        RetType ret;
        Impl{}(smem_ptr, ret.template get_as<VectorType>()[::tl::number<0>{}],
               ::tl::number<offset>{});

        return ret;
    }
};

// ========== Impl structs (builtin-based) ==========

struct DsreadmFormatAttributeImpl_M32x16_B16
{
    static constexpr ::tl::index_t kMN = 32;
    static constexpr ::tl::index_t kK  = 16;

    static constexpr ::tl::index_t kMNStoreLane = 16;
    static constexpr ::tl::index_t kKStoreLane  = 4;

    static constexpr ::tl::index_t kMN0StorePerLane = 2;
    static constexpr ::tl::index_t kMN1StorePerLane = 1;
    static constexpr ::tl::index_t kKStorePerLane   = 4;

    static constexpr ::tl::index_t kMNInterleave = 1;
    static constexpr ::tl::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ::tl::index_t offset>
    TL_DEVICE void operator()(TL_LDS_ADDR T* smem,
                                  ::tl::ext_vector_t<T, kVectorLength>& ret,
                                  ::tl::number<offset>) const
    {
        // element:0x2, row:0x2, col:0x1, alt:0x0
        TL_LDS_ADDR short* ptr = reinterpret_cast<TL_LDS_ADDR short*>(smem);
        ret = ::tl::bit_cast<::tl::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_format_u16(
                ptr, offset,
                static_cast<char>(0x2), static_cast<char>(0x1), static_cast<char>(0x0)));
    }
};

struct DsreadmFormatAttributeImpl_M32x16_B16_ALT2
{
    static constexpr ::tl::index_t kMN = 32;
    static constexpr ::tl::index_t kK  = 16;

    static constexpr ::tl::index_t kMNStoreLane = 16;
    static constexpr ::tl::index_t kKStoreLane  = 4;

    static constexpr ::tl::index_t kMN0StorePerLane = 1;
    static constexpr ::tl::index_t kMN1StorePerLane = 2;
    static constexpr ::tl::index_t kKStorePerLane   = 4;

    static constexpr ::tl::index_t kMNInterleave = 2;
    static constexpr ::tl::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ::tl::index_t offset>
    TL_DEVICE void operator()(TL_LDS_ADDR T* smem,
                                  ::tl::ext_vector_t<T, kVectorLength>& ret,
                                  ::tl::number<offset>) const
    {
        // element:0x2, row:0x2, col:0x1, alt:0x1
        TL_LDS_ADDR short* ptr = reinterpret_cast<TL_LDS_ADDR short*>(smem);
        ret = ::tl::bit_cast<::tl::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_format_u16(
                ptr, offset,
                static_cast<char>(0x2), static_cast<char>(0x1), static_cast<char>(0x1)));
    }
};

struct DsreadmFormatAttributeImpl_MT32x16_B16
{
    static constexpr ::tl::index_t kMN = 16;
    static constexpr ::tl::index_t kK  = 32;

    static constexpr ::tl::index_t kMNStoreLane = 16;
    static constexpr ::tl::index_t kKStoreLane  = 4;

    static constexpr ::tl::index_t kMN0StorePerLane = 1;
    static constexpr ::tl::index_t kMN1StorePerLane = 1;
    static constexpr ::tl::index_t kKStorePerLane   = 8;

    static constexpr ::tl::index_t kMNInterleave = 1;
    static constexpr ::tl::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ::tl::index_t offset>
    TL_DEVICE void operator()(TL_LDS_ADDR T* smem,
                                  ::tl::ext_vector_t<T, kVectorLength>& ret,
                                  ::tl::number<offset>) const
    {
        // element:0x2, row:0x2, col:0x1, alt:0x0
#if 0  // TEMP: inline asm when builtin causes compile error on gfx938
        ::tl::hcu_ds_read_matrix_trans_format_asm_impl(
            reinterpret_cast<uintptr_t>(smem), ret, ::tl::number<offset>{},
            ::tl::number<0x2>{}, ::tl::number<0x2>{}, ::tl::number<0x1>{},
            ::tl::number<0x0>{});
#else
        TL_LDS_ADDR short* ptr = reinterpret_cast<TL_LDS_ADDR short*>(smem);
        ret = ::tl::bit_cast<::tl::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_trans_format_u16(
                ptr, offset,
                static_cast<char>(0x2), static_cast<char>(0x1), static_cast<char>(0x0)));
#endif
    }
};

struct DsreadmFormatAttributeImpl_MT16x32_B16_ALT2
{
    static constexpr ::tl::index_t kMN = 32;
    static constexpr ::tl::index_t kK  = 16;

    static constexpr ::tl::index_t kMNStoreLane = 16;
    static constexpr ::tl::index_t kKStoreLane  = 4;

    static constexpr ::tl::index_t kMN0StorePerLane = 1;
    static constexpr ::tl::index_t kMN1StorePerLane = 2;
    static constexpr ::tl::index_t kKStorePerLane   = 4;

    static constexpr ::tl::index_t kMNInterleave = 2;
    static constexpr ::tl::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ::tl::index_t offset>
    TL_DEVICE void operator()(TL_LDS_ADDR T* smem,
                                  ::tl::ext_vector_t<T, kVectorLength>& ret,
                                  ::tl::number<offset>) const
    {
        // element:0x2, row:0x1, col:0x2, alt:0x1
        TL_LDS_ADDR short* ptr = reinterpret_cast<TL_LDS_ADDR short*>(smem);
        ret = ::tl::bit_cast<::tl::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_trans_format_u16(
                ptr, offset,
                static_cast<char>(0x1), static_cast<char>(0x2), static_cast<char>(0x1)));
    }
};

// b8 related

// DS_S_READ_M32X32_B8
struct DsreadmFormatAttributeImpl_M32x32_B8
{
    static constexpr ::tl::index_t kMN = 32;
    static constexpr ::tl::index_t kK  = 32;

    static constexpr ::tl::index_t kMNStoreLane = 16;
    static constexpr ::tl::index_t kKStoreLane  = 4;

    static constexpr ::tl::index_t kMN0StorePerLane = 2;
    static constexpr ::tl::index_t kMN1StorePerLane = 1;
    static constexpr ::tl::index_t kKStorePerLane   = 8;

    static constexpr ::tl::index_t kMNInterleave = 1;
    static constexpr ::tl::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ::tl::index_t offset>
    TL_DEVICE void operator()(TL_LDS_ADDR T* smem,
                                  ::tl::ext_vector_t<T, kVectorLength>& ret,
                                  ::tl::number<offset>) const
    {
        // element:0x1, row:2, col:2, alt:0
        TL_LDS_ADDR unsigned char* ptr =
            reinterpret_cast<TL_LDS_ADDR unsigned char*>(smem);
        ret = ::tl::bit_cast<::tl::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_format_u8(
                ptr, offset,
                static_cast<char>(0x2), static_cast<char>(0x2), static_cast<char>(0x0)));
    }
};

// DS_S_READ_M32X32_B8_ALT2
struct DsreadmFormatAttributeImpl_M32x32_B8_ALT2
{
    static constexpr ::tl::index_t kMN = 32;
    static constexpr ::tl::index_t kK  = 32;

    static constexpr ::tl::index_t kMNStoreLane = 16;
    static constexpr ::tl::index_t kKStoreLane  = 4;

    static constexpr ::tl::index_t kMN0StorePerLane = 1;
    static constexpr ::tl::index_t kMN1StorePerLane = 2;
    static constexpr ::tl::index_t kKStorePerLane   = 8;

    static constexpr ::tl::index_t kMNInterleave = 2;
    static constexpr ::tl::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ::tl::index_t offset>
    TL_DEVICE void operator()(TL_LDS_ADDR T* smem,
                                  ::tl::ext_vector_t<T, kVectorLength>& ret,
                                  ::tl::number<offset>) const
    {
        // element:0x1, row:2, col:1, alt:1
        TL_LDS_ADDR unsigned char* ptr =
            reinterpret_cast<TL_LDS_ADDR unsigned char*>(smem);
        ret = ::tl::bit_cast<::tl::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_format_u8(
                ptr, offset,
                static_cast<char>(0x2), static_cast<char>(0x2), static_cast<char>(0x0)));
    }
};

// DS_S_READ_M64X16_B8_ALT4
struct DsreadmFormatAttributeImpl_M64x16_B8_ALT4
{
    static constexpr ::tl::index_t kMN = 64;
    static constexpr ::tl::index_t kK  = 16;

    static constexpr ::tl::index_t kMNStoreLane = 16;
    static constexpr ::tl::index_t kKStoreLane  = 8;

    static constexpr ::tl::index_t kMN0StorePerLane = 1;
    static constexpr ::tl::index_t kMN1StorePerLane = 4;
    static constexpr ::tl::index_t kKStorePerLane   = 4;

    static constexpr ::tl::index_t kMNInterleave = 4;
    static constexpr ::tl::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ::tl::index_t offset>
    TL_DEVICE void operator()(TL_LDS_ADDR T* smem,
                                  ::tl::ext_vector_t<T, kVectorLength>& ret,
                                  ::tl::number<offset>) const
    {
        // element:0x1, row:0x3, col:0x1, alt:0x2
        TL_LDS_ADDR unsigned char* ptr =
            reinterpret_cast<TL_LDS_ADDR unsigned char*>(smem);
        ret = ::tl::bit_cast<::tl::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_trans_format_u8(
                ptr, offset,
                static_cast<char>(0x3), static_cast<char>(0x1), static_cast<char>(0x2)));
    }
};

// DS_S_READ_MT64X16_B8
struct DsreadmFormatAttributeImpl_MT16x64_B8
{
    static constexpr ::tl::index_t kMN = 16;
    static constexpr ::tl::index_t kK  = 64;

    static constexpr ::tl::index_t kMNStoreLane = 16;
    static constexpr ::tl::index_t kKStoreLane  = 4;

    static constexpr ::tl::index_t kMN0StorePerLane = 1;
    static constexpr ::tl::index_t kMN1StorePerLane = 1;
    static constexpr ::tl::index_t kKStorePerLane   = 16;

    static constexpr ::tl::index_t kMNInterleave = 1;
    static constexpr ::tl::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ::tl::index_t offset>
    TL_DEVICE void operator()(TL_LDS_ADDR T* smem,
                                  ::tl::ext_vector_t<T, kVectorLength>& ret,
                                  ::tl::number<offset>) const
    {
        // element:0x1, row:0x3, col:0x1, alt:0x0
        TL_LDS_ADDR unsigned char* ptr =
            reinterpret_cast<TL_LDS_ADDR unsigned char*>(smem);
        ret = ::tl::bit_cast<::tl::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_trans_format_u8(
                ptr, offset,
                static_cast<char>(0x3), static_cast<char>(0x1), static_cast<char>(0x0)));
    }
};

// DS_S_READ_MT32X32_B8_ALT2
struct DsreadmFormatAttributeImpl_MT32x32_B8_ALT2
{
    static constexpr ::tl::index_t kMN = 32;
    static constexpr ::tl::index_t kK  = 32;

    static constexpr ::tl::index_t kMNStoreLane = 16;
    static constexpr ::tl::index_t kKStoreLane  = 4;

    static constexpr ::tl::index_t kMN0StorePerLane = 1;
    static constexpr ::tl::index_t kMN1StorePerLane = 2;
    static constexpr ::tl::index_t kKStorePerLane   = 8;

    static constexpr ::tl::index_t kMNInterleave = 2;
    static constexpr ::tl::index_t kVectorLength =
        kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, ::tl::index_t offset>
    TL_DEVICE void operator()(TL_LDS_ADDR T* smem,
                                  ::tl::ext_vector_t<T, kVectorLength>& ret,
                                  ::tl::number<offset>) const
    {
        // element:0x1, row:0x2, col:0x2, alt:0x1
        TL_LDS_ADDR unsigned char* ptr =
            reinterpret_cast<TL_LDS_ADDR unsigned char*>(smem);
        ret = ::tl::bit_cast<::tl::ext_vector_t<T, kVectorLength>>(
            __builtin_hcu_ds_read_matrix_trans_format_u8(
                ptr, offset,
                static_cast<char>(0x2), static_cast<char>(0x2), static_cast<char>(0x1)));
    }
};

} // namespace mls
} // namespace tl
