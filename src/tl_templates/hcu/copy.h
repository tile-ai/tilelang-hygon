#pragma once

#include <cstdint>
#include <tl_templates/hcu/common.h>

#include <tl_templates/hcu/core/arch/amd_buffer_addressing.hpp>

using f32 = float;
// using f16 = _Float16;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;

using index_t = u32;

using tl::int32x4_t;

namespace tl {

namespace detail {

template <typename T, int ReadSize>
TL_DEVICE void hcu_direct_load_global_to_lds_impl_offen(
    const T *global_base_ptr, std::int32_t global_offset, T *lds_base_ptr,
    std::int32_t lds_offset, bool is_valid, std::int32_t src_element_space_size,
    std::uint8_t wave_lds_wrap_offset) {
  static_assert(ReadSize == 4 || ReadSize == 8 || ReadSize == 16);
  (void)wave_lds_wrap_offset;

  const int32x4_t src_resource = make_wave_buffer_resource(
      global_base_ptr,
      static_cast<u32>(
          static_cast<std::uint64_t>(static_cast<u32>(src_element_space_size)) *
          sizeof(T)));

  const std::int32_t element_offset_bytes =
      is_valid ? global_offset * static_cast<std::int32_t>(sizeof(T))
               : src_element_space_size * static_cast<std::int32_t>(sizeof(T));

  T *lds_ptr = lds_base_ptr + lds_offset;
  auto const lds_ptr_sgpr =
      __builtin_amdgcn_readfirstlane(reinterpret_cast<uintptr_t>(lds_ptr));

  if constexpr (ReadSize == 4) {
    asm volatile(
        "s_mov_b32 m0, %0; \n\t"
        "buffer_load_dword %1, %2, 0 offen lds;\n\t" ::"s"(lds_ptr_sgpr),
        "v"(element_offset_bytes), "s"(src_resource)
        : "memory");
  } else if constexpr (ReadSize == 8) {
    asm volatile(
        "s_mov_b32 m0, %0; \n\t"
        "buffer_load_dwordx2 %1, %2, 0 offen lds;\n\t" ::"s"(lds_ptr_sgpr),
        "v"(element_offset_bytes), "s"(src_resource)
        : "memory");
  } else {
    asm volatile(
        "s_mov_b32 m0, %0; \n\t"
        "buffer_load_dwordx4 %1, %2, 0 offen lds;\n\t" ::"s"(lds_ptr_sgpr),
        "v"(element_offset_bytes), "s"(src_resource)
        : "memory");
  }
}

} // namespace detail

// 与 codegen_hcu.cc / CK `hcu_direct_load_global_to_lds` 参数语义一致；仅实现
// UseIdxenLoad=false。
template <typename T, int NumElemsPerThread, bool UseIdxenLoad = false>
TL_DEVICE void hcu_direct_load_global_to_lds(
    const T *global_base_ptr, std::int32_t global_offset, T *lds_base_ptr,
    std::int32_t lds_offset, bool is_valid, std::int32_t src_element_space_size,
    std::uint8_t wave_lds_wrap_offset = 0) {
  static_assert(sizeof(T) * NumElemsPerThread == 4 ||
                    sizeof(T) * NumElemsPerThread == 8 ||
                    sizeof(T) * NumElemsPerThread == 16,
                "ReadSize must be 4, 8 or 16 bytes");
  static_assert(
      !UseIdxenLoad,
      "tl::hcu_direct_load_global_to_lds: Idxen path not implemented");

  constexpr int kReadSize = sizeof(T) * NumElemsPerThread;
  detail::hcu_direct_load_global_to_lds_impl_offen<T, kReadSize>(
      global_base_ptr, global_offset, lds_base_ptr, lds_offset, is_valid,
      src_element_space_size, wave_lds_wrap_offset);
}

namespace detail {

template <int NumUint32PerThread>
TL_DEVICE void hcu_cp_async_gs_via_direct_lds_with_resource(
    void *lds_base_ptr, void const *global_thread_ptr, int32x4_t src_resource,
    std::int32_t src_thread_byte_offset, bool is_valid) {
  static_assert(NumUint32PerThread == 1 || NumUint32PerThread == 2 ||
                NumUint32PerThread == 4);
  (void)global_thread_ptr;
  __attribute__((address_space(3))) int *const lds_ptr =
      reinterpret_cast<__attribute__((address_space(3))) int *>(
          reinterpret_cast<uintptr_t>(lds_base_ptr));
  const std::int32_t byte_offset =
      is_valid ? src_thread_byte_offset : src_resource.z;

  __builtin_amdgcn_raw_buffer_load_async_lds(
      src_resource, lds_ptr, NumUint32PerThread * sizeof(uint32_t),
      static_cast<uint32_t>(byte_offset), 0, 0, 0);
}

// 与 codegen_hcu.cc 约定：全局/global 传入「本线程区域起点」，须折算为 wave
// 统一基址 g_wave + 元素偏移 off（buffer_load + readfirstlane）。LDS 侧 impl
// 会做 lds_ptr = base + lds_offset， 与 (lds_wave, off) 代数上等价于 (l,
// 0)，故直接传本线程指针 l 与 lds_offset=0 即可。
template <int NumUint32PerThread>
TL_DEVICE void hcu_cp_async_gs_via_direct_lds(void *lds_base_ptr,
                                              void const *global_base_ptr,
                                              bool is_valid) {
  static_assert(NumUint32PerThread == 1 || NumUint32PerThread == 2 ||
                NumUint32PerThread == 4);
  static constexpr std::int32_t k_max_src_elems =
      static_cast<std::int32_t>(0xffffffffu / sizeof(uint32_t));

  auto *const g = static_cast<const uint32_t *>(global_base_ptr);
  uint32_t *const l = static_cast<uint32_t *>(lds_base_ptr);
  const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x);
  const auto *const g_wave =
      g - static_cast<size_t>(lane) * static_cast<size_t>(NumUint32PerThread);
  const std::int32_t off = lane * static_cast<std::int32_t>(NumUint32PerThread);

  // dword global→LDS 多数 HCU/gfx 可用；dwordx2/x4 依赖
  // ISA，不支持时退回同步向量读写以免非法指令或静默错误。
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) ||       \
    defined(__gfx946__)
#define TL_HCU_GLOBAL_TO_LDS_ASYNC_MULTIDWORD 1
#else
#define TL_HCU_GLOBAL_TO_LDS_ASYNC_MULTIDWORD 0
#endif

  if constexpr (NumUint32PerThread == 1) {
    hcu_direct_load_global_to_lds<uint32_t, 1, false>(
        g_wave, off, l, 0, is_valid, k_max_src_elems, 0);
  } else if constexpr (TL_HCU_GLOBAL_TO_LDS_ASYNC_MULTIDWORD) {
    if constexpr (NumUint32PerThread == 2) {
      hcu_direct_load_global_to_lds<uint32_t, 2, false>(
          g_wave, off, l, 0, is_valid, k_max_src_elems, 0);
    } else {
      hcu_direct_load_global_to_lds<uint32_t, 4, false>(
          g_wave, off, l, 0, is_valid, k_max_src_elems, 0);
    }
  } else if constexpr (NumUint32PerThread == 2) {
    *(uint2 *)lds_base_ptr =
        is_valid ? *(const uint2 *)global_base_ptr : make_uint2(0, 0);
  } else {
    *(uint4 *)lds_base_ptr =
        is_valid ? *(const uint4 *)global_base_ptr : make_uint4(0, 0, 0, 0);
  }
#undef TL_HCU_GLOBAL_TO_LDS_ASYNC_MULTIDWORD
}

} // namespace detail

// A committed TileLang async-copy group maps to one AMDGPU async mark.
TL_DEVICE void cp_async_commit() { __builtin_amdgcn_asyncmark(); }

// Global Memory only fence
__device__ void async_gld_fence(index_t cnt) {
  asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

// Global Memory and Shared Memory fence
__device__ void async_gld_sld_fence(index_t cnt) {
  asm volatile("s_waitcnt lgkmcnt(%0)" : : "n"(cnt) : "memory");
}

__device__ void wave_barrier() { asm volatile("s_barrier" : : : "memory"); }

template <int N = 0> TL_DEVICE void cp_async_wait() {
  static_assert(N >= 0 && N <= 65535,
                "async-copy group depth must fit unsigned short");
  __builtin_amdgcn_wait_asyncmark(static_cast<unsigned short>(N));
}

namespace detail {

template <int N, int SmemOffset>
TL_DEVICE void cp_async_gs_idxen_impl(uint32_t lds_base, int32x4_t src_resource,
                                      std::int32_t src_thread_byte_offset,
                                      std::int32_t idxen) {
  static_assert(N == 4 || N == 8 || N == 16,
                "idxen async copy only supports 4, 8, or 16 bytes");
  __attribute__((address_space(3))) int *lds_ptr =
      reinterpret_cast<__attribute__((address_space(3))) int *>(
          static_cast<uintptr_t>(lds_base + SmemOffset));
  __builtin_amdgcn_struct_buffer_load_async_lds(
      src_resource, lds_ptr, N, static_cast<uint32_t>(idxen),
      static_cast<uint32_t>(src_thread_byte_offset), 0, 0, 0);
}

template <int N> TL_DEVICE void clear_cp_async_lds(void *lds_lane_ptr) {
  if constexpr (N == 4) {
    *reinterpret_cast<uint32_t *>(lds_lane_ptr) = 0;
  } else if constexpr (N == 8) {
    typedef uint32_t uint32x2_t __attribute__((ext_vector_type(2)));
    *reinterpret_cast<uint32x2_t *>(lds_lane_ptr) = uint32x2_t{0, 0};
  } else {
    typedef uint32_t uint32x4_t __attribute__((ext_vector_type(4)));
    *reinterpret_cast<uint32x4_t *>(lds_lane_ptr) = uint32x4_t{0, 0, 0, 0};
  }
}

} // namespace detail

template <int N, int SmemOffset = 0>
TL_DEVICE void cp_async_gs_idxen(void *lds_base_ptr, int32x4_t src_resource,
                                 std::int32_t src_thread_byte_offset,
                                 std::int32_t idxen) {
  const uint32_t lds_base = __builtin_amdgcn_readfirstlane(
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds_base_ptr)));
  detail::cp_async_gs_idxen_impl<N, SmemOffset>(lds_base, src_resource,
                                                src_thread_byte_offset, idxen);
}

template <int N, int SmemOffset = 0>
TL_DEVICE void cp_async_gs_idxen(uint32_t lds_base, int32x4_t src_resource,
                                 std::int32_t src_thread_byte_offset,
                                 std::int32_t idxen) {
  detail::cp_async_gs_idxen_impl<N, SmemOffset>(lds_base, src_resource,
                                                src_thread_byte_offset, idxen);
}

template <int N, int SmemOffset = 0>
TL_DEVICE void
cp_async_gs_idxen_conditional(void *lds_base_ptr, int32x4_t src_resource,
                              std::int32_t src_thread_byte_offset,
                              std::int32_t idxen, bool condition) {
  if (condition) {
    cp_async_gs_idxen<N, SmemOffset>(lds_base_ptr, src_resource,
                                     src_thread_byte_offset, idxen);
  } else {
    detail::clear_cp_async_lds<N>(lds_base_ptr);
  }
}

template <int N, int SmemOffset = 0>
TL_DEVICE void cp_async_gs_idxen_conditional(
    void *lds_lane_ptr, uint32_t lds_base, int32x4_t src_resource,
    std::int32_t src_thread_byte_offset, std::int32_t idxen, bool condition) {
  if (condition) {
    cp_async_gs_idxen<N, SmemOffset>(lds_base, src_resource,
                                     src_thread_byte_offset, idxen);
  } else {
    detail::clear_cp_async_lds<N>(lds_lane_ptr);
  }
}

template <int N>
TL_DEVICE void cp_async_gs(void *lds_base_ptr, void const *global_base_ptr) {
  static_assert(N == 4 || N == 8 || N == 16,
                "tl::cp_async_gs: only N in {4,8,16}");
  detail::hcu_cp_async_gs_via_direct_lds<N / 4>(lds_base_ptr, global_base_ptr,
                                                true);
}

template <int N>
TL_DEVICE void cp_async_gs(void *lds_base_ptr, void const *global_thread_ptr,
                           int32x4_t src_resource,
                           std::int32_t src_thread_byte_offset) {
  static_assert(N == 4 || N == 8 || N == 16);
  detail::hcu_cp_async_gs_via_direct_lds_with_resource<N / 4>(
      lds_base_ptr, global_thread_ptr, src_resource, src_thread_byte_offset,
      true);
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void *lds_base_ptr,
                                       void const *global_base_ptr, bool cond) {
  static_assert(N == 4 || N == 8 || N == 16,
                "tl::cp_async_gs_conditional: only N in {4,8,16}");
  detail::hcu_cp_async_gs_via_direct_lds<N / 4>(lds_base_ptr, global_base_ptr,
                                                cond);
}

template <int N>
TL_DEVICE void
cp_async_gs_conditional(void *lds_base_ptr, void const *global_thread_ptr,
                        int32x4_t src_resource,
                        std::int32_t src_thread_byte_offset, bool condition) {
  static_assert(N == 4 || N == 8 || N == 16);
  detail::hcu_cp_async_gs_via_direct_lds_with_resource<N / 4>(
      lds_base_ptr, global_thread_ptr, src_resource, src_thread_byte_offset,
      condition);
}

template <typename T, tl::index_t N, bool oob_conditional_check = true>
TL_DEVICE tl::thread_buffer<T, N>
hcu_buffer_load(const T *p_src_wave, tl::index_t src_thread_element_offset,
                bool src_thread_element_valid,
                tl::index_t src_element_space_size) {
  const int32x4_t src_wave_buffer_resource =
      make_wave_buffer_resource(p_src_wave);

  tl::index_t src_thread_addr_offset = [&]() {
    if constexpr (oob_conditional_check)
      return src_thread_element_valid ? src_thread_element_offset * sizeof(T)
                                      : 0xffffffff;
    else
      return src_thread_element_offset * sizeof(T);
  }();
  return tl::hcu_buffer_load_impl<T, N>(src_wave_buffer_resource,
                                        src_thread_addr_offset, 0);
}

template <typename T, tl::index_t N, bool oob_conditional_check = true>
TL_DEVICE void hcu_buffer_store(const tl::thread_buffer<T, N> &src_thread_data,
                                T *p_dst_wave,
                                const tl::index_t dst_thread_element_offset,
                                const bool dst_thread_element_valid,
                                const tl::index_t dst_element_space_size) {
  const int32x4_t dst_wave_buffer_resource =
      make_wave_buffer_resource(p_dst_wave);

  tl::index_t dst_thread_addr_offset = [&]() {
    if constexpr (oob_conditional_check)
      return dst_thread_element_valid ? dst_thread_element_offset * sizeof(T)
                                      : 0xffffffff;
    else
      return dst_thread_element_offset * sizeof(T);
  }();
  tl::hcu_buffer_store_impl<T, N>(src_thread_data, dst_wave_buffer_resource,
                                  dst_thread_addr_offset, 0);
}

} // namespace tl
