#pragma once

// MLS resource descriptor helpers (trimmed from ck_tile hcu_matrix_addressing).
// Load instructions live in tl_mls_atom_*.hpp (__builtin_hcu_matrix_load_*).

#include <tl_templates/hcu/core/numeric/integer.hpp>
#include <tl_templates/hcu/core/numeric/integral_constant.hpp>
#include <tl_templates/hcu/core/numeric/vector_type.hpp>
#include <tl_templates/hcu/core/utility/bit_cast.hpp>

namespace tl {
namespace mls {

union mls_addr_union {
  struct {
    int32_t addr_lo;
    int32_t addr_hi;
  };

  uintptr_t addr;
};

TL_DEVICE void move_mls_addr_base(int32x4_t &mls_res,
                                  const index_t addr_byte_offset) {
  mls_addr_union addr_union{{mls_res.x, mls_res.y}};

  addr_union.addr += addr_byte_offset;
  mls_res.x = addr_union.addr_lo;
  mls_res.y = addr_union.addr_hi;
}

// Set mls_res addr to base_addr + addr_byte_offset (absolute, not add).
// Used by move_base_to for LICM-friendly load pattern.
TL_DEVICE void update_mls_addr_base(int32x4_t &mls_res, uintptr_t base_addr,
                                    index_t addr_byte_offset) {
  mls_addr_union addr_union;
  addr_union.addr = base_addr + addr_byte_offset;
  mls_res.x = addr_union.addr_lo;
  mls_res.y = addr_union.addr_hi;
}

struct __attribute__((packed)) mls_resource {
  const void *ptr;

  uint32_t stride;

  union {
    struct {
      uint32_t m_filter : 8;
      uint32_t nm_filter : 8;
      uint32_t cache_swizzle_enable : 1;
      uint32_t mfmt : 2;
      uint32_t reserved : 13;
    };

    uint32_t DW3_DATA;
  } DW3_CONFIG_UNION;
};

template <index_t mfmt>
TL_DEVICE int32x4_t make_mls_resource(const void *ptr,
                                      const uint32_t stride, // in elements
                                      const index_t m_filter,
                                      const index_t nm_filter,
                                      number<mfmt>) // interleave
{
  mls_resource res{ptr, 0, {{0, 0, 0, 0, 0}}};

  res.stride = stride;
  res.DW3_CONFIG_UNION.m_filter = m_filter;
  res.DW3_CONFIG_UNION.nm_filter = nm_filter;
  res.DW3_CONFIG_UNION.cache_swizzle_enable = 1;
  res.DW3_CONFIG_UNION.mfmt = mfmt;

  int32x4_t r = bit_cast<int32x4_t>(res);
  r.x = __builtin_amdgcn_readfirstlane(r.x);
  r.y = __builtin_amdgcn_readfirstlane(r.y);
  r.z = __builtin_amdgcn_readfirstlane(r.z);
  r.w = __builtin_amdgcn_readfirstlane(r.w);
  return r;
}

namespace detail {

template <index_t Interleave> struct mfmt_traits;
template <> struct mfmt_traits<1> {
  static constexpr auto value = number<0>{};
};
template <> struct mfmt_traits<2> {
  static constexpr auto value = number<1>{};
};
template <> struct mfmt_traits<4> {
  static constexpr auto value = number<2>{};
};
template <> struct mfmt_traits<8> {
  static constexpr auto value = number<3>{};
};

} // namespace detail
} // namespace mls
} // namespace tl
