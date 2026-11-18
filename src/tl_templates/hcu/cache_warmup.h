#pragma once

#include <cstdint>
#include <tl_templates/hcu/common.h>

namespace tl {

struct CacheWarmupRequest {
  int32x4_t resource;
  unsigned byte_offset;
};

template <typename T>
TL_DEVICE CacheWarmupRequest cache_warmup(const T *src, std::int64_t row_offset,
                                          std::int64_t num_rows,
                                          std::int64_t row_size,
                                          std::int64_t num_tile_rows) {
  // Sampling interval in bytes and maximum number of sample locations.
  constexpr std::uint64_t kPageBytes = 2ull * 1024 * 1024;
  constexpr unsigned kMaxPages = 16;

  const std::int64_t remaining_rows = num_rows - row_offset;
  const std::int64_t valid_rows =
      num_tile_rows < remaining_rows ? num_tile_rows : remaining_rows;
  const std::uint64_t span_bytes = static_cast<std::uint64_t>(valid_rows) *
                                   static_cast<std::uint64_t>(row_size) *
                                   sizeof(T);

  const std::uint64_t page_count =
      span_bytes >= sizeof(unsigned)
          ? 1 + (span_bytes - sizeof(unsigned)) / kPageBytes
          : 0;
  const unsigned pages =
      page_count > kMaxPages ? kMaxPages : static_cast<unsigned>(page_count);
  const unsigned thread =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const unsigned lane = thread % warpSize;
  // Each wave samples the same locations; remaining lanes reuse offset zero.
  const unsigned page = lane < pages ? lane : 0;
  unsigned byte_offset = static_cast<unsigned>(page * kPageBytes);

  const T *tile_base = src + static_cast<std::uint64_t>(row_offset) * row_size;

  auto resource = make_wave_buffer_resource(tile_base, 0xfffffff8u);
  resource.w = (7 << 12) | (4 << 15);
  return {resource, byte_offset};
}

} // namespace tl
