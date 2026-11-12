#pragma once

#include <cstdint>
#include <tl_templates/hcu/common.h>

namespace tl {

template <typename T>
TL_DEVICE unsigned cache_warmup(const T *src, std::int64_t row_offset,
                                std::int64_t num_rows, std::int64_t row_size,
                                std::int64_t num_tile_rows) {
  // Sampling interval in bytes and maximum number of sample locations.
  constexpr std::uint64_t kPageBytes = 2ull * 1024 * 1024;
  constexpr unsigned kMaxPages = 16;

  // Skip empty regions and requests outside the source row range.
  if (num_rows <= 0 || row_size <= 0 || num_tile_rows <= 0 || row_offset < 0 ||
      row_offset >= num_rows) {
    return 0;
  }

  const std::int64_t remaining_rows = num_rows - row_offset;
  const std::int64_t valid_rows =
      num_tile_rows < remaining_rows ? num_tile_rows : remaining_rows;
  const std::uint64_t span_bytes = static_cast<std::uint64_t>(valid_rows) *
                                   static_cast<std::uint64_t>(row_size) *
                                   sizeof(T);
  if (span_bytes < sizeof(unsigned)) {
    return 0;
  }

  const std::uint64_t page_count = 1 + (span_bytes - 1) / kPageBytes;
  const unsigned pages =
      page_count > kMaxPages ? kMaxPages : static_cast<unsigned>(page_count);
  const unsigned thread =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const unsigned lane = thread % warpSize;
  // Each wave samples the same locations; remaining lanes reuse offset zero.
  const unsigned page = lane < pages ? lane : 0;
  unsigned byte_offset = static_cast<unsigned>(page * kPageBytes);
  if (byte_offset > span_bytes - sizeof(unsigned)) {
    byte_offset = 0;
  }

  const T *tile_base = src + static_cast<std::uint64_t>(row_offset) * row_size;

  auto resource = make_wave_buffer_resource(tile_base, 0xfffffff8u);
  resource.w = (7 << 12) | (4 << 15);

  // Issue the request without an explicit wait; codegen keeps the result live.
  unsigned value;
  asm volatile("buffer_load_dword %0, %1, %2, 0 offen"
               : "=v"(value)
               : "v"(byte_offset), "s"(resource)
               : "memory");
  return value;
}

} // namespace tl
