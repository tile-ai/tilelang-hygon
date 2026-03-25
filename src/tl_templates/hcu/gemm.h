#pragma once

#include <type_traits>
#include "common.h"

namespace tl {

// Trait to determine the MMAC instruction to use based on data type
template <typename T> struct MmacTraits;

// Specialization for int8
template <> struct MmacTraits<int8_t> {
  template <typename AccType>
  static TL_DEVICE void mmac_op(const int8_t *b, const int8_t *a, AccType *c) {
    int32x2 *a_packed = reinterpret_cast<int32x2 *>(const_cast<int8_t *>(a));
    int32x2 *b_packed = reinterpret_cast<int32x2 *>(const_cast<int8_t *>(b));
#if defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx946__)
    // default: lit en, clamp disable, lts disable
    *c = __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts(*a_packed, *b_packed, *c, 1, 0, 0);
#else
    *c = __builtin_hcu_mmac_i32_16x16x32_i8(*a_packed, *b_packed, *c);
#endif
  }
};

// Specialization for half/float16
template <> struct MmacTraits<half> {
  template <typename AccType>
  static TL_DEVICE void mmac_op(const half *b, const half *a, AccType *c) {
#if defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx946__)
    *c = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(*((float16x4 *)a),
                                             *((float16x4 *)b), *c, 1, 0);
#else
    *c = __builtin_hcu_mmac_f32_16x16x16_f16(*((float16x4 *)a),
                                             *((float16x4 *)b), *c);
#endif
  }
};

// Specialization for bfloat16_t
template <> struct MmacTraits<bfloat16_t> {
  template <typename AccType>
  static TL_DEVICE void mmac_op(const bfloat16_t *b, const bfloat16_t *a,
                                AccType *c) {
    bfloat16x4_vec b_vec, a_vec;

    // Reinterpret the pointers
    short *b_short = reinterpret_cast<short *>(const_cast<bfloat16_t *>(b));
    short *a_short = reinterpret_cast<short *>(const_cast<bfloat16_t *>(a));

    // Copy the data
    for (int i = 0; i < 4; ++i) {
      b_vec[i] = b_short[i];
      a_vec[i] = a_short[i];
    }

    // Call the intrinsic and store the result directly to c
#if defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx946__)
    *c = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a_vec, b_vec, *c, 1, 0);
#else
    *c = __builtin_hcu_mmac_f32_16x16x16_bf16(a_vec, b_vec, *c);
#endif
  }
};

// Specialization for float
template <> struct MmacTraits<float> {
  template <typename AccType>
  static TL_DEVICE void mmac_op(const float *b, const float *a, AccType *c) {
    *c = __builtin_hcu_mmac_16x16x8_f32(*((float32x2 *)a),
                                        *((float32x2 *)b), *c);
  }
};

#if defined(HIP_FP8_ENABLED)
// Specialization for fp8_e4_t
template <> struct MmacTraits<fp8_e4_t> {
  template <typename AccType>
  static TL_DEVICE void mmac_op(const fp8_e4_t *b, const fp8_e4_t *a,
                                AccType *c) {
    int32x2 a_val = *reinterpret_cast<const int32x2 *>(a);
    int32x2 b_val = *reinterpret_cast<const int32x2 *>(b);
#if defined(__HIP_DEVICE_COMPILE__) && (!defined(__gfx938__) && !defined(__gfx92a__) && !defined(__gfx946__))
#error "fp8_e4_t MMAC operations are only supported on gfx938, gfx92a, and gfx946 architectures"
#elif defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx946__)
    *c = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(a_val, b_val, *c, 1, 0);
#endif
  }
};

// Specialization for fp8_e5_t
template <> struct MmacTraits<fp8_e5_t> {
  template <typename AccType>
  static TL_DEVICE void mmac_op(const fp8_e5_t *b, const fp8_e5_t *a,
                                AccType *c) {
    int32x2 a_val = *reinterpret_cast<const int32x2 *>(a);
    int32x2 b_val = *reinterpret_cast<const int32x2 *>(b);
#if defined(__HIP_DEVICE_COMPILE__) && (!defined(__gfx938__) && !defined(__gfx92a__) && !defined(__gfx946__))
#error "fp8_e5_t MMAC operations are only supported on gfx938, gfx92a, and gfx946 architectures"
#elif defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx946__)
    *c = __builtin_hcu_mmac_f32_16x16x32_bf8_bf8_lit_lts(a_val, b_val, *c, 1, 0);
#endif
  }
};
#endif

// ref to bitblas/tl/mfma_macro_generator.py::kPack
template <int M, int N, int K, int num_warp_m, int num_warp_n, bool TransposeA,
          bool TransposeB, bool clear_accum, int kPack, typename A_type,
          typename B_type, typename C_type, typename AccDataType = float,
          int min_n_per_warp = 16>
class GemmTensorOp {
public:
  // Note: clear_accum=true is not fully supported in HIP implementation
  // but we'll handle it by manually clearing the accumulator
  // static_assert(!clear_accum, "clear_accum=true is not supported yet");

  static constexpr int micro_size_x = 16;
  static constexpr int micro_size_y = 16;
  static constexpr int micro_size_k = 32 / sizeof(A_type);
  static constexpr int vec_size = 8 / sizeof(A_type);

  // This part comes from the Codegen
  static constexpr int M_Tile = M;
  static constexpr int N_Tile = N;
  static constexpr int K_Tile = K;

  static constexpr int block_row_warps = num_warp_m;
  static constexpr int block_col_warps = num_warp_n;
  static constexpr int block_col_warps_no_recompute =
      std::min(num_warp_n, N_Tile / min_n_per_warp);

  static constexpr int inner_k = K_Tile / (micro_size_k * kPack);
  static constexpr int warp_rows = M_Tile / (block_row_warps * micro_size_x);
  static constexpr int warp_cols = N_Tile / (block_col_warps_no_recompute * micro_size_y);

  // The kPadA, kPadB, kPadC & kBlockPerCu should also come from the Codegen
  // part.
  static constexpr bool kPadA = true;
  static constexpr bool kPadB = true;
  static constexpr bool kPadC = true;

  static constexpr int BANK_SIZE_BYTES = 128;

  static constexpr int warp_size = 64;

  TL_DEVICE static constexpr auto reverse_index_map(int thread_id,
                                                    int local_id) {
    return std::make_pair(thread_id % 16,
                          (thread_id / 16) * (vec_size * kPack) + local_id);
  }

  TL_DEVICE static constexpr auto reverse_index_map_transposed(int thread_id,
                                                               int local_id) {
    return std::make_pair((thread_id / 16) * (vec_size * kPack) + local_id,
                          thread_id % 16);
  }

  /*
   * Detailed Implementation please
   * checkout bitblas/tl/utils.py:get_swizzle_layout
   */
  template <int continuous = 32, int element_size = 2>
  TL_DEVICE static auto make_mmac_swizzle_layout(const int row, const int col) {
    const auto dtype_bits = element_size * 8;

    const int numBanks = 32;
    const int bankBitWidth = 32;
    const int SIMDWidth = 16;
    const int vecSize = vec_size * kPack;
    const int innerDimLength = continuous;
    const int typeWidthInBit = dtype_bits;

    const int elemsPerOneBanksRow = (numBanks * bankBitWidth) / typeWidthInBit;
    const int perPhase = std::max(1, elemsPerOneBanksRow / innerDimLength);
    const int maxPhase =
        std::min(SIMDWidth / perPhase, innerDimLength / vecSize);

    const int phase = (row / perPhase) % maxPhase;
    const int colOffSwizzled = (((col / vecSize) ^ phase) * vecSize);
    const int colOffOrdered = col % vecSize;
    const int colOff = colOffSwizzled + colOffOrdered;

    return std::make_pair(row, colOff);
  }

  template <int continuous = 32, int element_size = 2>
  TL_DEVICE static constexpr auto make_layout_padded(const int row,
                                                     const int col) {
    return std::make_pair(row, col);
  }

  template <int continuous = 32, int element_size = 2>
  TL_DEVICE static constexpr auto make_swizzle_layout(const int row,
                                                      const int col) {
    auto [n_row, n_col] =
        make_mmac_swizzle_layout<continuous, element_size>(row, col);
    return n_row * continuous + n_col;
  }

#if 0
  //FIXME: This shuffle function is not correct but just for testing,
  //       we leave it here for reference.
  static TL_DEVICE void vectorize_c_local(int lane, C_type *C_local) {
    // For each float in the float32x4 vector, compute which thread has the
    // value we need in consecutive layout and shuffle it
    for (int i = 0; i < warp_rows; ++i) {
      for (int j = 0; j < warp_cols; ++j) {
        float permuted_vec[4];
        auto acc_ptr = ((float32x4 *)C_local) + ((i * warp_cols) + j);
        permuted_vec[0] = ck_tile::warp_shuffle(((C_type*)acc_ptr)[0], (lane + 16) % warp_size);
        //permuted_vec[1] = ck_tile::warp_shuffle(acc_ptr[0], (lane + 32) % warp_size);
        //permuted_vec[2] = ck_tile::warp_shuffle(acc_ptr[0], (lane + 48) % warp_size);
        //permuted_vec[3] = ck_tile::warp_shuffle(acc_ptr[0], (lane + 64) % warp_size);
        ((C_type*)acc_ptr)[0] = permuted_vec[0];
       }
    }
  }
#endif

  static TL_DEVICE void body(A_type *A_shared, B_type *B_shared,
                             C_type *C_local) {
    auto tid = threadIdx.x;
    auto warp_id = tid / warp_size;
    auto warp_n = warp_id / block_row_warps;
    // we allways recompute on n warps when total warps > max warps needed
    if constexpr (block_col_warps_no_recompute != block_col_warps) {
      warp_n = warp_n % block_col_warps_no_recompute;
    }
    auto warp_m = warp_id % block_row_warps;
    constexpr auto warp_row_tiles = warp_rows * micro_size_x;
    constexpr auto warp_col_tiles = warp_cols * micro_size_y;

    auto lane_id = tid % warp_size;
    auto tx = lane_id;

    constexpr auto local_size_a = (micro_size_x * micro_size_k) / warp_size;
    constexpr auto local_size_b = (micro_size_y * micro_size_k) / warp_size;
    constexpr auto local_size_c = (micro_size_x * micro_size_y) / warp_size;

    constexpr auto last_dim_a = TransposeA ? M_Tile : K_Tile;
    constexpr auto last_dim_b = TransposeB ? K_Tile : N_Tile;

    A_type A_local[warp_rows * kPack * local_size_a];
    B_type B_local[warp_cols * kPack * local_size_b];

    for (int ki = 0; ki < inner_k; ki++) {
      // Fetch A into register
      for (int i = 0; i < warp_rows; i++) {
        const auto l = warp_m * warp_row_tiles + i * micro_size_x;
        const auto r = ki * (kPack * micro_size_k);
        for (int local_id = 0; local_id < (kPack * local_size_a); local_id++) {
          if constexpr (TransposeA) {
            auto [row, col] = reverse_index_map_transposed(lane_id, local_id);
            A_local[i * kPack * local_size_a + local_id] =
                A_shared[make_swizzle_layout<last_dim_a, sizeof(A_type)>(
                    r + row, l + col)];
          } else {
            auto [row, col] = reverse_index_map(lane_id, local_id);
            A_local[i * kPack * local_size_a + local_id] =
                A_shared[make_swizzle_layout<last_dim_a, sizeof(A_type)>(
                    l + row, r + col)];
          }
        }
      }
      // Fetch B into register
      for (int j = 0; j < warp_cols; j++) {
        const auto l = warp_n * warp_col_tiles + j * micro_size_y;
        const auto r = ki * (kPack * micro_size_k);
        for (int local_id = 0; local_id < (kPack * local_size_b); local_id++) {
          if constexpr (TransposeB) {
            auto [row, col] = reverse_index_map(lane_id, local_id);
            B_local[j * kPack * local_size_b + local_id] =
                B_shared[make_swizzle_layout<last_dim_b, sizeof(B_type)>(
                    l + row, r + col)];
          } else {
            auto [row, col] = reverse_index_map_transposed(lane_id, local_id);
            B_local[j * kPack * local_size_b + local_id] =
                B_shared[make_swizzle_layout<last_dim_b, sizeof(B_type)>(
                    r + row, l + col)];
          }
        }
      }
      // Compute
      for (int kp = 0; kp < kPack; kp++) {
        for (int i = 0; i < warp_rows; ++i) {
          for (int j = 0; j < warp_cols; ++j) {
            auto acc_ptr = ((float32x4 *)C_local) + ((i * warp_cols) + j);
            auto b_ptr = ((B_type *)B_local) + (j * kPack + kp) * vec_size;
            auto a_ptr = ((A_type *)A_local) + (i * kPack + kp) * vec_size;

            // Use the trait to select the correct MMAC instruction, either fp8,
            // fp16 or bf16 currently
            MmacTraits<A_type>::mmac_op(b_ptr, a_ptr, acc_ptr);
          }
        }
      }
    }
  }

  static TL_DEVICE void body_rs(A_type *A_local, B_type *B_shared,
                                C_type *C_local) {
    auto tid = threadIdx.x;
    auto warp_id = tid / warp_size;
    auto warp_n = warp_id / block_row_warps;
    // we allways recompute on n warps when total warps > max warps needed
    if constexpr (block_col_warps_no_recompute != block_col_warps) {
      warp_n = warp_n % block_col_warps_no_recompute;
    }
    constexpr auto warp_row_tiles = warp_rows * micro_size_x;
    constexpr auto warp_col_tiles = warp_cols * micro_size_y;

    auto lane_id = tid % warp_size;
    auto tx = lane_id;

    constexpr auto local_size_a = (micro_size_x * micro_size_k) / warp_size;
    constexpr auto local_size_b = (micro_size_y * micro_size_k) / warp_size;
    constexpr auto local_size_c = (micro_size_x * micro_size_y) / warp_size;

    constexpr auto last_dim_a = TransposeA ? M_Tile : K_Tile;
    constexpr auto last_dim_b = TransposeB ? K_Tile : N_Tile;

    B_type B_local[warp_cols * kPack * local_size_b];

    for (int ki = 0; ki < inner_k; ki++) {
      // Fetch B into register
      for (int j = 0; j < warp_cols; j++) {
        const auto l = warp_n * warp_col_tiles + j * micro_size_y;
        const auto r = ki * kPack * micro_size_k;
        for (int local_id = 0; local_id < kPack * local_size_b; local_id++) {
          if constexpr (TransposeB) {
            auto [row, col] = reverse_index_map(lane_id, local_id);
            B_local[j * kPack * local_size_b + local_id] =
                B_shared[make_swizzle_layout<last_dim_b, sizeof(B_type)>(
                    l + row, r + col)];
          } else {
            auto [row, col] = reverse_index_map_transposed(lane_id, local_id);
            B_local[j * kPack * local_size_b + local_id] =
                B_shared[make_swizzle_layout<last_dim_b, sizeof(B_type)>(
                    r + row, l + col)];
          }
        }
      }

      // Compute
      for (int kp = 0; kp < kPack; kp++) {
        for (int i = 0; i < warp_rows; ++i) {
          for (int j = 0; j < warp_cols; ++j) {
            auto acc_ptr = ((float32x4 *)C_local) + ((i * warp_cols) + j);
            auto b_ptr = ((B_type *)B_local) + (j * kPack + kp) * vec_size;
            auto a_ptr = ((A_type *)A_local) +
                         (ki * warp_rows * kPack + i * kPack + kp) * vec_size;

            // Use the trait to select the correct MMAC instruction, either fp8,
            // fp16 or bf16 currently
            MmacTraits<A_type>::mmac_op(b_ptr, a_ptr, acc_ptr);
          }
        }
      }
    }
  }

  static TL_DEVICE void body_rr(A_type *A_local, B_type *B_local,
                                C_type *C_local) {
    for (int ki = 0; ki < inner_k; ki++) {
      // Compute
      for (int kp = 0; kp < kPack; kp++) {
        for (int i = 0; i < warp_rows; ++i) {
          for (int j = 0; j < warp_cols; ++j) {
            auto acc_ptr = ((float32x4 *)C_local) + ((i * warp_cols) + j);
            auto b_ptr = ((B_type *)B_local) +
                         (ki * warp_cols * kPack + j * kPack + kp) * vec_size;
            auto a_ptr = ((A_type *)A_local) +
                         (ki * warp_rows * kPack + i * kPack + kp) * vec_size;
            // Use the trait to select the correct MMAC instruction, either fp8,
            // fp16 or bf16 currently
            MmacTraits<A_type>::mmac_op(b_ptr, a_ptr, acc_ptr);
          }
        }
      }
    }
  }

  static TL_DEVICE void body_sr(A_type *A_shared, B_type *B_local,
                             C_type *C_local) {
    auto tid = threadIdx.x;
    auto warp_id = tid / warp_size;
    auto warp_m = warp_id % block_row_warps;
    constexpr auto warp_row_tiles = warp_rows * micro_size_x;
    auto lane_id = tid % warp_size;

    constexpr auto local_size_a = (micro_size_x * micro_size_k) / warp_size;
    constexpr auto last_dim_a = TransposeA ? M_Tile : K_Tile;

    A_type A_local[warp_rows * kPack * local_size_a];

    for (int ki = 0; ki < inner_k; ki++) {
      // Fetch A into register
      for (int i = 0; i < warp_rows; i++) {
        const auto l = warp_m * warp_row_tiles + i * micro_size_x;
        const auto r = ki * (kPack * micro_size_k);
        for (int local_id = 0; local_id < (kPack * local_size_a); local_id++) {
          if constexpr (TransposeA) {
            auto [row, col] = reverse_index_map_transposed(lane_id, local_id);
            A_local[i * kPack * local_size_a + local_id] =
                A_shared[make_swizzle_layout<last_dim_a, sizeof(A_type)>(
                    r + row, l + col)];
          } else {
            auto [row, col] = reverse_index_map(lane_id, local_id);
            A_local[i * kPack * local_size_a + local_id] =
                A_shared[make_swizzle_layout<last_dim_a, sizeof(A_type)>(
                    l + row, r + col)];
          }
        }
      }
      // Compute
      for (int kp = 0; kp < kPack; kp++) {
        for (int i = 0; i < warp_rows; ++i) {
          for (int j = 0; j < warp_cols; ++j) {
            auto acc_ptr = ((float32x4 *)C_local) + ((i * warp_cols) + j);
            auto b_ptr = ((B_type *)B_local) +
                         (ki * warp_cols * kPack + j * kPack + kp) * vec_size;
            auto a_ptr = ((A_type *)A_local) + (i * kPack + kp) * vec_size;

            // Use the trait to select the correct MMAC instruction, either fp8,
            // fp16 or bf16 currently
            MmacTraits<A_type>::mmac_op(b_ptr, a_ptr, acc_ptr);
          }
        }
      }
    }
  }
};

// GemmTensorOp with num_warp_k support
// Warp id traversal order: warp_m -> warp_n -> warp_k (innermost to outermost)
template <int M, int N, int K, int num_warp_m, int num_warp_n, int num_warp_k,
          bool TransposeA, bool TransposeB, bool clear_accum, int kPack,
          typename A_type, typename B_type, typename C_type,
          typename AccDataType = float>
class GemmTensorOpKPartition {
public:
  static constexpr int micro_size_x = 16;
  static constexpr int micro_size_y = 16;
  static constexpr int micro_size_k = 32 / sizeof(A_type);
  static constexpr int vec_size = 8 / sizeof(A_type);

  static constexpr int M_Tile = M;
  static constexpr int N_Tile = N;
  static constexpr int K_Tile = K;

  static constexpr int block_row_warps = num_warp_m;
  static constexpr int block_col_warps = num_warp_n;
  static constexpr int block_k_warps = num_warp_k;

  // Each warp handles K / num_warp_k
  static constexpr int warp_k = K_Tile / num_warp_k;
  static constexpr int inner_k = warp_k / (micro_size_k * kPack);
  static constexpr int warp_rows = M_Tile / (block_row_warps * micro_size_x);
  static constexpr int warp_cols =
      N_Tile / (block_col_warps * micro_size_y);

  static constexpr bool kPadA = true;
  static constexpr bool kPadB = true;
  static constexpr bool kPadC = true;

  static constexpr int BANK_SIZE_BYTES = 128;
  static constexpr int warp_size = 64;

  TL_DEVICE static constexpr auto reverse_index_map(int thread_id,
                                                    int local_id) {
    return std::make_pair(thread_id % 16,
                          (thread_id / 16) * (vec_size * kPack) + local_id);
  }

  TL_DEVICE static constexpr auto reverse_index_map_transposed(int thread_id,
                                                               int local_id) {
    return std::make_pair((thread_id / 16) * (vec_size * kPack) + local_id,
                          thread_id % 16);
  }

  template <int continuous = 32, int element_size = 2>
  TL_DEVICE static auto make_mmac_swizzle_layout(const int row, const int col) {
    const auto dtype_bits = element_size * 8;

    const int numBanks = 32;
    const int bankBitWidth = 32;
    const int SIMDWidth = 16;
    const int vecSize = vec_size * kPack;
    const int innerDimLength = continuous;
    const int typeWidthInBit = dtype_bits;

    const int elemsPerOneBanksRow = (numBanks * bankBitWidth) / typeWidthInBit;
    const int perPhase = std::max(1, elemsPerOneBanksRow / innerDimLength);
    const int maxPhase =
        std::min(SIMDWidth / perPhase, innerDimLength / vecSize);

    const int phase = (row / perPhase) % maxPhase;
    const int colOffSwizzled = (((col / vecSize) ^ phase) * vecSize);
    const int colOffOrdered = col % vecSize;
    const int colOff = colOffSwizzled + colOffOrdered;

    return std::make_pair(row, colOff);
  }

  template <int continuous = 32, int element_size = 2>
  TL_DEVICE static constexpr auto make_layout_padded(const int row,
                                                     const int col) {
    return std::make_pair(row, col);
  }

  template <int continuous = 32, int element_size = 2>
  TL_DEVICE static constexpr auto make_swizzle_layout(const int row,
                                                      const int col) {
    auto [n_row, n_col] =
        make_mmac_swizzle_layout<continuous, element_size>(row, col);
    return n_row * continuous + n_col;
  }

  static TL_DEVICE void body_rs(A_type *A_local, B_type *B_shared,
                                C_type *C_local) {
    auto tid = threadIdx.x;
    auto warp_id = tid / warp_size;

    // Warp id traversal order: warp_m -> warp_n -> warp_k (innermost to outermost)
    auto warp_m = warp_id % block_row_warps;
    auto warp_n = (warp_id / block_row_warps) % block_col_warps;
    auto warp_k_idx = warp_id / (block_row_warps * block_col_warps);

    constexpr auto warp_row_tiles = warp_rows * micro_size_x;
    constexpr auto warp_col_tiles = warp_cols * micro_size_y;

    auto lane_id = tid % warp_size;
    auto tx = lane_id;

    constexpr auto local_size_a = (micro_size_x * micro_size_k) / warp_size;
    constexpr auto local_size_b = (micro_size_y * micro_size_k) / warp_size;
    constexpr auto local_size_c = (micro_size_x * micro_size_y) / warp_size;

    constexpr auto last_dim_a = TransposeA ? M_Tile : K_Tile;
    constexpr auto last_dim_b = TransposeB ? K_Tile : N_Tile;

    B_type B_local[warp_cols * kPack * local_size_b];

    // Each warp handles K / num_warp_k, so we need to offset by warp_k_idx * warp_k
    const int k_offset = warp_k_idx * warp_k;

    for (int ki = 0; ki < inner_k; ki++) {
      // Fetch B into register
      for (int j = 0; j < warp_cols; j++) {
        const auto l = warp_n * warp_col_tiles + j * micro_size_y;
        // Add k_offset to account for warp_k partitioning
        const auto r = k_offset + ki * kPack * micro_size_k;
        for (int local_id = 0; local_id < kPack * local_size_b; local_id++) {
          if constexpr (TransposeB) {
            auto [row, col] = reverse_index_map(lane_id, local_id);
            B_local[j * kPack * local_size_b + local_id] =
                B_shared[make_swizzle_layout<last_dim_b, sizeof(B_type)>(
                    l + row, r + col)];
          } else {
            auto [row, col] = reverse_index_map_transposed(lane_id, local_id);
            B_local[j * kPack * local_size_b + local_id] =
                B_shared[make_swizzle_layout<last_dim_b, sizeof(B_type)>(
                    r + row, l + col)];
          }
        }
      }

      // Compute
      for (int kp = 0; kp < kPack; kp++) {
        for (int i = 0; i < warp_rows; ++i) {
          for (int j = 0; j < warp_cols; ++j) {
            auto acc_ptr = ((float32x4 *)C_local) + ((i * warp_cols) + j);
            auto b_ptr = ((B_type *)B_local) + (j * kPack + kp) * vec_size;
            // A_local is already partitioned per warp, so no k_offset needed
            auto a_ptr = ((A_type *)A_local) +
                         (ki * warp_rows * kPack + i * kPack + kp) * vec_size;

            // Use the trait to select the correct MMAC instruction
            MmacTraits<A_type>::mmac_op(b_ptr, a_ptr, acc_ptr);
          }
        }
      }
    }
  }

  static TL_DEVICE void body_rr(A_type *A_local, B_type *B_local,
                                C_type *C_local) {
    // For body_rr, A and B are already in registers (local memory).
    // The caller should have prepared A_local and B_local with the correct
    // K range for this warp (warp_k_idx * warp_k to (warp_k_idx + 1) * warp_k).
    // The inner_k is already computed based on warp_k, so we just need to
    // iterate over it similar to the original body_rr.
    for (int ki = 0; ki < inner_k; ki++) {
      // Compute
      for (int kp = 0; kp < kPack; kp++) {
        for (int i = 0; i < warp_rows; ++i) {
          for (int j = 0; j < warp_cols; ++j) {
            auto acc_ptr = ((float32x4 *)C_local) + ((i * warp_cols) + j);
            auto b_ptr = ((B_type *)B_local) +
                         (ki * warp_cols * kPack + j * kPack + kp) * vec_size;
            auto a_ptr = ((A_type *)A_local) +
                         (ki * warp_rows * kPack + i * kPack + kp) * vec_size;
            // Use the trait to select the correct MMAC instruction
            MmacTraits<A_type>::mmac_op(b_ptr, a_ptr, acc_ptr);
          }
        }
      }
    }
  }

  static TL_DEVICE void body_ss(A_type *A_shared, B_type *B_shared,
                                C_type *C_local) {
    auto tid = threadIdx.x;
    auto warp_id = tid / warp_size;

    // Warp id traversal order: warp_m -> warp_n -> warp_k (innermost to outermost)
    auto warp_m = warp_id % block_row_warps;
    auto warp_n = (warp_id / block_row_warps) % block_col_warps;
    auto warp_k_idx = warp_id / (block_row_warps * block_col_warps);

    constexpr auto warp_row_tiles = warp_rows * micro_size_x;
    constexpr auto warp_col_tiles = warp_cols * micro_size_y;

    auto lane_id = tid % warp_size;
    auto tx = lane_id;

    constexpr auto local_size_a = (micro_size_x * micro_size_k) / warp_size;
    constexpr auto local_size_b = (micro_size_y * micro_size_k) / warp_size;
    constexpr auto local_size_c = (micro_size_x * micro_size_y) / warp_size;

    constexpr auto last_dim_a = TransposeA ? M_Tile : K_Tile;
    constexpr auto last_dim_b = TransposeB ? K_Tile : N_Tile;

    A_type A_local[warp_rows * kPack * local_size_a];
    B_type B_local[warp_cols * kPack * local_size_b];

    // Each warp handles K / num_warp_k, so we need to offset by warp_k_idx * warp_k
    const int k_offset = warp_k_idx * warp_k;

    for (int ki = 0; ki < inner_k; ki++) {
      // Fetch A into register
      for (int i = 0; i < warp_rows; i++) {
        const auto l = warp_m * warp_row_tiles + i * micro_size_x;
        // Add k_offset to account for warp_k partitioning
        const auto r = k_offset + ki * (kPack * micro_size_k);
        for (int local_id = 0; local_id < (kPack * local_size_a); local_id++) {
          if constexpr (TransposeA) {
            auto [row, col] = reverse_index_map_transposed(lane_id, local_id);
            A_local[i * kPack * local_size_a + local_id] =
                A_shared[make_swizzle_layout<last_dim_a, sizeof(A_type)>(
                    r + row, l + col)];
          } else {
            auto [row, col] = reverse_index_map(lane_id, local_id);
            A_local[i * kPack * local_size_a + local_id] =
                A_shared[make_swizzle_layout<last_dim_a, sizeof(A_type)>(
                    l + row, r + col)];
          }
        }
      }
      // Fetch B into register
      for (int j = 0; j < warp_cols; j++) {
        const auto l = warp_n * warp_col_tiles + j * micro_size_y;
        // Add k_offset to account for warp_k partitioning
        const auto r = k_offset + ki * (kPack * micro_size_k);
        for (int local_id = 0; local_id < (kPack * local_size_b); local_id++) {
          if constexpr (TransposeB) {
            auto [row, col] = reverse_index_map(lane_id, local_id);
            B_local[j * kPack * local_size_b + local_id] =
                B_shared[make_swizzle_layout<last_dim_b, sizeof(B_type)>(
                    l + row, r + col)];
          } else {
            auto [row, col] = reverse_index_map_transposed(lane_id, local_id);
            B_local[j * kPack * local_size_b + local_id] =
                B_shared[make_swizzle_layout<last_dim_b, sizeof(B_type)>(
                    r + row, l + col)];
          }
        }
      }
      // Compute
      for (int kp = 0; kp < kPack; kp++) {
        for (int i = 0; i < warp_rows; ++i) {
          for (int j = 0; j < warp_cols; ++j) {
            auto acc_ptr = ((float32x4 *)C_local) + ((i * warp_cols) + j);
            auto b_ptr = ((B_type *)B_local) + (j * kPack + kp) * vec_size;
            auto a_ptr = ((A_type *)A_local) + (i * kPack + kp) * vec_size;

            // Use the trait to select the correct MMAC instruction
            MmacTraits<A_type>::mmac_op(b_ptr, a_ptr, acc_ptr);
          }
        }
      }
    }
  }
};

} // namespace tl

namespace tl {

// Type tag for warp_k partitioning: distinguishes from bool (trans_A) to avoid overload ambiguity
template <int N>
struct WarpKParam { static constexpr int value = N; };

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int kPack, int min_n_per_warp = 16,
          typename A_type, typename B_type, typename C_type>
TL_DEVICE void gemm_ss(A_type *pA, B_type *pB, C_type *accum) {
  using Compute =
      GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B,
                   clear_accum, kPack, A_type, B_type, C_type, float,
                   min_n_per_warp>;
  Compute::body(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int kPack, int min_n_per_warp = 16,
          typename A_type, typename B_type, typename C_type>
TL_DEVICE void gemm_rs(A_type *pA, B_type *pB, C_type *accum) {
  using Compute =
      GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B,
                   clear_accum, kPack, A_type, B_type, C_type, float,
                   min_n_per_warp>;
  Compute::body_rs(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int kPack, int min_n_per_warp = 16,
          typename A_type, typename B_type, typename C_type>
TL_DEVICE void gemm_rr(A_type *pA, B_type *pB, C_type *accum) {
  using Compute =
      GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B,
                   clear_accum, kPack, A_type, B_type, C_type, float,
                   min_n_per_warp>;
  Compute::body_rr(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int kPack, int min_n_per_warp = 16,
          typename A_type, typename B_type, typename C_type>
TL_DEVICE void gemm_sr(A_type *pA, B_type *pB, C_type *accum) {
  using Compute =
      GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B,
                   clear_accum, kPack, A_type, B_type, C_type, float,
                   min_n_per_warp>;
  Compute::body_sr(pA, pB, accum);
}

// gemm_rs with warp_k partitioning (WarpKParam<N> as 6th param distinguishes from bool trans_A in gemm_rs no k partition)
template <int M, int N, int K, int num_warp_m, int num_warp_n,
          typename WarpKTag,  // WarpKParam<num_warp_k>
          bool trans_A, bool trans_B, bool clear_accum, int kPack, typename A_type,
          typename B_type, typename C_type>
TL_DEVICE void gemm_rs(A_type *pA, B_type *pB, C_type *accum) {
  constexpr int num_warp_k = WarpKTag::value;
  using Compute =
      GemmTensorOpKPartition<M, N, K, num_warp_m, num_warp_n, num_warp_k, trans_A, trans_B,
                   clear_accum, kPack, A_type, B_type, C_type>;
  Compute::body_rs(pA, pB, accum);
}

// gemm_ss with warp_k partitioning
template <int M, int N, int K, int num_warp_m, int num_warp_n,
          typename WarpKTag,
          bool trans_A, bool trans_B, bool clear_accum, int kPack, typename A_type,
          typename B_type, typename C_type>
TL_DEVICE void gemm_ss(A_type *pA, B_type *pB, C_type *accum) {
  constexpr int num_warp_k = WarpKTag::value;
  using Compute =
      GemmTensorOpKPartition<M, N, K, num_warp_m, num_warp_n, num_warp_k, trans_A, trans_B,
                   clear_accum, kPack, A_type, B_type, C_type>;
  Compute::body_ss(pA, pB, accum);
}

// gemm_rr with warp_k partitioning
template <int M, int N, int K, int num_warp_m, int num_warp_n,
          typename WarpKTag,
          bool trans_A, bool trans_B, bool clear_accum, int kPack, typename A_type,
          typename B_type, typename C_type>
TL_DEVICE void gemm_rr(A_type *pA, B_type *pB, C_type *accum) {
  constexpr int num_warp_k = WarpKTag::value;
  using Compute =
      GemmTensorOpKPartition<M, N, K, num_warp_m, num_warp_n, num_warp_k, trans_A, trans_B,
                   clear_accum, kPack, A_type, B_type, C_type>;
  Compute::body_rr(pA, pB, accum);
}

} // namespace tl
