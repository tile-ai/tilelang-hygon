#pragma once

#include <tl_templates/hcu/core.hpp>

#include <tl_templates/hcu/gemm.h>
#include <tl_templates/hcu/mls/tilelang_ds_read_format.hpp>

namespace tl {

/*
 * GemmMlsTensorOpAB: GEMM with both inputs from MLS LDS, loaded via
 * ds_read_format_tensor.
 *
 * Template params:
 *   M, N, K: tile dimensions
 *   num_warp_m, num_warp_n: warp partitioning
 *   TransposeA, TransposeB: matrix layout (gemm convention)
 *   kPack: K packing factor
 *   MlsTileA: tl::sequence<MlsTileM, MlsTileKA> - MLS tile for A (BlockSize = M
 * x K) MlsTileB: tl::sequence<MlsTileN, MlsTileKB> - MLS tile for B (BlockSize
 * = N x K) AltA, AltB: MLS format (Alt in ds_read sense). TransA, TransB
 * derived: TransA=!TransposeA, TransB=TransposeB A_type, B_type, C_type,
 * AccDataType: data types HcuArch: HCU architecture (default gfx938)
 *
 * Other scenarios (a reg + b mls lds, a swizzle lds + b mls lds) need separate
 * templates - they do not require MlsTileB, AltB (or MlsTileA, AltA).
 */
template <int M, int N, int K, int num_warp_m, int num_warp_n, bool TransposeA,
          bool TransposeB, int kPack, typename MlsTileA, typename MlsTileB,
          tl::index_t AltA, tl::index_t AltB, typename A_type, typename B_type,
          typename C_type, typename AccDataType = float,
          tl::hcu_target_enum HcuArch = tl::hcu_target_enum::gfx938>
class GemmMlsTensorOpAB {
  static constexpr bool TransA = !TransposeA;
  static constexpr bool TransB = TransposeB;
  static constexpr tl::index_t kMinNPerWarp = TransB ? 16 : 32;

  using GemmOp = GemmTensorOp<M, N, K, num_warp_m, num_warp_n, TransposeA,
                              TransposeB, false, kPack, A_type, B_type, C_type,
                              AccDataType, kMinNPerWarp>;

public:
  static constexpr int micro_size_x = GemmOp::micro_size_x;
  static constexpr int micro_size_y = GemmOp::micro_size_y;
  static constexpr int micro_size_k = GemmOp::micro_size_k;
  static constexpr int vec_size = GemmOp::vec_size;
  static constexpr int inner_k = GemmOp::inner_k;
  static constexpr int warp_rows = GemmOp::warp_rows;
  static constexpr int warp_cols = GemmOp::warp_cols;

  using ATraits =
      tl::mls::ds_read_format_traits<tl::sequence<M, K>, tl::sequence<M, K>,
                                     MlsTileA, num_warp_m, 1, A_type, AltA,
                                     TransA, HcuArch>;
  static constexpr tl::index_t A_local_size = ATraits::GemmTensorSize;

  static constexpr tl::index_t WarpN_no_recompute =
      std::min(num_warp_n, N / kMinNPerWarp);
  using BTraits =
      tl::mls::ds_read_format_traits<tl::sequence<N, K>, tl::sequence<N, K>,
                                     MlsTileB, WarpN_no_recompute, 1, B_type,
                                     AltB, TransB, HcuArch>;
  static constexpr tl::index_t B_local_size = BTraits::GemmTensorSize;

  static_assert(kPack == 1, "gemm_mls currently requires kPack=1");
  static_assert(A_local_size == inner_k * warp_rows * vec_size,
                "A GemmTensorSize must match gemm body_rr layout");
  static_assert(B_local_size == inner_k * warp_cols * vec_size,
                "B GemmTensorSize must match gemm body_rr layout");

  /*
   * body: both A and B from MLS LDS, load via ds_read_format_tensor.
   * A_lds, B_lds: MLS-formatted LDS pointers (block-level).
   * C_local: output accumulator (caller-allocated).
   */
  static TL_DEVICE void body(TL_LDS_ADDR A_type *A_lds,
                             TL_LDS_ADDR B_type *B_lds, C_type *C_local) {
    A_type A_local[A_local_size];
    B_type B_local[B_local_size];

    tl::mls::ds_read_format_tensor_a<tl::sequence<M, K>, tl::sequence<M, K>,
                                     MlsTileA, num_warp_m, 1, A_type, AltA,
                                     TransA, HcuArch>(A_lds, A_local);

    tl::mls::ds_read_format_tensor_b<
        tl::sequence<N, K>, tl::sequence<N, K>, MlsTileB,
        num_warp_m * num_warp_n, num_warp_n, 1, B_type, AltB, TransB, HcuArch>(
        B_lds, B_local);

    GemmOp::body_rr(A_local, B_local, C_local);
  }
};

/*
 * gemm_mls_mls: (A mls lds, B mls lds).
 * Naming: gemm_<a_src>_<b_src>, r=reg, s=swizzle lds, mls=mls lds.
 * Also: gemm_r_mls (A reg, B mls), gemm_s_mls (A swizzle lds, B mls).
 */
template <int M, int N, int K, int num_warp_m, int num_warp_n, bool TransposeA,
          bool TransposeB, int kPack, typename MlsTileA, typename MlsTileB,
          tl::index_t AltA, tl::index_t AltB, typename A_type, typename B_type,
          typename C_type, typename AccDataType = float,
          tl::hcu_target_enum HcuArch = tl::hcu_target_enum::gfx938>
TL_DEVICE void gemm_mls_mls(TL_LDS_ADDR A_type *A_lds,
                            TL_LDS_ADDR B_type *B_lds, C_type *C_local) {
  GemmMlsTensorOpAB<M, N, K, num_warp_m, num_warp_n, TransposeA, TransposeB,
                    kPack, MlsTileA, MlsTileB, AltA, AltB, A_type, B_type,
                    C_type, AccDataType, HcuArch>::body(A_lds, B_lds, C_local);
}

/*
 * GemmMlsTensorOpB: GEMM with B from MLS LDS. A can be from register or swizzle
 * LDS.
 *
 * body_r_mls: A from register, B from MLS LDS.
 * body_s_mls: A from swizzle LDS, B from MLS LDS.
 * MlsTileB: tl::sequence<MlsTileN, MlsTileKB> - MLS tile for B.
 */
template <int M, int N, int K, int num_warp_m, int num_warp_n, bool TransposeA,
          bool TransposeB, int kPack, typename MlsTileB, tl::index_t AltB,
          typename A_type, typename B_type, typename C_type,
          typename AccDataType = float,
          tl::hcu_target_enum HcuArch = tl::hcu_target_enum::gfx938>
class GemmMlsTensorOpB {
  static constexpr bool TransB = TransposeB;
  static constexpr tl::index_t kMinNPerWarp = TransB ? 16 : 32;

  using GemmOp = GemmTensorOp<M, N, K, num_warp_m, num_warp_n, TransposeA,
                              TransposeB, false, kPack, A_type, B_type, C_type,
                              AccDataType, kMinNPerWarp>;

public:
  static constexpr int micro_size_x = GemmOp::micro_size_x;
  static constexpr int micro_size_k = GemmOp::micro_size_k;
  static constexpr int inner_k = GemmOp::inner_k;
  static constexpr int warp_rows = GemmOp::warp_rows;
  static constexpr int warp_cols = GemmOp::warp_cols;
  static constexpr int vec_size = GemmOp::vec_size;

  static constexpr tl::index_t WarpN_no_recompute =
      std::min(num_warp_n, N / kMinNPerWarp);
  using BTraits =
      tl::mls::ds_read_format_traits<tl::sequence<N, K>, tl::sequence<N, K>,
                                     MlsTileB, WarpN_no_recompute, 1, B_type,
                                     AltB, TransB, HcuArch>;
  static constexpr tl::index_t B_local_size = BTraits::GemmTensorSize;

  static_assert(kPack == 1, "gemm_mls currently requires kPack=1");
  static constexpr tl::index_t A_local_size = inner_k * warp_rows * vec_size;
  static_assert(B_local_size == inner_k * warp_cols * vec_size,
                "B GemmTensorSize must match gemm body_rr layout");

  /*
   * body_r_mls: A from register, B from MLS LDS.
   */
  static TL_DEVICE void body_r_mls(A_type *A_local, TL_LDS_ADDR B_type *B_lds,
                                   C_type *C_local) {
    B_type B_local[B_local_size];
    tl::mls::ds_read_format_tensor_b<
        tl::sequence<N, K>, tl::sequence<N, K>, MlsTileB,
        num_warp_m * num_warp_n, num_warp_n, 1, B_type, AltB, TransB, HcuArch>(
        B_lds, B_local);
    GemmOp::body_rr(A_local, B_local, C_local);
  }

  /*
   * body_s_mls: A from swizzle LDS, B from MLS LDS.
   */
  static TL_DEVICE void body_s_mls(TL_LDS_ADDR A_type *A_shared,
                                   TL_LDS_ADDR B_type *B_lds, C_type *C_local) {
    constexpr int local_size_a =
        (micro_size_x * micro_size_k) / GemmOp::warp_size;
    constexpr auto last_dim_a =
        TransposeA ? static_cast<int>(M) : static_cast<int>(K);

    auto tid = threadIdx.x;
    auto warp_id = tid / GemmOp::warp_size;
    auto warp_m = warp_id % num_warp_m;
    auto lane_id = tid % GemmOp::warp_size;
    auto warp_row_tiles = warp_rows * micro_size_x;

    A_type A_local[A_local_size];

    for (int ki = 0; ki < inner_k; ki++) {
      for (int i = 0; i < warp_rows; i++) {
        const auto l = warp_m * warp_row_tiles + i * micro_size_x;
        const auto r = ki * (kPack * micro_size_k);
        for (int local_id = 0; local_id < (kPack * local_size_a); local_id++) {
          int row, col;
          if constexpr (TransposeA) {
            auto p = GemmOp::reverse_index_map_transposed(lane_id, local_id);
            row = p.first;
            col = p.second;
            A_local[(ki * warp_rows + i) * vec_size + local_id] =
                A_shared[GemmOp::template make_swizzle_layout<
                    last_dim_a, sizeof(A_type)>(r + row, l + col)];
          } else {
            auto p = GemmOp::reverse_index_map(lane_id, local_id);
            row = p.first;
            col = p.second;
            A_local[(ki * warp_rows + i) * vec_size + local_id] =
                A_shared[GemmOp::template make_swizzle_layout<
                    last_dim_a, sizeof(A_type)>(l + row, r + col)];
          }
        }
      }
    }

    body_r_mls(A_local, B_lds, C_local);
  }
};

/*
 * gemm_r_mls: (A reg, B mls lds).
 */
template <int M, int N, int K, int num_warp_m, int num_warp_n, bool TransposeA,
          bool TransposeB, int kPack, typename MlsTileB, tl::index_t AltB,
          typename A_type, typename B_type, typename C_type,
          typename AccDataType = float,
          tl::hcu_target_enum HcuArch = tl::hcu_target_enum::gfx938>
TL_DEVICE void gemm_r_mls(A_type *A_local, TL_LDS_ADDR B_type *B_lds,
                          C_type *C_local) {
  GemmMlsTensorOpB<M, N, K, num_warp_m, num_warp_n, TransposeA, TransposeB,
                   kPack, MlsTileB, AltB, A_type, B_type, C_type, AccDataType,
                   HcuArch>::body_r_mls(A_local, B_lds, C_local);
}

/*
 * gemm_s_mls: (A swizzle lds, B mls lds).
 */
template <int M, int N, int K, int num_warp_m, int num_warp_n, bool TransposeA,
          bool TransposeB, int kPack, typename MlsTileB, tl::index_t AltB,
          typename A_type, typename B_type, typename C_type,
          typename AccDataType = float,
          tl::hcu_target_enum HcuArch = tl::hcu_target_enum::gfx938>
TL_DEVICE void gemm_s_mls(TL_LDS_ADDR A_type *A_shared,
                          TL_LDS_ADDR B_type *B_lds, C_type *C_local) {
  GemmMlsTensorOpB<M, N, K, num_warp_m, num_warp_n, TransposeA, TransposeB,
                   kPack, MlsTileB, AltB, A_type, B_type, C_type, AccDataType,
                   HcuArch>::body_s_mls(A_shared, B_lds, C_local);
}

} // namespace tl
