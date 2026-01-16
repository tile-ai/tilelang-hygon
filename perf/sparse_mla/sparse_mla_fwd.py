# ruff: noqa
import torch
import tilelang
from tilelang import language as T
# from utils import assert_tensors_similar
import functools

# def get_configs():
#     import itertools
#     block_I = [16, 32]
#     threads = [128, 256]
#     num_split = [1, 2, 4, 8, 16]
#     num_stages = [1]

#     _configs = list(itertools.product(block_I, threads, num_split, num_stages))

#     return [{
#         "block_I": c[0],
#         "threads": c[1],
#         "num_split": c[2],
#         "num_stages": c[3],
#     } for c in _configs]

# @tilelang.autotune(configs=get_configs())

config_map_cu80 = {
    1: {
        "block_I": 32,
        "threads": 256,
        "num_split": 32,
        "num_stages": 1,
    },
    2: {
        "block_I": 32,
        "threads": 256,
        "num_split": 32,
        "num_stages": 1,
    },
    3: {
        "block_I": 32,
        "threads": 256,
        "num_split": 16,
        "num_stages": 1,
    },
    4: {
        "block_I": 32,
        "threads": 256,
        "num_split": 16,
        "num_stages": 1,
    },
    8: {
        "block_I": 32,
        "threads": 256,
        "num_split": 8,
        "num_stages": 1,
    },
    16: {
        "block_I": 32,
        "threads": 256,
        "num_split": 4,
        "num_stages": 1,
    },
    32: {
        "block_I": 32,
        "threads": 256,
        "num_split": 4,
        "num_stages": 1,
    },
    48: {
        "block_I": 32,
        "threads": 256,
        "num_split": 8,
        "num_stages": 1,
    },
    64: {
        "block_I": 32,
        "threads": 256,
        "num_split": 8,
        "num_stages": 1,
    },
    80: {
        "block_I": 32,
        "threads": 256,
        "num_split": 1,
        "num_stages": 1,
    },
    96: {
        "block_I": 32,
        "threads": 256,
        "num_split": 4,
        "num_stages": 1,
    },
    112: {
        "block_I": 32,
        "threads": 256,
        "num_split": 2,
        "num_stages": 1,
    },
    128: {
        "block_I": 32,
        "threads": 256,
        "num_split": 4,
        "num_stages": 1,
    }
}

@tilelang.jit(
    # if we set output idx, it will cuase error when create output tensor in cython wrapper when cuda_graph is used
    # out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def sparse_mla_fwd(
    num_heads,
    dim,
    tail_dim,
    topk,
    num_split=1,
    *,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=32,
    num_stages=1,
    threads=128,
    kv_stride=1,
    dtype="float16",
):
    assert dim == tilelang.math.next_power_of_2(
        dim
    ), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim
    ), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert (
        topk % block_I == 0
    ), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if num_split > 1:
        assert (
            topk % (num_split * block_I) == 0
        ), f"topk={topk} must be divisible by num_split * block_I={num_split} * {block_I}"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    # print(f"num_split={num_split}")
    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    head_kv = num_heads // kv_group
    q_shape = [batch, seq_len, num_heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, num_heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    glse_shape = [batch, seq_len, num_heads, num_split]
    output_partial_shape = [batch, seq_len, num_heads, num_split, dim]
    indices_dtype = "int32"
    dtype = dtype
    accum_dtype = "float"

    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1
    BI = block_I
    topk_per_split = topk // num_split if num_split > 1 else topk
    NI = tilelang.cdiv(topk_per_split, block_I)
    D = dim
    D_spilt = dim // 4
    D_tail = tail_dim
    max_block_m = 16

    if head_kv > max_block_m:
        assert head_kv % max_block_m == 0, f"head_kv should be a multiple of {max_block_m}"
        REPLICATE_H = head_kv // max_block_m
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else max_block_m
    kv_vectorized = max(min(min((H_per_block * D_spilt) // threads, (BI * D_spilt) // threads), 8), 1)
    # if BI is 16, can not kpack for gemm2
    kv_vectorized = 4 if (BI < 32 and kv_vectorized == 8) else kv_vectorized

    threads_per_line = D_spilt // kv_vectorized
    warps_line_stride = threads // threads_per_line
    kv_serial_count = BI // warps_line_stride
    kpack = min((kv_vectorized + 3) // 4, 2)

    # print(f"kv_serial_count={kv_serial_count}, warps_line_stride={warps_line_stride}, threads_per_line={threads_per_line}, kv_vectorized={kv_vectorized}, kpack={kpack}")

    @T.macro
    def sparse_mla(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        q_start_index_s: T.Tensor([1], indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_spilt0_shared = T.alloc_shared([H_per_block, D_spilt], dtype)
            Q_spilt1_shared = T.alloc_shared([H_per_block, D_spilt], dtype)
            Q_spilt2_shared = T.alloc_shared([H_per_block, D_spilt], dtype)
            Q_spilt3_shared = T.alloc_shared([H_per_block, D_spilt], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            
            KV_spilt0_shared = T.alloc_shared([BI, D_spilt], dtype)
            KV_spilt1_shared = T.alloc_shared([BI, D_spilt], dtype)
            KV_spilt2_shared = T.alloc_shared([BI, D_spilt], dtype)
            KV_spilt3_shared = T.alloc_shared([BI, D_spilt], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            mask = T.alloc_fragment([BI], "bool")

            acc_o0 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
            acc_o1 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
            acc_o2 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
            acc_o3 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)

            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
            indices_local = T.alloc_local([1], indices_dtype)
            indices_mask = T.alloc_local([1], indices_dtype)
            indices_tail = T.alloc_local([1], indices_dtype)
            valid_NI = T.alloc_fragment([1], "int")

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            # q_i = q_start_index_s[0] + s_i
            # max_kv_i = (q_i + 1 - kv_stride) // kv_stride
            # kv_i = (q_i + 1 - kv_stride) // kv_stride
            # max_kv_i = kv_i if (kv_i <= seq_len_kv - 1) else seq_len_kv - 1

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * max_block_m)
            H1 = H0 + H_per_block

            tx = T.get_thread_binding()
            T.copy(Q[b_i, s_i, H0:H1, :D_spilt], Q_spilt0_shared, coalesced_width=kv_vectorized)
            T.copy(Q[b_i, s_i, H0:H1, D_spilt:2*D_spilt], Q_spilt1_shared, coalesced_width=kv_vectorized)
            T.copy(Q[b_i, s_i, H0:H1, 2*D_spilt:3*D_spilt], Q_spilt2_shared, coalesced_width=kv_vectorized)
            T.copy(Q[b_i, s_i, H0:H1, 3*D_spilt:4*D_spilt], Q_spilt3_shared, coalesced_width=kv_vectorized)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            T.fill(acc_o0, 0)
            T.fill(acc_o1, 0)
            T.fill(acc_o2, 0)
            T.fill(acc_o3, 0)
            T.fill(sumexp, 1)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan
            T.fill(valid_NI, 0)

            for i_i in T.serial(NI):
                first_indices = Indices[b_i, s_i, g_i, i_i * BI]
                # if first_indices <= max_kv_i and first_indices >= 0:
                if first_indices >= 0:
                    valid_NI[0] += 1

            for i_i in T.Pipelined(valid_NI[0], num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    indices_mask[0] = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                    # mask[bi_i] = indices_mask[0] <= max_kv_i and indices_mask[0] >= 0
                    mask[bi_i] = indices_mask[0] >= 0

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )

                # for bi_i, d_i in T.Parallel(BI, D_spilt):
                #     indices_local[0] = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                #     indices_local[0] = T.if_then_else(indices_local[0] <= max_kv_i and indices_local[0] >= 0, indices_local[0], 0)
                #     KV_spilt0_shared[bi_i, d_i] = KV[b_i, indices_local[0], g_i, d_i]
                #     KV_spilt1_shared[bi_i, d_i] = KV[b_i, indices_local[0], g_i, D_spilt + d_i]
                #     KV_spilt2_shared[bi_i, d_i] = KV[b_i, indices_local[0], g_i, 2*D_spilt + d_i]
                #     KV_spilt3_shared[bi_i, d_i] = KV[b_i, indices_local[0], g_i, 3*D_spilt + d_i]
                for u in T.serial(kv_serial_count):
                    line_stride = u * warps_line_stride
                    indices_local[0] = Indices[b_i, s_i, g_i, i_i * BI + line_stride + tx // threads_per_line]
                    # indices_local[0] = T.if_then_else(indices_local[0] <= max_kv_i and indices_local[0] >= 0, indices_local[0], 0)
                    indices_local[0] = T.if_then_else(indices_local[0] >= 0, indices_local[0], 0)
                    for v in T.vectorized(kv_vectorized):
                        KV_spilt0_shared[line_stride + tx // threads_per_line,
                                        (tx % threads_per_line) * kv_vectorized + v] = KV[b_i, indices_local[0], g_i,
                                        (tx % threads_per_line) * kv_vectorized + v]
                        KV_spilt1_shared[line_stride + tx // threads_per_line,
                                        (tx % threads_per_line) * kv_vectorized + v] = KV[b_i, indices_local[0], g_i,
                                        D_spilt + (tx % threads_per_line) * kv_vectorized + v]
                        KV_spilt2_shared[line_stride + tx // threads_per_line,
                                        (tx % threads_per_line) * kv_vectorized + v] = KV[b_i, indices_local[0], g_i,
                                        2*D_spilt + (tx % threads_per_line) * kv_vectorized + v]
                        KV_spilt3_shared[line_stride + tx // threads_per_line,
                                        (tx % threads_per_line) * kv_vectorized + v] = KV[b_i, indices_local[0], g_i,
                                        3*D_spilt + (tx % threads_per_line) * kv_vectorized + v]

                for bi_i, d_i in T.Parallel(BI, D_tail):
                    indices_tail[0] = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                    # indices_tail[0] = T.if_then_else(indices_tail[0] <= max_kv_i and indices_tail[0] >= 0, indices_tail[0], 0)
                    indices_tail[0] = T.if_then_else(indices_tail[0] >= 0, indices_tail[0], 0)
                    K_tail_shared[bi_i, d_i] = KV[
                        b_i, indices_tail[0], g_i, D + d_i
                    ]

                T.gemm(Q_spilt0_shared, KV_spilt0_shared, acc_s, transpose_B=True, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_spilt1_shared, KV_spilt1_shared, acc_s, transpose_B=True, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_spilt2_shared, KV_spilt2_shared, acc_s, transpose_B=True, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_spilt3_shared, KV_spilt3_shared, acc_s, transpose_B=True, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_tail_shared, K_tail_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D_spilt):
                    acc_o0[h_i, d_i] = acc_o0[h_i, d_i] * alpha[h_i]
                    acc_o1[h_i, d_i] = acc_o1[h_i, d_i] * alpha[h_i]
                    acc_o2[h_i, d_i] = acc_o2[h_i, d_i] * alpha[h_i]
                    acc_o3[h_i, d_i] = acc_o3[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_spilt0_shared, acc_o0, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(S_shared, KV_spilt1_shared, acc_o1, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(S_shared, KV_spilt2_shared, acc_o2, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(S_shared, KV_spilt3_shared, acc_o3, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)
            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D_spilt):
                acc_o0[h_i, d_i] /= sumexp[h_i]
                acc_o1[h_i, d_i] /= sumexp[h_i]
                acc_o2[h_i, d_i] /= sumexp[h_i]
                acc_o3[h_i, d_i] /= sumexp[h_i]

            acc_oshared0 = T.alloc_shared([H_per_block, D_spilt], dtype)
            acc_oshared1 = T.alloc_shared([H_per_block, D_spilt], dtype)
            acc_oshared2 = T.alloc_shared([H_per_block, D_spilt], dtype)
            acc_oshared3 = T.alloc_shared([H_per_block, D_spilt], dtype)
            T.annotate_layout({
                acc_oshared0: tilelang.layout.make_hcu_swizzled_layout(acc_oshared0, major_pack=2),
                acc_oshared1: tilelang.layout.make_hcu_swizzled_layout(acc_oshared1, major_pack=2),
                acc_oshared2: tilelang.layout.make_hcu_swizzled_layout(acc_oshared2, major_pack=2),
                acc_oshared3: tilelang.layout.make_hcu_swizzled_layout(acc_oshared3, major_pack=2),
            })
            T.copy(acc_o0, acc_oshared0)
            T.copy(acc_o1, acc_oshared1)
            T.copy(acc_o2, acc_oshared2)
            T.copy(acc_o3, acc_oshared3)
            T.copy(acc_oshared0, Output[b_i, s_i, H0:H1, :D_spilt])
            T.copy(acc_oshared1, Output[b_i, s_i, H0:H1, D_spilt:2*D_spilt])
            T.copy(acc_oshared2, Output[b_i, s_i, H0:H1, 2*D_spilt:3*D_spilt])
            T.copy(acc_oshared3, Output[b_i, s_i, H0:H1, 3*D_spilt:4*D_spilt])

    @T.macro
    def sparse_mla_split(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        q_start_index_s: T.Tensor([1], indices_dtype),  # type: ignore
        glse: T.Tensor(glse_shape, dtype),  # type: ignore
        Output_partial: T.Tensor(output_partial_shape, dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch * kv_group, num_split, threads=threads) as (
            bx,
            by,
            bz,
            # split_idx,
        ):
            Q_spilt0_shared = T.alloc_shared([H_per_block, D_spilt], dtype)
            Q_spilt1_shared = T.alloc_shared([H_per_block, D_spilt], dtype)
            Q_spilt2_shared = T.alloc_shared([H_per_block, D_spilt], dtype)
            Q_spilt3_shared = T.alloc_shared([H_per_block, D_spilt], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            
            KV_spilt0_shared = T.alloc_shared([BI, D_spilt], dtype)
            KV_spilt1_shared = T.alloc_shared([BI, D_spilt], dtype)
            KV_spilt2_shared = T.alloc_shared([BI, D_spilt], dtype)
            KV_spilt3_shared = T.alloc_shared([BI, D_spilt], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            mask = T.alloc_fragment([BI], "bool")

            acc_o0 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
            acc_o1 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
            acc_o2 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
            acc_o3 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)

            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
            indices_local = T.alloc_local([1], indices_dtype)
            indices_mask = T.alloc_local([1], indices_dtype)
            indices_tail = T.alloc_local([1], indices_dtype)

            split_idx = bz
            b_i = by if kv_group == 1 else (by // kv_group)
            g_i = 0 if kv_group == 1 else (by % kv_group)
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            # q_i = q_start_index_s[0] + s_i
            # max_kv_i = (q_i + 1 - kv_stride) // kv_stride
            # kv_i = (q_i + 1 - kv_stride) // kv_stride
            # max_kv_i = kv_i if (kv_i <= seq_len_kv - 1) else seq_len_kv - 1

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * max_block_m)
            H1 = H0 + H_per_block

            first_indices = Indices[b_i, s_i, g_i, split_idx * topk_per_split]
            # if first_indices > max_kv_i and first_indices >= 0:
            if first_indices < 0:
                T.fill(sumexp, -T.infinity(accum_dtype))
                T.fill(acc_o0, 0)
                T.fill(acc_o1, 0)
                T.fill(acc_o2, 0)
                T.fill(acc_o3, 0)
                T.copy(sumexp, glse[b_i, s_i, H0:H1, split_idx])
                T.copy(acc_o0, Output_partial[b_i, s_i, H0:H1, split_idx, :D_spilt])
                T.copy(acc_o1, Output_partial[b_i, s_i, H0:H1, split_idx, D_spilt:2*D_spilt])
                T.copy(acc_o2, Output_partial[b_i, s_i, H0:H1, split_idx, 2*D_spilt:3*D_spilt])
                T.copy(acc_o3, Output_partial[b_i, s_i, H0:H1, split_idx, 3*D_spilt:4*D_spilt])
            else:
                tx = T.get_thread_binding()
                T.copy(Q[b_i, s_i, H0:H1, :D_spilt], Q_spilt0_shared, coalesced_width=kv_vectorized)
                T.copy(Q[b_i, s_i, H0:H1, D_spilt:2*D_spilt], Q_spilt1_shared, coalesced_width=kv_vectorized)
                T.copy(Q[b_i, s_i, H0:H1, 2*D_spilt:3*D_spilt], Q_spilt2_shared, coalesced_width=kv_vectorized)
                T.copy(Q[b_i, s_i, H0:H1, 3*D_spilt:4*D_spilt], Q_spilt3_shared, coalesced_width=kv_vectorized)
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

                T.fill(acc_o0, 0)
                T.fill(acc_o1, 0)
                T.fill(acc_o2, 0)
                T.fill(acc_o3, 0)
                T.fill(sumexp, 1)
                T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

                for i_i in T.Pipelined(NI, num_stages=num_stages):
                    idx_in_split = split_idx * topk_per_split + i_i * BI
                    for bi_i in T.Parallel(BI):
                        indices_mask[0] = Indices[b_i, s_i, g_i, idx_in_split + bi_i]
                        # mask[bi_i] = indices_mask[0] <= max_kv_i and indices_mask[0] >= 0
                        mask[bi_i] = indices_mask[0] >= 0

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            mask[bi_i], 0, -T.infinity(acc_s.dtype)
                        )
                    # for bi_i, d_i in T.Parallel(BI, D_spilt, coalesced_width=8):
                    #     indices_local[0] = Indices[b_i, s_i, g_i, idx_in_split + bi_i]
                    #     indices_local[0] = T.if_then_else(indices_local[0] <= max_kv_i, indices_local[0], 0)
                    #     KV_spilt0_shared[bi_i, d_i] = KV[b_i, indices_local[0], g_i, d_i]
                    #     KV_spilt1_shared[bi_i, d_i] = KV[b_i, indices_local[0], g_i, D_spilt + d_i]
                    #     KV_spilt2_shared[bi_i, d_i] = KV[b_i, indices_local[0], g_i, 2*D_spilt + d_i]
                    #     KV_spilt3_shared[bi_i, d_i] = KV[b_i, indices_local[0], g_i, 3*D_spilt + d_i]

                    for u in T.serial(kv_serial_count):
                        line_stride = u * warps_line_stride
                        indices_local[0] = Indices[b_i, s_i, g_i, idx_in_split + line_stride + tx // threads_per_line]
                        # indices_local[0] = T.if_then_else(indices_local[0] <= max_kv_i and indices_local[0] >= 0, indices_local[0], 0)
                        indices_local[0] = T.if_then_else(indices_local[0] >= 0, indices_local[0], 0)
                        for v in T.vectorized(kv_vectorized):
                            KV_spilt0_shared[line_stride + tx // threads_per_line,
                                            (tx % threads_per_line) * kv_vectorized + v] = KV[b_i, indices_local[0], g_i,
                                            (tx % threads_per_line) * kv_vectorized + v]
                            KV_spilt1_shared[line_stride + tx // threads_per_line,
                                            (tx % threads_per_line) * kv_vectorized + v] = KV[b_i, indices_local[0], g_i,
                                            D_spilt + (tx % threads_per_line) * kv_vectorized + v]
                            KV_spilt2_shared[line_stride + tx // threads_per_line,
                                            (tx % threads_per_line) * kv_vectorized + v] = KV[b_i, indices_local[0], g_i,
                                            2*D_spilt + (tx % threads_per_line) * kv_vectorized + v]
                            KV_spilt3_shared[line_stride + tx // threads_per_line,
                                            (tx % threads_per_line) * kv_vectorized + v] = KV[b_i, indices_local[0], g_i,
                                            3*D_spilt + (tx % threads_per_line) * kv_vectorized + v]

                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        indices_tail[0] = Indices[b_i, s_i, g_i, idx_in_split + bi_i]
                        # indices_tail[0] = T.if_then_else(indices_tail[0] <= max_kv_i and indices_tail[0] >= 0, indices_tail[0], 0)
                        indices_tail[0] = T.if_then_else(indices_tail[0] >= 0, indices_tail[0], 0)
                        K_tail_shared[bi_i, d_i] = KV[
                            b_i, indices_tail[0], g_i, D + d_i
                        ]
                                            
                    T.gemm(Q_spilt0_shared, KV_spilt0_shared, acc_s, transpose_B=True, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(Q_spilt1_shared, KV_spilt1_shared, acc_s, transpose_B=True, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(Q_spilt2_shared, KV_spilt2_shared, acc_s, transpose_B=True, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(Q_spilt3_shared, KV_spilt3_shared, acc_s, transpose_B=True, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(Q_tail_shared, K_tail_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(
                            acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                        )
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D_spilt):
                        acc_o0[h_i, d_i] = acc_o0[h_i, d_i] * alpha[h_i]
                        acc_o1[h_i, d_i] = acc_o1[h_i, d_i] * alpha[h_i]
                        acc_o2[h_i, d_i] = acc_o2[h_i, d_i] * alpha[h_i]
                        acc_o3[h_i, d_i] = acc_o3[h_i, d_i] * alpha[h_i]

                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_spilt0_shared, acc_o0, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(S_shared, KV_spilt1_shared, acc_o1, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(S_shared, KV_spilt2_shared, acc_o2, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(S_shared, KV_spilt3_shared, acc_o3, k_pack=kpack, policy=T.GemmWarpPolicy.FullCol)

                # Rescale
                for h_i, d_i in T.Parallel(H_per_block, D_spilt):
                    acc_o0[h_i, d_i] /= sumexp[h_i]
                    acc_o1[h_i, d_i] /= sumexp[h_i]
                    acc_o2[h_i, d_i] /= sumexp[h_i]
                    acc_o3[h_i, d_i] /= sumexp[h_i]

                acc_oshared0 = T.alloc_shared([H_per_block, D_spilt], dtype)
                acc_oshared1 = T.alloc_shared([H_per_block, D_spilt], dtype)
                acc_oshared2 = T.alloc_shared([H_per_block, D_spilt], dtype)
                acc_oshared3 = T.alloc_shared([H_per_block, D_spilt], dtype)
                T.annotate_layout({
                    acc_oshared0: tilelang.layout.make_hcu_swizzled_layout(acc_oshared0, major_pack=2),
                    acc_oshared1: tilelang.layout.make_hcu_swizzled_layout(acc_oshared1, major_pack=2),
                    acc_oshared2: tilelang.layout.make_hcu_swizzled_layout(acc_oshared2, major_pack=2),
                    acc_oshared3: tilelang.layout.make_hcu_swizzled_layout(acc_oshared3, major_pack=2),
                })
                T.copy(acc_o0, acc_oshared0)
                T.copy(acc_o1, acc_oshared1)
                T.copy(acc_o2, acc_oshared2)
                T.copy(acc_o3, acc_oshared3)
                T.copy(acc_oshared0, Output_partial[b_i, s_i, H0:H1, split_idx, :D_spilt])
                T.copy(acc_oshared1, Output_partial[b_i, s_i, H0:H1, split_idx, D_spilt:2*D_spilt])
                T.copy(acc_oshared2, Output_partial[b_i, s_i, H0:H1, split_idx, 2*D_spilt:3*D_spilt])
                T.copy(acc_oshared3, Output_partial[b_i, s_i, H0:H1, split_idx, 3*D_spilt:4*D_spilt])

                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                T.copy(sumexp, glse[b_i, s_i, H0:H1, split_idx])

    @T.macro
    def combine(
        glse: T.Tensor(glse_shape, dtype),  # type: ignore
        Output_partial: T.Tensor(output_partial_shape, dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, num_heads, batch, threads=128) as (bx, by, bz):
            po_local = T.alloc_fragment([dim], dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_local_split = T.alloc_local([1], accum_dtype)
            lse_logsum_local = T.alloc_local([1], accum_dtype)
            lse_max_local = T.alloc_local([1], accum_dtype)
            scale_local = T.alloc_local([1], accum_dtype)

            # T.annotate_layout({
            #     lse_logsum_local: T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
            # })
            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local[0] = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_max_local[0] = T.max(lse_max_local[0], glse[bz, bx, by, k])
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split[0] = glse[bz, bx, by, k]
                lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
            lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po_local[i] = Output_partial[bz, bx, by, k, i]
                lse_local_split[0] = glse[bz, bx, by, k]
                scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                for i in T.Parallel(dim):
                    o_accum_local[i] += po_local[i] * scale_local[0]
            for i in T.Parallel(dim):
                Output[bz, bx, by, i] = o_accum_local[i]

    @T.prim_func
    def main_split(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        q_start_index_s: T.Tensor([1], indices_dtype),  # type: ignore
        glse: T.Tensor(glse_shape, dtype),  # type: ignore
        Output_partial: T.Tensor(output_partial_shape, dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
    ):
        sparse_mla_split(Q, KV, Indices, q_start_index_s, glse, Output_partial)
        combine(glse, Output_partial, Output)

    @T.prim_func
    def main_no_split(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        q_start_index_s: T.Tensor([1], indices_dtype),  # type: ignore
        glse: T.Tensor(glse_shape, dtype),  # type: ignore
        Output_partial: T.Tensor(output_partial_shape, dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
    ):
        sparse_mla(Q, KV, Indices, q_start_index_s, Output)

    if num_split > 1:
        return main_split
    else:
        return main_no_split

@functools.lru_cache
def get_best_config_cu80(batch, seq_len):
    cu_count = 80
    key = batch * seq_len
    best_key = min(config_map_cu80.keys(), key=lambda x: abs(x - key))
    config = config_map_cu80[best_key]
    block_I, threads, num_split, num_stages = config["block_I"], config["threads"], config["num_split"], config["num_stages"]
    if key != best_key and (key * num_split) % cu_count != 0 and (key * num_split) % cu_count < 32 and (key * num_split) // cu_count < 8:
        max_scoe = (key + 79) // 80 * 64
        num_split = 1
        splits = [1, 2, 4, 8, 16] if key < 32 else [1, 2, 4, 8]
        for split in splits:
            sore_ = (key * split + 79) // 80 * (64 // split)
            if sore_ < max_scoe:
                max_scoe = sore_
                num_split = split
    
    print(f"Using best config: best_key={best_key}, block_I={block_I}, threads={threads}, num_split={num_split}, num_stages={num_stages}")
    return block_I, threads, num_split, num_stages

def sparse_mla_fwd_interface(q,
                             kv,
                             indices,
                             kv_stride=1,
                             sm_scale=None,
                             return_p_sum: bool = False,
                             d_v=512,
                             block_I=None,
                             num_stages=None,
                             threads=None,
                             num_split=None,
                             dtype="float16"):
    is_causal = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    if block_I is None or num_stages is None or threads is None or num_split is None:
        block_I, threads, num_split, num_stages = get_best_config_cu80(batch, seq_len)
 
    kernel = sparse_mla_fwd(
        heads,
        dim,
        tail_dim,
        topk,
        dtype=dtype,
        num_split=num_split,
        kv_group=kv_group,
        sm_scale=sm_scale,
        is_causal=is_causal,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
        kv_stride=kv_stride)

    return kernel, num_split



def ref_sparse_mla_fwd_interface(q, kv, indices, output_dtype, q_start_s_index=0, kv_stride=1, sm_scale=None, is_casual=True):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(
        q_start_s_index, sq + q_start_s_index, dtype=torch.int32,
        device="cuda").view(-1, 1) >= torch.arange(
            kv_stride - 1, sk * kv_stride, kv_stride, dtype=torch.int32, device="cuda").view(1, -1)

    indices = torch.where(indices >= 0, indices, sk)
    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, :kv_stride - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(output_dtype)