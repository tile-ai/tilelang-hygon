import argparse
import tilelang.testing
import block_sparse_attn_triton
import example_tilelang_block_sparse_attn
import example_tilelang_sparse_gqa_decode_varlen_indice
import example_tilelang_sparse_gqa_decode_varlen_mask
import example_tilelang_sparse_gqa_decode_paged


def test_block_sparse_attn_triton():
    block_sparse_attn_triton.main()


def test_example_tilelang_block_sparse_attn():
    example_tilelang_block_sparse_attn.main()


def test_example_tilelang_sparse_gqa_decode_varlen_indice():
    example_tilelang_sparse_gqa_decode_varlen_indice.main(batch=1, max_cache_seqlen=2048)


def test_example_tilelang_sparse_gqa_decode_varlen_mask():
    example_tilelang_sparse_gqa_decode_varlen_mask.main(batch=1, max_cache_seqlen=2048)


def test_example_tilelang_sparse_gqa_decode_paged():
    args = argparse.Namespace(
        batch=1,
        heads=32,
        heads_kv=8,
        max_cache_seqlen=2048,
        dim=128,
        dim_v=128,
        sparse_ratio=0.0,
        block_N=64,
        page_block_size=256,
        num_pages=8,
    )
    example_tilelang_sparse_gqa_decode_paged.main(args)


if __name__ == "__main__":
    tilelang.testing.main()
