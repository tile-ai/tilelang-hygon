import tilelang.testing
import example_mamba_chunk_state


def test_example_mamba_chunk_state():
    example_mamba_chunk_state.main(batch=1, heads=8, seq_len=512)


if __name__ == "__main__":
    tilelang.testing.main()
