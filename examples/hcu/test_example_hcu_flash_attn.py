import tilelang.testing
import example_hcu_flash_attn_bwd
import example_hcu_flash_attn_fwd


def test_example_hcu_flash_attn_bwd():
    example_hcu_flash_attn_bwd.main(seq_len=1024)


def test_example_hcu_flash_attn_fwd():
    example_hcu_flash_attn_fwd.main(seq_len=1024)


if __name__ == "__main__":
    tilelang.testing.main()
