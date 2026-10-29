"""Tests for validating inverses of TileLang layouts."""

from itertools import product

import pytest

from tilelang import tvm
from tilelang.layout import Fragment, Layout


def assert_round_trip(layout):
    """Check both directions over small domains in tests, not during lowering."""
    inverse = layout.inverse()
    analyzer = tvm.arith.Analyzer()
    is_fragment = isinstance(layout, Fragment)
    if is_fragment:
        assert int(layout.replicate_size) == 1
    input_shape = tuple(int(dim) for dim in layout.get_input_shape())
    output_shape = tuple(int(dim) for dim in inverse.get_input_shape())

    def evaluate(expressions):
        return tuple(int(analyzer.simplify(expr)) for expr in expressions)

    def forward(logical):
        physical = evaluate(layout.map_forward_index(list(logical)))
        if is_fragment:
            physical += evaluate(layout.map_forward_thread(list(logical)))
        return physical

    for logical in product(*(range(dim) for dim in input_shape)):
        physical = forward(logical)
        assert all(0 <= value < extent for value, extent in zip(physical, output_shape))
        expected = logical + (0,) if is_fragment else logical
        assert evaluate(inverse.map_forward_index(list(physical))) == expected
    for physical in product(*(range(dim) for dim in output_shape)):
        logical = evaluate(inverse.map_forward_index(list(physical)))
        if is_fragment:
            assert logical[-1] == 0
            logical = logical[:-1]
        assert all(0 <= value < extent for value, extent in zip(logical, input_shape))
        assert forward(logical) == physical


def test_inverse_accepts_bijective_split():
    layout = Layout([128], lambda i: [i // 8, i % 8])

    inverse = layout.inverse()

    assert [int(dim) for dim in inverse.get_input_shape()] == [16, 8]
    assert [int(dim) for dim in inverse.get_output_shape()] == [128]


def test_inverse_rejects_non_round_tripping_cyclic_shift():
    layout = Layout([128], lambda i: (i + 1) % 128)

    with pytest.raises(Exception, match="non-round-tripping inverse"):
        layout.inverse()


def test_explicit_inverse_accepts_non_self_inverse_map():
    layout = Layout([128], lambda i: (i + 1) % 128)
    inverse = Layout([128], lambda i: (i - 1) % 128)
    checked = layout.with_inverse(inverse)
    analyzer = tvm.arith.Analyzer()
    assert int(analyzer.simplify(checked.inverse().map_forward_index([0])[0])) == 127
    assert int(analyzer.simplify(checked.map_forward_index([0])[0])) == 1
    assert_round_trip(checked)


def test_explicit_inverse_does_not_enumerate_large_domain():
    extent = 1 << 19  # Larger than the automatic inverse's exact-check limit.
    layout = Layout([extent], lambda i: (i + 1) % extent)
    supplied = Layout([extent], lambda i: (i - 1) % extent)
    inverse = layout.with_inverse(supplied).inverse()
    analyzer = tvm.arith.Analyzer()
    assert int(analyzer.simplify(inverse.map_forward_index([0])[0])) == extent - 1


def test_explicit_inverse_rejects_domain_mismatch():
    layout = Layout([128], lambda i: i)
    with pytest.raises(Exception, match="inverse input shape"):
        layout.with_inverse(Layout([64], lambda i: i))


def test_explicit_inverse_rejects_input_rank_mismatch():
    layout = Layout([128], lambda i: i)
    with pytest.raises(Exception, match="inverse input shape"):
        layout.with_inverse(Layout([16, 8], lambda i, j: i * 8 + j))


def test_explicit_inverse_rejects_output_rank_mismatch():
    layout = Layout([128], lambda i: i)
    with pytest.raises(Exception, match="inverse output rank"):
        layout.with_inverse(Layout([128], lambda i: [i, 0]))


def test_explicit_fragment_inverse_survives_loop_fusion_reshape():
    fragment = Fragment([32], forward_fn=lambda i: ((i + 1) % 16, ((i + 1) // 16) % 2))
    inverse = Layout([2, 16], lambda slot, thread: [(slot * 16 + thread - 1) % 32, 0])
    checked = fragment.with_inverse(inverse)
    reshaped = checked.reshape([4, 8])
    values = reshaped.inverse().map_forward_index([0, 0])
    analyzer = tvm.arith.Analyzer()
    assert [int(analyzer.simplify(x)) for x in values] == [3, 7, 0]
    assert_round_trip(reshaped)


def test_explicit_fragment_inverse_with_xor_thread_permutation():
    fragment = Fragment([8, 8], forward_fn=lambda i, j: (i ^ j, i))
    assert int(fragment.get_thread_size()) == 8
    checked = fragment.with_inverse(Layout([8, 8], lambda slot, thread: [slot, slot ^ thread, 0]))
    assert [int(dim) for dim in checked.inverse().get_input_shape()] == [8, 8]
    assert_round_trip(checked)


if __name__ == "__main__":
    pytest.main([__file__])
