from tvm import tirx

from tilelang.language.distributed import get_num_ranks, get_rank


def test_get_rank_builds_distributed_intrinsic():
    expr = get_rank()

    assert expr.dtype == "int32"
    assert expr.op.same_as(tirx.op.Op.get("tl.get_rank"))


def test_get_num_ranks_builds_distributed_intrinsic():
    expr = get_num_ranks()

    assert expr.dtype == "int32"
    assert expr.op.same_as(tirx.op.Op.get("tl.get_num_ranks"))
