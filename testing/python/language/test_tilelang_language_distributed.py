from tvm import tirx

from tilelang.language.distributed import get_block, get_num_ranks, get_rank


def test_get_rank_builds_distributed_intrinsic():
    expr = get_rank()

    assert expr.dtype == "int32"
    assert expr.op.same_as(tirx.op.Op.get("tl.get_rank"))


def test_get_num_ranks_builds_distributed_intrinsic():
    expr = get_num_ranks()

    assert expr.dtype == "int32"
    assert expr.op.same_as(tirx.op.Op.get("tl.get_num_ranks"))


def test_get_block_builds_distributed_intrinsic():
    src = tirx.Var("src", "handle")
    dst = tirx.Var("dst", "handle")
    expr = get_block(src, dst, tirx.IntImm("int32", 16), tirx.IntImm("int32", 1))

    assert expr.op.same_as(tirx.op.Op.get("tl.get_block"))
    assert len(expr.args) == 4
