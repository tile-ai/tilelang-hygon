from tvm import tirx
import tilelang.language.common as T

__all__ = ["rng_init", "rng_rand", "rng_rand_float"]


def rng_init(seed, seq=None, off=0, generator="hiprandStatePhilox4_32_10_t") -> tirx.PrimExpr:
    """Initialize HCU hipRAND random number generator state

    Parameters
    ----------
    seed : PrimExpr
        Random seed value.
    seq : PrimExpr
        Sequence number for parallel random number generation.
    off : PrimExpr
        Offset number for parallel random number generation.
    generator : StringImm
        Set random generator.

    Returns
    -------
    init : PrimExpr
        The random number generator initialization intrinsic.
    """
    assert generator in ["hiprandStateMRG32k3a_t", "hiprandStatePhilox4_32_10_t", "hiprandStateXORWOW_t"]
    seed = tirx.convert(seed)
    if seq is None:
        bx = T.get_block_binding()
        ex = T.get_thread_extent()
        tx = T.get_thread_binding()
        id = tx + bx * ex
        seq = tirx.convert(id)
    else:
        seq = tirx.convert(seq)
    off = tirx.convert(off)
    return tirx.call_intrin("void", tirx.op.Op.get("tl.rng_init"), seed, seq, off, generator)


def rng_rand() -> tirx.PrimExpr:
    """Generate a 32-bit unsigned random integer

    Returns
    -------
    random_value : PrimExpr
        A 32-bit unsigned random integer.
    """
    return tirx.call_intrin("uint32", tirx.op.Op.get("tl.rng_rand"))


def rng_rand_float(bit=32, dist="uniform") -> tirx.PrimExpr:
    """Generate a random float

    Parameters
    ----------
    bit : int = [32, 64]
        Bitwidth of random float.
    dist : StringImm = ["uniform", "normal"]
        Random distribution.

    Returns
    -------
    random_value : PrimExpr
        A random float.
    """
    assert bit in [32, 64]
    assert dist in ["uniform", "normal"]
    return tirx.call_intrin("float" + str(bit), tirx.op.Op.get("tl.rng_rand_float"), dist)
