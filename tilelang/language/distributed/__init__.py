"""Backend-neutral distributed kernel-language APIs."""

from .common import get_num_ranks, get_rank

__all__ = ["get_rank", "get_num_ranks"]
