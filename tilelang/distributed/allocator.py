"""Distributed allocator protocol and backend registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Allocation:
    """A non-owning allocation inside a distributed arena."""

    ptr: int
    offset: int
    nbytes: int
    generation: int


class DistributedAllocator(ABC):
    """Lifetime contract shared by distributed communication backends."""

    @property
    @abstractmethod
    def generation(self) -> int: ...

    @abstractmethod
    def allocate(self, nbytes: int, alignment: int | None = None) -> Allocation: ...

    @abstractmethod
    def close(self) -> None: ...


def get_distributed_allocator(backend: str = "ipc", **kwargs) -> DistributedAllocator:
    """Create a distributed allocator selected by its runtime backend name."""
    if backend == "ipc":
        from .backends.ipc import IpcAllocator

        return IpcAllocator(**kwargs)
    raise ValueError(f"Unsupported distributed allocator backend: {backend}")
