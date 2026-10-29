"""Distributed allocator protocol and backend registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Callable
from threading import RLock


@dataclass(frozen=True)
class Allocation:
    """A non-owning allocation inside a distributed arena."""

    ptr: int
    offset: int
    nbytes: int
    generation: int
    owner: object | None = None
    device: int | None = None


@dataclass(frozen=True, eq=False)
class _BackendRegistration:
    """Immutable process identity; backend IDs must agree with native plugins."""

    name: str
    backend_id: int
    factory: Callable[..., DistributedAllocator]


_backend_registry: dict[str, _BackendRegistration] = {}
_backend_registry_lock = RLock()


def register_distributed_backend(name: str, backend_id: int, factory: Callable[..., DistributedAllocator]) -> _BackendRegistration:
    """Register once; duplicate names/IDs and replacement are rejected.

    Native and Python plugins choose the same stable, nonzero uint32 ID. IPC
    reserves ID 1. IDs identify implementations, not registration order.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Distributed backend registration requires a non-empty name")
    if isinstance(backend_id, bool) or not isinstance(backend_id, int) or not 0 < backend_id < (1 << 32):
        raise ValueError("Distributed backend ID must be a nonzero uint32")
    if not callable(factory):
        raise TypeError("Distributed backend factory must be callable")
    with _backend_registry_lock:
        if name in _backend_registry:
            raise ValueError(f"Distributed backend is already registered and cannot be replaced: {name}")
        for entry in _backend_registry.values():
            if entry.backend_id == backend_id:
                raise ValueError(f"Distributed backend ID {backend_id} is already registered to {entry.name}")
        entry = _BackendRegistration(name, backend_id, factory)
        _backend_registry[name] = entry
        return entry


def list_distributed_backends() -> dict[str, _BackendRegistration]:
    """Return a snapshot; mutating it cannot modify the process registry."""
    with _backend_registry_lock:
        return dict(sorted(_backend_registry.items()))


def get_distributed_backend_registration(name: str) -> _BackendRegistration:
    with _backend_registry_lock:
        try:
            return _backend_registry[name]
        except KeyError:
            supported = ", ".join(sorted(_backend_registry))
            raise ValueError(f"Unsupported distributed allocator backend: {name}; supported backends: {supported}") from None


class DistributedAllocator(ABC):
    """Lifetime contract shared by distributed communication backends."""

    def _bind_distributed_backend(self, name: str) -> None:
        entry = get_distributed_backend_registration(name)
        previous = getattr(self, "_distributed_backend_registration", None)
        if previous is not None and previous is not entry:
            raise RuntimeError("Distributed allocator backend_id cannot be rebound")
        self._distributed_backend_registration = entry

    @property
    def backend_registration(self) -> _BackendRegistration:
        return self._distributed_backend_registration

    @property
    def backend_id(self) -> int:
        return self.backend_registration.backend_id

    @property
    @abstractmethod
    def generation(self) -> int: ...

    @abstractmethod
    def allocate(self, nbytes: int, alignment: int | None = None) -> Allocation: ...

    @abstractmethod
    def close(self) -> None: ...


def get_distributed_allocator(backend: str = "ipc", **kwargs) -> DistributedAllocator:
    """Create a distributed allocator selected by its runtime backend name."""
    entry = get_distributed_backend_registration(backend)
    allocator = entry.factory(**kwargs)
    if not isinstance(allocator, DistributedAllocator):
        raise TypeError(f"Distributed backend {backend} factory must return a DistributedAllocator")
    allocator._bind_distributed_backend(backend)
    return allocator
