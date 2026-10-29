"""Backend-neutral module binding and initialization/launch stream ordering."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Any
from collections.abc import Callable
from weakref import WeakValueDictionary
from .allocator import get_distributed_backend_registration
from .metadata import metadata_layout


@contextmanager
def tvm_hcu_stream(device: int, stream: int):
    """Temporarily bind TVM's ROCm device stream to a PyTorch HIP stream."""
    from tilelang import tvm

    get_stream = tvm.ffi.get_global_func("tl.hcu.distributed.get_tvm_stream")
    set_stream = tvm.ffi.get_global_func("tl.hcu.distributed.set_tvm_stream")
    previous = int(get_stream(device))
    set_stream(device, int(stream))
    try:
        yield
    finally:
        set_stream(device, previous)


@dataclass
class _DistributedMetadataBinding:
    module: Any
    owner: Any
    context: Any
    device: int
    backend: str
    generation: int
    metadata_tensor: Any
    registration: Any = None
    launch_lock: Any = field(default_factory=Lock)
    ready_event: Any = None

    def validate(self, allocator, backend=None):
        allocator._require_open()
        if (
            self.owner is not allocator
            or self.context is not allocator.group
            or self.device != allocator.device
            or self.backend != allocator.backend_name
            or (backend is not None and self.backend != backend)
        ):
            raise RuntimeError("Distributed kernel cannot be rebound to a different allocator, context, device or backend")
        if (
            self.registration is not allocator.backend_registration
            or allocator.backend_id != self.registration.backend_id
            or get_distributed_backend_registration(self.backend) is not self.registration
        ):
            raise RuntimeError("Distributed kernel registry identity/backend_id cannot be rebound")


# A module/device has one owner while any initialized adapter retains its
# binding. Native TVM Module equality/hash use the underlying Object identity,
# so different Python wrappers for the same module share this entry. Each
# initializer strongly retains its binding, module and owner; the weak registry
# coordinates adapters without retaining dead kernels/arenas for process life.
_module_bindings: WeakValueDictionary = WeakValueDictionary()
_module_bindings_lock = RLock()


class DistributedMetadataInitializer:
    """Initialize a distributed module's device metadata once per allocator generation."""

    def __init__(self, requirements=None) -> None:
        self.requirements = requirements
        self.binding: _DistributedMetadataBinding | None = None

    def initialize(self, module: Any, allocator: Any, *, backend=None) -> bool:
        """Launch the module helper when its cached allocator generation is stale.

        Returns ``True`` when a helper launch occurred and ``False`` when the
        module/device pair was already initialized for this generation.
        """
        with _module_bindings_lock:
            return self._initialize(module, allocator, backend=backend)

    def _initialize(self, module: Any, allocator: Any, *, backend) -> bool:
        backend = allocator.backend_name if backend is None else backend
        allocator._require_open()
        if allocator.backend_name != backend:
            raise RuntimeError("Distributed kernel requires its selected allocator backend")
        if self.binding is not None:
            self.binding.validate(allocator, backend)
            if self.binding.module != module:
                raise RuntimeError("Distributed kernel cannot switch its initialized module")
        registration = get_distributed_backend_registration(backend)
        if allocator.backend_registration is not registration or allocator.backend_id != registration.backend_id:
            raise RuntimeError("Distributed allocator registry identity/backend_id cannot be rebound")
        module_key = (module, allocator.device)
        binding = _module_bindings.get(module_key)
        if binding is not None:
            binding.validate(allocator, backend)
            if binding.generation == allocator.generation:
                self.binding = binding
                return False
        from tilelang import tvm

        native_id = int(tvm.ffi.get_global_func("tl.GetHcuDistributedBackendId")(backend))
        if native_id != registration.backend_id:
            raise RuntimeError(
                f"Distributed backend registry ID mismatch for {backend}: native={native_id}, Python={registration.backend_id}"
            )
        from .metadata import validate_envelope, incompatible, requirements_for_backend

        requirements = self.requirements or requirements_for_backend(backend)
        spec = requirements["init_spec"]
        try:
            helper = module.get_function(spec["initialize"]["name"], query_imports=False)
        except AttributeError as error:
            raise incompatible("module does not contain the common distributed initializer") from error
        if helper is None:
            raise incompatible("module does not contain the common distributed initializer")
        payload = allocator.metadata
        layout = metadata_layout()
        validate_envelope(payload, registration.backend_id)
        if payload[layout["rank"]] != allocator.rank or payload[layout["world_size"]] != allocator.world_size:
            raise RuntimeError("Distributed metadata rank/world_size is inconsistent with allocator")
        import ctypes
        import torch

        metadata_tensor = torch.tensor(payload, dtype=torch.uint64, device=f"cuda:{allocator.device}")
        # FunctionInfo declares an opaque handle: pass the device pointer,
        # not a Tensor/NDArray descriptor.
        helper_arg = ctypes.c_void_p(metadata_tensor.data_ptr())
        stream_ptr = int(tvm.ffi.get_global_func("tl.hcu.distributed.get_tvm_stream")(allocator.device))
        init_stream = torch.cuda.ExternalStream(stream_ptr, device=allocator.device)
        produced = torch.cuda.Event()
        produced.record(torch.cuda.current_stream(allocator.device))
        init_stream.wait_event(produced)
        result = torch.empty(layout["status_words"], dtype=torch.uint64, device=f"cuda:{allocator.device}")
        helper(helper_arg, len(payload), ctypes.c_void_p(result.data_ptr()))
        init_stream.synchronize()
        error, detail = result.cpu().tolist()
        if error:
            raise incompatible(f"device initializer rejected metadata: status={error}, observed version={detail}")
        if binding is None:
            binding = _DistributedMetadataBinding(
                module,
                allocator,
                allocator.group,
                allocator.device,
                backend,
                allocator.generation,
                metadata_tensor,
                registration=registration,
            )
        else:
            # Update the shared record only after a successful helper call, so
            # all adapters observe the same owner generation and payload.
            binding.generation = allocator.generation
            binding.metadata_tensor = metadata_tensor
        # Initialization may use another host thread/stream than launch.
        binding.ready_event = torch.cuda.Event()
        binding.ready_event.record(init_stream)
        self.binding = binding
        _module_bindings[module_key] = binding
        return True

    def launch(self, launch: Callable[[], Any], *, backend=None) -> Any:
        """Validate once and enqueue work after metadata initialization.

        The shared lock serializes host submissions, not GPU completion.
        Kernel authors own peer validity and communication synchronization.
        """
        import torch

        binding = self.binding
        if binding is None:
            raise RuntimeError("Distributed kernel must be initialized before launch")
        allocator = binding.owner
        with binding.launch_lock:
            binding.validate(allocator, backend)
            if binding.generation != allocator.generation:
                raise RuntimeError("Distributed allocator generation changed; call kernel.initialize() before launch")
            with torch.cuda.device(binding.device):
                stream = torch.cuda.current_stream(binding.device)
                if binding.ready_event is not None:
                    stream.wait_event(binding.ready_event)
                with tvm_hcu_stream(binding.device, stream.cuda_stream):
                    return launch()
