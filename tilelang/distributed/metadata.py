"""Versioned distributed envelope and artifact compatibility checks."""
from functools import cache
import json
import re
from pathlib import Path

from tilelang import tvm


class DistributedABIError(RuntimeError):
    """An artifact/payload must be rebuilt rather than silently reused."""


def incompatible(message):
    return DistributedABIError(
        f"Incompatible distributed ABI: {message}. Rebuild the native library and recompile "
        "with a fresh TILELANG_CACHE_DIR (or clear the incompatible cache/export)."
    )


@cache
def metadata_layout():
    """Read the sole native-header ABI definition through the compiled host FFI."""
    get = tvm.ffi.get_global_func("tl.hcu.distributed.metadata_layout", allow_missing=True)
    if get is None:
        raise incompatible("native library has no versioned distributed ABI descriptor")
    return json.loads(str(get()))


def validate_envelope(payload, backend_id):
    layout = metadata_layout()
    if len(payload) < layout["payload"]:
        raise incompatible("missing versioned metadata envelope (legacy or truncated payload)")
    version = payload[layout["version"]]
    if version != layout["abi_version"]:
        raise incompatible(f"metadata version {version}; expected {layout['abi_version']}")
    if payload[layout["backend"]] != backend_id:
        raise incompatible(f"metadata backend {payload[layout['backend']]}; expected {backend_id}")
    if len(payload) > layout["capacity"]:
        raise RuntimeError("Distributed metadata exceeds device capacity")
    if payload[layout["payload_size"]] != len(payload) - layout["payload"]:
        raise RuntimeError("Distributed metadata payload_size/count mismatch")
    rank, world_size = payload[layout["rank"]], payload[layout["world_size"]]
    if not 0 <= rank < world_size <= (1 << 31) - 1:
        raise RuntimeError("Distributed metadata rank/world_size is inconsistent")
    if payload[layout["status"]] != layout["success"] or payload[layout["status_detail"]] != 0:
        raise RuntimeError("Distributed metadata initial status must be clear")


def _source_text(source):
    if not hasattr(source, "text"):
        return source or ""
    if source.text is not None:
        return source.text
    if source.path is not None and Path(source.path).is_file():
        return Path(source.path).read_text()
    raise incompatible("cached/exported kernel has no source ABI descriptor")


def source_uses_distributed_metadata(source):
    """Detect runtime use, including markers from legacy artifacts."""
    # Legacy remote-source markers still require strict artifact validation.
    return re.search(
        r"^// (?:tilelang_distributed_metadata|tilelang_distributed_abi|tilelang_ipc_remote_source):",
        _source_text(source), re.MULTILINE,
    ) is not None


class UnsupportedDistributedExecutionError(DistributedABIError):
    """Runtime compatibility rejection; cache loaders must not swallow it as a miss.

    Uses the existing compatibility-error hierarchy without changing wire ABI,
    serialized requirements, or the source ABI validation rules.
    """


def reject_unsupported_execution_backend(source, execution_backend_name, *, supported):
    """Reject actual distributed use when the execution spec lacks its hooks."""
    if supported:
        return
    source = _source_text(source)
    if not source_uses_distributed_metadata(source):
        return
    names = re.findall(
        r"\bvoid\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(",
        source or "",
    )
    kernels = list(dict.fromkeys(name for name in names if not name.startswith("__tilelang_")))
    kernel_name = ", ".join(kernels) or "<unknown>"
    raise UnsupportedDistributedExecutionError(
        f"UnsupportedDistributedExecutionError (RuntimeError): kernel {kernel_name} uses distributed runtime "
        "hooks (binding/initialize and allocator lifecycle validation), "
        f"but execution backend {execution_backend_name!r} does not support them. "
        "Use a distributed-runtime-capable execution backend, such as execution_backend='tvm_ffi'."
    )


def validate_artifact_source(source, target):
    """Reject incompatible distributed artifacts before dlopen, including direct reload."""
    if target.kind.name != "hcu" or not str(target.attrs.get("dist_backend", "")):
        return
    source = _source_text(source)
    if not source_uses_distributed_metadata(source):
        return
    markers = re.findall(r"^// tilelang_distributed_abi: (\d+):(\d+)$", source or "", re.MULTILINE)
    if len(markers) != 1:
        raise incompatible("cached/exported kernel is missing its versioned distributed ABI descriptor")
    version, backend_id = map(int, markers[0])
    expected_version = metadata_layout()["abi_version"]
    expected_backend = int(tvm.ffi.get_global_func("tl.GetHcuDistributedBackendId")(str(target.attrs["dist_backend"])))
    if (version, backend_id) != (expected_version, expected_backend):
        raise incompatible(
            f"cached/exported version/backend {version}/{backend_id}; expected {expected_version}/{expected_backend}"
        )


def requirements_for_backend(name):
    """Structured native codegen/init contract, independent of payload layout."""
    return json.loads(str(tvm.ffi.get_global_func("tl.GetHcuDistributedRequirements")(name)))


def requirements_for_target(target):
    if target.kind.name != "hcu" or not str(target.attrs.get("dist_backend", "")):
        return None
    return requirements_for_backend(str(target.attrs["dist_backend"]))


def save_library_requirements(library_path, requirements):
    """Export the requirements as a hash-bound, versioned JSON artifact member."""
    if requirements is None:
        return
    from hashlib import sha256
    import os
    import tempfile

    path = Path(str(library_path) + ".distributed.json")
    record = {"requirements": requirements, "library_sha256": sha256(Path(library_path).read_bytes()).hexdigest()}
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp-")
    try:
        with os.fdopen(descriptor, "w") as file:
            json.dump(record, file, sort_keys=True, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_library_requirements(library_path, target):
    """Reject missing/foreign requirements before dlopen, including direct exports."""
    expected = requirements_for_target(target)
    if expected is None:
        return None
    from hashlib import sha256

    path = Path(str(library_path) + ".distributed.json")
    try:
        record = json.loads(path.read_text())
        requirements = record["requirements"]
        if requirements != expected:
            raise incompatible(f"serialized distributed requirements {requirements}; expected {expected}")
        if record["library_sha256"] != sha256(Path(library_path).read_bytes()).hexdigest():
            raise incompatible("serialized requirements do not match the exported library hash")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise incompatible(f"missing or malformed serialized requirements in {path}") from error
    return requirements
