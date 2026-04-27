import contextlib
import ctypes
import logging
import os
import sys
import warnings
from pathlib import Path


def _compute_version() -> str:
    """Return the package version without being polluted by unrelated installs.

    Preference order:
    1) If running from a source checkout (VERSION file present at repo root),
       use the dynamic version from version_provider (falls back to plain VERSION).
    2) Otherwise, use importlib.metadata for the installed distribution.
    3) As a last resort, return a dev sentinel.
    """
    try:
        repo_root = Path(__file__).resolve().parent.parent
        version_file = repo_root / "VERSION"
        if version_file.is_file():
            try:
                from version_provider import dynamic_metadata  # type: ignore

                return dynamic_metadata("version")
            except Exception:
                # Fall back to the raw VERSION file if provider isn't available.
                return version_file.read_text().strip()
    except Exception:
        # If any of the above fails, fall through to installed metadata.
        pass

    try:
        from importlib.metadata import version as _dist_version  # py3.8+

        return _dist_version("tilelang")
    except Exception as exc:
        warnings.warn(
            f"tilelang version metadata unavailable ({exc!r}); using development version.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "0.0.dev0"


__version__ = _compute_version()
del _compute_version


logger = logging.getLogger(__name__)


def set_log_level(level):
    """Set the logging level for the module's logger.

    Args:
        level (str or int): Can be the string name of the level (e.g., 'INFO') or the actual level (e.g., logging.INFO).
        OPTIONS: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)


def _init_logger():
    """Initialize the logger specific for this module with custom settings and a Tqdm-based handler."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    class TqdmLoggingHandler(logging.Handler):
        """Custom logging handler that directs log output to tqdm progress bar to avoid interference."""

        def __init__(self, level=logging.NOTSET):
            """Initialize the handler with an optional log level."""
            super().__init__(level)

        def emit(self, record):
            """Emit a log record. Messages are written to tqdm to ensure output in progress bars isn't corrupted."""
            try:
                msg = self.format(record)
                if tqdm is not None:
                    tqdm.write(msg)
            except Exception:
                self.handleError(record)

    handler = TqdmLoggingHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s  [TileLang:%(name)s:%(levelname)s] (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    set_log_level("INFO")


from .env import env as env  # noqa: F401

# Skip logger initialization in light import mode
if not env.is_light_import():
    _init_logger()

del _init_logger


def _libtorch_cuda_present(lib_dir: Path) -> bool:
    if sys.platform.startswith("win32"):
        return (lib_dir / "torch_cuda.dll").is_file()
    return (lib_dir / "libtorch_cuda.so").is_file() or bool(
        next(lib_dir.glob("libtorch_cuda*.so"), None)
    )


def _libtorch_hip_present(lib_dir: Path) -> bool:
    if sys.platform.startswith("win32"):
        return (lib_dir / "torch_hip.dll").is_file()
    return (lib_dir / "libtorch_hip.so").is_file() or bool(
        next(lib_dir.glob("libtorch_hip*.so"), None)
    )


def _maybe_disable_torch_c_dlpack() -> None:
    """Align with ``tvm_ffi._optional_torch_c_dlpack.load_torch_c_dlpack_extension``.

    That function first tries ``import torch_c_dlpack_ext`` (pip). That package often
    dlopens **CUDA**-named libs before tvm-ffi's ROCm JIT path runs, so on ROCm-only
    PyTorch a missing ``libtorch_cuda.so`` still breaks import even when
    ``torch.version.hip`` is set.

    We mirror ``torch.cuda.is_available()`` + cuda/hip version branching, then:
    - **CUDA build** (`torch.version.cuda`): require ``libtorch_cuda`` under ``torch/lib``.
    - **ROCm build** (`torch.version.hip`, cuda None): if ``torch_c_dlpack_ext`` is
      installed, require **CUDA** libs (pip path); otherwise require **HIP** libs
      (tvm-ffi JIT ``device=rocm`` path only).
    When ``torch.cuda.is_available()`` is false we do nothing (``cpu`` path).
    """
    if os.environ.get("TVM_FFI_DISABLE_TORCH_C_DLPACK") is not None:
        return
    try:
        import importlib.util

        import torch

        lib_dir = Path(torch.__file__).resolve().parent / "lib"
        if not lib_dir.is_dir():
            os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
            return

        if not torch.cuda.is_available():
            return

        if torch.version.cuda is not None:
            device = "cuda"
        elif torch.version.hip is not None:
            device = "rocm"
        else:
            os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
            return

        if device == "cuda":
            ok = _libtorch_cuda_present(lib_dir)
        else:
            # ROCm: pip torch_c_dlpack_ext runs first and typically needs CUDA .so.
            has_pip_ext = importlib.util.find_spec("torch_c_dlpack_ext") is not None
            if has_pip_ext:
                ok = _libtorch_cuda_present(lib_dir)
            else:
                ok = _libtorch_hip_present(lib_dir)

        if not ok:
            os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
    except Exception:
        pass


def _patch_tvm_ffi_torch_c_dlpack_loader() -> None:
    """Monkey-patch tvm_ffi before `import tvm` (does not modify upstream TVM).

    Official TVM's relax/base_py_module calls load_torch_c_dlpack_extension() again;
    respect TVM_FFI_DISABLE_TORCH_C_DLPACK and swallow ImportError/OSError.
    """
    try:
        import tvm_ffi._optional_torch_c_dlpack as m
    except ImportError:
        return
    _orig = m.load_torch_c_dlpack_extension

    def _safe():
        if os.environ.get("TVM_FFI_DISABLE_TORCH_C_DLPACK", "0") != "0":
            return None
        try:
            return _orig()
        except (ImportError, OSError):
            return None

    m.load_torch_c_dlpack_extension = _safe


@contextlib.contextmanager
def _lazy_load_lib():
    import torch  # noqa: F401 # preload torch to avoid dlopen errors

    old_flags = sys.getdlopenflags()
    old_init = ctypes.CDLL.__init__

    def lazy_init(self, name, mode=ctypes.DEFAULT_MODE, *args, **kwargs):
        return old_init(self, name, mode | os.RTLD_LAZY, *args, **kwargs)

    sys.setdlopenflags(old_flags | os.RTLD_LAZY)
    ctypes.CDLL.__init__ = lazy_init
    try:
        yield
    finally:
        sys.setdlopenflags(old_flags)
        ctypes.CDLL.__init__ = old_init


# Skip heavy imports in light import mode
if not env.is_light_import():
    with _lazy_load_lib():
        from .env import enable_cache, disable_cache, is_cache_enabled  # noqa: F401

        _maybe_disable_torch_c_dlpack()
        _patch_tvm_ffi_torch_c_dlpack_loader()

        import tvm
        import tvm.base  # noqa: F401
        from tvm import DataType  # noqa: F401

        # Setup tvm search path before importing tvm
        from . import libinfo

        def _load_tile_lang_lib():
            """Load Tile Lang lib"""
            if sys.platform.startswith("win32") and sys.version_info >= (3, 8):
                for path in libinfo.get_dll_directories():
                    os.add_dll_directory(path)
            lib_path = libinfo.find_lib_path("tilelang")
            return ctypes.CDLL(lib_path), lib_path

        # only load once here
        if env.SKIP_LOADING_TILELANG_SO == "0":
            _LIB, _LIB_PATH = _load_tile_lang_lib()

    from .jit import jit, JITKernel, compile, par_compile  # noqa: F401
    from .profiler import Profiler  # noqa: F401
    from .cache import clear_cache  # noqa: F401
    from .utils import (
        TensorSupplyType,  # noqa: F401
        deprecated,  # noqa: F401
        build_date,  # noqa: F401
    )
    from .layout import (
        Layout,  # noqa: F401
        Fragment,  # noqa: F401
    )
    from . import (
        analysis,  # noqa: F401
        transform,  # noqa: F401
        language,  # noqa: F401
        engine,  # noqa: F401
        tools,  # noqa: F401
    )
    from .language import dtypes  # noqa: F401
    from .autotuner import autotune  # noqa: F401
    from .transform import PassConfigKey  # noqa: F401
    from .engine import lower, register_cuda_postproc, register_hip_postproc, register_c_postproc  # noqa: F401
    from .math import *  # noqa: F403
    from . import ir  # noqa: F401
    from . import tileop  # noqa: F401

del _lazy_load_lib
