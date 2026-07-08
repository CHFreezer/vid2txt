"""CUDA library preloading for ctranslate2 / faster-whisper.

ctranslate2 loads CUDA libraries at runtime via dlopen / LoadLibrary.
When CUDA is installed via pip (``nvidia-cuda-runtime-cu12``,
``nvidia-cublas-cu12``, etc.), those libraries live deep inside
``site-packages`` and are NOT on the default search path.

This module discovers pip-installed nvidia packages and makes their
libraries visible to the runtime linker — *before* any CUDA-dependent
package imports.

Call :func:`setup` once, at the very top of the entry point.
It is idempotent and safe to call on any platform / any Python.
"""

import sys
import os


def setup() -> None:
    """Discover and preload CUDA libraries from pip-installed nvidia packages.

    - **Windows**: registers DLL directories via ``os.add_dll_directory()``.
    - **Linux**: preloads ``.so`` files via ``ctypes.CDLL()`` so that
      ``dlopen`` resolves them when ``ctranslate2`` imports.
    - **macOS**: no-op — ``ctranslate2`` runs CPU-only on macOS (Apple
      Silicon does not support NVIDIA CUDA).

    Discovery uses ``importlib.metadata`` (the pip package manifest), so
    the list of libraries is never hardcoded — it adapts automatically
    when nvidia packages are added, removed, or upgraded to a new CUDA
    major version.
    """
    if _discovery_done():
        return

    if sys.platform == "darwin":
        return  # ctranslate2 uses CPU / Apple Accelerate on macOS

    # ------------------------------------------------------------------
    # Phase 1: discover nvidia library directories via pip manifests
    # ------------------------------------------------------------------
    lib_dirs = _discover_from_pip()

    # ------------------------------------------------------------------
    # Phase 2: fallback — filesystem scan for editable / legacy installs
    # ------------------------------------------------------------------
    if not lib_dirs:
        lib_dirs = _discover_from_filesystem()

    if not lib_dirs:
        return

    # ------------------------------------------------------------------
    # Phase 3: platform-specific loading
    # ------------------------------------------------------------------
    # Sort by priority: cudart → cudnn → cublas → rest
    sorted_dirs = sorted(lib_dirs.items(), key=lambda kv: kv[1])

    if sys.platform == "win32":
        _load_windows(sorted_dirs)
    else:
        _load_linux(sorted_dirs)

    _mark_done()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_from_pip() -> dict[str, int]:
    """Find nvidia CUDA library dirs via ``importlib.metadata``.

    Returns ``{path: priority}`` — lower priority loads first.
    """
    from importlib.metadata import distributions

    lib_dirs: dict[str, int] = {}

    for dist in distributions():
        name = dist.metadata.get("Name", "")
        if not name.startswith("nvidia-cu"):
            continue
        if dist.files is None:
            continue  # editable / legacy install — fallback handles it

        for f in dist.files:
            f_str = str(f).replace("\\", "/")

            if sys.platform == "win32":
                if "/bin/" not in f_str or not f_str.endswith(".dll"):
                    continue
            else:
                if "/lib/" not in f_str or ".so" not in f_str:
                    continue

            try:
                resolved = dist.locate_file(f)
                lib_dir = str(resolved.parent.resolve())
                if lib_dir not in lib_dirs:
                    lib_dirs[lib_dir] = _priority(name)
            except Exception:
                pass

    return lib_dirs


def _discover_from_filesystem() -> dict[str, int]:
    """Scan ``sys.path`` for ``nvidia/*/bin`` or ``nvidia/*/lib`` dirs.

    Covers editable installs and edge cases where ``dist.files`` is empty.
    """
    lib_dirs: dict[str, int] = {}
    subdir = "bin" if sys.platform == "win32" else "lib"

    for pkg_path in sys.path:
        nvidia_root = os.path.join(pkg_path, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for sub in os.listdir(nvidia_root):
            candidate = os.path.join(nvidia_root, sub, subdir)
            if os.path.isdir(candidate) and candidate not in lib_dirs:
                lib_dirs[candidate] = _priority(sub)

    return lib_dirs


# ---------------------------------------------------------------------------
# Platform loaders
# ---------------------------------------------------------------------------

def _load_windows(dirs: list[tuple[str, int]]) -> None:
    """Register DLL directories with the Windows process loader."""
    for lib_dir, _ in dirs:
        try:
            os.add_dll_directory(lib_dir)
        except OSError:
            pass


def _load_linux(dirs: list[tuple[str, int]]) -> None:
    """Preload ``.so`` files so ``dlopen`` finds them for ctranslate2."""
    import ctypes

    for lib_dir, _ in dirs:
        try:
            for entry in os.scandir(lib_dir):
                if not entry.is_file():
                    continue
                name = entry.name
                # Match lib*.so or lib*.so.12 etc.; skip Python extension modules
                if not (name.startswith("lib") and (name.endswith(".so") or ".so." in name)):
                    continue
                try:
                    ctypes.CDLL(entry.path)
                except OSError:
                    pass
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _priority(name: str) -> int:
    """Return load priority for a package name (lower = load earlier).

    ``cudart`` must load before ``cublas`` / ``cudnn`` (dependency order).
    """
    name_lower = name.lower()
    if "cuda_runtime" in name_lower or "cudart" in name_lower:
        return 0
    if "cudnn" in name_lower:
        return 1
    if "cublas" in name_lower:
        return 2
    return 3


_SETUP_DONE = False


def _discovery_done() -> bool:
    return _SETUP_DONE


def _mark_done() -> None:
    global _SETUP_DONE
    _SETUP_DONE = True
