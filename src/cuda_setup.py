"""CUDA library bootstrap for ctranslate2 / faster-whisper.

Ensures pip-installed nvidia CUDA libraries are discoverable by the
runtime linker before any CUDA-dependent import runs.

Design
------
- **Windows**: ``os.add_dll_directory`` registers pip NVIDIA ``bin/``
  dirs with the loader; ``PATH`` is extended for subprocess / fallback.
- **Linux**: ``LD_LIBRARY_PATH`` style fixups are usually handled by
  pip wheel install scripts — this module is a no-op unless extra
  directories are needed.
- **macOS**: no-op.

Inspired by the cross-platform patterns used in Unsloth, llama-cpp,
and HuggingFace Spaces.
"""

import os
import sys
import glob
import platform


def setup() -> None:
    """Call once before any CUDA import."""
    if _done():
        return

    if sys.platform == "win32":
        _setup_windows()
    elif sys.platform == "linux":
        _setup_linux()

    _mark_done()


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def _setup_windows() -> None:
    import ctypes

    nvidia_dirs = _find_nvidia_dll_dirs()

    for d in nvidia_dirs:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass

    # Extend PATH for subprocess / edge cases
    existing = os.environ.get("PATH", "")
    for d in nvidia_dirs:
        if d not in existing:
            os.environ["PATH"] = d + os.pathsep + existing

    # Preload key DLLs in dependency order.  ctranslate2 may load
    # CUDA libraries with flags that bypass add_dll_directory; a ctypes
    # preload with the default loader makes them resident first.
    # Use glob patterns instead of hardcoding CUDA major versions.
    _preload_patterns = [
        "cudart64_*.dll",
        "cublas64_*.dll",
        "cublasLt64_*.dll",
    ]
    for d in nvidia_dirs:
        for pat in _preload_patterns:
            for path in glob.glob(os.path.join(d, pat)):
                try:
                    ctypes.CDLL(path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Linux
# ---------------------------------------------------------------------------

def _setup_linux() -> None:
    nvidia_dirs = _find_nvidia_so_dirs()

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    for d in nvidia_dirs:
        if d not in existing:
            os.environ["LD_LIBRARY_PATH"] = d + os.pathsep + existing


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _find_nvidia_dll_dirs() -> list[str]:
    """Find ``nvidia/*/bin`` directories under site-packages (Windows)."""
    seen = set()
    dirs: list[str] = []

    for site in _site_packages_dirs():
        nvidia_root = os.path.join(site, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue

        # Legacy layout: nvidia/<pkg>/bin
        for d in glob.glob(os.path.join(nvidia_root, "*", "bin")):
            _add_unique(dirs, d, seen)
        # Conda repack layout: nvidia/<pkg>/Library/bin
        for d in glob.glob(os.path.join(nvidia_root, "*", "Library", "bin")):
            _add_unique(dirs, d, seen)

    # PyTorch-bundled CUDA DLLs
    for site in _site_packages_dirs():
        torch_lib = os.path.join(site, "torch", "lib")
        if os.path.isdir(torch_lib):
            _add_unique(dirs, torch_lib, seen)

    return dirs


def _find_nvidia_so_dirs() -> list[str]:
    """Find ``nvidia/*/lib`` directories under site-packages (Linux)."""
    seen = set()
    dirs: list[str] = []

    for site in _site_packages_dirs():
        nvidia_root = os.path.join(site, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue

        for d in glob.glob(os.path.join(nvidia_root, "*", "lib")):
            _add_unique(dirs, d, seen)

    for site in _site_packages_dirs():
        torch_lib = os.path.join(site, "torch", "lib")
        if os.path.isdir(torch_lib):
            _add_unique(dirs, torch_lib, seen)

    return dirs


def _site_packages_dirs() -> list[str]:
    """All site-packages directories reachable from ``sys.path``."""
    seen = set()
    dirs: list[str] = []
    for p in sys.path:
        if p.endswith("site-packages") and p not in seen:
            seen.add(p)
            dirs.append(p)
    return dirs


def _add_unique(dirs: list[str], d: str, seen: set[str]) -> None:
    key = os.path.normcase(os.path.normpath(d))
    if key not in seen:
        seen.add(key)
        dirs.append(d)


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

_SETUP_DONE = False


def _done() -> bool:
    return _SETUP_DONE


def _mark_done() -> None:
    global _SETUP_DONE
    _SETUP_DONE = True
