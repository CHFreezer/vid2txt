"""CUDA DLL preloading for Windows.

ctranslate2's C extension loads CUDA DLLs at runtime via LoadLibrary,
which doesn't search the nvidia pip-package bin directories automatically.
This module preloads every DLL from those directories into the process
so that subsequent LoadLibrary calls resolve immediately.

Call :func:`setup` once, before importing any CUDA-dependent package
(faster-whisper, ctranslate2, torch, etc.). It is idempotent — safe to
call multiple times.
"""

import sys
import os


def setup() -> None:
    """Preload nvidia CUDA DLLs from all installed nvidia-cu* pip packages.

    Discovery uses ``importlib.metadata`` (the pip package manifest), so
    the list of DLLs is never hardcoded — it adapts automatically when
    nvidia packages are added, removed, or upgraded to a new CUDA major
    version.  A filesystem-scan fallback handles editable / legacy installs.
    """
    if sys.platform != "win32":
        return

    from importlib.metadata import distributions

    # ------------------------------------------------------------------
    # Phase 1: discover every nvidia CUDA bin directory via pip manifests
    # ------------------------------------------------------------------
    bin_dirs: dict[str, int] = {}          # path -> priority (lower = earlier)
    found_any = False
    for dist in distributions():
        name = dist.metadata.get("Name", "")
        if not name.startswith("nvidia-cu"):
            continue
        if dist.files is None:
            continue                        # editable / legacy install, skip
        for f in dist.files:
            f_str = str(f).replace("\\", "/")
            if "/bin/" not in f_str or not f_str.endswith(".dll"):
                continue
            try:
                resolved = dist.locate_file(f)
                bin_dir = str(resolved.parent.resolve())
                if bin_dir not in bin_dirs:
                    # cudart must load before cublas (dependency order)
                    priority = 0 if "cuda_runtime" in bin_dir.lower() else \
                               1 if "cublas" in bin_dir.lower() else 2
                    bin_dirs[bin_dir] = priority
                    found_any = True
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Phase 2: fallback — filesystem scan for cases where pip manifest
    #           doesn't list files (e.g. editable installs)
    # ------------------------------------------------------------------
    if not found_any:
        for pkg_path in sys.path:
            nvidia_root = os.path.join(pkg_path, "nvidia")
            if not os.path.isdir(nvidia_root):
                continue
            for sub in os.listdir(nvidia_root):
                bin_candidate = os.path.join(nvidia_root, sub, "bin")
                if os.path.isdir(bin_candidate) and bin_candidate not in bin_dirs:
                    priority = 0 if "cuda_runtime" in bin_candidate.lower() else \
                               1 if "cublas" in bin_candidate.lower() else 2
                    bin_dirs[bin_candidate] = priority
                    found_any = True

    if not bin_dirs:
        return

    # ------------------------------------------------------------------
    # Phase 3: register directories + preload every DLL
    # ------------------------------------------------------------------
    import ctypes

    for bin_dir, _ in sorted(bin_dirs.items(), key=lambda kv: kv[1]):
        try:
            os.add_dll_directory(bin_dir)
        except OSError:
            pass
        try:
            for entry in os.scandir(bin_dir):
                if entry.is_file() and entry.name.endswith(".dll"):
                    try:
                        ctypes.CDLL(entry.path)
                    except OSError:
                        pass
        except OSError:
            pass
