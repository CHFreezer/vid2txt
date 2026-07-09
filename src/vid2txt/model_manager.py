"""Whisper model discovery and download with progress.

faster-whisper normally downloads models on first use and disables the
tqdm progress bar.  This module takes over downloading so the WebUI can
show real progress and let the user control *when* to download.
"""

import os
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger("vid2txt")

# Hugging Face repo for each model size
_REPO_PREFIX = "Systran/faster-whisper-"

# Files that must exist for a model to be considered "downloaded"
_REQUIRED_FILES = ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")


# ---------------------------------------------------------------------------
# Cache discovery
# ---------------------------------------------------------------------------

def _custom_model_path(base: str, size: str) -> Path:
    return Path(base) / f"faster-whisper-{size}"


def _is_complete(model_dir: Path) -> bool:
    return all((model_dir / f).exists() for f in _REQUIRED_FILES)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_models(model_path: str) -> dict[str, dict]:
    """Return download status for every supported model size.

    Returns a dict keyed by size name, e.g.::

        {"small": {"downloaded": True, "path": "/.../snapshots/abc123"}, ...}
    """
    from .config import SUPPORTED_MODELS

    result: dict[str, dict] = {}
    custom_base = Path(model_path).resolve()

    for size in SUPPORTED_MODELS:
        custom = _custom_model_path(custom_base, size)
        if _is_complete(custom):
            result[size] = {"downloaded": True, "path": str(custom)}
        else:
            result[size] = {"downloaded": False, "path": str(custom)}

    return result


def download_model(
    size: str,
    model_path: str,
    progress_callback=None,
) -> str:
    """Download *size* to ``<model_path>/faster-whisper-<size>/``.

    *progress_callback(ratio)* is called periodically (0.0 … 1.0),
    driven by the real download progress via huggingface_hub's tqdm.

    Returns the local model path.
    """
    from huggingface_hub import snapshot_download

    repo_id = _REPO_PREFIX + size
    local_dir = str(_custom_model_path(model_path, size))
    local_dir_abs = os.path.abspath(local_dir)

    logger.info("Downloading %s → %s", repo_id, local_dir_abs)

    allow_patterns = [
        "config.json",
        "preprocessor_config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.*",
    ]

    # Derive a tqdm class that feeds progress_callback from the real download
    TqdmClass = None
    if progress_callback:
        import tqdm

        _last_ratio = [0.0]
        _download_progress = _last_ratio  # exposed for external polling

        class _ProgressTqdm(tqdm.tqdm):
            def __init__(self, total=None, **kw):
                kw.setdefault("file", open(os.devnull, "w"))
                super().__init__(total=total or 0, **kw)

            def update(self, n=1):
                super().update(n)
                if self.total and self.total > 0:
                    ratio = min(self.n / self.total, 1.0)
                    if ratio - _last_ratio[0] > 0.01 or ratio >= 1.0:
                        _last_ratio[0] = ratio
                        progress_callback(ratio)

            def close(self):
                if _last_ratio[0] < 1.0:
                    progress_callback(1.0)
                super().close()

        TqdmClass = _ProgressTqdm

    kw = dict(
        repo_id=repo_id,
        local_dir=local_dir_abs,
        allow_patterns=allow_patterns,
    )
    if TqdmClass is not None:
        kw["tqdm_class"] = TqdmClass

    return snapshot_download(**kw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir_size(path: Path) -> int:
    """Total bytes of all files under *path* (0 if path doesn't exist)."""
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


_MODEL_SIZES_MB = {
    "tiny": 150,
    "tiny.en": 150,
    "base": 220,
    "base.en": 220,
    "small": 580,
    "small.en": 580,
    "medium": 1800,
    "medium.en": 1800,
    "large-v3": 3500,
}


def _expected_model_size(size: str) -> int:
    """Estimated download size in bytes for a model (for progress estimation)."""
    mb = _MODEL_SIZES_MB.get(size, 2000)
    return mb * 1024 * 1024
