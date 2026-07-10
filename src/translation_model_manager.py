"""Hy-MT2 translation model discovery and download.

Follows the same pattern as ``src/model_manager.py`` — single-file downloads
with tqdm progress bars and Xet-disabled downloads for reliable progress.
"""

import logging
import os
from pathlib import Path

from .config import (
    TRANSLATION_MODEL_REPOS,
    SUPPORTED_TRANSLATION_MODELS,
    DEFAULT_TRANSLATION_MODEL_DIR,
)

logger = logging.getLogger("vid2txt")


def get_model_path(model_key: str, base_path: str) -> Path:
    """Return the expected file path for translation model *model_key* under *base_path*."""
    info = TRANSLATION_MODEL_REPOS[model_key]
    return Path(base_path) / info["filename"]


def is_model_downloaded(model_key: str, base_path: str) -> bool:
    """Check whether *model_key* is fully downloaded under *base_path*."""
    return get_model_path(model_key, base_path).exists()


def list_translation_models(base_path: str) -> dict[str, dict]:
    """Return download status for every supported translation model."""
    base = Path(base_path).resolve()
    result: dict[str, dict] = {}
    for key in SUPPORTED_TRANSLATION_MODELS:
        info = TRANSLATION_MODEL_REPOS[key]
        p = base / info["filename"]
        result[key] = {
            "downloaded": p.exists(),
            "path": str(p),
            "size_gb": info["size_gb"],
        }
    return result


def download_translation_model(model_key: str, base_path: str) -> str:
    """Download *model_key* GGUF file to *base_path*.

    Downloads a single file (one tqdm bar) — unlike ``snapshot_download``
    whose parallel ``thread_map`` + ``_AggregatedTqdm`` freezes the bar.
    """
    import huggingface_hub.constants as _hf_constants

    # Disable Xet only for this download — restore original value afterwards
    _xet_original = _hf_constants.HF_HUB_DISABLE_XET
    _hf_constants.HF_HUB_DISABLE_XET = True

    from huggingface_hub import hf_hub_download

    info = TRANSLATION_MODEL_REPOS[model_key]
    repo_id = info["repo"]
    filename = info["filename"]
    local_dir = os.path.abspath(base_path)
    os.makedirs(local_dir, exist_ok=True)

    logger.info("Downloading %s/%s → %s", repo_id, filename, local_dir)

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
        )
        dest = os.path.join(local_dir, filename)
        logger.info("Download complete: %s", dest)
        return dest
    finally:
        _hf_constants.HF_HUB_DISABLE_XET = _xet_original
