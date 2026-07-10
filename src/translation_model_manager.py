"""M2M100 CTranslate2 translation model discovery and download.

Follows the same pattern as ``src/model_manager.py`` for Whisper models.
The model is stored as CTranslate2 files (model.bin + config.json +
shared_vocabulary.json).
"""

import fnmatch
import logging
import os
from pathlib import Path

from .config import (
    TRANSLATION_MODEL_REPO,
    REQUIRED_TRANSLATION_MODEL_FILES,
    DEFAULT_TRANSLATION_MODEL_DIR,
)

logger = logging.getLogger("vid2txt")

_ALLOW_PATTERNS = [
    "config.json",
    "model.bin",
    "shared_vocabulary.json",
    "sentencepiece.bpe.model",
]


def is_model_downloaded(base_path: str) -> bool:
    """Check whether the translation model is fully downloaded."""
    md = Path(base_path)
    return all((md / f).exists() for f in REQUIRED_TRANSLATION_MODEL_FILES)


def get_model_dir(base_path: str) -> str:
    """Return the expected model directory path."""
    return os.path.abspath(base_path)


def download_translation_model(base_path: str) -> str:
    """Download the CTranslate2 M2M100 model to *base_path*.

    Downloads files one at a time (one tqdm bar each).
    """
    import huggingface_hub.constants as _hf_constants

    _xet_original = _hf_constants.HF_HUB_DISABLE_XET
    _hf_constants.HF_HUB_DISABLE_XET = True

    from huggingface_hub import HfApi, hf_hub_download

    repo_id = TRANSLATION_MODEL_REPO
    local_dir = os.path.abspath(base_path)
    os.makedirs(local_dir, exist_ok=True)

    logger.info("Downloading %s → %s", repo_id, local_dir)

    # Discover which files to download
    api = HfApi()
    all_files = api.list_repo_files(repo_id)
    filtered = []
    for f in sorted(all_files):
        for pat in _ALLOW_PATTERNS:
            if fnmatch.fnmatch(f, pat):
                filtered.append(f)
                break

    try:
        for filename in filtered:
            logger.info("  %s", filename)
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
            )
        logger.info("Download complete: %s", local_dir)
        return local_dir
    finally:
        _hf_constants.HF_HUB_DISABLE_XET = _xet_original
