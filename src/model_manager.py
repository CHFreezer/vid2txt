"""Whisper model discovery and download.

faster-whisper normally downloads models on first use.  This module
takes over downloading so the WebUI can let the user control *when*
to download and to which directory.
"""

import fnmatch
import logging
import os
from pathlib import Path

from .config import REQUIRED_MODEL_FILES

logger = logging.getLogger("vid2txt")

_REPO_PREFIX = "Systran/faster-whisper-"

_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


def _custom_model_path(base: str, size: str) -> Path:
    return Path(base) / f"faster-whisper-{size}"


def _is_complete(model_dir: Path) -> bool:
    if not all((model_dir / f).exists() for f in REQUIRED_MODEL_FILES):
        return False
    # Some models ship vocabulary.txt, others vocabulary.json
    return (model_dir / "vocabulary.txt").exists() or (model_dir / "vocabulary.json").exists()


def list_models(whisper_model_path: str) -> dict[str, dict]:
    """Return download status for every supported model size."""
    from .config import SUPPORTED_MODELS

    result: dict[str, dict] = {}
    custom_base = Path(whisper_model_path).resolve()

    for size in SUPPORTED_MODELS:
        custom = _custom_model_path(custom_base, size)
        if _is_complete(custom):
            result[size] = {"downloaded": True, "path": str(custom)}
        else:
            result[size] = {"downloaded": False, "path": str(custom)}

    return result


def download_model(size: str, whisper_model_path: str) -> str:
    """Download *size* to ``<whisper_model_path>/faster-whisper-<size>/``.

    Downloads files one at a time so each gets its own clean tqdm bar —
    unlike ``snapshot_download`` whose parallel ``thread_map`` +
    ``_AggregatedTqdm`` causes the progress bar to appear frozen.
    """
    import huggingface_hub.constants as _hf_constants

    # Disable Xet only for this download — restore original value afterwards
    # so we don't permanently mutate third-party global state.
    _xet_original = _hf_constants.HF_HUB_DISABLE_XET
    _hf_constants.HF_HUB_DISABLE_XET = True

    from huggingface_hub import HfApi, hf_hub_download

    repo_id = _REPO_PREFIX + size
    local_dir = str(_custom_model_path(whisper_model_path, size))
    local_dir_abs = os.path.abspath(local_dir)

    logger.info("Downloading %s → %s", repo_id, local_dir_abs)

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
        # Download one at a time — each file gets its own tqdm bar
        for filename in filtered:
            logger.info("  %s", filename)
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir_abs,
            )

        logger.info("Download complete: %s", local_dir_abs)
        return local_dir_abs
    finally:
        _hf_constants.HF_HUB_DISABLE_XET = _xet_original
