"""Persistent user settings stored as JSON.

Settings are loaded once at WebUI startup and saved whenever the user
changes a preference.  The file is git-ignored so each developer keeps
their own defaults.

Call :func:`set_config_path` to use a custom path (e.g. for tests).  If
never called, defaults to ``<project_root>/vid2txt_config.json``.
"""

import json
import os
from pathlib import Path

from .config import (
    DEFAULT_MODEL, DEFAULT_LANGUAGE, DEFAULT_TARGET_LANG,
    DEFAULT_WHISPER_MODEL_DIR, DEFAULT_TRANSLATION_MODEL_DIR,
)

# Project root is 2 levels up from this file (src/settings.py → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = str(_PROJECT_ROOT / "vid2txt_config.json")

# Mutable — call set_config_path() to override
_config_path: str = _DEFAULT_CONFIG_PATH

_DEFAULTS = {
    "device": "cpu",
    "whisper_model_path": DEFAULT_WHISPER_MODEL_DIR,
    "model": DEFAULT_MODEL,
    "language": DEFAULT_LANGUAGE,
    # Translation
    "translate_enabled": False,
    "target_lang": DEFAULT_TARGET_LANG,
    "translation_model_path": DEFAULT_TRANSLATION_MODEL_DIR,
}


def set_config_path(path: str) -> None:
    """Override the config file path used by :func:`load` and :func:`save`.

    Call before any other settings operations (e.g. at process startup).
    """
    global _config_path
    _config_path = path


def load() -> dict:
    """Return the current settings dict (defaults merged with saved values)."""
    settings = dict(_DEFAULTS)
    try:
        with open(_config_path, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return settings
    settings.update(saved)
    return settings


def _write_atomic(data: dict) -> None:
    """Write *data* to the current config file atomically (tmp + replace)."""
    tmp_path = _config_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_path, _config_path)


def save(device: str | None = None, whisper_model_path: str | None = None,
         model: str | None = None, language: str | None = None,
         translate_enabled: bool | None = None,
         target_lang: str | None = None,
         translation_model_path: str | None = None) -> None:
    """Persist one or more settings.  Pass ``None`` to keep the current value."""
    current = load()
    if device is not None:
        current["device"] = device
    if whisper_model_path is not None:
        current["whisper_model_path"] = whisper_model_path
    if model is not None:
        current["model"] = model
    if language is not None:
        current["language"] = language
    if translate_enabled is not None:
        current["translate_enabled"] = translate_enabled
    if target_lang is not None:
        current["target_lang"] = target_lang
    if translation_model_path is not None:
        current["translation_model_path"] = translation_model_path

    _write_atomic(current)
