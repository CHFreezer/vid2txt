"""Persistent user settings stored as JSON in the project root.

Settings are loaded once at WebUI startup and saved whenever the user
changes a preference.  The file is git-ignored so each developer keeps
their own defaults.
"""

import json
import os
from pathlib import Path

# Project root is 2 levels up from this file (src/settings.py → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = str(_PROJECT_ROOT / "vid2txt_config.json")

_DEFAULTS = {
    "device": "cpu",
    "model_path": "./models",
    "model": "base",
    "language": "auto",
}


def load() -> dict:
    """Return the current settings dict (defaults merged with saved values)."""
    settings = dict(_DEFAULTS)
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return settings
    settings.update(saved)
    return settings


def save(device: str | None = None, model_path: str | None = None,
         model: str | None = None, language: str | None = None) -> None:
    """Persist one or more settings.  Pass ``None`` to keep the current value."""
    current = load()
    if device is not None:
        current["device"] = device
    if model_path is not None:
        current["model_path"] = model_path
    if model is not None:
        current["model"] = model
    if language is not None:
        current["language"] = language

    # Atomic write: write to temp file then replace, avoiding corruption
    # and read-modify-write races with concurrent readers.
    tmp_path = SETTINGS_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(current, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_path, SETTINGS_FILE)
