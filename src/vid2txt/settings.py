"""Persistent user settings stored as JSON in the project root.

Settings are loaded once at WebUI startup and saved whenever the user
changes a preference.  The file is git-ignored so each developer keeps
their own defaults.
"""

import json
import os
from dataclasses import dataclass

SETTINGS_FILE = os.path.join(os.getcwd(), "vid2txt_config.json")

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

    with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
        json.dump(current, fh, indent=2, ensure_ascii=False)
