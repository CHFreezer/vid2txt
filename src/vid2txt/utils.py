"""Utility functions for vid2txt."""

import os
import re
import shutil
import time
import logging
from pathlib import Path

logger = logging.getLogger("vid2txt")

# Bilibili URL patterns
_BILIBILI_PATTERNS = [
    re.compile(r"https?://(?:www\.)?bilibili\.com/video/([A-Za-z0-9]+)"),
    re.compile(r"https?://(?:www\.)?b23\.tv/([A-Za-z0-9]+)"),
]

# Characters invalid in Windows filenames
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def validate_bilibili_url(url: str) -> bool:
    """Check whether *url* is a valid Bilibili video URL."""
    return any(p.search(url) for p in _BILIBILI_PATTERNS)


def format_timestamp(seconds: float) -> str:
    """Convert float seconds to SRT timestamp format: HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def get_output_basename(title: str, video_id: str = "") -> str:
    """Generate a safe filesystem basename from a video title.

    Strips invalid filename characters and truncates to a reasonable length.
    Falls back to the video ID if the title yields an empty string.
    """
    # Strip invalid characters
    clean = _INVALID_FILENAME_CHARS.sub("", title).strip()
    # Collapse whitespace
    clean = re.sub(r"\s+", " ", clean)
    if not clean and video_id:
        clean = video_id
    if not clean:
        clean = "untitled"
    # Truncate
    from .config import MAX_BASENAME_LENGTH
    if len(clean) > MAX_BASENAME_LENGTH:
        clean = clean[:MAX_BASENAME_LENGTH].rstrip()
    return clean


def cleanup_temp_dir(temp_dir: str, *, retries: int = 3, delay: float = 0.5) -> None:
    """Remove a temporary directory, with retries for Windows file-locking."""
    if not os.path.isdir(temp_dir):
        return
    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(temp_dir)
            return
        except PermissionError:
            if attempt < retries:
                time.sleep(delay)
            else:
                logger.warning("Could not remove temp dir: %s (file may be locked)", temp_dir)


def check_dependencies() -> list[str]:
    """Verify external dependencies are available.

    Returns a list of missing dependency descriptions (empty = all good).
    """
    missing: list[str] = []
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg (install via: choco install ffmpeg)")
    if not shutil.which("yt-dlp"):
        missing.append("yt-dlp (install via: choco install yt-dlp)")
    return missing
