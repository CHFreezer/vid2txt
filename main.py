#!/usr/bin/env python3
"""vid2txt - Extract spoken text from Bilibili videos using Whisper speech recognition.

Usage:
    python main.py <bilibili_url> [options]
"""

import sys
from src.vid2txt.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
