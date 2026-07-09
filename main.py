#!/usr/bin/env python3
"""vid2txt - Extract spoken text from Bilibili videos using Whisper speech recognition.

Usage:
    python main.py <bilibili_url> [options]
"""

import sys
from src.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
