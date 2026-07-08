#!/usr/bin/env python3
"""Launch the vid2txt WebUI.

Usage:
    python webui.py

The browser will open automatically at http://127.0.0.1:7860
"""

import sys
import os

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# CUDA DLL preloading must happen before any CUDA imports
from src.vid2txt.cuda_setup import setup as _setup_cuda

_setup_cuda()

from src.vid2txt.webui import main

if __name__ == "__main__":
    main()
