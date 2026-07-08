@echo off
title vid2txt WebUI
cd /d "%~dp0"

echo ========================================
echo   vid2txt WebUI Launcher
echo ========================================
echo.

call "C:\Users\chfre\miniconda3\condabin\conda.bat" activate "%~dp0.conda"
if errorlevel 1 (
    echo [ERROR] Conda activation failed
    pause
    exit /b 1
)

echo [INFO] Starting WebUI at http://127.0.0.1:7860
echo.
python webui.py
pause
