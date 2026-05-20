@echo off
title KineticsForge Launcher
echo.
echo  ====================================
echo   KINETICSFORGE PLATFORM LAUNCHER
echo  ====================================
echo.

cd /d "%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11+ from python.org
    pause
    exit /b 1
)

:: Install dependencies if needed
echo [1/3] Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo       Installing dependencies...
    pip install fastapi uvicorn[standard] numpy pydantic python-multipart openpyxl --quiet
)

:: Extract checkpoints from Kaggle results
echo [2/3] Extracting trained checkpoints...
python scripts\extract_checkpoints.py 2>nul

:: Launch server
echo [3/3] Starting KineticsForge server...
echo.
echo  ============================================
echo   Server starting at http://localhost:8000
echo   Open your browser to http://localhost:8000
echo   Press Ctrl+C to stop
echo  ============================================
echo.

python serve.py
pause
