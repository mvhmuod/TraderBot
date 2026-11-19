@echo off
rem Start TraderBot (paper trading only - simulated orders, no real money).
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Could not find the Python environment in ".venv".
    echo Create it first - see README.md for the setup steps.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" app.py
if errorlevel 1 (
    echo.
    echo TraderBot closed with an error. See the messages above.
    pause
)
