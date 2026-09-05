@echo off
REM Launch AVTR - Automatic Vision Tab Router (fail-safe development gate).
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0..\venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" set "PROJECT_PYTHON=python"
"%PROJECT_PYTHON%" production_app.py %*
if errorlevel 1 pause
