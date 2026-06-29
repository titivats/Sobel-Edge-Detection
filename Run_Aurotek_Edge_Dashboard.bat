@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "VENV_PYTHON=%PROJECT_ROOT%venv\Scripts\python.exe"

cd /d "%PROJECT_ROOT%"

if not exist "%VENV_PYTHON%" (
    echo Creating Python virtual environment...
    py -3 -m venv "%PROJECT_ROOT%venv"
    if errorlevel 1 (
        echo Failed to create venv with py launcher. Trying python...
        python -m venv "%PROJECT_ROOT%venv"
        if errorlevel 1 goto :error
    )
)

echo Installing dependencies...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :error
"%VENV_PYTHON%" -m pip install -r "%PROJECT_ROOT%requirements.txt"
if errorlevel 1 goto :error

set "PYTHONPATH=%PROJECT_ROOT%src;%PYTHONPATH%"
start "" "%VENV_PYTHON%" -m aurotek_edge_detection.dashboard_app
exit /b 0

:error
echo.
echo Dashboard startup failed. Check the error message above.
pause
exit /b 1
