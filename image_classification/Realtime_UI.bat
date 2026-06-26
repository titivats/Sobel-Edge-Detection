@echo off
setlocal

set "ROOT=%~dp0.."
set "PYTHON=%ROOT%\venv\Scripts\python.exe"
set "UI=%ROOT%\src\realtime_predict_ui.py"

if not exist "%PYTHON%" (
    echo ERROR: Python environment not found:
    echo %PYTHON%
    pause
    exit /b 1
)

if not exist "%~dp0runs\pass_ng_classifier\weights\best.pt" (
    echo ERROR: Trained classification model not found:
    echo %~dp0runs\pass_ng_classifier\weights\best.pt
    pause
    exit /b 1
)

set "PYTHONPATH=%ROOT%\src"
"%PYTHON%" "%UI%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Real-time UI stopped with an error.
    pause
)
exit /b %EXIT_CODE%
