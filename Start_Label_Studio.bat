@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "PYTHON_CMD=python"
set "LS_PORT=8080"
set "DEBUG=0"
set "HOST="

cd /d "%PROJECT_ROOT%"

%PYTHON_CMD% -m pip show label-studio >nul 2>&1
if errorlevel 1 (
    echo Installing Label Studio...
    %PYTHON_CMD% -m pip install label-studio
    if errorlevel 1 goto :error
)

echo Starting Label Studio on http://localhost:%LS_PORT%
echo.
echo Keep this window open while using Label Studio.
echo Press Ctrl+C to stop.
echo.

start "" "http://localhost:%LS_PORT%"
%PYTHON_CMD% "%PROJECT_ROOT%tools\run_label_studio.py" start --host 127.0.0.1 --port %LS_PORT%
exit /b 0

:error
echo.
echo Label Studio startup failed. Check the error message above.
echo If the message says "Application Control policy has blocked this file",
echo please ask IT/Admin to allow Python pip binary extensions (*.pyd), especially numpy.
pause
exit /b 1
