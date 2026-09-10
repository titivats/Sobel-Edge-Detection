@echo off
setlocal
set "TEST_PYTHON=%~dp0..\venv\Scripts\python.exe"
if not exist "%TEST_PYTHON%" set "TEST_PYTHON=%~dp0..\..\venv\Scripts\python.exe"
if not exist "%TEST_PYTHON%" (
    echo Create venv and install requirements-dev.txt first; see README.md.
    exit /b 1
)
pushd "%~dp0..\RouterVisionStudio"
if errorlevel 1 exit /b 1
set "QT_QPA_PLATFORM=offscreen"
set "PYTHONDONTWRITEBYTECODE=1"
"%TEST_PYTHON%" -m unittest discover -s tests
set "TEST_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %TEST_EXIT_CODE%
