@echo off
setlocal

set "HERE=%~dp0"
set "PYTHON=%HERE%..\venv\Scripts\python.exe"
set "SCRIPT=%HERE%predict.py"
set "INPUT=%HERE%predict_images"

if not exist "%PYTHON%" (
    echo ERROR: Python environment not found:
    echo %PYTHON%
    pause
    exit /b 1
)

if not exist "%INPUT%" mkdir "%INPUT%"

dir /b /s "%INPUT%\*.bmp" "%INPUT%\*.png" "%INPUT%\*.jpg" "%INPUT%\*.jpeg" "%INPUT%\*.tif" "%INPUT%\*.tiff" "%INPUT%\*.webp" >nul 2>&1
if errorlevel 1 (
    echo No images found.
    echo Put new images in:
    echo %INPUT%
    pause
    exit /b 1
)

"%PYTHON%" "%SCRIPT%" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Prediction failed.
    pause
    exit /b %EXIT_CODE%
)

echo.
echo Prediction complete.
echo Results:
echo %HERE%prediction_results
pause
exit /b 0
