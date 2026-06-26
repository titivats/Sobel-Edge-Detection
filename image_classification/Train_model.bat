@echo off
setlocal

set "HERE=%~dp0"
set "PYTHON=%HERE%..\venv\Scripts\python.exe"
set "PREPARE=%HERE%prepare_dataset.py"
set "TRAIN=%HERE%train.py"

if not exist "%PYTHON%" (
    echo ERROR: Python environment not found:
    echo %PYTHON%
    pause
    exit /b 1
)

"%PYTHON%" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo ERROR: CUDA-enabled PyTorch is not available.
    echo Run scripts\install_gpu_torch.ps1 from the project folder first.
    pause
    exit /b 1
)

echo ========================================
echo 1/2 Preparing classification dataset
echo ========================================
"%PYTHON%" "%PREPARE%"
if errorlevel 1 (
    echo.
    echo Dataset preparation failed.
    pause
    exit /b 1
)

echo.
echo ========================================
echo 2/2 Training YOLO classifier
echo ========================================
"%PYTHON%" "%TRAIN%" %*
if errorlevel 1 (
    echo.
    echo Training failed.
    echo Make sure Label Studio contains both PASS and NG images.
    pause
    exit /b 1
)

echo.
echo Training complete.
echo Best model:
echo %HERE%runs\pass_ng_classifier\weights\best.pt
pause
exit /b 0
