$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python venv not found: $Python"
}

$env:PYTHONUTF8 = "1"
& $Python -m pip install `
    --force-reinstall `
    --no-deps `
    torch==2.12.1+cu126 `
    torchvision==0.27.1+cu126 `
    --index-url https://download.pytorch.org/whl/cu126

if ($LASTEXITCODE -ne 0) {
    throw "CUDA PyTorch installation failed with exit code $LASTEXITCODE"
}

& $Python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__); print(torch.cuda.get_device_name(0))"
exit $LASTEXITCODE
