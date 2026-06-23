param(
    [string] $ExeName = "Realtime_Sobel_YOLO"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$EntryPoint = Join-Path $ProjectRoot "src\realtime_predict_ui.py"
$BuildDir = Join-Path $ProjectRoot "build"
$IconPath = Join-Path $ProjectRoot "assets\edge_detection_monitor.ico"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

if (-not (Test-Path -LiteralPath $EntryPoint)) {
    throw "Realtime UI entry point not found: $EntryPoint"
}

Push-Location $ProjectRoot
try {
    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --windowed `
        --onefile `
        --name $ExeName `
        --icon $IconPath `
        --paths (Join-Path $ProjectRoot "src") `
        --distpath $ProjectRoot `
        --workpath $BuildDir `
        --collect-all ultralytics `
        --collect-all torch `
        --collect-all torchvision `
        $EntryPoint

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    Write-Host "Created: $(Join-Path $ProjectRoot "$ExeName.exe")"
}
finally {
    Pop-Location
}
