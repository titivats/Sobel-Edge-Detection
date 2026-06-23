param(
    [string] $ExeName = "Sobel_Fine_Tune"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$EntryPoint = Join-Path $ProjectRoot "sobel_tuning_ui.py"
$BuildDir = Join-Path $ProjectRoot "build"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

if (-not (Test-Path -LiteralPath $EntryPoint)) {
    throw "Tuning UI entry point not found: $EntryPoint"
}

Push-Location $ProjectRoot
try {
    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --name $ExeName `
        --paths (Join-Path $ProjectRoot "src") `
        --distpath $ProjectRoot `
        --workpath $BuildDir `
        $EntryPoint

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    Write-Host "Created: $(Join-Path $ProjectRoot "$ExeName.exe")"
}
finally {
    Pop-Location
}
