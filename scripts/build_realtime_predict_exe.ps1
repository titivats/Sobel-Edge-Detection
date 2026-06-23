param(
    [string] $ExeName = "Realtime_Sobel_YOLO",
    [ValidateSet("onedir", "onefile")]
    [string] $Mode = "onedir",
    [switch] $Console
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$EntryPoint = Join-Path $ProjectRoot "src\realtime_predict_ui.py"
$BuildDir = Join-Path $ProjectRoot "build"
$DistDir = if ($Mode -eq "onedir") { Join-Path $ProjectRoot "dist" } else { $ProjectRoot }
$IconPath = Join-Path $ProjectRoot "assets\edge_detection_monitor.ico"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

if (-not (Test-Path -LiteralPath $EntryPoint)) {
    throw "Realtime UI entry point not found: $EntryPoint"
}

Push-Location $ProjectRoot
try {
    $WindowModeArgs = if ($Console) { @("--console") } else { @("--windowed") }
    $BuildModeArgs = if ($Mode -eq "onedir") { @("--onedir") } else { @("--onefile") }
    $PyInstallerArgs = @(
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean"
    ) + $WindowModeArgs + $BuildModeArgs + @(
        "--name",
        $ExeName,
        "--icon",
        $IconPath,
        "--paths",
        (Join-Path $ProjectRoot "src"),
        "--distpath",
        $DistDir,
        "--workpath",
        $BuildDir,
        "--collect-all",
        "ultralytics",
        "--collect-all",
        "torch",
        "--collect-all",
        "torchvision",
        $EntryPoint
    )
    & $Python @PyInstallerArgs

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    if ($Mode -eq "onedir") {
        Write-Host "Created: $(Join-Path $DistDir "$ExeName\$ExeName.exe")"
    }
    else {
        Write-Host "Created: $(Join-Path $ProjectRoot "$ExeName.exe")"
    }
}
finally {
    Pop-Location
}
