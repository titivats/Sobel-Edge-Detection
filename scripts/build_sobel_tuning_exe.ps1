param(
    [string] $ExeName = "Finetune",
    [ValidateSet("onedir", "onefile")]
    [string] $Mode = "onedir",
    [switch] $Console
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$EntryPoint = Join-Path $ProjectRoot "src\fine_tune_entry.py"
$BuildDir = Join-Path $ProjectRoot "build"
$DistDir = Join-Path $BuildDir "finetune-dist"
$PackageDir = Join-Path $ProjectRoot "finetune"
$IconPath = Join-Path $ProjectRoot "assets\sobel_fine_tune_icon.ico"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

if (-not (Test-Path -LiteralPath $EntryPoint)) {
    throw "Tuning UI entry point not found: $EntryPoint"
}

Push-Location $ProjectRoot
try {
    $WindowModeArgs = if ($Console) { @("--console") } else { @("--windowed") }
    $BuildModeArgs = if ($Mode -eq "onedir") { @("--onedir") } else { @("--onefile") }
    $PyInstallerArgs = @(
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--specpath",
        $BuildDir
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
        $EntryPoint
    )
    & $Python @PyInstallerArgs

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    if ($Mode -eq "onedir") {
        $BuiltApp = Join-Path $DistDir $ExeName
        if (Test-Path -LiteralPath $PackageDir) {
            Remove-Item -LiteralPath $PackageDir -Recurse -Force
        }
        Move-Item -LiteralPath $BuiltApp -Destination $PackageDir
        Copy-Item -LiteralPath (Join-Path $ProjectRoot "assets") -Destination $PackageDir -Recurse
        Copy-Item -LiteralPath (Join-Path $ProjectRoot "settings") -Destination $PackageDir -Recurse
        Write-Host "Created: $(Join-Path $PackageDir "$ExeName.exe")"
    }
    else {
        $BuiltExe = Join-Path $DistDir "$ExeName.exe"
        New-Item -ItemType Directory -Path $PackageDir -Force | Out-Null
        Move-Item -LiteralPath $BuiltExe -Destination (Join-Path $PackageDir "$ExeName.exe") -Force
        Copy-Item -LiteralPath (Join-Path $ProjectRoot "assets") -Destination $PackageDir -Recurse -Force
        Copy-Item -LiteralPath (Join-Path $ProjectRoot "settings") -Destination $PackageDir -Recurse -Force
        Write-Host "Created: $(Join-Path $PackageDir "$ExeName.exe")"
    }

    if (Test-Path -LiteralPath $BuildDir) {
        Remove-Item -LiteralPath $BuildDir -Recurse -Force
    }
}
finally {
    Pop-Location
}
