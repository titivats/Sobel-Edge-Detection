param(
    [string] $InputPath = "",
    [string] $ProgramName = "MANUAL_TUNED"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "sobel_tuning_ui.py"

if ([string]::IsNullOrWhiteSpace($InputPath)) {
    $InputPath = Join-Path $ProjectRoot "SepData\camera"
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

& $Python $Runner --input $InputPath --program-name $ProgramName
exit $LASTEXITCODE
