param(
    [string] $Settings = "",
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "src\yolo_training.py"

if ([string]::IsNullOrWhiteSpace($Settings)) {
    $Settings = Join-Path $ProjectRoot "yolo\setting.txt"
}
elseif (-not [System.IO.Path]::IsPathRooted($Settings)) {
    $Settings = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Settings))
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python venv not found: $Python"
}

if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Training runner not found: $Runner"
}

if (-not (Test-Path -LiteralPath $Settings -PathType Leaf)) {
    throw "Settings file not found: $Settings"
}

$Arguments = @($Runner, "--settings", $Settings)
if ($DryRun) {
    $Arguments += "--dry-run"
}

& $Python @Arguments
exit $LASTEXITCODE
