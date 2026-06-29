param(
    [string] $InputPath = "E:\Project_Edge_detection\SepData\camera",
    [string] $OutputPath = "E:\Project_Edge_detection\outputs_intrusion_sobel",
    [double] $PixelsPerMm = 1,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ExtraArgs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "router_intrusion_measure.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

& $Python $Runner --input $InputPath --output $OutputPath --pixels-per-mm $PixelsPerMm @ExtraArgs
exit $LASTEXITCODE
