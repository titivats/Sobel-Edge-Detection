param(
    [Parameter(Mandatory = $true)]
    [string] $InputPath,
    [Parameter(Mandatory = $true)]
    [string] $OutputPath,
    [Parameter(Mandatory = $true)]
    [ValidateRange(0.000000001, 1000000000)]
    [double] $PixelsPerMm,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ExtraArgs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "router_intrusion_measure.py"

if (-not (Test-Path -LiteralPath $Python)) {
    $Python = Join-Path (Split-Path -Parent $ProjectRoot) "venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Create venv and install dependencies first; see README.md."
}

& $Python $Runner --input $InputPath --output $OutputPath --pixels-per-mm $PixelsPerMm @ExtraArgs
exit $LASTEXITCODE
