param(
    [string] $InputPath = "",
    [string] $OutputPath = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ExtraArgs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "src\batch_entry.py"

if ([string]::IsNullOrWhiteSpace($InputPath)) {
    $SepDataPath = Join-Path $ProjectRoot "SepData\Picture"
    $InputPath = if (Test-Path -LiteralPath $SepDataPath) {
        $SepDataPath
    }
    else {
        Join-Path $ProjectRoot "Input_files\Picture"
    }
}

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $ProjectRoot "Output_files"
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

& $Python $Runner --input $InputPath --output $OutputPath @ExtraArgs
exit $LASTEXITCODE
