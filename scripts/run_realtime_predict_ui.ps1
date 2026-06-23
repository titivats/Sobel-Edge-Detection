param(
    [string] $ImageDir = "",
    [string] $CsvDir = "",
    [string] $Model = "",
    [string] $OutputDir = "",
    [double] $Confidence = 0.25,
    [double] $PollSeconds = 2.0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "src\realtime_predict_ui.py"

if ([string]::IsNullOrWhiteSpace($ImageDir)) {
    $ImageDir = Join-Path $ProjectRoot "SepData\Picture"
}

if ([string]::IsNullOrWhiteSpace($CsvDir)) {
    $CsvDir = Join-Path $ProjectRoot "SepData\Product_Info"
}

if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = Join-Path $ProjectRoot "Yolo_train\runs\sobel_yolo\weights\best.pt"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot "Output_files\Sobel_Image_OBB"
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

& $Python $Runner --image-dir $ImageDir --csv-dir $CsvDir --model $Model --output-dir $OutputDir --confidence $Confidence --poll-seconds $PollSeconds
exit $LASTEXITCODE
