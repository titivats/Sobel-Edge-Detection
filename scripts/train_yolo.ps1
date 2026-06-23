param(
    [string] $Data = "",
    [string] $Model = "yolov8n.pt",
    [int] $Epochs = 50,
    [int] $ImageSize = 640,
    [int] $Batch = 16,
    [int] $Workers = 0,
    [string] $Project = "",
    [string] $Name = "sobel_yolo",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ExtraArgs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$Yolo = Join-Path $ProjectRoot "venv\Scripts\yolo.exe"

if ([string]::IsNullOrWhiteSpace($Data)) {
    $Data = Join-Path $ProjectRoot "Yolo_train\data.yaml"
}

if ([string]::IsNullOrWhiteSpace($Project)) {
    $Project = Join-Path $ProjectRoot "Yolo_train\runs"
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

if (-not (Test-Path -LiteralPath $Yolo)) {
    throw "YOLO command not found: $Yolo"
}

$YoloArgs = @(
    "train",
    "model=$Model",
    "data=$Data",
    "epochs=$Epochs",
    "imgsz=$ImageSize",
    "batch=$Batch",
    "workers=$Workers",
    "project=$Project",
    "name=$Name"
) + $ExtraArgs

& $Yolo @YoloArgs
exit $LASTEXITCODE
