param(
    [string] $Model = "",
    [string] $Source = "",
    [string] $Output = "",
    [double] $Confidence = 0.25,
    [string] $Project = "",
    [string] $Name = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ExtraArgs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$Yolo = Join-Path $ProjectRoot "venv\Scripts\yolo.exe"

if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = Join-Path $ProjectRoot "Yolo_train\runs\sobel_yolo\weights\best.pt"
}

if ([string]::IsNullOrWhiteSpace($Source)) {
    $Source = Join-Path $ProjectRoot "Input_files\Sobel_Image"
}

if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $ProjectRoot "Output_files\Sobel_Image_OBB"
}

if (-not [System.IO.Path]::IsPathRooted($Model)) {
    $Model = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Model))
}

if (-not [System.IO.Path]::IsPathRooted($Source)) {
    $Source = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Source))
}

if (-not [System.IO.Path]::IsPathRooted($Output)) {
    $Output = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Output))
}

if ([string]::IsNullOrWhiteSpace($Project)) {
    $Project = Split-Path -Parent $Output
}

if (-not [System.IO.Path]::IsPathRooted($Project)) {
    $Project = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Project))
}

if ([string]::IsNullOrWhiteSpace($Name)) {
    $Name = Split-Path -Leaf $Output
}

if (Test-Path -LiteralPath $Source -PathType Container) {
    $imageExtensions = @(".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
    $directImages = Get-ChildItem -LiteralPath $Source -File -ErrorAction SilentlyContinue |
        Where-Object { $imageExtensions -contains $_.Extension.ToLowerInvariant() }

    if (-not $directImages) {
        $latestSobelDir = Get-ChildItem -LiteralPath $Source -Recurse -Directory -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -eq "sobel_edges" -and
                (Get-ChildItem -LiteralPath $_.FullName -File -ErrorAction SilentlyContinue |
                    Where-Object { $imageExtensions -contains $_.Extension.ToLowerInvariant() } |
                    Select-Object -First 1)
            } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1

        if ($latestSobelDir) {
            $Source = $latestSobelDir.FullName
        }
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}

if (-not (Test-Path -LiteralPath $Yolo)) {
    throw "YOLO command not found: $Yolo"
}

if (-not (Test-Path -LiteralPath $Model)) {
    throw "YOLO model not found: $Model"
}

$YoloArgs = @(
    "predict",
    "model=$Model",
    "source=$Source",
    "conf=$Confidence",
    "project=$Project",
    "name=$Name",
    "save=True",
    "exist_ok=True"
) + $ExtraArgs

& $Yolo @YoloArgs
exit $LASTEXITCODE
