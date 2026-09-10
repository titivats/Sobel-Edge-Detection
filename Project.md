# Sobel & Vision Transformer — final_dev

## Beginner development cleanup (2026-09-10)

Start with [README.md](README.md) and [Where to edit](docs/PROJECT_STRUCTURE.md).
The former general-purpose Sobel experiment, unused recipe-planning templates,
and empty test placeholder have been removed. The active measurement CLI and
engineering workspace remain available as related optional tools.

The Sobel panel now uses one ordered parameter definition list, preserving its
behavior with less repeated construction code. Install `requirements-dev.txt`
and use `scripts/test.bat` to run the 119 application tests. The batch runner
does not require changing PowerShell's execution policy.

AVTR is a Windows desktop project for reviewing Router images, preparing Sobel
inputs, and classifying PCB cuts as GOOD or NG using DINOv2 Vision Transformer
features and a supervised linear classification head.

## Current workflow

1. **DATA SOURCE:** select an export containing Picture and Result; review the
   product, panel assignments, and missing-image warnings.
2. **SOBEL TUNING:** compare the original image with Sobel output. The eight
   parameters are arranged in one column with direct entry and vertical +/minus
   controls. Save the selected image as GOOD or NG between Previous and Next.
3. **TRAIN IMAGES:** use the same saved images and labels without selecting a
   second folder. Train, test predictions, and save tested settings. Page
   navigation buttons sit at the bottom right.

Image labels and preprocessing save status are separate. The image banner shows
GOOD/NG and SAVED / NOT SAVED, while the source machine result remains available
in a tooltip. Changed preprocessing must be saved again before it is eligible
for training; an unsuccessful write does not publish a new label.

## Model and validation

- Frozen DINOv2 ViT-S/14 backbone, supervised GOOD/NG head.
- At least five eligible saved examples per class to start training.
- Approximately 75% training and 25% validation per class.
- Current acceptance checks: validation accuracy at least 80%, and recall of
  each class at least 50%. These development gates are not production certification.
- GOOD-only anomaly detection was discussed as an alternative but is not implemented.
- Conveyor output remains simulation-only.

## Verification

The application passed 119 unittest cases on Windows with Qt offscreen.
Ruff lint/format checks and Python compilation passed for the repository code.

```powershell
cd RouterVisionStudio
$env:QT_QPA_PLATFORM = "offscreen"
..\venv\Scripts\python.exe -m unittest discover -s tests
```

Run tests in a separate terminal from the interactive application, or clear
`QT_QPA_PLATFORM` before launching the desktop UI.

See [OPERATIONS.md](RouterVisionStudio/OPERATIONS.md) for installation and usage.
Machine exports, local settings, labels, saved images, and model weights are
excluded by `.gitignore` and must be supplied locally.
