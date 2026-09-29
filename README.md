# AVTR — Automatic Vision Tab Router

Windows desktop workflow for reviewing AUROTEK AUO6000 PCB images, tuning Sobel edges,
and training a DINOv2-based GOOD / NG classifier. The production UI is in English.

## Current release — 2026-09-30

- Full HD (1920×1080) layout, maximized on launch, with previews fitted to the available space.
- Stable Sobel preview scale when navigating images or resizing the workspace.
- Shared image labels and preparation settings between **2 SOBEL TUNING** and **3 TRAIN IMAGES**.
- Training button: **START TRAIN THE MODEL**.
- Settings-save button: **START TEST & SAVE SETTINGS**. This is the renamed save action;
  first run **TEST ALL … IMAGES**, review the predictions, then save. Renaming does not
  combine testing and saving or bypass the validation gates.
- **SELECT PCB EDGE A/B** is available on **3 TRAIN IMAGES**, with Original / Sobel views
  and A / B / NOT SURE choices stored separately from GOOD / NG labels.
- The model-status message is above the training controls on the right, leaving more room for images.

This is a development application, not a certified production inspection system.
The conveyor adapter is simulation-only; no real PLC control is provided.

## Install and launch

From the repository root in Windows PowerShell, using Python 3.10 or newer:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:AVTR_SETTINGS_PASSWORD = '<your local Settings password>'
.\RouterVisionStudio\run.bat
```

Set your own local Settings password before launching. Settings access is disabled if
`AVTR_SETTINGS_PASSWORD` is absent. Do not commit passwords or local runtime configuration.
The first model-training/inference run may need internet access to download the DINOv2 backbone.

On a new checkout, choose a data source through **SETTING → 1 DATA SOURCE**.
Do not copy `config.example.json` unchanged: its machine paths are examples.

## Image workflow

1. **DATA SOURCE:** choose `SepData`, `Picture`, the corresponding `Sep.Net`, or their parent.
   Multiple exports require an explicit choice; a single ProductId and unambiguous panel assignments
   are required for the completed Settings workflow.
2. **SOBEL TUNING:** review Original / Sobel, adjust parameters, and save images as GOOD or NG.
3. **TRAIN IMAGES:** review the shared examples and press **START TRAIN THE MODEL**.
   At least five eligible saved examples of each class are required to start.
   This minimum alone does not establish model quality.
4. **TEST ALL … IMAGES:** test the matching model against the selected image set.
5. **START TEST & SAVE SETTINGS:** save only after successful testing with unchanged inputs.

Checkpoints must match the ProductId and `sobel-magnitude-v1` preprocessing contract.
Missing identity is rejected. The model quality gate requires validation accuracy of at least
80%, recall of at least 50% for each class, and consistent validation counts.
Missing images, changed source data, invalid model output, or low-confidence GOOD predictions
prevent release in the simulated production gate.

## Router references and measurement limits

The A/B selector needs an image-bound reference manifest and its original source files.
Use **LOAD REFERENCE DATA** in the selector. Hash, recipe, ProductId, panel and cut-point
checks must pass. A clean checkout does not include the local machine data or audit manifest.

A and B are calculated tangents of the recorded router-bit sweep. They are not automatically
detected physical PCB edges. A saved choice reuses the selected side for the same recipe/program/
cut-point context; coordinates are checked separately for each image.

`fiducial_projection.py` and `measurement_trial.py` are experimental utilities.
Physical camera calibration, actual encoder position at capture, general PCB-edge discrimination,
and integration of inner/outer MAX measurements into the production workflow remain unfinished.
Reference provenance retains `production_reference_verified=false`.

## Validation

```powershell
.\scripts\test.bat
.\venv\Scripts\ruff.exe check .
.\venv\Scripts\ruff.exe format --check .
```

The 2026-09-30 audit passed **177 tests**, Ruff lint, and Ruff formatting checks.
Layout checks cover 1920×1080, 1920×1040, 1536×864 and smaller-window behavior.
Qt tests use offscreen rendering; they do not certify physical measurement accuracy or model
performance on unseen production boards.

The separate local dataset audit found 32 images across two panels of 16 cuts, with no unassigned
images. Labels were GOOD 15 / NG 17; 31 images matched the inspected preparation settings.
One NG image had different saved Sobel/crop settings. The existing local checkpoint was rejected
for missing ProductId identity. Stored training flags do not establish a valid trained model.
All 64 A/B reference checks passed, without changing the unverified calibration status.
These local data and model files are not included in this repository.

## Repository map

- `RouterVisionStudio/production_app.py`: desktop application and workflow.
- `RouterVisionStudio/router_vision/`: source import, preprocessing, model, reference and UI modules.
- `RouterVisionStudio/tests/`: automated tests using temporary/synthetic data.
- `RouterVisionStudio/config.example.json`: configuration fields and example paths.
- `src/aurotek_edge_detection/`: separate experimental command-line intrusion analysis.

Machine exports, images, recipes, checkpoints, local labels and workflow JSON remain local.
This README replaces the older separate project/operations/measurement documents.
