# Aurotek Router Edge Intrusion Measurement

Sobel-based measurement project for Aurotek Router PCB tab cutting images.
The camera captures `.bmp` files into `SepData\camera`; this project converts each image to the selected Sobel edge view and measures tab intrusion into the black router background.

## Quick Run

For current trial calibration, where `1 px = 1 mm`:

```powershell
.\venv\Scripts\python.exe .\router_intrusion_measure.py --input "E:\Project_Edge_detection\SepData\camera" --pixels-per-mm 1
```

Or use the daily runner:

```powershell
.\scripts\run_router_intrusion.ps1 -PixelsPerMm 1
```

Use the real camera calibration before production:

```powershell
.\scripts\run_router_intrusion.ps1 -PixelsPerMm 20
```

## Current Outputs

Outputs are written to `outputs_intrusion_sobel\`:

- `intrusion_measurements.csv`: one result row per image.
- `sobel_edges\*_sobel_edge.png`: the Sobel edge image actually used for measurement.
- `annotated\*_sobel_intrusion.png`: Sobel image with focus zone, baseline, min, and max measurement lines.

The project no longer creates a `masks` folder for the Aurotek Router workflow.

## File Layout

```text
configs/                         Recipe templates and tuned defaults.
docs/                            Developer notes.
scripts/                         Operator/developer helper scripts.
src/aurotek_edge_detection/       Production Python package.
tools/                           Older experiments and reference scripts.
SepData/camera/                  Aurotek Router camera image input.
outputs_intrusion_sobel/         Runtime measurement output.
router_intrusion_measure.py       Backward-compatible production wrapper.
sobel_measure.py                 Backward-compatible wrapper for legacy experiment.
```

See `docs\PROJECT_STRUCTURE.md` for the full layout.

## Measurement Logic

The production entrypoint is `src\aurotek_edge_detection\router_intrusion_main.py`.
The implementation is split by responsibility:

- `command_line_interface.py`: command-line parsing and orchestration.
- `image_file_discovery.py`: supported image lookup from file/folder input.
- `intrusion_measurement.py`: one-image intrusion measurement workflow.
- `dark_background_mask.py`: black background mask creation.
- `focus_zone.py`: yellow focus zone calculation and masking.
- `slot_orientation_roi.py`: horizontal/vertical slot direction and search ROI.
- `sobel_edge_detection.py`: Sobel edge mask creation.
- `slot_boundary_detection.py`: top/bottom/left/right edge selection.
- `annotated_image_renderer.py`: annotated output image drawing.
- `measurement_image_output.py`: saves Sobel and annotated images.
- `measurement_statistics.py`: min/max and baseline helper math.
- `measurement_csv_writer.py`: CSV output.
- `measurement_data_models.py`: shared dataclasses.

High-level flow:

1. Read supported image files from `SepData\camera`.
2. Detect black background pixels inside the focus zone.
3. Decide whether the black slot is horizontal or vertical by projection inside the focus zone.
4. Generate Sobel X/Y edge masks.
5. Use the Sobel axis that matches the detected black slot direction.
6. Estimate the normal black-background baseline.
7. Measure `T/B/L/R min/max` intrusion in pixels and convert to mm with calibration.
8. Save CSV, Sobel edge image, and annotated image.

## Common Tuning

```powershell
.\scripts\run_router_intrusion.ps1 `
  -PixelsPerMm 20 `
  --black-threshold 55 `
  --focus-x-min-ratio 0.20 `
  --focus-x-max-ratio 0.85 `
  --focus-y-min-ratio 0.15 `
  --focus-y-max-ratio 0.85
```

Important options:

- `--pixels-per-mm`: camera calibration. This must be real before production.
- `--black-threshold`: grayscale threshold for black router background.
- `--focus-*-ratio`: yellow focus zone ratios.
- `--orientation auto|horizontal|vertical`: normally keep `auto`.
- `--display-edge-thickness`: visual Sobel thickness only; does not change measurement.

## Setup

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Recipe Direction

`configs\default_recipe.json` records the current tuned defaults.
For multiple PCB models, the next production step is to create one recipe per model, then match camera image timestamp with Aurotek machine log to load the correct recipe and S/N.
