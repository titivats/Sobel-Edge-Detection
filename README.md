# Aurotek Router Sobel Edge Detection

Sobel edge detection project for Aurotek Router camera images. The production
runner now creates Sobel edge images only. Intrusion analysis, CSV result
logging, millimeter calibration, cut classification, and inspection specs are
not part of the active workflow.

## Quick Run

Create Sobel edge images from the default image folder:

```powershell
.\venv\Scripts\python.exe .\sobel_edge_detect.py
```

Or use the helper script:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_sobel_edge_detection.ps1
```

Use a specific input file or folder:

```powershell
.\venv\Scripts\python.exe .\sobel_edge_detect.py `
  --input .\SepData\Picture `
  --output .\Output_files
```

Tune Sobel output:

```powershell
.\venv\Scripts\python.exe .\sobel_edge_detect.py `
  --sobel-threshold-ratio 0.12 `
  --edge-close-kernel 3 `
  --edge-close-iterations 1 `
  --edge-dilate-iterations 0 `
  --display-edge-thickness 1 `
  --edge-view all
```

Open the manual tuning UI:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_sobel_tuning_ui.ps1
```

Build the Fine Tune executable:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_sobel_tuning_exe.ps1
```

## Outputs

Outputs are written to a daily folder inside `Output_files\`.
For example, a run on 22-Jun-2026 writes to:

```text
Output_files\22-Jun-2026\sobel_edges\*_sobel_edge.png
```

No inspection CSV files are created by the active runner.

## YOLO Training

YOLO-format training data is stored in `Yolo_train\`:

```text
Yolo_train\
  data.yaml
  classes.txt
  images\
  labels\
```

Train a YOLO detection model:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\train_yolo.ps1 `
  -Epochs 50 `
  -ImageSize 640 `
  -Batch 4 `
  -Workers 0
```

The trained weights are written under:

```text
Yolo_train\runs\sobel_yolo\weights\best.pt
```

Run prediction with a trained model:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_yolo_predict.ps1 `
  -Model .\Yolo_train\runs\sobel_yolo\weights\best.pt `
  -Source .\Output_files `
  -Confidence 0.25
```

## Real-Time Result UI

Open a real-time preview window that watches router images, matches each image
to the latest `_YYYYMMDD_HHMMSS.csv` file in `SepData\Product_Info`, runs YOLO
on the Sobel edge view, and saves annotated bounding-box images:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_realtime_predict_ui.ps1
```

Defaults:

```text
Images:  SepData\Picture
CSV:     SepData\Product_Info
Model:   Yolo_train\runs\sobel_yolo\weights\best.pt
Output:  Output_files\Sobel_Image_OBB
```

Build the real-time UI executable:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_realtime_predict_exe.ps1
```

## File Layout

```text
configs/                         Recipe templates and tuned Sobel defaults.
scripts/                         Operator/developer helper scripts.
src/                             Production Python modules.
Function_test/                   Function tests.
SepData/Picture/                 Default image input folder.
Output_files/                    Runtime Sobel edge output.
sobel_edge_detect.py             Production Sobel-only wrapper.
```

## Active Modules

- `command_line_interface.py`: command-line parsing and Sobel-only orchestration.
- `image_file_discovery.py`: supported image lookup from file/folder input.
- `sobel_edge_detection.py`: Sobel X, Y, and combined edge masks.
- `sobel_edge_output.py`: saves Sobel edge images.
- `sobel_tuning_ui_main.py`: manual tuning UI.
- `sobel_fine_tune_gui.py`: Fine Tune GUI.
- `realtime_predict_ui.py`: executable entry point for Edge Detection Monitor.
- `realtime_app.py`: Tkinter UI layout, tabs, result cards, and queue handling.
- `realtime_predictor.py`: image scanning, YOLO prediction, and output writing.
- `realtime_product_info.py`: CSV timestamp matching, ProductInfo parsing, and caches.
- `realtime_overlay.py`: Sobel conversion, PASS/NG status, and image annotations.
- `realtime_config.py`: default paths, colors, app icon, and CLI arguments.

## Dependencies

```powershell
pip install -r requirements.txt
```
