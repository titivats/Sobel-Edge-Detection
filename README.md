# Aurotek Edge Detection Dashboard

Production-oriented dashboard for Aurotek machine vision. The app compares each original image with its Sobel edge preview, then uses the trained YOLO image-classification model to show `PASS`, `NG`, or `UNKNOWN`.

## Quick Start

Double-click:

```text
Run_Aurotek_Edge_Dashboard.bat
```

The launcher creates/uses `venv`, installs `requirements.txt`, and starts the dashboard.

## Default Paths

```text
Input images     E:\AI_Project\edge_detection\Input_files\Picture
Import .CSV      E:\AI_Project\edge_detection\Input_files\Product Info
Models Select    image_classification\runs\pass_ng_classifier\weights\best.pt
Output           outputs_intrusion_sobel
Label Studio     image_classification\Export JSON from label-studio
```

## Main Workflow

1. Put camera images in `Input_files\Picture`.
2. Put machine CSV files in `Input_files\Product Info`.
3. Train/update the classifier from Label Studio exports when needed.
4. Open `Run_Aurotek_Edge_Dashboard.bat`.
5. Use:
   - `Dashboard Real-time` for production review.
   - `Configuration` to select input, CSV, model, and output paths.
   - `Fine-Tune Model` to adjust Sobel preview parameters per image.

## Source Code Layout

```text
src/aurotek_edge_detection/
  dashboard_app.py          Tkinter dashboard UI.
  dashboard_paths.py        Default project paths.
  dashboard_theme.py        Red/white enterprise UI theme.
  dashboard_utils.py        Shared image and timestamp helpers.
  image_classifier.py       Lazy-loaded YOLO image-classification service.
  sobel_edge_detection.py   Sobel edge mask and display helpers.
  image_file_discovery.py   Supported image lookup.

image_classification/
  train.py                  Train YOLO image-classification model.
  prepare_dataset.py        Prepare dataset from Label Studio export.
  Export JSON from label-studio/
  runs/pass_ng_classifier/weights/best.pt

Input_files/
  Picture/                  Runtime image input.
  Product Info/             Runtime CSV input.

Output_files/               Production output folders.
outputs_intrusion_sobel/    Dashboard/generated Sobel output.
docs/                       Developer/operator notes.
```

## Developer Checks

```powershell
.\venv\Scripts\python.exe -m py_compile src\aurotek_edge_detection\dashboard_app.py
```

Use this before pushing UI changes.
