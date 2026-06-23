# Project Structure

This project has four runtime workflows. Keep new code close to the workflow it
belongs to, and keep shared path or image helpers in one place.

## Runtime Apps

```text
sobel_edge_detect.py
  Batch Sobel edge image generator.

sobel_tuning_ui.py
  Sobel Fine Tune UI. Shows Original Image and Sobel Edge Detection side by side.
  Built as dist/Sobel_YOLO_Fine_Tune/Sobel_YOLO_Fine_Tune.exe.

src/realtime_predict_ui.py
  Real-time Sobel + YOLO monitor UI.
  Built as dist/Realtime_Sobel_YOLO/Realtime_Sobel_YOLO.exe.

Yolo_train/training.py
  YOLO training command-line runner.
  Built as dist/Sobel_YOLO_Train/Sobel_YOLO_Train.exe when needed.
```

## Source Modules

```text
src/app_paths.py
  Shared project-root and runtime path lookup. Use this for default folders.

src/command_line_interface.py
  Arguments and orchestration for batch Sobel edge generation.

src/image_file_discovery.py
  Supported image search for files and folders.

src/sobel_edge_detection.py
  Core Sobel mask logic. Keep image-processing math here.

src/sobel_edge_output.py
  Writes Sobel edge images to Output_files.

src/sobel_fine_tune_gui.py
  Tkinter Fine Tune UI for adjusting black/white Sobel edge output.

src/sobel_tuning_ui_main.py
  Older OpenCV tuning window and recipe save helpers reused by the Tkinter UI.

src/realtime_config.py
  Real-time UI arguments, colors, and defaults.

src/realtime_app.py
  Tkinter real-time monitor layout, worker thread, and UI queue handling.

src/realtime_predictor.py
  Watches image folders and runs YOLO prediction.

src/realtime_overlay.py
  Converts images to Sobel view and draws PASS/NG overlays.

src/realtime_product_info.py
  Matches image timestamps to Product Info CSV files.
```

## Data And Runtime Folders

```text
configs/
  Recipe templates and saved tuning recipes.

Yolo_train/
  YOLO dataset, labels, data.yaml, and trained weights.

Input_files/
  Sample source data kept with the project.

Output_files/
  Runtime outputs. Do not treat this as source code.

dist/
  Built exe folders. Keep each exe inside its folder with _internal.

build/
  PyInstaller temporary files.
```

## Build Scripts

```text
scripts/build_sobel_tuning_exe.ps1
  Builds the Sobel Fine Tune UI as Sobel_YOLO_Fine_Tune.exe.

scripts/build_realtime_predict_exe.ps1
  Builds the real-time monitor as Realtime_Sobel_YOLO.exe.

scripts/build_yolo_training_exe.ps1
  Builds the YOLO training CLI as Sobel_YOLO_Train.exe.
```

Use `onedir` builds for normal operation. They open faster than `onefile`
because large dependencies do not need to be unpacked on every launch.
