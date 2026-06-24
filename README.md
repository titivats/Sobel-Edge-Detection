# Sobel Edge Detection

Production workflow for Sobel image tuning, YOLO training, and real-time
inspection.

## Operator Start

Run the numbered files from the project root:

```text
1_FINE_TUNE.bat
2_TRAIN_YOLO.bat
3_CHECK_TRAINING.bat
4_REALTIME_MONITOR.bat
```

## Fine Tune

`1_FINE_TUNE.bat` opens the Sobel tuning application. Saved images are written
under `Output_files\tuning_saved`, while recipes are written to `settings`.

## YOLO Training

1. Put images in `yolo\images`.
2. Put matching YOLO labels in `yolo\labels`.
3. Edit `yolo\setting.txt`.
4. Run `3_CHECK_TRAINING.bat`.
5. Run `2_TRAIN_YOLO.bat`.

Training results are written to:

```text
yolo\runs\<run_name>\weights\best.pt
```

The checked GTX 1060 3GB baseline is:

```ini
image_size = 320
batch = 1
workers = 0
device = 0
```

## Real-Time Monitor

`4_REALTIME_MONITOR.bat` opens the real-time Sobel and YOLO monitor.

Default paths:

```text
Images: Input_files\Picture
CSV: Input_files\Product Info
Model: best.pt
Output: Output_files\Sobel_Image_OBB
```

## Project Layout

```text
apps\          Packaged Windows applications
assets\        Application icons
Input_files\   Runtime images and Product Info CSV files
Output_files\  Generated images and logs
settings\      Sobel recipes
yolo\          Training dataset, settings, and runs
src\           Production source code
tests\         Automated tests
scripts\       Build and maintenance scripts
```
