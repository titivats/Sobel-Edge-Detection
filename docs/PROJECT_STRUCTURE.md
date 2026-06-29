# Project Structure

This project is organized around the production dashboard first, with the older Sobel measurement modules kept as reusable processing/backward-compatible code.

```text
edge_detection/
  Run_Aurotek_Edge_Dashboard.bat
      Operator launcher for the dashboard. Creates/uses venv and installs requirements.

  requirements.txt
      Runtime Python dependencies.

  README.md
      Operator-facing quick start.

  src/aurotek_edge_detection/
      dashboard_app.py
          Main Tkinter UI. Owns the three tabs:
          Dashboard Real-time, Configuration, Fine-Tune Model.

      dashboard_paths.py
          Central default paths used by the dashboard.

      dashboard_theme.py
          Central red/white enterprise color theme and ttk tab styling.

      dashboard_utils.py
          Shared utilities for timestamp parsing and Tk image conversion.

      image_classifier.py
          Lazy-loaded YOLO image-classification service.

      sobel_edge_detection.py
          Sobel mask generation and edge display thickening.

      image_file_discovery.py
          Supported image file lookup.

      router_intrusion_main.py / command_line_interface.py / intrusion_*.py
          Older measurement pipeline retained for compatibility and generated Sobel outputs.

  image_classification/
      prepare_dataset.py
          Converts Label Studio export data into a trainable image-classification dataset.

      train.py
          YOLO image-classification training script.

      Export JSON from label-studio/
          Put Label Studio JSON exports here.

      runs/pass_ng_classifier/weights/best.pt
          Default trained classification model selected by the dashboard.

  Input_files/
      Picture/
          Runtime camera image input.

      Product Info/
          Runtime machine CSV input used for timestamp context.

  Output_files/
      Sobel_Final_Result/
          Production output area for final reviewed images.

  outputs_intrusion_sobel/
      Runtime-generated Sobel output used by the current dashboard card loader.
```

## Long-Term Ownership

- Keep UI layout changes in `dashboard_app.py`.
- Put new fixed paths only in `dashboard_paths.py`.
- Put new colors/fonts/tab style only in `dashboard_theme.py`.
- Put reusable non-UI helpers in `dashboard_utils.py`.
- Put model loading/prediction behavior in `image_classifier.py`.
- Avoid adding more production logic to root-level scripts. Prefer package modules under `src/aurotek_edge_detection/`.

## Runtime Folders

The following folders contain machine data, training data, generated output, or local environments:

- `Input_files/`
- `Output_files/`
- `Output_Sobel/`
- `outputs_intrusion_sobel/`
- `image_classification/runs/`
- `venv/`

Do not clean or move these automatically during code refactors.
