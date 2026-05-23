# Project Structure

```text
Project_Edge_detection/
  configs/
    default_recipe.json          Default trial tuning values.
  docs/
    PROJECT_STRUCTURE.md         File layout and ownership.
  scripts/
    run_router_intrusion.ps1     PowerShell runner for daily operation.
  src/
    aurotek_edge_detection/
      command_line_interface.py  Command-line argument parsing and orchestration.
      dark_background_mask.py    Black background mask creation.
      focus_zone.py              Focus zone calculation and masking.
      image_file_discovery.py    Supported image file discovery.
      intrusion_measurement.py   Per-image measurement workflow.
      annotated_image_renderer.py Annotated image drawing.
      measurement_data_models.py Shared dataclasses.
      measurement_image_output.py Sobel/annotated image saving.
      measurement_statistics.py  Min/max and baseline helper math.
      measurement_csv_writer.py  CSV writing.
      router_intrusion_main.py   Small production entrypoint.
      slot_boundary_detection.py Sobel boundary selection.
      slot_orientation_roi.py    Slot direction and ROI selection.
      sobel_edge_detection.py    Sobel edge mask creation.
  tools/
    sobel_measure.py             Older general-purpose Sobel experiment.
  tests/
    .gitkeep                     Reserved for future automated tests.
  SepData/
    camera/                      Camera images from the Aurotek Router.
  outputs_intrusion_sobel/
    annotated/                   Sobel images with focus zone and measurements.
    sobel_edges/                 Saved Sobel edge images.
    intrusion_measurements.csv   Measurement result table.
  router_intrusion_measure.py    Backward-compatible wrapper.
  sobel_measure.py               Backward-compatible wrapper for legacy tool.
  requirements.txt               Python dependencies.
  README.md                      Operator and developer guide.
```

## Ownership

- `src/aurotek_edge_detection/router_intrusion_main.py` is the production entrypoint.
- `src/aurotek_edge_detection/intrusion_measurement.py` is the main per-image workflow.
- `src/aurotek_edge_detection/slot_boundary_detection.py` contains Sobel boundary selection.
- `src/aurotek_edge_detection/slot_orientation_roi.py` contains slot direction and ROI logic.
- `src/aurotek_edge_detection/sobel_edge_detection.py` creates the Sobel edge masks.
- `configs/default_recipe.json` is a recipe template. The current CLI still receives values through arguments; this file documents the tuned defaults for each model recipe.
- `SepData/`, `outputs_intrusion_sobel/`, and `venv/` are runtime folders, not source code.
- `tools/sobel_measure.py` is kept for reference and experiments. New Aurotek Router work should use `router_intrusion_measure.py`.
