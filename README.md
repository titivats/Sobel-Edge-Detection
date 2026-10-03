# AVTR — Automatic Vision Tab Router

Windows desktop workflow for reviewing AUROTEK AUO6000 PCB images, tuning Sobel edges,
and training a DINOv2-based GOOD / NG classifier. The production UI is in English.

## Current release — 2026-10-03

- **SPEC** on the Production screen opens recipe-specific **Max Inner / Max Outer (mm)**
  settings. **SAVE SPEC** saves edited recipes to the app configuration; switching recipes
  keeps pending edits, and **CANCEL** discards them. Values must be finite and non-negative.
  The button displays the most recently saved recipe and its limits, and opens that recipe.
  The Production toolbar uses **SETTING** to access the three workflow tabs.
  Production first requires a complete, confident GOOD model result, then measures the
  saved A/B side on each new original image and applies the inspected route's SPEC.
  A known exceeded limit is NG; unavailable or incomplete measurement is FAULT/HOLD.
  The latest SPEC displayed on the button is informational; inspection always selects
  the limits belonging to the actual board route.
- **START AUTO** polls for new stable Result files and their complete image sets. It waits
  for the next board instead of stopping at the end of the loaded queue. SQLite claims
  prevent duplicate processing across restarts. Results and images are stored in the app's
  `production_results.sqlite` and `production_evidence` folder, outside machine data.
- Full HD (1920×1080) layout, maximized on launch, with previews fitted to the available space.
- Stable Sobel preview scale when navigating images or resizing the workspace.
- Shared image labels and preparation settings between **2 SOBEL TUNING** and **3 TRAIN IMAGES**.
- Training button: **START TRAIN THE MODEL**.
- Settings-save button: **START TEST & SAVE SETTINGS**. This is the renamed save action;
  first run **TEST ALL … IMAGES**, review the predictions, then save. Renaming does not
  combine testing and saving or bypass the validation gates.
- **SELECT PCB EDGE A/B** is available on **2 SOBEL TUNING** for its selected image, with Original / Sobel views
  and A / B / NOT SURE choices stored separately from GOOD / NG labels.
- Select A/B or click near a detected edge in the Sobel view to highlight that side
  in cyan and immediately fit a blue straight reference from its observed side strips.
  Blue is a visual guide only. Measurement zero comes from normal PCB Sobel samples
  outside the recorded cut span and appears green when measuring. Red/yellow mark
  **INNER LINE MAX / OUTER LINE MAX**. Review and press **SAVE EDGE**;
  there is no confirmation checkbox. Each image needs its own save confirmation.
  **EDGE_ONLY** can be saved without mm values if normal PCB samples are insufficient.
  **MACHINE REFERENCE** optionally shows the selected side's recorded tangent.
  NOT SURE clears the measurement and revokes this image's confirmation when saved.
- **MEASUREMENT EDGE** shows **Inner Cut (MAX) / Outer Cut (MAX)** in mm for the
  selected side. Clicking rechecks source data; changing sides refreshes the result.
  Maxima are restricted to recorded Start–End travel including the bit radius at both ends.
  Partial results cover accepted samples only. Missing normal PCB support displays N/A;
  the old blue-guide straightness checks no longer gate measurements.
- Zoom the Sobel view with the mouse wheel or **+ / −**; right-drag pans and **FIT**
  resets the view. Original follows the same crop. **MAX INNER / MAX OUTER** focus
  the measured extrema, with red/yellow mm labels on the Sobel image. Display zoom
  does not change measurements or saved source coordinates. Zero/unavailable extrema
  have no focus target. Click selection requires a detected edge within 20 source pixels.
- Edge selection reuses each side's analysis within the current image dialog while
  rechecking source provenance before showing requested measurements. Reloading references
  clears cached analysis. Navigation consumes pending Sobel redraws, and entering training
  refreshes the preview once. Measurement values appear in one result panel.
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

### Live board workflow

1. Save tested Settings for the inspected ProductId / route and expected cut count.
2. Save the trainer's A/B choices for each cut in **2 SOBEL TUNING**. The template binds
   ProductId, recipe content hash, table, program, layer and cut point. A matching
   **SAVE EDGE** confirmation is required. The next board reuses only the side;
   its contour, normal PCB zero and geometry are read again.
3. Set the inspected recipe's **SPEC**. Use **CAPTURE DATA** on **1 DATA SOURCE** to
   select the folder containing per-capture JSON records. An empty configuration can
   use the reference manifest selected in edge review for historical replay.
4. Close **SETTING**, press **ACK / RESET**, then **START AUTO**. New files must remain
   unchanged for the configured stability interval (default 2 seconds). A Result plus
   every expected cut and matching capture geometry is required. Missing data is allowed
   a settling interval (default 30 seconds) before being recorded as FAULT/HOLD.
5. A board advances only after model and SPEC both pass. NG/FAULT stops AUTO.
   **RETRY HELD BOARD** on DATA SOURCE permits an explicit retry of the latest held or
   interrupted board; close Settings, ACK, then START AUTO. Prior attempts are retained.

The capture reader accepts the existing `nominal-router-reference-v1` manifests and
`recorded-router-captures-v1`. The latter has `source_files` with immutable path/SHA256
records, `scale_mm_per_px_xy`, and an `images` list. Each image supplies:

```json
{
  "name": "20261002_120010.bmp",
  "sha256": "<image SHA256>",
  "size_px": [1440, 1080],
  "csv_sn": "<CSV SN>",
  "result_file": "_20261002_120030.csv",
  "result_path": "C:/AUO6000/Router/Result/_20261002_120030.csv",
  "cut_point": 1,
  "reference_context": {
    "product_id": "PRODUCT-A",
    "recipe_name": "PRODUCT-A.rcp",
    "recipe_sha256": "<recipe SHA256>",
    "table": "LeftTable",
    "program_key": 1,
    "layer": "Cut",
    "cut_point": 1
  },
  "motion": {
    "alignment_applied": true,
    "start_end_xy_mm": [[0.0, 0.0], [4.0, 0.0]],
    "camera_xy_mm": [2.0, 0.0],
    "diameter_mm": 1.3,
    "offset_xy_mm": [0.0, 0.0],
    "rotation_rad": 0.0
  }
}
```

These values must come from the capture's machine records; example numbers are not
calibration or actual machine coordinates. Start/End already include alignment, so
offsets are not applied a second time. The reader checks OffsetX/Y and available
BitDiameter/RotateAngle against the hashed Result row. Recipe and Result must be hashed
sources. Reversed horizontal and vertical travel retain the A/B material-side convention.
Do not point manifests at continually changing whole logs/databases: export an immutable
capture snapshot instead. Duplicate matching records are rejected rather than picking one.

The supplied historical logs do not contain per-image segment IDs or actual camera XY.
This implementation reads live capture records; it does not invent those missing fields
or claim to export them from an unsupported proprietary machine format. Physical accuracy
validation and real PLC integration remain deferred at the user's request.

The A/B selector needs an image-bound reference manifest and its original source files.
Use **LOAD REFERENCE DATA** in the selector. Hash, recipe, ProductId, panel and cut-point
checks must pass. A clean checkout does not include the local machine data or audit manifest.

For horizontal travel A selects the upper PCB and B the lower PCB. For vertical travel
A selects the left PCB and B the right PCB. Reversing travel does not swap these sides.
A saved choice reuses the selected side for
the same recipe/program/cut-point context. Recorded router tangents are optional comparison
overlays; cyan highlights the detected contour and blue shows the fitted straight reference.
The blue visual estimate is separate from the measurement zero and does not set mm results.

`fiducial_projection.py` and `measurement_trial.py` are experimental utilities.
The A/B dialog uses Sobel gradients on original pixels and material connectivity, with
the manifest's X/Y scale. The longitudinal measurement interval is the projection of
recorded center travel plus the bit radius at each end; points displaced inward/outward
are not removed just because they lie outside the nominal cutter width. Normal PCB zero
is robustly fitted from observed samples outside that interval, within one diameter
(at least 32 pixels), with at least 12 samples on each side. This remains an estimate;
normal-edge residual and support are recorded. Vertical processing swaps axes and scales,
then restores original pixel coordinates. The blue guide never gates or defines this zero.
If normal PCB support is missing, EDGE_ONLY saves the readable contour without mm values.
Use NOT SURE when the edge cannot be identified. PARTIAL confirmations cover
only highlighted accepted samples and never fill missing columns. Read coverage is not
accuracy. Confirmed contours, normal PCB samples, cut geometry, baseline,
measurements and preview settings are stored with image/context hashes and algorithm version.
Changing the image, sources, result or preview settings requires a new confirmation. These
annotations do not train the GOOD/NG classifier or an edge model, and do not decide GOOD/NG.

Physical camera calibration, actual encoder position at capture and general PCB-edge
discrimination remain unfinished. Texture or copper can still be confused with the PCB edge.
Reference provenance retains `production_reference_verified=false`.

## Validation

The 2026-10-03 release check passed **250 tests**, Ruff lint and formatting checks.
Runtime configuration, trained models, production SQLite files and evidence images
remain excluded from Git.

```powershell
.\scripts\test.bat
.\venv\Scripts\ruff.exe check .
.\venv\Scripts\ruff.exe format --check .
```

The 2026-09-30 audit passed **177 tests**, Ruff lint, and Ruff formatting checks.
The subsequent A/B measurement integration passed 105 relevant tests, including known
inner/outer offsets, missing/partial edges, stale references and per-panel offset changes.
Those results refer to the earlier machine-tangent measurement. The earlier fixed 32-image
baseline trial had B: 15 complete / 7 partial / 10 rejected and A: 1 complete / 4 partial /
27 rejected cases. Baseline failures with readable contours can now appear as EDGE_ONLY.
The previous UI cleanup passed 100 relevant tests. The latest cut-path/normal-PCB-zero
revision passed 62 edge, UI and machine-reference tests, including reversed horizontal/
vertical travel, unequal X/Y scales, exclusion of defects outside travel, and blue-guide
independence. Real 32-image A/B checks returned 25 complete estimates, 37 partial results,
one contour-only result and one unreadable edge. Vertical behavior was checked using
synthetic data; actual vertical machine captures are not present in this dataset.
Tests use standard unittest execution with real PyTorch imports for library checks.
Live production fixtures use a classifier stub with actual Sobel measurements and Qt
workers to check new-board polling, model-to-SPEC decisions and history restoration.
They also reject changing capture data and prevent PASS when result recording fails.
The earlier host Application Control import failure did not recur. These checks do not
execute model training. Click selection, blue overlays,
save confirmation and EDGE_ONLY persistence are covered; real CUT 13 previews were checked
at 960×640, 1250×820 and 1920×1040.
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
