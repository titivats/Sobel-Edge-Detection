# AVTR — Automatic Vision Tab Router

Windows desktop workflow for reviewing AUROTEK AUO6000 PCB images, tuning Sobel edges,
and training a DINOv2-based GOOD / NG classifier. The production UI is in English.

### Windows x64 distribution

Download and run the installer, then open **AVTR Router Vision Studio** from Start. The public installer includes the CPU runtime, the pinned DINOv2 ViT-S/14 weights, and default settings. In Settings, select the machine export, train the classifier for the selected program, run TEST and SAVE SETTINGS before starting AUTO. The launcher accepts the existing Settings password environment variable or prompts to set a password for the current Windows user. [Windows x64 installer](release/AVTR-Setup-x64.exe) · [คู่มือติดตั้งภาษาไทย](release/วิธีติดตั้ง%20AVTR.txt) · [SHA-256 checksums](release/SHA256SUMS.txt)

For a public release build, run `scripts/windows/build_portable.py --public` with the tested Windows x64 Python 3.12 environment. Supply `--app` (the application folder), `--output` (an empty portable directory), `--vc-cab` (the x64 VC runtime minimum redistributable cabinet), and optionally `--hub` (the Torch hub cache). Use the resulting folder's `_runtime/python.exe` to run `scripts/windows/verify_portable.py --public <portable-directory>`. Compile `scripts/windows/avtr.nsi` with NSIS 3 using the absolute paths in `/DPAYLOAD_DIR=...`, `/DOUTPUT_EXE=...` and `/DBUILD_PYTHON=...`.

## Live trial update — 2026-11-18

Enter the number of CUTs shown by the machine program in **SETTING → DATA SOURCE → CUTs in machine program**. Capture counts are checked against this value on Save. **CAPTURE JSON FOLDER** selects the live capture feed. After testing, **START TEST & SAVE SETTINGS** pins the approved checkpoint by SHA-256. Retraining updates the candidate checkpoint. Restarting AUTO continues to use the last approved copy.

Repeat **SAVE EDGE** for each selected cut after upgrading to measurement v7. The dark groove must agree with the projected travel in the capture record. The normal PCB edge in the image supplies the measurement zero. Use reference parts to compare the observed groove and measured dimensions on the actual machine.

The first live scan records the existing older Result files for the selected route and directory in the local inspection database. A matching capture or a changed Result brings an archived board into scope. Later arrivals stay in scope across restarts. An approved program can wait for its first Result in an empty folder. A retry requires the original board's Result and captures before inspection can continue.

## Earlier release — 2026-10-03

- **RECIPE / PROGRAM** in **SETTING → DATA SOURCE** selects one recipe/product/table.
  Images are assigned against the full Result archive before filtering, so choosing a
  program does not reassign another program's images. Counts, previews, models and trials
  follow the selection; switching programs invalidates the previous trial.
  **SPEC** beside this dropdown edits **Max Inner / Max Outer (mm)** only for the selected
  program. The button displays its saved limits, never the latest limits from another program.
  **SAVE SPEC** persists finite, non-negative values; **CANCEL** discards edits.
  The Production toolbar uses **SETTING** to access the three workflow tabs.
  Production first requires a complete, confident GOOD model result, then measures the
  saved A/B side on each new original image and applies the inspected route's SPEC.
  A known exceeded limit is NG; unavailable or incomplete measurement is FAULT/HOLD.
  Production uses the exact saved program identity for model and SPEC selection. A different
  recipe is excluded even when its ProductId/table match. CAPTURE DATA and RETRY HELD BOARD
  buttons have been removed; capture validation and durable recovery checks remain enforced.
- **START AUTO** polls for new stable Result files and their complete image sets. It waits
  for the next board instead of stopping at the end of the loaded queue. SQLite claims
  prevent duplicate processing across restarts. Results and images are stored in the app's
  `production_results.sqlite` and `production_evidence` folder, outside machine data.
  Live image assignment uses capture SN/cut identities instead of acquisition-gap guesses.
  Unchanged Result/capture files are indexed without repeated parsing. Missing or unstable
  images time out; interrupted claims still block when their source files have disappeared.
  Source hashes are checked again when evidence is recorded. A result arriving after HOLD
  is recorded as FAULT, and only one app instance can use a runtime configuration.
  FAULT is displayed as **FAULT / HOLD**, separately from a measured/classified NG.
- Production archives every readable cut as native-resolution `cut_####_original.jpeg`
  and the exact model-crop `cut_####_sobel.jpeg`, using JPEG quality 100 and 4:4:4 sampling
  when supported. The worker writes these before committing the board result, reusing
  decoded crops for the overview. Each attempt has its own date/program/SN directory
  containing `overview.jpeg` and `inspection.json` with predictions, SPEC, measurements,
  source/output hashes and final status. Files are flushed and atomically replaced.
  Export or audit recording failure prevents PASS; a 256 MB free-space reserve holds
  production before archiving. `production_jpeg_quality` (90–100) and
  `production_min_free_mb` configure these defaults. Images are not automatically deleted;
  plan storage/backup capacity for sustained operation. Native resolution preserves camera
  detail; JPEG is lossy and enlarging beyond that detail cannot create sharper pixels.
- Runtime events and worker error traces are written as UTF-8 JSON lines in `logs/avtr.jsonl`,
  rotating at 10 MB with ten backups. SQLite remains the durable decision history.
  The 2026-10-03 suite passed 307 tests. A short synthetic stress run committed 200 boards
  and 1,000 JPEGs through the real Sobel/SPEC/archive pipeline with a stub predictor,
  reopening SQLite ten times with no pending claims. All 32 supplied images also exported
  at 1440×1080 (Original) / 640×640 (Sobel). These checks do not establish days-long uptime,
  real model accuracy, physical mm calibration or machine/PLC integration.
- Full HD (1920×1080) layout, maximized on launch, with previews fitted to the available space.
- Settings opens on DATA SOURCE and retains an already loaded source even when the selected
  program has no matching images. Initial loading and explicit source refresh run in a
  cancellable worker; training/testing/production controls remain locked while it loads.
  Source snapshots still detect Result changes before testing/saving. Directory metadata
  is reused for regular CSV files, and image previews decode each Original once.
  On the supplied 21,842-panel export, measured reopen time fell from 7.09 s to 0.003 s,
  while complete source loading fell from 5.01 s to 1.39 s (local measurements).
- SOBEL PARAMETERS also exposes Edge contrast (90–100%), Horizontal/Vertical edges (0–2),
  and Smoothing spread (0/AUTO–10). These edit the existing saved Sobel contract; changed
  parameters invalidate prior test approval and require matching image preparation/model.
  Train/Test cards stay compact at their content height, with actions adjacent to their status;
  smaller windows retain scrolling. The 2026-10-04 full suite passed 312 tests.
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

1. **DATA SOURCE:** choose the machine folder containing both `Sep.Net` and `SepData`.
   Selecting either subfolder or `SepData/Picture` resolves the same machine. Settings requires
   the pair and reads images only from `SepData/Picture`, excluding program resources.
   Choose **RECIPE / PROGRAM**, enter the CUT count from that program, and set its **SPEC**. A mixed archive is
   supported, but the completed workflow uses one selected recipe/product/table and only its
   matched images. Recipes without matching Result records cannot borrow another program's data.
   **ALL SOURCE IMAGES / SELECT RECIPE** previews the source images without assigning a program.
2. **SOBEL TUNING:** review Original / Sobel. **Edge smoothing** controls Gaussian blur,
   **Edge detail** controls Sobel kernel size, **Noise removal** removes faint pixels,
   and **Edge brightness** controls gain. For thinner lines such as CUT 04, try smoothing 3
   and detail 3 instead of 7. Smaller kernels can also expose more noise. Save the reviewed
   images as GOOD or NG. Opening this panel retains all stored preprocessing values.
3. **TRAIN IMAGES:** review the shared examples and press **START TRAIN THE MODEL**.
   At least five eligible saved examples of each class are required to start.
   This minimum alone does not establish model quality.
4. **TEST ALL … IMAGES:** test the matching model against the selected image set.
5. **START TEST & SAVE SETTINGS:** save only after successful testing with unchanged inputs.

Newly selected programs use separate checkpoint identities derived from recipe name,
ProductId and table, plus the `sobel-magnitude-v1` preprocessing contract. Older checkpoints
that identify only a ProductId are not reused by the new program selector; train and test
the selected program first. Existing checkpoint files and runtime configuration are preserved.
Production cannot inspect an unbound legacy configuration; save the selected program's
tested Settings before starting AUTO.
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
3. Set the selected program's **SPEC** on **DATA SOURCE**. Select **CAPTURE JSON FOLDER**
   for the live capture records. Existing configurations can also specify an explicit JSON file.
4. Close **SETTING**, press **ACK / RESET**, then **START AUTO**. New files must remain
   unchanged for the configured stability interval (default 2 seconds). A Result plus
   every expected cut and matching capture geometry is required. Missing data is allowed
   a settling interval (default 30 seconds) before being recorded as FAULT/HOLD.
5. A board advances only after model and SPEC both pass. NG/FAULT stops AUTO.
   For a PENDING or SOURCE_CHANGED board, inspect the cause and open **SETTING → DATA SOURCE**.
   The contextual retry button authorizes another inspection for the displayed held board.
   Close Settings, acknowledge and restart AUTO. Earlier attempts remain in the history.

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

The 2026-10-03 resilience revision adds interrupted/late results, malformed configuration,
changing source files, duplicate identities, Unicode paths and a 100-board synthetic soak.
All **291 tests** passed, along with repository-wide Ruff lint and formatting checks.
The soak runs real Sobel measurements and SQLite recording with a classifier stub,
including 90 GOOD and 10 SPEC NG results, without duplicate attempts.
An intake-only benchmark with 2,001 synthetic Result files reduced median polling time
from 803 ms to 153 ms; this is not real-machine throughput or neural inference performance.
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

The earlier local dataset audit found 32 images across two panels of 16 cuts and rejected
an old checkpoint without identity. The 2026-10-03 local checkpoint check passed identity
and report quality gates, with 80% accuracy on five validation images. This small validation
set does not establish performance on unseen production boards. Stored training flags alone
do not establish a valid trained model.
All 64 A/B reference checks passed, without changing the unverified calibration status.
These local data and model files are not included in this repository.

## Repository map

- `RouterVisionStudio/production_app.py`: desktop application and workflow.
- `router_vision/workers.py`: intake, inspection, trial and training workers.
- `router_vision/capture_index.py`, `capture_records.py`: indexed capture identities and geometry validation.
- `router_vision/live_source.py`: stable-file intake and recovery checks.
- `router_vision/production_measurement.py`, `spec.py`: classifier/Sobel/SPEC decision stages.
- `router_vision/production_store.py`, `production_evidence.py`: claims, durable history and evidence.
- `router_vision/production_archive.py`, `runtime_log.py`: native cut JPEGs and rotating event logs.
- `router_vision/images.py`: common image loading, including Windows Unicode paths.
- `RouterVisionStudio/router_vision/`: source import, preprocessing, model, reference and UI modules.
- `RouterVisionStudio/tests/`: automated tests using temporary/synthetic data.
- `RouterVisionStudio/config.example.json`: configuration fields and example paths.
- `src/aurotek_edge_detection/`: separate experimental command-line intrusion analysis.

Machine exports, images, recipes, checkpoints, local labels and workflow JSON remain local.
This README replaces the older separate project/operations/measurement documents.
The obsolete `app.py` dashboard and its exclusive baseline/green-mask modules were removed.
The documented offline CLI remains separate from the AVTR production gate. Importing the
package no longer initializes PyTorch or the UI until those components are requested.

## Sep.Net and SepData inputs

The 2026-10-03 check of the supplied AUO6000 export confirmed folder discovery and
Sobel decoding for all 32 images. None matched its Result records: images were captured
on 2026-09-03, while Result filenames ended on 2026-05-16. Its Result archive contained
nine ProductIds and no supported capture JSON was found in SepData. This export needs
matching Result records, an unambiguous product and per-cut capture data before use in
the complete training/AUTO workflow. Its Eqp.cfg pixel scale also differs from the local
app configuration; this check did not verify physical calibration or change either value.

| Location | Required by AVTR |
| --- | --- |
| `SepData/Picture` | Original camera images. |
| `SepData/Result` | Matching panel CSV with SN, recipe and offsets; timestamps must correspond to the images. |
| `SepData/Recipe` | Captured `*.rcp` version, bound by content hash. |
| `SepData/Config/Eqp.cfg` | Source setup: camera pixel scales and picture-save flags. |
| `SepData/Database/ProcessReport.db` | Current historical reference and cross-checks; not queried by fresh capture records. |
| `SepData/Log` | Logs pinned by the current historical manifest; acquisition/alignment audits. |
| `SepData/Temp` | Historical Eqp snapshots pinned by the current manifest. |
| `Sep.Net/MC88X_Log` | Optional historical controller-motion audit. |

The supplied manifest currently pins `Sep_Debug_20260903.log` and
`Temp/20260903/TempEqp20260903T110944.0391436.cfg` as well as the database and local
recipe/Result files. Removing them invalidates that reference. New-board inspection
also requires immutable capture JSON with the geometry described above; current Eqp.cfg
or changing whole logs/databases do not provide a per-image camera position by themselves.

AVTR does not import the Sep.Net executable/DLL/language resources or SepData's EcoSmart,
LayoutScan, Sep_CMD and WorkSataeFolder data. Other database/config/license files belong
to the machine. Absence of an AVTR dependency is not permission to delete machine files.
Keep historical originals separate from the active daily export so an unlimited archive
does not have to be listed on each poll. Config, models, SQLite and evidence belong outside
machine-owned folders; the write guard protects the complete selected SepData export.
