# AVTR — Automatic Vision Tab Router

AVTR is a Windows desktop application for reviewing Aurotek Router AUO6000 exports,
tuning Sobel edge images, labeling GOOD/NG examples, and testing a DINOv2 vision
classifier. Run `production_app.py` or `run.bat`.

## Install and open

From the repository root, using Python 3.10 or newer:

```powershell
py -m venv venv
.\venv\Scripts\python.exe -m pip install -r .\RouterVisionStudio\requirements.txt
$env:AVTR_SETTINGS_PASSWORD = "choose-your-own-password"
[Environment]::SetEnvironmentVariable("AVTR_SETTINGS_PASSWORD", $env:AVTR_SETTINGS_PASSWORD, "User")
.\RouterVisionStudio\run.bat
```

The environment variable configures this terminal and future terminals. There is
no built-in password. Without it, Settings remains locked. On a fresh install,
open Settings and select a data source; the application does not automatically
scan an assumed machine path. `config.example.json` is an optional starting point
for manual configuration; replace its example paths before launching.

## Settings workflow

### Technician image training

Use **TRAIN IMAGES** on the main screen to open the training page through the
existing Settings authentication. This is a workflow shortcut, not a separate
role or permission system. A technician with Settings access can prepare and
train examples without using the Recipe reference editor or camera mapping.

1. In **1 DATA SOURCE**, select the Router root containing Picture and Result.
   The app needs the product identity from the results.
2. In **2 SOBEL TUNING**, review Original/Sobel and choose **SAVE AS GOOD** or
   **SAVE AS NG** between Previous and Next. These save current preprocessing
   and the selected image's label. Labels and saved images carry over to page 3
   automatically; no second folder selection is needed. The label actions remain
   in page 2, and page 3 provides image navigation, training and testing.
3. Review the GOOD/NG counts and model training status in the training card.
   Press **TRAIN & SAVE MODEL** when enough eligible examples are available.
4. Run **4a TEST MODEL**, review predictions, then use **4b SAVE TESTED SETTINGS**.
   The bottom-right buttons navigate between Sobel Tuning and Train Images.

Sobel parameters use one column with direct numeric entry and vertical +/minus
buttons (hold to repeat). Numeric fields support the mouse wheel. The panel
scrolls vertically on short screens, with no horizontal scrollbar. Blur size and
Sobel kernel retain their discrete choices. The current image banner shows the
user's GOOD/NG label and SAVED / NOT SAVED state; the machine's original Router
result is available in its tooltip. After a parameter edit, the saved class
button returns to SAVE AS GOOD/NG until matching preprocessing is saved again.

Labeling does not silently retune preprocessing: current Sobel settings are used
for the preview and save. If edges need adjustment, use the Sobel tab and review
examples under the new settings. An image or label save failure keeps the
selection in place. If only the label write fails, the saved Sobel image remains
available for retry. The next-step banner, quick-guide button, and advanced
settings/edge-measurement button are hidden in the final development UI.
The hidden confidence control retains its configured value for testing.
Training does not certify a model for production, and current minimum sample
counts and quality gates are unchanged.

### Detailed preparation and testing

1. **DATA SOURCE:** choose the export root containing `Picture`, `Result`, and
   optionally `Recipe`, `Log`, and `Config`. Both the newer AUO6000 CSV without a
   table column and the older table-based CSV are supported. Review unmatched
   image/panel warnings before continuing.
2. **SOBEL TUNING:** compare Original and Sobel, adjust the eight controls, and
   save reviewed images. Preview updates automatically. SAVED and TRAINED are
   separate states. A model has one Sobel configuration: only labeled images
   saved with the currently selected configuration and crop are eligible for
   training. Images saved with other settings remain available for further work.
   X/Y edge weights and brightness use 0.001 steps, shown and saved at the same
   precision; blur strength and normalization use 0.01 steps. Older records that
   already rounded a value cannot recover the lost digits; review and save again.
3. **TRAIN IMAGES:** prepare at least five eligible GOOD and five eligible NG
   images in page 2, then select **TRAIN & SAVE MODEL**. A linear head is trained on
   frozen DINOv2 features. The first run downloads the pinned backbone source and
   pretrained weights; subsequent runs use the local cache.
4. Select **4a TEST MODEL**. Use the image selector or its arrows to inspect each
   Original/Sobel/prediction pair. A low-confidence GOOD remains `PREDICT: GOOD`
   and is marked `BELOW THRESHOLD`; it is not accepted as a passing result.
5. **SAVE SETTINGS** becomes available after a complete, valid test. It saves the
   selected source, its expected panel image count, and the GOOD threshold, then
   reloads the operator queue and matching ProductId model. Press **ACK / RESET**
   to clear HOLD before inspection. Changed inputs require another test.

AUO6000 models are keyed by ProductId. Legacy table-specific queues remain
distinct. Checkpoints also carry the model identity, crop and Sobel parameters;
renaming a checkpoint does not change its identity. Older checkpoints without
identity metadata require retraining. Inference preserves checkpoint preprocessing.

## Model screening and current limits

### Reference edge measurement (experimental)

The reference editor remains in the code and tests, but its entry button is
hidden in the final development UI. The following describes the retained
experimental editor for developers; it is not a current operator workflow.
It does not modify production GOOD/NG, labels, Sobel preprocessing or model weights.

1. Click **MANUAL LINE** and drag along the nominal, known-good PCB boundary when using manual mode.
   Drag the endpoints to change its angle; MOVE buttons shift it one pixel normal
   to the line. Do not fit the reference to a defect. Wheel zooms; dragging away
   from the endpoints pans; **FIT IMAGE** restores the view.
2. Use **FLIP PCB SIDE** so the pink arrow points into material. Select the
   appropriate bright/dark boundary polarity. Adjust the search band and minimum
   contrast while checking the detected edge against the Original image.
3. Blue is the reference. Red marks inward deviation over the trial limit;
   orange marks protrusion over the limit. Green is within the trial limits,
   not a production GOOD verdict. Maximum deviations and edge coverage update
   automatically. Unreliable or clipped scans make the overall measurement
   invalid; displayed extrema then describe only the accepted portions.
4. Measurements start in original-resolution pixels. Enter separate X/Y mm/px
   scales and explicitly verify them against a known dimension at the actual
   PCB plane to enable mm. No default machine scale is assumed. Confirm manual
   reference alignment separately. Neither confirmation is restored on reopening.
5. **SAVE REFERENCE FOR THIS IMAGE** saves the line and controls to the local
   `reference_lines.json`. References are source-signature-bound and per image in
   this prototype. They are not automatically transferred across boards or cut
   points. There is no automatic registration or curved-reference support yet.

The detector samples grayscale material transitions after a fixed 3x3 blur,
independently of the adjustable Sobel/ViT preprocessing. It measures visible 2-D
outline deviation, not out-of-plane burr height. Sampling is at most one pixel
apart along a segment up to 4095 pixels long. Interpolated decimal output is not
an accuracy guarantee. Reflections, copper tracks and solder-mask boundaries can
be confused with the physical PCB edge. Validate optical calibration, detection,
repeatability and tolerances before considering any production integration.

#### Reference from the Router program

Choose **CHOOSE RECIPE LINE** to use machine-program geometry instead of
drawing a reference. The matching panel's recipe is suggested when available;
**CHOOSE RECIPE** also accepts recipes in customer subfolders. Choose a
straight candidate segment in the recipe overview. The existing binary reader
scans the first 40,000 bytes heuristically; candidate geometry and its meaning
must be checked against the Router program, not treated as a certified parser.

Enter the machine X/Y coordinate corresponding to the original image center,
the verified X/Y mm-per-pixel scales, the angle of image +X in machine XY, and
the camera Y direction. Use inspection-camera coordinates with the correct
camera/tool offset. These values are not inferred from the inspected edge or
borrowed from another machine. This planar mapping supports rotation and separate
axis scales; it does not correct perspective or lens distortion.

Set the signed nominal-edge offset from segment A to B: positive is its left
normal in machine XY. If the recipe represents a cutter centerline, establish
the correct side and compensation from the Router program/tool setup; do not
assume the centerline is the finished PCB edge or apply the tool radius twice.
Zero means the recipe segment itself is the intended reference.

After checking the mapping, choose **USE THIS LINE**. Only the visible
portion is projected; out-of-view or missing-scale mappings are refused. The
line is never stretched to fit an image. Recipe endpoints and MOVE controls are
locked; change the recipe mapping to reposition them. Check alignment against
the Original image before interpreting measurements. mm still requires explicit
scale verification. **MANUAL LINE** leaves recipe mode.

Saved references include the recipe path, SHA-256, selected segment, mapping and
edge offset. Reload and save recheck recipe content and geometry. Alignment and
scale verification are not restored as confirmed. The reference remains bound
to one source image and does not change GOOD/NG or PLC decisions. No actual
AUO6000 camera mapping has been validated in the current sample export.

The default view shows the recipe and Original photo side by side, with numbered
recipe lines and **PREVIOUS / NEXT** navigation. Coordinates are in tooltips;
technical mapping controls are collapsed under **CAMERA SETUP**. When mapping
is missing, **CAMERA SETUP NEEDED** explains the block and **USE THIS LINE** is
disabled. No reference is drawn on the photo until the mapping is verified.
This simplifies the screen; it does not supply missing camera calibration or
automatically associate recipe line numbers with image cut-point numbers.
The measurement editor keeps manual drawing, search controls and unit calibration
under **ADVANCED MEASUREMENT SETTINGS**. Existing saved settings retain their behavior.

To open the sample recipe editor directly:

```powershell
cd RouterVisionStudio
..\venv\Scripts\python.exe -m router_vision.reference_ui ..\sobel\Picture\20260903_111059.bmp --recipe ..\sobel\Recipe\199944000-Optoput-Test.rcp
```

To open just the trial editor without starting inspection:

```powershell
cd RouterVisionStudio
..\venv\Scripts\python.exe -m router_vision.reference_ui "D:\path\to\image.bmp"
```

### Classifier screening

The development screening gate requires at least 80% overall validation accuracy,
both classes in validation, and at least 50% recall for each class. Failed training
does not overwrite the previous model. Unreadable labeled inputs abort training.
These are development checks, not production qualification criteria. Validation
currently splits individual images; production qualification still needs an
independent, representative holdout organized by board/lot and a defect-recall
target agreed for the process. Confidence is a model score, not a measured defect
probability.

The operator screen reviews retained exports. It is not a live camera/PLC service.
The current queue stops at its end instead of silently replaying panels. Reloading
a queue can intentionally revisit historical panels. New-format image-to-result
association uses timestamps with a bounded delay; no direct image-to-panel ID is
provided by that export format. The saved expected image count comes from the
reviewed export and must match the actual recipe before any hardware integration.

The conveyor interface is simulation-only; physical PLC output is disabled.
Incomplete images, invalid or non-finite predictions, unknown classes, low GOOD
confidence, and worker errors produce HOLD. Recipe/day/model changes cannot clear
an existing HOLD or retain a previous PASS release. Workstation shutdown requests
cooperative cancellation and waits for running workers to finish.

## Local data and verification

Machine exports are read-only. Local `config.json`, `sobel_workflow.json`, saved
Sobel images and models stay outside the machine source. Runtime data, passwords,
machine exports and weights are not included in Git. Model writes use temporary
files and replace the checkpoint only after serialization succeeds.

Sobel saves create a new PNG version before committing its workflow record. A
failed record write keeps the previous image and label/status record unchanged;
successful saves retain previous PNG versions in `sobel_finetune` (disk usage can
grow). Unicode output folders, including Thai names, are supported. New records
also check output size/modification time so a replaced or damaged export is not
silently treated as SAVED. These checks are not cryptographic integrity checks.
If a model is saved but its training-status record fails to save, the UI reports
failure rather than claiming TRAINED; resolve the storage error and train again.

```powershell
cd RouterVisionStudio
..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover parsers, grouping, model identity/persistence, numerical failure
cases, Settings state transitions, bounded prediction rendering, and Qt layout.
They do not require downloading or training DINOv2. `app.py` is the earlier
Engineering Studio for measurement/calibration experiments; it is not part of the
operator decision path. Historical notes in the legacy README are not current
production performance claims.
