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
3. **VISION TRANSFORMER:** label at least five eligible GOOD and five eligible NG
   images, then select **TRAIN & SAVE MODEL**. A fresh linear head is trained on
   frozen DINOv2 features. The first run downloads the pinned backbone source and
   pretrained weights; subsequent runs use the local cache.
4. Select **TEST ALL IMAGES**. Use the image selector or its arrows to inspect each
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
