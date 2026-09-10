# AVTR - Sobel & Vision Transformer

Windows desktop application for preparing PCB cut images and classifying them as GOOD or NG with DINOv2 ViT features.

## Start the application

Run these commands from this repository folder (the folder containing this README):

```powershell
py -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
$env:AVTR_SETTINGS_PASSWORD = "choose-your-own-password"
.\RouterVisionStudio\run.bat
```

Set the Settings password in each new terminal, or configure it as a user environment variable. Do not put a real password in source code. Select your own machine export in DATA SOURCE after opening Settings.

## Workflow

1. **DATA SOURCE:** select the Router export and check product/panel assignments.
2. **SOBEL TUNING:** adjust parameters, then SAVE AS GOOD or SAVE AS NG. The label and SAVED / NOT SAVED state follow the selected image and current settings.
3. **TRAIN IMAGES:** use the shared saved images and labels to train, test, and save tested settings. At least five eligible images of each class are required to start.

The model uses frozen DINOv2 ViT features and a supervised linear GOOD/NG head. GOOD-only anomaly detection is not implemented. Conveyor control remains simulation-only.

## Start developing

Read [Where to edit](docs/PROJECT_STRUCTURE.md), then change one behavior at a time.

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\scripts\test.bat
.\venv\Scripts\python.exe -m ruff check RouterVisionStudio src
.\venv\Scripts\python.exe -m ruff format --check RouterVisionStudio src
```

The test runner also supports the parent workspace's existing venv. It restores the Qt environment after testing so the desktop app can open normally.

- [Project behavior](Project.md)
- [Operator instructions](RouterVisionStudio/OPERATIONS.md)
- [Optional measurement CLI](docs/EDGE_MEASUREMENT.md)

Machine exports, local settings, labels, saved Sobel images, and model weights are local runtime data, excluded by `.gitignore`.
