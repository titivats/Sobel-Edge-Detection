# Where to edit

Start with `RouterVisionStudio/production_app.py`. It builds the current desktop application and connects buttons to data and model code.

| Change | File to read first |
| --- | --- |
| Buttons, tabs, parameter controls, status text | `RouterVisionStudio/production_app.py` |
| Sobel, ViT features, training, checkpoints | `RouterVisionStudio/router_vision/model.py` |
| GOOD/NG decisions for a panel | `RouterVisionStudio/router_vision/production.py` |
| Image discovery and panel matching | `RouterVisionStudio/router_vision/auo6000.py` |
| Machine Result CSV parsing | `RouterVisionStudio/router_vision/machine.py` |
| Settings defaults and loading | `RouterVisionStudio/router_vision/config.py` |
| Simulated HOLD/RELEASE | `RouterVisionStudio/router_vision/interlock.py` |
| Protection of machine export files | `RouterVisionStudio/router_vision/guard.py` |
| Reference measurement UI | `RouterVisionStudio/router_vision/reference_ui.py` |
| Recipe geometry | `RouterVisionStudio/router_vision/recipe_reference.py` |
| Regression tests | `RouterVisionStudio/tests/` |

## Main files

- `RouterVisionStudio/run.bat`: desktop launcher.
- `requirements.txt`: desktop dependencies; includes `RouterVisionStudio/requirements.txt`.
- `requirements-dev.txt`: application dependencies plus Ruff.
- `scripts/test.bat`: application tests with an offscreen Qt platform.
- `ruff.toml`: formatting and lint rules.
- `Project.md`: project behavior and scope.
- `RouterVisionStudio/OPERATIONS.md`: operator workflow.

## Related optional tools

`RouterVisionStudio/app.py` is the earlier engineering workspace. Its measurement and calibration modules are still related to the project; they are not the current operator entry point.

`src/aurotek_edge_detection/`, `router_intrusion_measure.py`, and `pyproject.toml` provide an independent measurement CLI. See [the CLI guide](EDGE_MEASUREMENT.md). They do not decide GOOD/NG for the desktop app.

## Local data: do not edit as code

- `venv/`: installed libraries.
- `Router/`, `sobel/`: original machine exports.
- `RouterVisionStudio/config.json`: local settings.
- `RouterVisionStudio/sobel_workflow.json`: image and label records.
- `RouterVisionStudio/sobel_finetune/`: saved Sobel images and earlier versions.
- `RouterVisionStudio/models/`: trained models.
- `__pycache__/`, `.ruff_cache/`: disposable generated caches.

Age alone does not make a saved image or model safe to delete: workflow records may still reference it.

## Small change workflow

1. Find the behavior in the table above.
2. Make one change; avoid editing data or installed libraries.
3. Run `scripts/test.bat` and Ruff.
4. Open the app and check the affected screen.
5. Review `git diff` before committing.
