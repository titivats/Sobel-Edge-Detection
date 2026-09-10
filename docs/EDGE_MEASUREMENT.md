# Optional edge measurement CLI

This tool measures cut intrusion independently of the desktop ViT classifier.

```powershell
.\venv\Scripts\python.exe .\router_intrusion_measure.py --help
.\venv\Scripts\python.exe .\router_intrusion_measure.py --input "C:\Data\Picture" --output ".\outputs_intrusion" --pixels-per-mm 20
```

Replace paths and calibration with real values. Output includes CSV measurements and generated images. For CLI-only installation, the dependencies declared in `pyproject.toml` are sufficient.

The PowerShell wrapper requires explicit paths and calibration:

```powershell
.\scripts\run_router_intrusion.ps1 -InputPath "C:\Data\Picture" -OutputPath ".\outputs_intrusion" -PixelsPerMm 20
```

The package entry point is `src/aurotek_edge_detection/router_intrusion_main.py`. `intrusion_measurement.py` coordinates processing and the boundary/edge modules implement measurement.

The old general-purpose Sobel experiment and unused planning recipe templates were removed. This CLI receives settings through command-line arguments.
