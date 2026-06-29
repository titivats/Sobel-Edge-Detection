from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_DIR = PROJECT_ROOT / "Input_files" / "Picture"
DEFAULT_CSV_DIR = PROJECT_ROOT / "Input_files" / "Product Info"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs_intrusion_sobel"
DEFAULT_CLASSIFICATION_MODEL = (
    PROJECT_ROOT / "image_classification" / "runs" / "pass_ng_classifier" / "weights" / "best.pt"
)
