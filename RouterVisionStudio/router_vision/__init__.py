"""Router Vision Studio - inspect, label and classify PCB depaneling cuts."""

__version__ = "3.0.0"

from .analysis import Baseline, BaselineStore, calibrate, inspect
from .config import AppConfig, read_machine_flags, read_pixel_size
from .guard import ProtectedPathError, check_write_target, protected_roots
from .labeling import DEFAULT_CLASSES, Label, LabelStore
from .machine import Run, available_days, index_pictures, load_day
from .model import (
    PREPROCESS_VERSION,
    CropBox,
    CutClassifier,
    FeatureExtractor,
    SobelConfig,
    TrainReport,
    pick_device,
)
from .production import ImageDecision, PanelDecision, classify_panel
from .positions import PositionLabels
from .toolpath import Segment, Toolpath, find_recipe, fit_bounds, load_toolpath
from .vision import CutMeasurement, measure_image, render_overlay

__all__ = [
    "AppConfig", "read_machine_flags", "read_pixel_size",
    "Run", "available_days", "index_pictures", "load_day",
    "Baseline", "BaselineStore", "calibrate", "inspect",
    "CutMeasurement", "measure_image", "render_overlay",
    "Toolpath", "Segment", "fit_bounds",
    "load_toolpath", "find_recipe",
    "LabelStore", "Label", "DEFAULT_CLASSES",
    "CutClassifier", "FeatureExtractor", "CropBox", "SobelConfig",
    "TrainReport", "PREPROCESS_VERSION", "pick_device",
    "ImageDecision", "PanelDecision", "classify_panel",
    "PositionLabels",
    "ProtectedPathError", "check_write_target", "protected_roots",
]
