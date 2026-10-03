"""AVTR package; load public components only when requested."""

from importlib import import_module

__version__ = "3.0.0"

_EXPORTS = {
    "config": ("AppConfig", "read_machine_flags", "read_pixel_size"),
    "guard": ("ProtectedPathError", "check_write_target", "protected_roots"),
    "labeling": ("DEFAULT_CLASSES", "Label", "LabelStore"),
    "machine": ("Run", "available_days", "index_pictures", "load_day"),
    "model": (
        "PREPROCESS_VERSION",
        "CropBox",
        "CutClassifier",
        "FeatureExtractor",
        "SobelConfig",
        "TrainReport",
        "pick_device",
    ),
    "production": ("ImageDecision", "PanelDecision", "classify_panel"),
    "toolpath": ("Segment", "Toolpath", "find_recipe", "fit_bounds", "load_toolpath"),
}
__all__ = [name for names in _EXPORTS.values() for name in names]


def __getattr__(name):
    for module, names in _EXPORTS.items():
        if name in names:
            value = getattr(import_module(f".{module}", __name__), name)
            globals()[name] = value
            return value
    raise AttributeError(f"{__name__!r} has no attribute {name!r}")
