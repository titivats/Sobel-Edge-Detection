"""Application configuration: where the machine data lives and how strict to be."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_CONFIG_NAME = "config.json"
DEFAULT_BASELINE_NAME = "baselines.json"


@dataclass
class AppConfig:
    # --- where the data is -------------------------------------------------
    picture_dir: str = r"E:\router\Picture"
    result_dir: str = r"E:\router\Result"
    recipe_dir: str = r"E:\router\Recipe"
    eqp_cfg_path: str = r"E:\router\Config\Eqp.cfg"

    # Where trained models are kept - one per ProductId and table. Empty means
    # the app's own models folder. It must never point inside machine data.
    model_dir: str = ""

    # Saved together with the tested data source by the AVTR Settings workflow.
    expected_images_by_route: dict[str, int] = field(default_factory=dict)
    active_recipe: str = ""

    # --- scale -------------------------------------------------------------
    # mm per pixel, read from <PixelSize X=".." Y=".."/> in Eqp.cfg.
    # This is the only thing that turns pixels into millimetres - never guess it.
    pixel_size_mm: float = 0.008692709

    # --- scan window and solder-mask classification ------------------------
    # The scan covers the whole picture, one row at a time: y1 is clamped to the
    # picture height, so 4000 simply means "down to the bottom".
    scan_y0: int = 0
    scan_y1: int = 4000
    scan_step: int = 1
    green_delta: int = 5  # a mask pixel needs G - R greater than this
    green_min: int = 25  # and G within this band, which excludes gold pads
    green_max: int = 140
    min_valid_rows: int = 300  # scan rows that must find both edges

    # --- judgement ---------------------------------------------------------
    width_tol_mm: float = 0.20  # absolute spec limit on slot width deviation
    edge_tol_mm: float = 0.20  # absolute spec limit on sideways shift
    sigma_k: float = 5.0  # also fail at K x the sd measured at calibration (0 = off)
    align_warn_mm: float = 0.15  # OffsetX/Y in the run CSV beyond this raises a warning

    # --- model-only mass-production gate ---------------------------------
    # RELEASE is allowed only when every image is classified GOOD at or above
    # this confidence. Lower-confidence GOOD results fail closed as FAULT/HOLD.
    good_confidence_min: float = 0.95

    # Operator-adjustable Sobel preprocessing used by the production model.
    # These values must be revalidated whenever they differ from training.
    sobel_blur_ksize: int = 3
    sobel_blur_sigma: float = 0.0
    sobel_ksize: int = 3
    sobel_gradient_x_weight: float = 1.0
    sobel_gradient_y_weight: float = 1.0
    sobel_clip_percentile: float = 99.5
    sobel_edge_gain: float = 1.0
    sobel_noise_floor: int = 0

    max_runs_per_scan: int = 200
    calib_max_runs: int = 40

    # --- background timers, as left by the operator ------------------------
    show_overlay: bool = True  # draw the measurement on pictures on screen
    index_refresh_min: int = 10  # how often the picture index rebuilds
    auto_calibrate: bool = False  # auto calibration running when the app opens
    calibrate_every_min: int = 10  # how often auto calibration looks for gaps
    calibrate_on_bit_change: bool = True  # rebuild a baseline after a bit-wear alarm

    # ----------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "AppConfig":
        cfg = cls()
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return cfg
            known = {f.name for f in fields(cls)}
            for key, value in raw.items():
                if key in known:
                    setattr(cfg, key, value)
        return cfg

    def save(self, path: Path) -> None:
        # never let the app's own settings land inside the machine data
        from .guard import check_write_target, protected_roots

        check_write_target(path, protected_roots(self))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(asdict(self), indent=2, allow_nan=False), encoding="utf-8"
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    # convenience ----------------------------------------------------------
    @property
    def px_per_mm(self) -> float:
        return 1.0 / self.pixel_size_mm if self.pixel_size_mm else 0.0

    def px_to_mm(self, px: float) -> float:
        return px * self.pixel_size_mm

    def mm_to_px(self, mm: float) -> float:
        return mm / self.pixel_size_mm if self.pixel_size_mm else 0.0


# --------------------------------------------------------------------------
# Eqp.cfg readers
# --------------------------------------------------------------------------

_PIXEL_SIZE_RE = re.compile(r'<PixelSize\s+X="([0-9.eE+-]+)"\s+Y="([0-9.eE+-]+)"')

# Flags worth showing the operator: they explain why images may be missing and
# how loose the machine's own alignment guard is set.
INTERESTING_FLAGS = (
    "EnableAfterCuttingTakePicture",
    "AfterCuttingTakePictureSaveFolder",
    "UseAfterCuttingInspect",
    "AlignMaxOffset",
    "EnableAlignValueDetect",
    "AlignRetryTimes",
    "UpdatedTime",
)


def read_pixel_size(eqp_cfg_path: str | Path) -> tuple[float, float]:
    """Return (x_mm_per_px, y_mm_per_px) from Eqp.cfg."""
    path = Path(eqp_cfg_path)
    if not path.exists():
        raise FileNotFoundError(f"Eqp.cfg not found: {path}")
    match = _PIXEL_SIZE_RE.search(path.read_text(encoding="utf-8", errors="replace"))
    if not match:
        raise ValueError("No <PixelSize .../> element found in Eqp.cfg")
    return float(match.group(1)), float(match.group(2))


def read_machine_flags(eqp_cfg_path: str | Path) -> dict[str, str]:
    path = Path(eqp_cfg_path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for name in INTERESTING_FLAGS:
        m = re.search(rf'{name}="([^"]*)"', text)
        out[name] = m.group(1) if m else "<absent>"
    return out
