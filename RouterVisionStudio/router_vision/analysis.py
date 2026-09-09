"""Calibration and inspection.

Within one run the Nth picture is always the same inspection position on the
board, so measurements are only ever compared index to index.  Calibration
learns the mean and spread of each index over known-good runs and discards the
positions that do not frame a slot reliably.  Inspection then measures those
same indices and reports the deviation in millimetres.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .config import AppConfig
from .guard import check_write_target
from .machine import Run
from .vision import CutMeasurement, measure_image

# An index only earns a place in the baseline if it frames a real slot and
# repeats tightly across runs.
MIN_BASELINE_WIDTH_PX = 50.0
MAX_BASELINE_SD_PX = 12.0
MIN_CALIB_RUNS = 5


@dataclass
class IndexBaseline:
    width_mean: float
    width_sd: float
    left_mean: float
    left_sd: float
    samples: int
    # Both edges are recorded, because which one is worth judging depends on the
    # picture: a cut with a tab or a contour leaves one side scalloped, and only
    # the straight side means anything. Defaults keep older baseline files loadable.
    right_mean: float = 0.0
    right_sd: float = 0.0
    rough_side: str = ""  # the side that was scalloped on these panels

    @property
    def straight_side(self) -> str:
        return "right" if self.rough_side == "left" else "left"

    def edge(self, side: str) -> float:
        """Where an edge sat at calibration. Falls back for older files."""
        if side == "right":
            return self.right_mean or self.left_mean + self.width_mean
        return self.left_mean

    def edge_sd(self, side: str) -> float:
        return self.right_sd if side == "right" and self.right_mean else self.left_sd


@dataclass
class Baseline:
    pics_per_run: int
    runs_used: int
    pixel_size_mm: float
    calibrated_at: str
    indices: dict[str, IndexBaseline] = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["indices"] = {k: asdict(v) for k, v in self.indices.items()}
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Baseline":
        indices = {k: IndexBaseline(**v) for k, v in d.get("indices", {}).items()}
        return cls(
            pics_per_run=d["pics_per_run"],
            runs_used=d["runs_used"],
            pixel_size_mm=d["pixel_size_mm"],
            calibrated_at=d["calibrated_at"],
            indices=indices,
        )


class BaselineStore:
    def __init__(self, path: Path, protected: list[Path] | None = None):
        self.path = path
        self.protected = protected or []
        self.data: dict[str, Baseline] = {}
        self.load()

    def load(self) -> None:
        self.data = {}
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for key, value in raw.items():
            try:
                self.data[key] = Baseline.from_json(value)
            except (KeyError, TypeError):
                continue

    def save(self) -> None:
        check_write_target(self.path, self.protected)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v.to_json() for k, v in self.data.items()}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get(self, key: str) -> Baseline | None:
        return self.data.get(key)

    def put(self, key: str, baseline: Baseline) -> None:
        self.data[key] = baseline
        self.save()


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def calibrate(
    cfg: AppConfig,
    runs: list[Run],
    progress=None,
    cancelled=None,
) -> Baseline:
    """Build a baseline from passing runs that all captured the same picture count."""
    usable = [r for r in runs if r.passed and r.pictures]
    if not usable:
        raise ValueError("No passing runs with pictures in this selection.")

    # the dominant picture count defines the shape of a normal run
    pics_per_run = Counter(len(r.pictures) for r in usable).most_common(1)[0][0]
    usable = [r for r in usable if len(r.pictures) == pics_per_run][: cfg.calib_max_runs]
    if len(usable) < MIN_CALIB_RUNS:
        raise ValueError(f"Only {len(usable)} usable runs - need at least {MIN_CALIB_RUNS}.")

    widths: dict[int, list[float]] = defaultdict(list)
    lefts: dict[int, list[float]] = defaultdict(list)
    rights: dict[int, list[float]] = defaultdict(list)
    rough: dict[int, Counter] = defaultdict(Counter)

    for done, run in enumerate(usable, start=1):
        if cancelled and cancelled():
            raise InterruptedError("Cancelled.")
        for k in range(pics_per_run):
            m = _measure(cfg, run.pictures[k])
            if not m.valid:
                continue
            widths[k].append(float(m.width))
            lefts[k].append(float(m.slot_left))
            rights[k].append(float(m.slot_right))
            rough[k][m.rough_side] += 1
        if progress:
            progress(done, len(usable))

    need = max(MIN_CALIB_RUNS, int(len(usable) * 0.6))
    indices: dict[str, IndexBaseline] = {}
    for k in sorted(widths):
        width_samples, left_samples, right_samples = widths[k], lefts[k], rights[k]
        if len(width_samples) < need:
            continue
        w_mean = statistics.fmean(width_samples)
        w_sd = statistics.pstdev(width_samples) if len(width_samples) > 1 else 0.0

        # A position whose pictures agree that one side is scalloped is judged on
        # its straight edge alone, so the usual width sanity check would throw
        # away a position that is perfectly measurable.
        side, hits = rough[k].most_common(1)[0]
        index_rough = side if side and hits > len(width_samples) / 2 else ""
        edge = right_samples if index_rough == "left" else left_samples
        edge_sd = statistics.pstdev(edge) if len(edge) > 1 else 0.0
        if index_rough:
            if edge_sd > MAX_BASELINE_SD_PX:
                continue  # even the straight side is not repeatable here
        elif w_mean < MIN_BASELINE_WIDTH_PX or w_sd > MAX_BASELINE_SD_PX:
            continue  # this position does not show a usable slot
        indices[str(k)] = IndexBaseline(
            width_mean=round(w_mean, 2),
            width_sd=round(w_sd, 2),
            left_mean=round(statistics.fmean(left_samples), 2),
            left_sd=round(statistics.pstdev(left_samples) if len(left_samples) > 1 else 0.0, 2),
            right_mean=round(statistics.fmean(right_samples), 2),
            right_sd=round(statistics.pstdev(right_samples) if len(right_samples) > 1 else 0.0, 2),
            rough_side=index_rough,
            samples=len(width_samples),
        )

    if not indices:
        raise ValueError("No picture index was stable enough to calibrate.")

    return Baseline(
        pics_per_run=pics_per_run,
        runs_used=len(usable),
        pixel_size_mm=cfg.pixel_size_mm,
        calibrated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        indices=indices,
    )


# --------------------------------------------------------------------------
# inspection
# --------------------------------------------------------------------------

STATUS_OK = "OK"
STATUS_NG_WIDTH = "NG WIDTH"
STATUS_NG_SHIFT = "NG SHIFT"
STATUS_WARN_ALIGN = "WARN ALIGN"
STATUS_NO_BASELINE = "NO BASELINE"
STATUS_NO_DATA = "NO DATA"
STATUS_SKIP = "SKIP"


@dataclass
class IndexResult:
    index: int
    ok: bool
    path: str
    width_px: int = 0
    left_px: int = 0
    width_dev_mm: float = 0.0
    edge_dev_mm: float = 0.0
    rough_side: str = ""  # side dropped as scalloped; width is not judged then

    @property
    def width_judged(self) -> bool:
        return self.ok and not self.rough_side


@dataclass
class RunResult:
    run: Run
    status: str = STATUS_OK
    note: str = ""
    checked: int = 0
    width_dev_mm: float = 0.0
    edge_dev_mm: float = 0.0
    details: list[IndexResult] = field(default_factory=list)


def _measure(cfg: AppConfig, path: str) -> CutMeasurement:
    return measure_image(
        path,
        y0=cfg.scan_y0,
        y1=cfg.scan_y1,
        step=cfg.scan_step,
        green_delta=cfg.green_delta,
        green_min=cfg.green_min,
        green_max=cfg.green_max,
        min_valid_rows=cfg.min_valid_rows,
    )


def inspect(
    cfg: AppConfig,
    runs: list[Run],
    store: BaselineStore,
    progress=None,
    cancelled=None,
) -> list[RunResult]:
    results: list[RunResult] = []

    for done, run in enumerate(runs, start=1):
        if cancelled and cancelled():
            break
        if progress:
            progress(done, len(runs))

        res = RunResult(run=run)
        baseline = store.get(run.key)

        if baseline is None:
            res.status = STATUS_NO_BASELINE
            res.note = "calibrate this recipe/table first"
        elif len(run.pictures) != baseline.pics_per_run:
            res.status = STATUS_SKIP
            res.note = f"expected {baseline.pics_per_run} pictures, found {len(run.pictures)}"
        else:
            worst_w = worst_e = 0.0
            ng_width: list[int] = []
            ng_shift: list[int] = []

            for key, ib in sorted(baseline.indices.items(), key=lambda kv: int(kv[0])):
                k = int(key)
                if k >= len(run.pictures):
                    continue
                m = _measure(cfg, run.pictures[k])
                if not m.valid:
                    res.details.append(IndexResult(index=k, ok=False, path=run.pictures[k]))
                    continue

                res.checked += 1
                # One scalloped side makes the width meaningless, so only the
                # straight edge is judged there. The picture decides; the baseline
                # answers for pictures where the scallop was not obvious.
                rough = m.rough_side or ib.rough_side
                side = "right" if rough == "left" else "left"
                edge_px = m.slot_right if side == "right" else m.slot_left

                dw_px = m.width - ib.width_mean
                de_px = edge_px - ib.edge(side)
                dw_mm = cfg.px_to_mm(dw_px)
                de_mm = cfg.px_to_mm(de_px)
                if not rough and abs(dw_mm) > abs(worst_w):
                    worst_w = dw_mm
                if abs(de_mm) > abs(worst_e):
                    worst_e = de_mm

                # absolute spec limit, tightened by how repeatable this index is
                w_lim = cfg.mm_to_px(cfg.width_tol_mm)
                e_lim = cfg.mm_to_px(cfg.edge_tol_mm)
                if cfg.sigma_k > 0 and ib.width_sd > 0:
                    w_lim = min(w_lim, cfg.sigma_k * ib.width_sd)
                if cfg.sigma_k > 0 and ib.edge_sd(side) > 0:
                    e_lim = min(e_lim, cfg.sigma_k * ib.edge_sd(side))
                if not rough and abs(dw_px) > w_lim:
                    ng_width.append(k)
                if abs(de_px) > e_lim:
                    ng_shift.append(k)

                res.details.append(
                    IndexResult(
                        index=k,
                        ok=True,
                        path=run.pictures[k],
                        width_px=m.width,
                        left_px=edge_px,
                        width_dev_mm=0.0 if rough else round(dw_mm, 4),
                        edge_dev_mm=round(de_mm, 4),
                        rough_side=rough,
                    )
                )

            res.width_dev_mm = round(worst_w, 4)
            res.edge_dev_mm = round(worst_e, 4)

            if res.checked == 0:
                res.status = STATUS_NO_DATA
                res.note = "no calibrated index could be measured"
            elif ng_width:
                res.status = STATUS_NG_WIDTH
                idx = ",".join(str(i) for i in ng_width)
                res.note = f"slot width off at index {idx} - over-cut / bit wear / debris"
            elif ng_shift:
                res.status = STATUS_NG_SHIFT
                idx = ",".join(str(i) for i in ng_shift)
                res.note = f"slot shifted at index {idx} - cut off centre"

        _apply_alignment_checks(cfg, run, res)
        results.append(res)

    return results


def _apply_alignment_checks(cfg: AppConfig, run: Run, res: RunResult) -> None:
    """Cross-check the alignment numbers the machine already recorded.

    These come straight from the run CSV, cost nothing, and cover every run
    including the ones with no usable pictures.
    """
    notes: list[str] = [res.note] if res.note else []

    worst = max(abs(run.offset_x), abs(run.offset_y))
    if worst > cfg.align_warn_mm:
        if res.status == STATUS_OK:
            res.status = STATUS_WARN_ALIGN
        notes.append(f"alignment offset {worst:.3f} mm")

    # A pass with a perfectly zero offset means the fiducial correction never
    # landed - the board was cut with no alignment compensation at all.
    if run.offset_x == 0 and run.offset_y == 0 and run.passed:
        if res.status == STATUS_OK:
            res.status = STATUS_WARN_ALIGN
        notes.append("passed with zero alignment offset")

    res.note = "; ".join(notes)
