"""Image-derived PCB edge measurement for trainer review, separate from classification."""

import cv2
import numpy as np

ALGORITHM_VERSION = "sobel-local-pcb-zero-v6-cut-path-axes"


def validate_image(image):
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 160
    ):
        raise ValueError("Expected a readable uint8 BGR image at least 160 pixels per side.")


def checked_scale(normal, scales):
    scales = np.asarray(scales, dtype=float)
    normal = np.asarray(normal, dtype=float)
    if scales.shape != (2,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Pixel scales must be two finite positive numbers.")
    if (
        normal.shape != (2,)
        or not np.isfinite(normal).all()
        or not np.isclose(np.linalg.norm(normal), 1)
    ):
        raise ValueError("Expected a finite unit normal.")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        factor = 1 / np.hypot(*(normal / scales))
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Pixel scale cannot be represented safely.")
    return float(factor)


def sample_ranges(x, mask):
    indices = np.flatnonzero(mask)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [[int(x[r[0]]), int(x[r[-1]])] for r in runs if len(r)]


def straight_baseline(x, y, wings, width):
    """Require distributed support and agreement on both sides; never trim defects to fit."""
    accepted = np.isfinite(y)
    masks = (x < width * wings[0], x >= width * wings[1])
    lines = []
    for mask in masks:
        indices = np.flatnonzero(mask)
        if len(indices) < 12 or np.mean(accepted[mask]) < 0.9:
            raise ValueError("Insufficient straight-edge coverage on both sides (90% required).")
        for block in np.array_split(indices, 3):
            if np.mean(accepted[block]) < 0.8:
                raise ValueError("Straight-edge support is not distributed across each side.")
        points = np.column_stack((x[mask & accepted], y[mask & accepted])).astype(np.float32)
        vx, vy, px, py = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        if abs(vx) < 1e-6:
            raise ValueError("Unsupported vertical baseline.")
        lines.append((float(vy / vx), float(py - vy / vx * px)))
    mid = float((x[0] + x[-1]) / 2)
    disagreement = abs((lines[0][0] - lines[1][0]) * mid + lines[0][1] - lines[1][1])
    angular_drift = abs(lines[0][0] - lines[1][0]) * (x[-1] - x[0])
    if disagreement > 2 or angular_drift > 4:
        raise ValueError("Left and right straight edges disagree in position or direction.")
    fit = accepted & (masks[0] | masks[1])
    vx, vy, px, py = cv2.fitLine(
        np.column_stack((x[fit], y[fit])).astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
    ).ravel()
    slope, intercept = float(vy / vx), float(py - vy / vx * px)
    deviation = (y - (slope * x + intercept)) / np.hypot(slope, 1)
    residual95 = float(np.percentile(np.abs(deviation[fit]), 95))
    # P95 alone hides narrow notches in the reference strips. Reject, do not smooth them away.
    residual_max = float(np.max(np.abs(deviation[fit])))
    if residual95 > 2 or residual_max > 3:
        raise ValueError("Side segments contain damage or do not support a straight baseline.")
    return (
        slope,
        intercept,
        deviation,
        fit,
        residual95,
        {
            "side_residual_max_px": residual_max,
            "side_line_disagreement_px": float(disagreement),
            "side_line_angular_drift_px": float(angular_drift),
        },
    )


def preview_reference(result, width, wings=(0.25, 0.75)):
    """Fit a visual guide from observed side strips; never relax measurement validation."""
    x, y = result["x"], result["y"]
    fit = result["accepted"] & ((x < width * wings[0]) | (x >= width * wings[1]))
    if result["slope"] is not None:
        slope, intercept = result["slope"], result["intercept"]
        status = "MEASUREMENT_BASELINE"
    else:
        # Do not use the central tab as a substitute for missing reference strips.
        if fit.sum() < 12 or np.ptp(x[fit]) < width * 0.1:
            return None
        vx, vy, px, py = cv2.fitLine(
            np.column_stack((x[fit], y[fit])).astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
        ).ravel()
        if abs(vx) < 1e-6:
            return None
        slope, intercept = float(vy / vx), float(py - vy / vx * px)
        status = "VISUAL_ESTIMATE_ONLY"
    return {
        "status": status,
        "source": "selected_edge_sobel_side_samples",
        "line_px": [[float(v), float(slope * v + intercept)] for v in (x[0], x[-1])],
        "sample_x_ranges": sample_ranges(x, fit),
        "sample_count": int(fit.sum()),
        "used_for_mm": status == "MEASUREMENT_BASELINE",
    }


def local_pcb_zero(result, lo, hi, margin, side):
    """Use observed normal PCB outside the swept span, independently of the blue guide."""
    x, y, accepted = result["x"], result["y"], result["accepted"]
    result.update(
        slope=None,
        intercept=None,
        residual95_px=None,
        deviation=np.full(len(x), np.nan),
        fit=np.zeros(len(x), dtype=bool),
        fit_sample_x_ranges=[],
        baseline_diagnostics={},
    )
    masks = ((x >= lo - margin) & (x < lo), (x > hi) & (x <= hi + margin))
    counts = [int((mask & accepted).sum()) for mask in masks]
    if min(counts) < 12:
        result["baseline_reason"] = (
            "Normal PCB edge needs at least 12 Sobel samples on each side of the cut."
        )
        return
    fit = accepted & (masks[0] | masks[1])
    vx, vy, px, py = cv2.fitLine(
        np.column_stack((x[fit], y[fit])).astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
    ).ravel()
    if abs(vx) < 1e-6:
        result["baseline_reason"] = "Normal PCB edge direction cannot be resolved."
        return
    slope, intercept = float(vy / vx), float(py - vy / vx * px)
    sign = 1 if side == "B" else -1
    deviation = sign * (y - (slope * x + intercept)) / np.hypot(slope, 1)
    result.update(
        slope=slope,
        intercept=intercept,
        deviation=deviation,
        fit=fit,
        normal=sign * np.array([-slope, 1.0]) / np.hypot(slope, 1),
        residual95_px=float(np.percentile(np.abs(deviation[fit]), 95)),
        baseline_reason="",
        fit_sample_x_ranges=sample_ranges(x, fit),
        baseline_diagnostics={
            "normal_pcb_sample_counts": counts,
            "normal_pcb_search_margin_px": float(margin),
        },
    )


def analyse(
    image,
    blur=3,
    fraction=0.4,
    wings=(0.25, 0.75),
    side="B",
    allow_edge_only=False,
    fit_baseline=True,
):
    validate_image(image)
    if side not in ("A", "B"):
        raise ValueError("Select side A or B.")
    if not isinstance(blur, int) or blur < 1 or blur > 15 or blur % 2 == 0:
        raise ValueError("Blur must be an odd integer from 1 to 15.")
    if not 0 < fraction < 1 or len(wings) != 2 or not 0.1 < wings[0] < 0.5 < wings[1] < 0.9:
        raise ValueError("Invalid threshold or side-strip fractions.")
    if side == "A":
        result = analyse(
            cv2.flip(image, 0),
            blur,
            fraction,
            wings,
            side="B",
            allow_edge_only=allow_edge_only,
            fit_baseline=fit_baseline,
        )
        height = image.shape[0]
        result["y"] = height - 1 - result["y"]
        if result["slope"] is not None:
            result["slope"] *= -1
            result["intercept"] = height - 1 - result["intercept"]
        result["normal"][1] *= -1
        result["gray"] = cv2.flip(result["gray"], 0)
        result["dy"] = -cv2.flip(result["dy"], 0)
        result["slot_seed_y"] = height - 1 - result["slot_seed_y"]
        result["slot_end_y"] = height - 1 - result["slot_end_y"]
        return result
    gray = cv2.GaussianBlur(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32), (blur, blur), 0
    )
    height, width = gray.shape
    x = np.arange(int(width * 0.1), int(width * 0.9))
    rows = np.median(gray[int(height * 0.25) : int(height * 0.75), x], axis=1)
    low, high = np.percentile(rows, [10, 90])
    candidates = np.flatnonzero(rows < low + 0.25 * (high - low))
    if not len(candidates):
        raise ValueError("No central dark slot found.")
    runs = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1) + 1)
    slot = max(runs, key=len) + int(height * 0.25)
    seed, end = int(np.median(slot)), int(slot[-1])
    if len(slot) < 20 or end + 50 >= height:
        raise ValueError("Slot is too narrow or too close to the image border.")
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    y = np.full(len(x), np.nan)
    dark = np.median(gray[seed - 8 : seed + 9, x], axis=0)
    material = np.median(gray[end + 20 : end + 45, x], axis=0)
    contrast_ok = material - dark >= 10
    bright = gray[seed : end + 50, x] > dark + fraction * (material - dark)
    bright[:, ~contrast_ok] = False
    # Only follow a component joined to the interior PCB band. A detached reflection
    # or dust spot must not become a protrusion, even if its Sobel gradient is strong.
    _, labels = cv2.connectedComponents(bright.astype(np.uint8), connectivity=8)
    band = labels[end + 20 - seed : end + 45 - seed]
    ids = np.unique(band)
    support = [(int(i), int(np.count_nonzero(np.any(band == i, axis=0)))) for i in ids if i]
    if not support:
        raise ValueError("No connected PCB material below the slot.")
    component, count = max(support, key=lambda item: item[1])
    if count < 0.8 * len(x):
        raise ValueError("PCB material connectivity is ambiguous or too fragmented.")
    connected = labels == component
    detached_columns = np.any(bright & ~connected, axis=0)
    for index, column in enumerate(x):
        if not contrast_ok[index]:
            continue
        # Threshold identifies the slot exit, not the final measured edge.
        sustained = np.convolve(
            connected[:, index].astype(int), np.ones(5, dtype=int), mode="valid"
        )
        exits = np.flatnonzero(sustained == 5)
        if not len(exits):
            continue
        coarse = seed + exits[0]
        lo, hi = max(seed, coarse - 6), min(height - 2, coarse + 7)
        peak = lo + int(np.argmax(dy[lo:hi, column]))
        a, b, c = dy[peak - 1 : peak + 2, column]
        if b < 3 or peak == lo or peak == hi - 1:
            continue
        # A peak must straddle this component boundary, not a nearby disconnected speck.
        nearby = slice(max(0, peak - 2 - seed), min(len(connected), peak + 7 - seed))
        if (
            not connected[nearby, index].any()
            or (bright[nearby, index] & ~connected[nearby, index]).any()
        ):
            continue
        denominator = a - 2 * b + c
        offset = 0.5 * (a - c) / denominator if abs(denominator) > 1e-6 else 0
        y[index] = peak + np.clip(offset, -0.5, 0.5)
    accepted = np.isfinite(y)
    baseline_reason = ""
    try:
        if not fit_baseline:
            raise ValueError("Measurement zero has not been selected.")
        slope, intercept, deviation, fit, residual95, diagnostics = straight_baseline(
            x, y, wings, width
        )
    except ValueError as error:
        if not allow_edge_only or not accepted.any():
            raise
        # Review the observed contour without inventing a baseline or millimetre maxima.
        baseline_reason = str(error)
        slope = intercept = residual95 = None
        deviation = np.full(len(x), np.nan)
        fit = np.zeros(len(x), dtype=bool)
        diagnostics = {}
    normal = (
        np.array([-slope, 1.0]) / np.hypot(slope, 1) if slope is not None else np.array([0.0, 1.0])
    )
    return {
        "x": x,
        "y": y,
        "accepted": accepted,
        "fit": fit,
        "normal": normal,
        "slope": slope,
        "intercept": intercept,
        "deviation": deviation,
        "residual95_px": residual95,
        "baseline_diagnostics": diagnostics,
        "baseline_reason": baseline_reason,
        "detached_bright_columns": int(detached_columns.sum()),
        "fit_sample_x_ranges": sample_ranges(x, fit),
        "slot_seed_y": seed,
        "slot_end_y": end,
        "gray": gray,
        "dy": dy,
    }


def measurement_summary(result, scales):
    factor = checked_scale(result["normal"], scales)
    complete = bool(result["accepted"].all())
    summary = {
        "measurement_status": "COMPLETE_ESTIMATE" if complete else "PARTIAL",
        "coverage": float(result["accepted"].mean()),
        "coverage_meaning": "accepted fraction of sampled columns, not measurement accuracy",
        "detector_verified": False,
        "baseline_source": "image_only",
        "calibration_verified": False,
        "production_decision": None,
        "max_scope": "accepted_samples_only",
        "scale_mm_per_px_xy": list(scales),
        "normal_mm_per_px": factor,
        "slope": result["slope"],
        "intercept": result["intercept"],
        "side_residual95_px": result["residual95_px"],
        "baseline_diagnostics": result["baseline_diagnostics"],
        "detached_bright_columns": result["detached_bright_columns"],
        "fit_sample_x_ranges": result["fit_sample_x_ranges"],
        "measured_x_range": [int(result["x"][0]), int(result["x"][-1])],
        "rejected_x_px": result["x"][~result["accepted"]].tolist(),
    }
    if result["slope"] is None or not result["accepted"].any():
        summary.update(
            measurement_status="EDGE_ONLY" if result["slope"] is None else "NO_EDGE",
            baseline_source=None,
            baseline_reason=result["baseline_reason"],
            normal_mm_per_px=None,
        )
        for name in ("inner", "outer"):
            summary.update(
                {
                    name + "_index": None,
                    name + "_px": None,
                    name + "_mm": None,
                    "complete_" + name + "_mm": None,
                }
            )
        return summary
    for name, sign in (("inner", 1), ("outer", -1)):
        values = result["deviation"] * sign
        indices = np.flatnonzero(result["accepted"] & (values > 1e-6))
        index = int(indices[np.argmax(values[indices])]) if len(indices) else None
        summary[name + "_index"] = index
        summary[name + "_px"] = float(values[index]) if index is not None else 0.0
        summary[name + "_mm"] = summary[name + "_px"] * factor
        summary["complete_" + name + "_mm"] = summary[name + "_mm"] if complete else None
    return summary
