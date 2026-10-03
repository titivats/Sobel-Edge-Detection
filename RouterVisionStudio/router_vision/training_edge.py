"""Trainer-selected material side, separate from classifier labels and pixels."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .image_edge import (
    ALGORITHM_VERSION,
    analyse,
    local_pcb_zero,
    measurement_summary,
    preview_reference,
)
from .machine_reference import load_manifest, reference_for_image, sha256
from .reference import EdgeProfile, measure_reference, normal_scale

CHOICES = ("UNSURE", "A", "B")
CONTEXT_FIELDS = ("product_id", "recipe_sha256", "table", "program_key", "layer", "cut_point")


def measure_image_edge(image, review, edge):
    """Find the selected edge in this image; machine coordinates are not fitted to it."""
    if edge not in ("A", "B"):
        raise ValueError("Select A or B before measuring.")
    reference = review[edge]
    vertical = reference["orientation"] == "vertical"
    working_image = cv2.transpose(image) if vertical else image
    result = analyse(working_image, side=edge, allow_edge_only=True, fit_baseline=False)
    guide = preview_reference(result, working_image.shape[1])
    path = reference["cut_path"]
    lo, hi = np.ceil(path["axis_extent_px"][0]), np.floor(path["axis_extent_px"][1])
    in_path = (result["x"] >= lo) & (result["x"] <= hi)
    if not in_path.any():
        raise ValueError("Recorded cut travel has no overlap with the readable image scan.")
    complete_scope = bool(result["x"][0] <= lo and result["x"][-1] >= hi)
    scales = reference["scales"][::-1] if vertical else reference["scales"]
    local_pcb_zero(result, lo, hi, max(32, path["bit_diameter_mm"] / scales[0]), edge)
    zero_points = np.column_stack((result["x"][result["fit"]], result["y"][result["fit"]]))
    for name in ("x", "y", "accepted", "fit", "deviation"):
        result[name] = result[name][in_path]
    summary = measurement_summary(result, scales)
    if not complete_scope and summary["measurement_status"] != "EDGE_ONLY":
        summary.update(measurement_status="PARTIAL", complete_inner_mm=None, complete_outer_mm=None)
    x, y = result["x"], result["y"]
    points = np.column_stack((x, y))
    anchors = points - result["deviation"][:, None] * result["normal"]
    line = (
        [[float(v), result["slope"] * float(v) + result["intercept"]] for v in (x[0], x[-1])]
        if result["slope"] is not None
        else None
    )
    if guide is not None:
        (gx0, gy0), (gx1, gy1) = guide["line_px"]
        guide["line_px"] = [
            [float(v), float(gy0 + (v - gx0) * (gy1 - gy0) / (gx1 - gx0))] for v in (x[0], x[-1])
        ]
    normal = result["normal"]
    if vertical:
        zero_points = zero_points[:, ::-1]
        points, anchors, normal = points[:, ::-1], anchors[:, ::-1], normal[::-1]
        if line is not None:
            line = [point[::-1] for point in line]
        if guide is not None:
            guide["line_px"] = [point[::-1] for point in guide["line_px"]]
        summary["measured_y_range"] = summary.pop("measured_x_range")
        summary["rejected_y_px"] = summary.pop("rejected_x_px")
        summary["fit_sample_y_ranges"] = summary.pop("fit_sample_x_ranges")
    summary.update(
        edge=edge,
        algorithm_version=ALGORITHM_VERSION,
        reference_line_px=line,
        preview_reference=guide,
        orientation=reference["orientation"],
        pcb_side_name=reference["pcb_side_name"],
        cut_path=path,
        complete_cut_scope=complete_scope,
        max_scope="accepted_sobel_samples_within_cut_span",
        scale_mm_per_px_xy=reference["scales"],
        baseline_equation="x = slope * y + intercept" if vertical else "y = slope * x + intercept",
        baseline_source="normal_pcb_sobel_outside_cut" if line is not None else None,
        normal_pcb_points_px=zero_points.tolist(),
        inner_line_max_mm=summary["inner_mm"],
        outer_line_max_mm=summary["outer_mm"],
        max_indices={name: summary[name + "_index"] for name in ("inner", "outer")},
    )
    profile = EdgeProfile(
        valid=bool(line is not None and result["accepted"].all() and complete_scope),
        reason=result["baseline_reason"]
        or (
            "Cut travel extends outside the image scan."
            if not complete_scope
            else ""
            if result["accepted"].all()
            else "Some columns could not be measured."
        ),
        coverage=summary["coverage"],
        inward_px=summary["inner_px"],
        protrusion_px=summary["outer_px"],
        normal=normal,
        anchors=anchors,
        points=points,
        deviations=result["deviation"],
        accepted=result["accepted"],
    )
    return summary, profile


def confirmation_key(review):
    source = review["A"]["provenance"]
    return _digest([review["key"], source["image_sha256"], str(source["csv_sn"])])


def _digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def edge_snapshot(review, measurement, profile, preview_sobel):
    """Persist exactly what was reviewed, not a contour to reuse on the next board."""
    source = review["A"]["provenance"]
    return {
        "schema": "trainer-image-edge-v1",
        "context": review["context"],
        "image_sha256": source["image_sha256"],
        "csv_sn": str(source["csv_sn"]),
        "manifest_sha256": review["manifest_sha256"],
        "edge": measurement["edge"],
        "algorithm_version": measurement["algorithm_version"],
        "preview_sobel": preview_sobel,
        "measurement": measurement,
        "accepted_edge_points_px": profile.points[profile.accepted].tolist(),
        "review_scope": (
            "highlighted accepted edge samples only; no baseline or mm measurement confirmed"
            if measurement["measurement_status"] == "EDGE_ONLY"
            else "selected Sobel edge and normal PCB zero within recorded cut travel"
        ),
    }


def confirmation_matches(record, snapshot):
    return (
        isinstance(record, dict)
        and record.get("human_confirmed") is True
        and record.get("snapshot") == snapshot
        and record.get("snapshot_sha256") == _digest(snapshot)
    )


def make_confirmation(snapshot):
    return {
        "human_confirmed": True,
        "snapshot": snapshot,
        "snapshot_sha256": _digest(snapshot),
        "confirmed_at": datetime.now().isoformat(timespec="seconds"),
        "calibration_verified": False,
        "used_for_classifier_training": False,
    }


def saved_confirmation_edge(review, confirmations, preview_sobel):
    """Read a current per-image save badge without rerunning edge detection."""
    record = confirmations.get(confirmation_key(review), {})
    snapshot = record.get("snapshot") if isinstance(record, dict) else None
    if not isinstance(snapshot, dict) or not confirmation_matches(record, snapshot):
        return None
    source = review["A"]["provenance"]
    if (
        snapshot.get("edge") not in ("A", "B")
        or snapshot.get("algorithm_version") != ALGORITHM_VERSION
        or snapshot.get("context") != review["context"]
        or snapshot.get("manifest_sha256") != review["manifest_sha256"]
        or snapshot.get("image_sha256") != source["image_sha256"]
        or snapshot.get("csv_sn") != str(source["csv_sn"])
        or snapshot.get("preview_sobel") != preview_sobel
    ):
        return None
    return snapshot["edge"]


def validate_confirmation(record, image, review, edge):
    if not isinstance(record, dict) or not isinstance(record.get("snapshot"), dict):
        raise ValueError("A visual edge confirmation is required.")
    measurement, profile = measure_image_edge(image, review, edge)
    expected = edge_snapshot(review, measurement, profile, record["snapshot"].get("preview_sobel"))
    if not confirmation_matches(record, expected):
        raise ValueError("The confirmed edge changed. Reopen the image and review again.")


def measure_selected_edge(image, review, edge):
    """Observed maxima on original pixels; never a production decision or calibration."""
    if edge not in ("A", "B"):
        raise ValueError("Select A or B before measuring.")
    reference = review[edge]
    profile = measure_reference(
        image, reference["line"], side=reference["side"], radius=45, min_contrast=3.0
    )
    if not len(profile.anchors):
        raise ValueError(profile.reason)
    factor = normal_scale(profile.normal, *reference["scales"])
    accepted = bool(profile.accepted.any())
    inner = profile.inward_px * factor if accepted else None
    outer = profile.protrusion_px * factor if accepted else None
    maxima = {}
    for name, sign in (("inner", 1), ("outer", -1)):
        indices = np.flatnonzero(profile.accepted & (profile.deviations * sign > 0))
        maxima[name] = (
            int(indices[np.argmax(profile.deviations[indices] * sign)]) if len(indices) else None
        )
    return {
        "edge": edge,
        "reference_line_px": reference["line"],
        "pcb_side": reference["side"],
        "scale_mm_per_px_xy": reference["scales"],
        "normal_mm_per_px": factor,
        "search_radius_px": 45,
        "min_contrast": 3.0,
        "coverage": profile.coverage,
        "measurement_status": (
            "COMPLETE_ESTIMATE" if profile.valid else "PARTIAL" if accepted else "NO_EDGE"
        ),
        "inner_line_max_mm": inner,
        "outer_line_max_mm": outer,
        "complete_inner_mm": inner if profile.valid else None,
        "complete_outer_mm": outer if profile.valid else None,
        "line_max_scope": "accepted_samples_only",
        "max_indices": maxima,
        "calibration_verified": False,
        "production_decision": None,
        "provenance": reference["provenance"],
    }, profile


def context_key(context):
    values = {name: context[name] for name in CONTEXT_FIELDS}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()


def load_edge_review(
    manifest_path, image_path, *, product_id, recipe_path, cut_point, csv_sn, manifest=None
):
    """Bind the displayed tangents to this image, recipe version and program."""
    manifest = load_manifest(manifest_path) if manifest is None else manifest
    a = reference_for_image(manifest, image_path, "A")
    b = reference_for_image(manifest, image_path, "B")
    record = next(r for r in manifest["images"] if r["name"] == Path(image_path).name)
    context = record.get("reference_context", {})
    if not isinstance(context, dict) or any(
        context.get(name) in (None, "") for name in CONTEXT_FIELDS
    ):
        raise ValueError("Reference data needs product, recipe and cutting-program identity.")
    if (
        context["product_id"] != product_id
        or context["recipe_sha256"] != sha256(recipe_path)
        or context["recipe_sha256"] != record["projection"]["source_recipe_sha256"]
        or context["cut_point"] != cut_point
        or record["cut_point"] != cut_point
        or str(record["csv_sn"]) != str(csv_sn)
        or Path(context.get("recipe_name", "")).name.casefold() != Path(recipe_path).name.casefold()
    ):
        raise ValueError("Reference data does not match this image's product / recipe / cut point.")
    return {
        "context": context,
        "key": context_key(context),
        "A": a,
        "B": b,
        "manifest_path": manifest["manifest_path"],
        "manifest_sha256": manifest["manifest_sha256"],
    }


def saved_edge(choices, review):
    record = choices.get(review["key"], {})
    if (
        not isinstance(record, dict)
        or record.get("context") != review["context"]
        or record.get("edge") not in CHOICES
    ):
        return None
    return record["edge"]


def selection_record(review, image_path, edge):
    if edge not in CHOICES:
        raise ValueError("Select A, B or NOT SURE.")
    return {
        "context": review["context"],
        "edge": edge,
        "reviewed_image": str(Path(image_path).resolve()),
        "reviewed_image_sha256": review["A"]["provenance"]["image_sha256"],
        "reviewed_at": datetime.now().isoformat(timespec="seconds"),
        "reference_manifest": review["manifest_path"],
        "reference_manifest_sha256": review["manifest_sha256"],
        "calibration_verified": False,
    }
