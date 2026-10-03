"""Predict first, then measure fresh Sobel edges against the active route's SPEC."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .image_edge import ALGORITHM_VERSION
from .machine_reference import load_manifest, sha256
from .production import STATUS_FAULT, STATUS_GOOD, STATUS_NG, classify_panel
from .recipe_spec_ui import validated_limits
from .training_edge import (
    confirmation_matches,
    load_edge_review,
    measure_image_edge,
    saved_edge,
)


def file_fingerprint(paths):
    files = {str(Path(p).resolve()): sha256(p) for p in paths}
    signature = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return signature, files


class CaptureReferences:
    """Load fresh, immutable capture records; never copy a prior board's geometry."""

    def __init__(self, location):
        self.location = Path(location) if location else None

    def paths(self):
        if self.location is None:
            return []
        return sorted(self.location.glob("*.json")) if self.location.is_dir() else [self.location]

    def review(self, path, run, recipe, index):
        candidates = []
        # Inspect identity first; unrelated or superseded manifests cannot mask a valid capture.
        for manifest_path in self.paths():
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                matches = [
                    record
                    for record in raw.get("images", [])
                    if record.get("name") == Path(path).name
                    and str(record.get("csv_sn")) == run.sn
                    and record.get("cut_point") == index
                    and record.get("reference_context", {}).get("product_id") == run.product_id
                    and (
                        not run.table
                        or record.get("reference_context", {}).get("table", "").strip() == run.table
                    )
                    and record.get("result_file", run.result_file) == run.result_file
                ]
                if matches:
                    candidates.append(manifest_path)
            except (OSError, ValueError, AttributeError, TypeError):
                continue
        if len(candidates) != 1:
            raise ValueError(
                f"Cut {index}: needs exactly one matching capture record (found {len(candidates)})."
            )
        review = load_edge_review(
            candidates[0],
            path,
            product_id=run.product_id,
            recipe_path=recipe,
            cut_point=index,
            csv_sn=run.sn,
        )
        # load_manifest validates every source. Pin those sources for the final recheck too.
        manifest = load_manifest(candidates[0])
        if manifest["manifest_sha256"] != review["manifest_sha256"]:
            raise ValueError("Capture record changed while loading its geometry.")
        review["source_paths"] = [s["path"] for s in manifest["source_files"]] + [
            str(candidates[0])
        ]
        review["source_hashes"] = {
            str(Path(s["path"]).resolve()): s["sha256"] for s in manifest["source_files"]
        }
        review["source_hashes"][str(candidates[0].resolve())] = manifest["manifest_sha256"]
        return review


def approved_side(state, review):
    side = saved_edge(state.get("choices", {}), review)
    if side not in ("A", "B"):
        raise ValueError("Trainer has not saved A/B for this recipe version / program / cut.")
    choice = state["choices"][review["key"]]
    for record in state.get("confirmations", {}).values():
        snapshot = record.get("snapshot") if isinstance(record, dict) else None
        if (
            isinstance(snapshot, dict)
            and snapshot.get("context") == review["context"]
            and snapshot.get("edge") == side
            and snapshot.get("image_sha256") == choice.get("reviewed_image_sha256")
            and snapshot.get("algorithm_version") == ALGORITHM_VERSION
            and confirmation_matches(record, snapshot)
        ):
            return side
    raise ValueError("A/B selection needs a matching trainer SAVE EDGE confirmation.")


def inspect_with_spec(
    classifier, run, expected, threshold, limits, references, recipe, state, cancelled=None
):
    paths = [*run.pictures, recipe]
    before, hashes = file_fingerprint(paths)
    evidence = {
        "source_hashes": hashes,
        "recipe_sha256": sha256(recipe),
        "threshold": threshold,
        "route": run.key,
        "model_stage": None,
        "measurement_stage": "NOT_RUN",
        "measurements": [],
        "algorithm_version": ALGORITHM_VERSION,
        "calibration_verified": False,
    }
    result = classify_panel(classifier, run, expected, threshold, cancelled=cancelled)
    evidence["model_stage"] = result.status
    if result.status == STATUS_GOOD:
        faults, violations = [], []
        try:
            spec = validated_limits(limits.get("inner_max_mm"), limits.get("outer_max_mm"))
            evidence["spec"] = spec
        except (ValueError, AttributeError) as exc:
            result.status, result.note = STATUS_FAULT, f"SPEC unavailable for {run.key}: {exc}"
        else:
            for index, path in enumerate(run.pictures, 1):
                if cancelled and cancelled():
                    raise InterruptedError("Inspection cancelled")
                entry = {"cut_point": index, "path": path}
                try:
                    review = references.review(path, run, recipe, index)
                    side = approved_side(state, review)
                    source_hashes = review["source_hashes"]
                    if any(sha256(p) != h for p, h in source_hashes.items()):
                        raise ValueError("Capture sources changed before measurement.")
                    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("Image cannot be read.")
                    summary, profile = measure_image_edge(image, review, side)
                    entry.update(
                        edge=side,
                        context=review["context"],
                        measurement=summary,
                        source_hashes=source_hashes,
                        capture_provenance=review[side]["provenance"],
                    )
                    inner, outer = (
                        summary.get("inner_line_max_mm"),
                        summary.get("outer_line_max_mm"),
                    )
                    if any(v is None or not math.isfinite(v) or v < 0 for v in (inner, outer)):
                        raise ValueError("Inner/Outer values are unavailable.")
                    # A known violation is NG even if another part of the contour is unreadable.
                    if inner > spec["inner_max_mm"] or outer > spec["outer_max_mm"]:
                        violations.append(index)
                        entry["status"] = STATUS_NG
                    elif (
                        not profile.valid
                        or summary.get("measurement_status") != "COMPLETE_ESTIMATE"
                    ):
                        raise ValueError(
                            "Incomplete measured cut span; observed values cannot establish a PASS."
                        )
                    else:
                        entry["status"] = STATUS_GOOD
                    if any(sha256(p) != h for p, h in source_hashes.items()):
                        raise ValueError("Capture sources changed during measurement.")
                    evidence["source_hashes"].update(source_hashes)
                except (OSError, ValueError, KeyError, TypeError, cv2.error) as exc:
                    faults.append(index)
                    entry.update(status=STATUS_FAULT, reason=str(exc))
                evidence["measurements"].append(entry)
            evidence["measurement_stage"] = (
                STATUS_FAULT if faults else STATUS_NG if violations else STATUS_GOOD
            )
            result.status = evidence["measurement_stage"]
            result.note = (
                "; ".join(
                    (
                        ["Measurement unavailable at cut " + ",".join(map(str, faults))]
                        if faults
                        else []
                    )
                    + (
                        ["SPEC exceeded at cut " + ",".join(map(str, violations))]
                        if violations
                        else []
                    )
                )
                if faults or violations
                else f"All {expected} cuts passed model and Inner/Outer SPEC."
            )
    if file_fingerprint(paths)[0] != before or any(
        sha256(p) != h for p, h in evidence["source_hashes"].items()
    ):
        result.status, result.note = STATUS_FAULT, "Input files changed during inspection."
    evidence["prediction"] = [asdict(d) for d in result.details]
    return result, evidence
