"""Model-only panel decision used by the mass-production gate.

Metrology and baseline comparison intentionally do not participate here.  A
panel is released only when its image set is complete and every Sobel-DINOv2
prediction is GOOD with sufficient confidence. Anything incomplete or
ambiguous fails closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .machine import Run
from .model import CutClassifier

STATUS_GOOD = "GOOD"
STATUS_NG = "NG"
STATUS_FAULT = "FAULT"


@dataclass
class ImageDecision:
    index: int
    path: str
    label: str
    confidence: float


@dataclass
class PanelDecision:
    run: Run
    status: str = STATUS_FAULT
    note: str = ""
    checked: int = 0
    details: list[ImageDecision] = field(default_factory=list)


def classify_panel(
    classifier: CutClassifier,
    run: Run,
    expected_images: int,
    good_confidence_min: float = 0.95,
    cancelled=None,
) -> PanelDecision:
    """Classify one complete panel, applying an asymmetric fail-safe rule."""
    result = PanelDecision(run=run)

    if not run.passed:
        result.status = STATUS_NG
        result.note = run.message or "router reported FAIL"
        return result
    if expected_images <= 0:
        result.note = "invalid expected image count"
        return result
    if len(run.pictures) != expected_images:
        result.note = f"expected {expected_images} images, found {len(run.pictures)}"
        return result
    if set(classifier.classes) != {"GOOD", "NG"}:
        result.note = "model classes must be exactly GOOD and NG"
        return result

    try:
        threshold = float(good_confidence_min)
    except (TypeError, ValueError, OverflowError):
        threshold = math.nan
    if (
        isinstance(good_confidence_min, bool)
        or not math.isfinite(threshold)
        or not 0.5 <= threshold <= 1.0
    ):
        result.note = "invalid GOOD confidence threshold (expected 50% to 100%)"
        return result
    raw_predictions = classifier.predict(run.pictures, cancelled=cancelled)
    try:
        predictions = list(raw_predictions)
    except TypeError:
        result.note = "model returned an invalid prediction set"
        return result
    if len(predictions) != len(run.pictures):
        result.note = (
            "model returned an incomplete prediction set: "
            f"expected {len(run.pictures)}, found {len(predictions)}"
        )
        return result

    unreadable: list[int] = []
    unexpected: list[int] = []
    malformed: list[int] = []
    invalid_confidence: list[int] = []
    ng: list[int] = []
    uncertain: list[int] = []
    for index, (path, prediction) in enumerate(zip(run.pictures, predictions)):
        try:
            label, raw_confidence = prediction
        except (TypeError, ValueError):
            malformed.append(index)
            result.details.append(ImageDecision(index, path, "", math.nan))
            continue

        label_is_text = isinstance(label, str)
        label_text = label if label_is_text else repr(label)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError, OverflowError):
            confidence = math.nan
        confidence_is_valid = (
            not isinstance(raw_confidence, bool)
            and math.isfinite(confidence)
            and 0.0 <= confidence <= 1.0
        )
        result.details.append(ImageDecision(index, path, label_text, confidence))

        if not confidence_is_valid:
            invalid_confidence.append(index)
        if not label:
            unreadable.append(index)
        elif not label_is_text or label not in {"GOOD", "NG"}:
            unexpected.append(index)
        elif label == "NG":
            ng.append(index)
        elif confidence_is_valid and confidence < threshold:
            uncertain.append(index)

    result.checked = len(result.details)
    issues: list[str] = []
    if unreadable:
        issues.append("unreadable/blank Sobel input at cut point " + _indices(unreadable))
    if unexpected:
        issues.append("invalid/unknown model label at cut point " + _indices(unexpected))
    if malformed:
        issues.append("malformed model prediction at cut point " + _indices(malformed))
    if invalid_confidence:
        issues.append(
            "invalid confidence (expected a finite value from 0.0 to 1.0) at cut point "
            + _indices(invalid_confidence)
        )
    if ng:
        issues.append("NG predicted at cut point " + _indices(ng))
    if uncertain:
        issues.append(f"GOOD confidence below {threshold:.1%} at cut point " + _indices(uncertain))

    # Report every issue found on this Panel/SN. Any data-quality or confidence
    # problem makes the panel a FAULT; a cleanly classified NG-only panel is NG.
    if issues:
        result.status = (
            STATUS_FAULT
            if unreadable or unexpected or malformed or invalid_confidence or uncertain
            else STATUS_NG
        )
        result.note = "; ".join(issues)
    else:
        result.status = STATUS_GOOD
        lowest = min((d.confidence for d in result.details), default=0.0)
        result.note = f"all {result.checked} cut points GOOD; lowest confidence {lowest:.1%}"
    return result


def _indices(values: list[int]) -> str:
    """Format zero-based model indexes as operator-facing cut-point numbers."""
    return ",".join(str(value + 1) for value in values)
