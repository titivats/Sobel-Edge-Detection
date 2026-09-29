"""Stable panel/model identifiers and legacy route image-count inference."""

from __future__ import annotations

from collections import Counter

from .machine import Run

MODEL_PREFIX = "cut_classifier_"


def panel_id(run: Run) -> str:
    return run.sn.strip() or run.run_id.strip() or run.result_file


def model_file_name(key: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return f"{MODEL_PREFIX}{safe}.pt"


def product_from_route_key(key: str) -> str:
    """Return ProductId from either ``ProductId`` or ``ProductId|Table``."""
    return str(key).split("|", 1)[0].strip()


def inferred_expected_images(runs: list[Run]) -> int:
    """Infer the normal image count while incomplete runs still fail closed."""
    counts = Counter(len(run.pictures) for run in runs if run.pictures)
    if not counts:
        return 0
    most_common = counts.most_common()
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return 0
    return int(most_common[0][0])
