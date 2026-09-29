"""Trainer-selected material side, separate from classifier labels and pixels."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .machine_reference import load_manifest, reference_for_image, sha256

CHOICES = ("UNSURE", "A", "B")
CONTEXT_FIELDS = ("product_id", "recipe_sha256", "table", "program_key", "layer", "cut_point")


def context_key(context):
    values = {name: context[name] for name in CONTEXT_FIELDS}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()


def load_edge_review(manifest_path, image_path, *, product_id, recipe_path, cut_point, csv_sn):
    """Bind the displayed tangents to this image, recipe version and program."""
    manifest = load_manifest(manifest_path)
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
