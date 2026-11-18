"""Native-resolution per-cut evidence, written in the inspection worker."""

from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .guard import check_write_target, protected_roots
from .images import read_image
from .machine_reference import sha256
from .production_evidence import save_image


def archive_panel(result, classifier, directory, cfg, claim, *, cancelled=None):
    root = Path(directory)
    check_write_target(root, protected_roots(cfg))
    root.mkdir(parents=True, exist_ok=True)
    reserve = cfg.production_min_free_mb * 1024 * 1024
    source_bytes = sum(Path(p).stat().st_size for p in result.run.pictures if Path(p).is_file())
    if shutil.disk_usage(root).free < reserve + source_bytes * 3:
        raise OSError("Insufficient disk space to archive this board; production held.")
    now = datetime.now(timezone.utc)
    program = cfg.active_program_key or result.run.key
    group = hashlib.sha256(program.encode()).hexdigest()[:16]
    sn = re.sub(r"[^a-zA-Z0-9_-]", "_", str(result.run.sn))[:64] or "unknown"
    folder = (
        root
        / now.strftime("%Y-%m-%d")
        / f"program_{group}"
        / f"SN_{sn}_{now:%H%M%S%f}_{uuid.uuid4().hex}"
    )
    predictions = {item.path: item for item in result.details}
    cuts, prepared = [], {}
    for index, source in enumerate(result.run.pictures, 1):
        if cancelled and cancelled():
            raise InterruptedError("Image archiving cancelled.")
        record = {"cut_point": index, "source": source, "files": []}
        cuts.append(record)
        try:
            source_hash = sha256(source)
        except OSError:
            record["error"] = "Source image is missing or unreadable."
            continue
        original = read_image(source)
        if original is None:
            record["error"] = "Source image is unreadable."
            continue
        edge = (
            classifier.extractor.edge_map_array(original, classifier.crop)
            if hasattr(classifier.extractor, "edge_map_array")
            else classifier.extractor.edge_map(source, classifier.crop)
        )
        record["source_sha256"] = source_hash
        record["original_size_px"] = list(original.shape[:2][::-1])
        prediction = predictions.get(source)
        record["prediction"] = asdict(prediction) if prediction is not None else None
        for kind, pixels in (("original", original), ("sobel", edge)):
            if pixels is None:
                record["error"] = "Sobel evidence is unavailable."
                continue
            path = folder / f"cut_{index:04d}_{kind}.jpeg"
            save_image(path, pixels, cfg)
            record["files"].append(
                {
                    "kind": kind,
                    "path": str(path),
                    "size_px": list(pixels.shape[:2][::-1]),
                    "sha256": sha256(path),
                }
            )
        if sha256(source) != source_hash:
            raise OSError(f"Image changed during archiving: {Path(source).name}")
        # Copy only the crop; a NumPy view would retain each full-size camera frame.
        prepared[source] = classifier.crop.apply(original).copy(), edge
    return {
        "directory": str(folder),
        "claim": claim,
        "program": program,
        "time_utc": now.isoformat(),
        "jpeg_quality": cfg.production_jpeg_quality,
        "original_resolution_preserved": True,
        "sobel_is_model_crop": True,
        "complete": len(cuts) == len(result.run.pictures)
        and all(len(c["files"]) == 2 for c in cuts),
        "cuts": cuts,
    }, prepared
