"""Incremental board intake: explicit capture identities, stable files and durable claims."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from .auo6000 import IMAGE_EXTENSIONS, scan_auo6000_dataset
from .machine import read_run
from .machine_reference import sha256
from .production_measurement import CaptureReferences
from .production_store import panel_key


@dataclass
class LivePanel:
    run: object
    key: str
    signature: str
    hashes: dict
    readiness_error: str = ""


class LiveSource:
    def __init__(
        self,
        cfg,
        store,
        *,
        stable_seconds=2,
        settle_seconds=30,
        clock=time.monotonic,
        capture_location=None,
    ):
        for name, value in (("stability", stable_seconds), ("settling", settle_seconds)):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Live {name} interval must be finite and non-negative.")
        self.cfg, self.store = deepcopy(cfg), store
        self.stable_seconds, self.settle_seconds, self.clock = stable_seconds, settle_seconds, clock
        self.seen = {}
        self.first_panel_seen = {}
        self.hash_cache = {}
        self.run_cache = {}
        self.invalid_result_seen = {}
        self._assignment_stamp = None
        self._assigned = {}
        self.references = (
            CaptureReferences(capture_location) if capture_location is not None else None
        )

    def _stable(self, path, now):
        stat = path.stat()
        stamp = stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        previous = self.seen.get(str(path))
        if previous is None or previous[0] != stamp:
            self.seen[str(path)] = stamp, now
            return False
        return stat.st_size > 0 and now - previous[1] >= self.stable_seconds

    def _hash(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        stamp = stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        previous = self.hash_cache.get(str(path))
        if previous is None or previous[0] != stamp:
            digest = sha256(path)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != stamp:
                raise OSError(f"Source changed while hashing: {path.name}")
            self.hash_cache[str(path)] = stamp, digest
        return self.hash_cache[str(path)][1]

    def _run(self, path):
        stamp = self.seen[str(path)][0]
        cached = self.run_cache.get(str(path))
        if cached is None or cached[0] != stamp:
            run = read_run(path, strict=True)
            if self._hash(path) != run.source_sha256:
                raise ValueError("Result changed while its board metadata was being read.")
            self.run_cache[str(path)] = stamp, run
        return deepcopy(self.run_cache[str(path)][1])

    def _historical_assignments(self, picture, result):
        """Timestamp grouping is retained only for old callers without capture metadata."""
        snapshot = tuple(sorted((path, stamp[0]) for path, stamp in self.seen.items()))
        if snapshot != self._assignment_stamp:
            dataset = scan_auo6000_dataset(picture)
            if Path(dataset.result_dir).resolve() != result:
                raise ValueError("Live Result does not match the AUO6000 Picture export.")
            assigned = defaultdict(list)
            for image in dataset.images:
                if image.result_file:
                    assigned[Path(image.result_file).name].append(image)
            self._assigned, self._assignment_stamp = assigned, snapshot
        return self._assigned

    def poll(self, route, expected):
        if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
            raise ValueError("Expected cut count must be configured before live inspection.")
        states = self.store.states(route)
        if any(state == "PENDING" for state, _ in states.values()):
            raise ValueError("Board has an unfinished inspection; explicit recovery is required.")
        if any(state == "SOURCE_CHANGED" for state, _ in states.values()):
            raise ValueError("Board source files changed; operator review and retry are required.")
        now = self.clock()
        picture, result = Path(self.cfg.picture_dir).resolve(), Path(self.cfg.result_dir).resolve()
        if not picture.is_dir() or not result.is_dir():
            raise OSError("Live Picture / Result folders are unavailable.")
        paths = sorted(
            p.resolve()
            for p in picture.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        result_paths = sorted(
            p.resolve()
            for p in result.iterdir()
            if p.name.startswith("_") and p.suffix.lower() == ".csv"
        )
        active_paths = {str(p) for p in [*paths, *result_paths]}
        stable_images = {str(p) for p in paths if self._stable(p, now)}
        stable_results = [p for p in result_paths if self._stable(p, now)]
        stable_captures = set()
        if self.references is not None:
            self.references.refresh()
            for p in self.references.paths():
                if p.is_file():
                    active_paths.add(str(p))
                    if self._stable(p, now):
                        stable_captures.add(str(p.resolve()))
        for cache in (self.seen, self.hash_cache, self.run_cache, self.invalid_result_seen):
            for key in cache.keys() - active_paths:
                del cache[key]
        by_name = defaultdict(list)
        for path in paths:
            by_name[path.name].append(str(path))
        assigned = (
            self._historical_assignments(picture, result) if self.references is None else None
        )
        ready, active_panels = [], set()
        for result_path in stable_results:
            try:
                run = self._run(result_path)
            except (ValueError, UnicodeError) as exc:
                since = self.invalid_result_seen.setdefault(str(result_path), now)
                if now - since < self.settle_seconds:
                    continue
                raise ValueError(f"Invalid stable Result {result_path.name}: {exc}") from exc
            self.invalid_result_seen.pop(str(result_path), None)
            if run.key != route:
                continue
            key = panel_key(result, run)
            active_panels.add(key)
            self.first_panel_seen.setdefault(key, now)
            overdue = now - self.first_panel_seen[key] >= self.settle_seconds
            error, records = "", []
            if self.references is not None:
                captures = self.references.records_for_run(run)
                cuts = [record.get("cut_point") for _, record in captures]
                if (
                    len(cuts) != expected
                    or any(type(cut) is not int for cut in cuts)
                    or set(cuts) != set(range(1, expected + 1))
                ):
                    error = f"Expected {expected} unique captures, received {len(cuts)}."
                else:
                    for _, record in sorted(captures, key=lambda item: item[1]["cut_point"]):
                        matches = by_name.get(record.get("name"), [])
                        if len(matches) != 1:
                            error = "Capture image is missing or its filename is ambiguous."
                            break
                        records.append(matches[0])
            else:
                items = sorted(assigned[run.result_file], key=lambda item: item.cut_point)
                records = [item.path for item in items]
                if len(items) != expected or {i.cut_point for i in items} != set(
                    range(1, expected + 1)
                ):
                    error = f"Expected {expected} unique cuts, received {len(items)}."
                if any(item.panel_sn != run.sn for item in items):
                    error = "Panel SN mismatch in image assignment."
            if any(p not in stable_images for p in records):
                error = "Cut images have not become stable within the allowed interval."
            if error and not overdue:
                continue
            run.pictures = [p for p in records if p in stable_images]
            hashes = {str(result_path): self._hash(result_path)}
            hashes.update({p: self._hash(p) for p in run.pictures})
            signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            prior = states.get(key)
            if prior and prior[0] != "RETRY_APPROVED":
                if prior[1] != signature:
                    self.store.quarantine(
                        key, f"Panel {run.sn} source files changed after inspection."
                    )
                    raise ValueError(
                        f"Panel {run.sn} source files changed; operator review required."
                    )
                continue
            if self.references is not None and not error:
                try:
                    recipes = [
                        p
                        for p in Path(self.cfg.recipe_dir).rglob("*.rcp")
                        if p.name.casefold() == Path(run.recipe).name.casefold()
                    ]
                    if len(recipes) != 1:
                        raise ValueError("Recipe file is missing or ambiguous.")
                    for index, path in enumerate(run.pictures, 1):
                        review = self.references.review(path, run, recipes[0], index)
                        if review["manifest_path"] not in stable_captures:
                            raise ValueError("Capture record is still being written.")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    if not overdue:
                        continue
                    error = f"Capture data unavailable: {exc}"
            ready.append(LivePanel(run, key, signature, hashes, error))
        self.first_panel_seen = {
            key: value for key, value in self.first_panel_seen.items() if key in active_panels
        }
        ready.sort(key=lambda panel: panel.run.end)
        return ready
