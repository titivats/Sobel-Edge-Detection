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
from .production import repeated_cut_image
from .production_measurement import CaptureReferences
from .production_store import panel_key
from .program_selection import recipe_name


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
        self.result_wait_since = {}
        self.archive_cutoff_ns = time.time_ns() - int(max(stable_seconds, settle_seconds) * 1e9)
        self.archive_stamps = None
        self.archive_route = None
        self.legacy_hashes = {}
        self.retry_wait_since = {}
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
            try:
                run = read_run(path, strict=True)
            except (ValueError, UnicodeError) as exc:
                self.run_cache[str(path)] = stamp, str(exc)
                raise
            if self._hash(path) != run.source_sha256:
                raise ValueError("Result changed while its board metadata was being read.")
            self.run_cache[str(path)] = stamp, run
        run = self.run_cache[str(path)][1]
        if isinstance(run, str):
            raise ValueError(run)
        return deepcopy(run)

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

    def _capture_signature(self, hashes, captures, recipe_files, active_paths):
        dependencies = {str(p) for p in recipe_files}
        manifests = {path for path, _ in captures}
        for path in manifests:
            dependencies.update(self.references.source_paths(path))
        for _, record in captures:
            dependencies.update(record["source_paths"])
        active_paths.update(dependencies)
        actual = {p: self._hash(p) if Path(p).is_file() else None for p in dependencies}
        records = sorted((str(p), r["capture_sha256"]) for p, r in captures)
        signature = (
            "v2:"
            + hashlib.sha256(
                json.dumps([hashes, records, actual], sort_keys=True).encode()
            ).hexdigest()
        )
        # The durable signature covers this board. In-flight verification also pins
        # the exact manifest bytes that the measurement worker must read.
        hashes.update({p: digest for p, digest in actual.items() if digest})
        hashes.update({str(path): self._hash(path) for path in manifests})
        return signature

    def _completed_inputs_changed(self, key, previous, signature, old_signature, active_paths):
        if previous.startswith("v2:"):
            return previous != signature
        if key not in self.legacy_hashes:
            self.legacy_hashes[key] = self.store.legacy_source_hashes(key)
        historical = self.legacy_hashes[key]
        active_paths.update(historical)
        return previous != old_signature or any(
            not Path(p).is_file() or self._hash(p) != digest for p, digest in historical.items()
        )

    def _capture_hashes(self, run, recipe_files, stable_captures, hashes):
        if len(recipe_files) != 1:
            raise ValueError("Recipe file is missing or ambiguous.")
        for index, path in enumerate(run.pictures, 1):
            review = self.references.review(path, run, recipe_files[0], index)
            manifest = Path(review["manifest_path"])
            if str(manifest) not in stable_captures:
                raise ValueError("Capture record is still being written.")
            if hashes[str(manifest)] != self.references.manifest_sha256(manifest):
                raise ValueError("Capture manifest changed while preparing the board.")
            if any(
                hashes.get(p, digest) != digest for p, digest in review["source_hashes"].items()
            ):
                raise ValueError("Capture source changed while preparing the board.")
            hashes.update(review["source_hashes"])

    def poll(self, route, expected):
        if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
            raise ValueError("Expected cut count must be configured before live inspection.")
        tracked = self.store.tracked_panels(route)
        states = {key: (state, signature) for key, state, signature, _ in tracked}
        tracked_paths = {Path(path): key for key, _, _, path in tracked if path}
        has_unlocated_claims = any(not path for _, _, _, path in tracked)
        retry_keys = {key for key, (state, _) in states.items() if state == "RETRY_APPROVED"}
        retry_paths = {p for p, key in tracked_paths.items() if key in retry_keys}
        all_retries_located = bool(retry_keys) and len(retry_paths) == len(retry_keys)
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
        stable_results = {p for p in result_paths if self._stable(p, now)}
        stable_captures = set()
        if self.references is not None:
            self.references.refresh()
            for p in self.references.paths():
                if p.is_file():
                    active_paths.add(str(p))
                    if self._stable(p, now):
                        stable_captures.add(str(p.resolve()))
        if self.archive_route != route:
            self.archive_stamps = {}
            if self.references is not None:
                initial_files = {
                    p.name: self.seen[str(p)][0]
                    for p in result_paths
                    if self.seen[str(p)][0][1] < self.archive_cutoff_ns
                }
                self.archive_stamps = self.store.intake_baseline(result, route, initial_files)
            self.archive_route = route
        self.archive_stamps = {
            p: stamp
            for p, stamp in self.archive_stamps.items()
            if str(p) in active_paths and self.seen[str(p)][0] == stamp
        }
        for cache in (self.seen, self.run_cache, self.invalid_result_seen, self.result_wait_since):
            for key in cache.keys() - active_paths:
                del cache[key]
        by_name = defaultdict(list)
        for path in paths:
            by_name[path.name].append(str(path))
        assigned = (
            self._historical_assignments(picture, result) if self.references is None else None
        )
        recipes = defaultdict(list)
        if self.references is not None:
            for path in Path(self.cfg.recipe_dir).rglob("*.rcp"):
                recipes[path.name.casefold()].append(path.resolve())
        ready, active_panels = [], set()
        for result_path in result_paths:
            if all_retries_located and result_path not in retry_paths:
                continue
            known_key = tracked_paths.get(result_path)
            archival = (
                result_path in self.archive_stamps
                and known_key is None
                and result_path.name not in self.references.result_files
            )
            if archival and not self.references.has_unnamed_results and not has_unlocated_claims:
                continue
            if result_path not in stable_results:
                if not archival:
                    since = self.result_wait_since.setdefault(str(result_path), now)
                    if now > since and now - since >= max(self.stable_seconds, self.settle_seconds):
                        reason = f"Result is empty or did not become stable: {result_path.name}"
                        if known_key:
                            self.store.quarantine(known_key, reason)
                        raise ValueError(reason)
                continue
            self.result_wait_since.pop(str(result_path), None)
            try:
                run = self._run(result_path)
            except (ValueError, UnicodeError) as exc:
                if archival:
                    continue
                since = self.invalid_result_seen.setdefault(str(result_path), now)
                if now - since < self.settle_seconds:
                    continue
                reason = f"Invalid stable Result {result_path.name}: {exc}"
                if known_key:
                    self.store.quarantine(known_key, reason)
                raise ValueError(reason) from exc
            self.invalid_result_seen.pop(str(result_path), None)
            key = panel_key(result, run)
            if known_key and known_key != key:
                self.store.quarantine(known_key, f"Board identity changed in {result_path.name}.")
                raise ValueError(
                    f"Board identity changed in {result_path.name}; operator review required."
                )
            if retry_keys and key not in retry_keys:
                continue
            if run.key != route:
                continue
            if self.cfg.active_program_key:
                if route != self.cfg.active_recipe or recipe_name(run.recipe) != recipe_name(
                    self.cfg.active_program_recipe
                ):
                    continue
            captures = self.references.records_for_run(run) if self.references is not None else []
            if archival and not captures and key not in states:
                continue
            active_panels.add(key)
            self.first_panel_seen.setdefault(key, now)
            overdue = now - self.first_panel_seen[key] >= self.settle_seconds
            error, records = "", []
            if self.references is not None:
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
                    if not error:
                        error = repeated_cut_image(records)
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
            old_signature = signature
            recipe_files = recipes[Path(run.recipe).name.casefold()]
            if self.references is not None:
                signature = self._capture_signature(hashes, captures, recipe_files, active_paths)
            prior = states.get(key)
            if prior and prior[0] != "RETRY_APPROVED":
                if self._completed_inputs_changed(
                    key, prior[1], signature, old_signature, active_paths
                ):
                    self.store.quarantine(
                        key, f"Panel {run.sn} source files changed after inspection."
                    )
                    raise ValueError(
                        f"Panel {run.sn} source files changed; operator review required."
                    )
                continue
            if self.references is not None and not error:
                try:
                    self._capture_hashes(run, recipe_files, stable_captures, hashes)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    if not overdue:
                        continue
                    error = f"Capture data unavailable: {exc}"
            ready.append(LivePanel(run, key, signature, hashes, error))
        self.first_panel_seen = {
            key: value for key, value in self.first_panel_seen.items() if key in active_panels
        }
        for path in self.hash_cache.keys() - active_paths:
            del self.hash_cache[path]
        self.retry_wait_since = {
            key: self.retry_wait_since.get(key, now) for key in retry_keys - active_panels
        }
        if retry_keys and not active_panels:
            missing = all_retries_located and not retry_paths.intersection(result_paths)
            eligible = not all_retries_located or bool(retry_paths & stable_results)
            expired = eligible and all(
                now > start and now - start >= self.settle_seconds
                for start in self.retry_wait_since.values()
            )
            if missing or expired:
                raise ValueError(
                    "Approved retry is waiting for the original board's Result. Restore its files "
                    "and select its saved program before ACK / RESET and START AUTO."
                )
        ready.sort(key=lambda panel: panel.run.end)
        return ready
