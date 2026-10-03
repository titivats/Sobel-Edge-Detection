"""Poll stable AUO6000 panel files without resetting the inspection queue."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .auo6000 import IMAGE_EXTENSIONS, scan_auo6000_dataset
from .machine import available_days, load_runs
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
        self.cfg, self.store = cfg, store
        self.stable_seconds, self.settle_seconds, self.clock = stable_seconds, settle_seconds, clock
        self.seen = {}
        self.first_panel_seen = {}
        self.hash_cache = {}
        self.references = (
            CaptureReferences(capture_location) if capture_location is not None else None
        )

    def _stable(self, path, now):
        path = Path(path)
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        previous = self.seen.get(str(path))
        if previous is None or previous[0] != stamp:
            self.seen[str(path)] = stamp, now
            return False
        return stat.st_size > 0 and now - previous[1] >= self.stable_seconds

    def poll(self, route, expected):
        if expected <= 0:
            raise ValueError("Expected cut count must be configured before live inspection.")
        now = self.clock()
        picture = Path(self.cfg.picture_dir)
        result = Path(self.cfg.result_dir)
        if not picture.is_dir() or not result.is_dir():
            raise OSError("Live Picture / Result folders are unavailable.")
        # Prime every picture, including images captured before the panel's Result appears.
        stable_images = {
            str(path.resolve())
            for path in picture.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and self._stable(path, now)
        }
        stable_results = {path.name for path in result.glob("_*.csv") if self._stable(path, now)}
        stable_captures = set()
        if self.references is not None:
            stable_captures = {
                str(path.resolve())
                for path in self.references.paths()
                if path.is_file() and self._stable(path, now)
            }
        try:
            dataset = scan_auo6000_dataset(picture)
        except OSError as exc:
            if "Result data changed during scanning" in str(exc):
                return []
            raise
        if Path(dataset.result_dir).resolve() != result.resolve():
            raise ValueError("Live Result folder must match the selected AUO6000 Picture export.")
        assigned = defaultdict(list)
        for item in dataset.images:
            if item.result_file:
                assigned[Path(item.result_file).name].append(item)
        ready = []
        for day in available_days(result):
            for run in load_runs(result, day):
                if run.key != route or run.result_file not in stable_results:
                    continue
                key = panel_key(result, run)
                self.first_panel_seen.setdefault(key, now)
                items = sorted(assigned[run.result_file], key=lambda item: item.cut_point)
                run.pictures = [item.path for item in items]
                if any(p not in stable_images for p in run.pictures):
                    continue
                error = ""
                if len(items) != expected or {i.cut_point for i in items} != set(
                    range(1, expected + 1)
                ):
                    if now - self.first_panel_seen[key] < self.settle_seconds:
                        continue
                    error = f"Expected {expected} unique cuts, received {len(items)}."
                if any(item.panel_sn != run.sn for item in items):
                    error = "Panel SN mismatch in image assignment."
                hashes = {}
                for path in [result / run.result_file, *run.pictures]:
                    path = Path(path).resolve()
                    stat = path.stat()
                    stamp = (stat.st_size, stat.st_mtime_ns)
                    previous = self.hash_cache.get(str(path))
                    if previous is None or previous[0] != stamp:
                        self.hash_cache[str(path)] = (stamp, sha256(path))
                    hashes[str(path)] = self.hash_cache[str(path)][1]
                signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
                prior = self.store.state(key)
                if prior and prior[0] != "RETRY_APPROVED":
                    if prior[1] != signature:
                        if prior[0] != "SOURCE_CHANGED":
                            self.store.quarantine(
                                key, f"Panel {run.sn} source files changed after inspection."
                            )
                        raise ValueError(
                            f"Previously claimed panel {run.sn} source files changed; operator review required."
                        )
                    if prior[0] == "PENDING":
                        raise ValueError(
                            f"Panel {run.sn} has an unfinished inspection; recovery is required."
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
                        if now - self.first_panel_seen[key] < self.settle_seconds:
                            continue
                        error = f"Capture data unavailable: {exc}"
                ready.append(LivePanel(run, key, signature, hashes, error))
        ready.sort(key=lambda panel: panel.run.end)
        return ready
