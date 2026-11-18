"""Indexed image-bound capture records, independent of the UI and decision stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .machine_reference import load_manifest
from .training_edge import load_edge_review


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _index_records(manifest):
    """Bind each cut to its own record and the shared geometry/source contract."""
    records = manifest.get("images", [])
    if not isinstance(records, list):
        raise ValueError("Capture images must be a list.")
    records = [
        r for r in records if isinstance(r, dict) and isinstance(r.get("reference_context"), dict)
    ]
    sources = {
        str(Path(s["path"]).resolve()): s.get("sha256")
        for s in manifest.get("source_files", [])
        if isinstance(s, dict) and isinstance(s.get("path"), str)
    }
    result_paths = {
        str(Path(r["result_path"]).resolve())
        for r in records
        if isinstance(r.get("result_path"), str)
    }
    image_names = {r["name"] for r in records if isinstance(r.get("name"), str)}
    image_sources = {}
    shared_sources = {}
    for path, digest in sources.items():
        name = Path(path).name
        if name in image_names:
            image_sources.setdefault(name, {})[path] = digest
        elif path not in result_paths:
            shared_sources[path] = digest

    # Appending a board adds its Result, pictures and cut records. These must not
    # invalidate another board, while changes to shared recipe/log inputs must.
    header = {
        key: value for key, value in manifest.items() if key not in ("images", "source_files")
    }
    header_sha = _digest([header, shared_sources])
    identities = []
    for record in records:
        context = record["reference_context"]
        identity = {
            key: record[key]
            for key in ("name", "csv_sn", "cut_point", "result_file")
            if key in record
        }
        identity["reference_context"] = {
            key: context[key] for key in ("product_id", "table") if key in context
        }
        cut_sources = dict(image_sources.get(record.get("name"), {}))
        if record.get("result_path"):
            result_path = str(Path(record["result_path"]).resolve())
            cut_sources[result_path] = sources.get(result_path)
        identity["capture_sha256"] = _digest([header_sha, cut_sources, record])
        identity["source_paths"] = tuple(cut_sources)
        identities.append(identity)
    return identities, shared_sources


class CaptureReferences:
    """Load fresh, immutable capture records; never copy a prior board's geometry."""

    def __init__(self, location):
        self.location = Path(location).resolve() if location else None
        self._cache = {}
        self._by_panel = {}
        self.result_files = set()
        self.has_unnamed_results = False
        self._initialized = False
        self._validated = {}

    def paths(self):
        if self.location is None:
            return []
        if self.location.is_dir():
            return sorted(p for p in self.location.iterdir() if p.suffix.casefold() == ".json")
        return [self.location]

    def refresh(self):
        """Index changed JSON files once per poll, without rereading old capture records."""
        self._validated.clear()
        current = {}
        changed = False
        for path in self.paths():
            try:
                stat = path.stat()
                stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                previous = self._cache.get(path)
                if previous is not None and previous[0] == stamp:
                    current[path] = previous
                    continue
                raw_bytes = path.read_bytes()
                identities, shared_sources = _index_records(json.loads(raw_bytes.decode("utf-8")))
                current[path] = (
                    stamp,
                    identities,
                    shared_sources,
                    hashlib.sha256(raw_bytes).hexdigest(),
                )
                changed = True
            except (OSError, ValueError, AttributeError, TypeError):
                # A writer may not have finished yet. Missing geometry never permits PASS.
                changed = True
        if changed or current.keys() != self._cache.keys() or not self._initialized:
            self._by_panel = {}
            self.result_files = set()
            self.has_unnamed_results = False
            for path, (_, records, _, _) in current.items():
                for record in records:
                    context = record.get("reference_context")
                    if isinstance(context, dict) and isinstance(context.get("product_id"), str):
                        identity = str(record.get("csv_sn")), context.get("product_id")
                        self._by_panel.setdefault(identity, []).append((path, record))
                        if isinstance(record.get("result_file"), str):
                            self.result_files.add(record["result_file"])
                        else:
                            self.has_unnamed_results = True
        self._cache = current
        self._initialized = True

    def source_paths(self, path):
        return self._cache[path][2]

    def manifest_sha256(self, path):
        return self._cache[path][3]

    def records_for_run(self, run):
        if not self._initialized:
            self.refresh()
        return [
            (path, record)
            for path, record in self._by_panel.get((str(run.sn), run.product_id), [])
            if (not run.table or record["reference_context"].get("table", "").strip() == run.table)
            and record.get("result_file", run.result_file) == run.result_file
        ]

    def review(self, path, run, recipe, index):
        candidates = {
            manifest_path
            for manifest_path, record in self.records_for_run(run)
            if record.get("name") == Path(path).name and record.get("cut_point") == index
        }
        candidates = sorted(candidates)
        if len(candidates) != 1:
            raise ValueError(
                f"Cut {index}: needs exactly one matching capture record (found {len(candidates)})."
            )
        manifest = self._validated.get(candidates[0])
        if manifest is None:
            manifest = load_manifest(candidates[0])
            self._validated[candidates[0]] = manifest
        review = load_edge_review(
            candidates[0],
            path,
            product_id=run.product_id,
            recipe_path=recipe,
            cut_point=index,
            csv_sn=run.sn,
            manifest=manifest,
        )
        # load_manifest validates every source. Pin those sources for the final recheck too.
        if run.table and review["context"]["table"] != run.table:
            raise ValueError("Capture table differs from the inspected board.")
        record = next(r for r in manifest["images"] if r["name"] == Path(path).name)
        if record.get("result_file", run.result_file) != run.result_file:
            raise ValueError("Capture Result identity changed while loading.")
        if manifest.get("capture_source_schema") and run.source_path:
            if Path(record["result_path"]).resolve() != Path(run.source_path).resolve():
                raise ValueError("Capture Result is outside the inspected export.")
        review["source_paths"] = [s["path"] for s in manifest["source_files"]] + [
            str(candidates[0])
        ]
        review["source_hashes"] = {
            str(Path(s["path"]).resolve()): s["sha256"] for s in manifest["source_files"]
        }
        review["source_hashes"][str(candidates[0].resolve())] = manifest["manifest_sha256"]
        return review
