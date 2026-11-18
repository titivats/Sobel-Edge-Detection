"""Durable panel claims, evidence and decisions outside machine-owned folders."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from .guard import check_write_target, protected_roots


def panel_key(result_dir, run):
    return hashlib.sha256(
        json.dumps(
            [str(Path(result_dir).resolve()), run.result_file, run.sn, run.run_id, run.key]
        ).encode()
    ).hexdigest()


def json_safe(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, Path)):
        return str(value)
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return value


class ProductionStore:
    def __init__(self, path, cfg):
        self.path = Path(path)
        self.owned_claims = {}
        check_write_target(self.path, protected_roots(cfg))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS panels ("
                "panel_key TEXT PRIMARY KEY, route TEXT NOT NULL, signature TEXT NOT NULL, "
                "state TEXT NOT NULL, started_at TEXT NOT NULL, completed_at TEXT, "
                "status TEXT, payload TEXT, overview TEXT, claim_id TEXT, result_path TEXT)"
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(panels)")}
            if "claim_id" not in columns:
                db.execute("ALTER TABLE panels ADD COLUMN claim_id TEXT")
            if "result_path" not in columns:
                db.execute("ALTER TABLE panels ADD COLUMN result_path TEXT")
                db.execute(
                    "UPDATE panels SET result_path=NULLIF(json_extract(payload, "
                    "'$.result.run.source_path'),'') WHERE json_valid(payload)"
                )
            db.execute(
                "CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, panel_key TEXT NOT NULL, signature TEXT NOT NULL, completed_at TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL, overview TEXT)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS panels_route_state ON panels(route,state)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS intake_baselines (result_dir TEXT NOT NULL,"
                " route TEXT NOT NULL, files TEXT NOT NULL, PRIMARY KEY(result_dir,route))"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS attempts_with_overview ON attempts(id DESC) WHERE overview IS NOT NULL"
            )

    def states(self, route):
        """Read the route's claims in one transaction instead of opening a DB per board."""
        with self.connect() as db:
            return {
                key: (state, signature)
                for key, state, signature in db.execute(
                    "SELECT panel_key,state,signature FROM panels WHERE route=?", (route,)
                )
            }

    def tracked_panels(self, route):
        with self.connect() as db:
            return db.execute(
                "SELECT panel_key,state,signature,result_path FROM panels WHERE route=?", (route,)
            ).fetchall()

    def legacy_source_hashes(self, key):
        """Use original evidence for old decisions whose signature only covered images/CSV."""
        with self.connect() as db:
            row = db.execute("SELECT payload FROM panels WHERE panel_key=?", (key,)).fetchone()
        if not row or not row[0]:
            return {}
        return json.loads(row[0]).get("evidence", {}).get("source_hashes", {})

    def intake_baseline(self, result_dir, route, initial_files):
        """Only files already archived at the first AUTO start can ever be skipped."""
        identity = str(result_dir), route
        with self.connect() as db:
            row = db.execute(
                "SELECT files FROM intake_baselines WHERE result_dir=? AND route=?", identity
            ).fetchone()
            if row is None:
                db.execute(
                    "INSERT OR IGNORE INTO intake_baselines VALUES (?,?,?)",
                    (*identity, json.dumps(initial_files)),
                )
                row = db.execute(
                    "SELECT files FROM intake_baselines WHERE result_dir=? AND route=?", identity
                ).fetchone()
        return {result_dir / name: tuple(stamp) for name, stamp in json.loads(row[0]).items()}

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def state(self, key):
        with self.connect() as db:
            return db.execute(
                "SELECT state, signature FROM panels WHERE panel_key=?", (key,)
            ).fetchone()

    def claim(self, key, route, signature, *, result_path=None):
        claim_id = uuid.uuid4().hex
        result_path = str(Path(result_path).resolve()) if result_path else None
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT state FROM panels WHERE panel_key=? AND route=?", (key, route)
            ).fetchone()
            if row and row[0] == "RETRY_APPROVED":
                db.execute(
                    "UPDATE panels SET state='PENDING',signature=?,started_at=?,completed_at=NULL,"
                    "claim_id=?,result_path=COALESCE(?,result_path) WHERE panel_key=?",
                    (signature, datetime.now(timezone.utc).isoformat(), claim_id, result_path, key),
                )
                self.owned_claims[key] = claim_id
                return
            db.execute(
                "INSERT INTO panels(panel_key,route,signature,state,started_at,claim_id,result_path) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    key,
                    route,
                    signature,
                    "PENDING",
                    datetime.now(timezone.utc).isoformat(),
                    claim_id,
                    result_path,
                ),
            )
            self.owned_claims[key] = claim_id

    def finish(self, key, result, evidence, overview=""):
        if result.status not in {"GOOD", "NG", "FAULT"}:
            raise ValueError("Invalid production decision status.")
        payload = json.dumps(
            json_safe({"result": result, "evidence": evidence}), allow_nan=False, ensure_ascii=False
        )
        with self.connect() as db:
            row = db.execute("SELECT route FROM panels WHERE panel_key=?", (key,)).fetchone()
            if row is None or row[0] != result.run.key:
                raise ValueError("Result route differs from the claimed board.")
            changed = db.execute(
                "UPDATE panels SET state='DONE',completed_at=?,status=?,payload=?,overview=?,"
                "result_path=COALESCE(result_path,?) "
                "WHERE panel_key=? AND state='PENDING' AND claim_id=?",
                (
                    datetime.now(timezone.utc).isoformat(),
                    result.status,
                    payload,
                    overview,
                    str(Path(result.run.source_path).resolve()) if result.run.source_path else None,
                    key,
                    self.owned_claims.get(key),
                ),
            ).rowcount
            if changed != 1:
                raise ValueError("Panel result has no matching pending claim.")
            row = db.execute(
                "SELECT signature,completed_at FROM panels WHERE panel_key=?", (key,)
            ).fetchone()
            db.execute(
                "INSERT INTO attempts(panel_key,signature,completed_at,status,payload,overview) VALUES(?,?,?,?,?,?)",
                (key, row[0], row[1], result.status, payload, overview),
            )
        self.owned_claims.pop(key, None)

    def blocked_board(self, route):
        with self.connect() as db:
            return db.execute(
                "SELECT panel_key,state,result_path FROM panels WHERE route=? AND state IN "
                "('PENDING','SOURCE_CHANGED') ORDER BY started_at, panel_key LIMIT 1",
                (route,),
            ).fetchone()

    def retry_held(self, route, *, blocked_key=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if blocked_key is None:
                row = db.execute(
                    "SELECT panel_key,state FROM panels WHERE route=? AND (state='PENDING' OR status IN ('FAULT','NG')) AND state!='RETRY_APPROVED' ORDER BY started_at DESC LIMIT 1",
                    (route,),
                ).fetchone()
            else:
                row = db.execute(
                    "SELECT panel_key,state FROM panels WHERE route=? AND panel_key=? "
                    "AND state IN ('PENDING','SOURCE_CHANGED')",
                    (route, blocked_key),
                ).fetchone()
            if not row:
                raise ValueError("No interrupted / held board is available to retry.")
            key, state = row
            if state == "PENDING":
                payload = json.dumps(
                    {
                        "result": None,
                        "evidence": {
                            "reason": "Operator requested retry of an interrupted inspection."
                        },
                    }
                )
                row = db.execute(
                    "SELECT signature FROM panels WHERE panel_key=?", (key,)
                ).fetchone()
                db.execute(
                    "INSERT INTO attempts(panel_key,signature,completed_at,status,payload) VALUES(?,?,?,?,?)",
                    (key, row[0], datetime.now(timezone.utc).isoformat(), "FAULT", payload),
                )
            db.execute("UPDATE panels SET state='RETRY_APPROVED' WHERE panel_key=?", (key,))
            return key

    def pending(self, route):
        with self.connect() as db:
            return db.execute(
                "SELECT panel_key FROM panels WHERE route=? AND state='PENDING'", (route,)
            ).fetchall()

    def quarantine(self, key, reason):
        with self.connect() as db:
            db.execute(
                "UPDATE panels SET state='SOURCE_CHANGED',status='FAULT' WHERE panel_key=?", (key,)
            )
            row = db.execute(
                "SELECT signature,payload,overview FROM panels WHERE panel_key=?", (key,)
            ).fetchone()
            if row and row[1]:
                payload = json.loads(row[1])
                payload["result"]["status"] = "FAULT"
                payload["result"]["note"] = reason
                payload["evidence"]["source_conflict"] = reason
                db.execute(
                    "INSERT INTO attempts(panel_key,signature,completed_at,status,payload,overview) VALUES(?,?,?,?,?,?)",
                    (
                        key,
                        row[0],
                        datetime.now(timezone.utc).isoformat(),
                        "FAULT",
                        json.dumps(payload),
                        row[2],
                    ),
                )

    def recent(self, count=5):
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload,overview,completed_at FROM attempts WHERE overview IS NOT NULL "
                "ORDER BY id DESC LIMIT ?",
                (count,),
            ).fetchall()
        return [(json.loads(p), overview, completed) for p, overview, completed in rows]
