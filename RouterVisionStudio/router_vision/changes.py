"""Detecting machine events that invalidate a measurement baseline.

A baseline records what a good cut looks like on this machine, with this bit,
through this camera.  Change any of those and the baseline is describing a
machine that no longer exists, so it has to be rebuilt.

What the machine actually records:

  bit change      yes - it raises "[Cut] Router Bit Is Abrasion" when the bit is
                  worn (804 of them across two years in the sample data), and
                  KnifeUsedLength in the config snapshots resets to zero when a
                  fresh bit goes in
  camera change   yes - PixelSize and CCDSpindleOffset live in Eqp.cfg, and the
                  Temp folder keeps a timestamped snapshot every time the
                  machine state changes
  fixture change  no - FixtureBarcode is empty on every run in the sample data
                  because the barcode option is switched off, so a fixture swap
                  leaves no trace. That one still needs a human to say so.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Alarm text that means the bit was worn out and about to be replaced.
BIT_ALARM_PATTERNS = (
    "router bit is abrasion",
    "router bit is slide",
    "put router bit fail",
    "change bit",
)
ALARM_NAME = re.compile(r"^Alarm\d*_(\d{8})_(\d{6})\.csv$", re.IGNORECASE)
CFG_NAME = re.compile(r"^TempEqp(\d{8})T(\d{6})", re.IGNORECASE)


@dataclass
class MachineEvent:
    when: datetime
    kind: str  # "bit" or "camera"
    detail: str

    def __str__(self) -> str:
        return f"{self.when:%Y-%m-%d %H:%M}  {self.kind}: {self.detail}"


def _stamp(date_part: str, time_part: str) -> datetime | None:
    try:
        return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def find_bit_events(
    result_dir: str | Path, since: datetime | None = None, limit: int = 200
) -> list[MachineEvent]:
    """Bit-wear alarms, newest first.

    Only files stamped after `since` are opened, so this stays cheap even with
    a hundred thousand result files on disk.
    """
    events: list[MachineEvent] = []
    try:
        entries = list(os.scandir(result_dir))
    except OSError:
        return events

    candidates = []
    for entry in entries:
        m = ALARM_NAME.match(entry.name)
        if not m:
            continue
        when = _stamp(m.group(1), m.group(2))
        if when is None or (since is not None and when <= since):
            continue
        candidates.append((when, entry.path))

    candidates.sort(reverse=True)
    for when, path in candidates[: limit * 4]:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                fh.readline()
                line = fh.readline()
        except OSError:
            continue
        if not line:
            continue
        cells = line.split(",")
        if len(cells) < 6:
            continue
        message = cells[5].strip()
        low = message.lower()
        if any(p in low for p in BIT_ALARM_PATTERNS):
            events.append(MachineEvent(when, "bit", message))
            if len(events) >= limit:
                break
    return events


def last_bit_change(result_dir: str | Path, since: datetime | None = None) -> datetime | None:
    events = find_bit_events(result_dir, since=since, limit=1)
    return events[0].when if events else None


def find_camera_events(
    temp_dir: str | Path, since: datetime | None = None, limit: int = 50
) -> list[MachineEvent]:
    """Snapshots where the camera calibration differs from the one before it."""
    snapshots: list[tuple[datetime, str]] = []
    try:
        for dirpath, _dirs, files in os.walk(temp_dir):
            for name in files:
                m = CFG_NAME.match(name)
                if not m:
                    continue
                when = _stamp(m.group(1), m.group(2))
                if when is None:
                    continue
                snapshots.append((when, os.path.join(dirpath, name)))
    except OSError:
        return []
    if not snapshots:
        return []
    snapshots.sort()

    # one snapshot per day is plenty to spot a calibration edit
    per_day: dict[str, tuple[datetime, str]] = {}
    for when, path in snapshots:
        per_day[when.strftime("%Y%m%d")] = (when, path)
    ordered = [per_day[k] for k in sorted(per_day)]

    pixel_re = re.compile(r'<PixelSize\s+X="([^"]+)"\s+Y="([^"]+)"')
    offset_re = re.compile(r'<CCDSpindleOffset X="([^"]+)" Y="([^"]+)"')

    events: list[MachineEvent] = []
    previous: tuple[str, str] | None = None
    for when, path in ordered:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        pm, om = pixel_re.search(text), offset_re.search(text)
        current = (pm.group(0) if pm else "", om.group(0) if om else "")
        if previous is not None and current != previous:
            what = []
            if current[0] != previous[0]:
                what.append("PixelSize changed")
            if current[1] != previous[1]:
                what.append("CCDSpindleOffset changed")
            if since is None or when > since:
                events.append(MachineEvent(when, "camera", "; ".join(what)))
        previous = current
    return events[-limit:]


def baseline_is_stale(calibrated_at: str, result_dir: str | Path) -> tuple[bool, str]:
    """Has anything happened since this baseline was measured that invalidates it?"""
    try:
        made = datetime.strptime(calibrated_at, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return False, ""
    when = last_bit_change(result_dir, since=made)
    if when is not None:
        return True, f"bit changed {when:%Y-%m-%d %H:%M}, after this baseline was made"
    return False, ""
