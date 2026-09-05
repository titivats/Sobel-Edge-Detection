"""Human names for inspection positions.

The machine records nothing about which sub-board a picture belongs to: the run
CSV carries one SN for the whole panel, the recipe names a single SubBoard_01
outline, and picture filenames are timestamps.  So the software cannot derive
"sub-board B" on its own without guessing.

What it does know is that picture index N is always the same physical spot on
the panel.  So an engineer names the positions once - "SubBoard B, left edge" -
and from then on every report speaks in the shop's own terms instead of
"index 3".  The labels live here, keyed by recipe and table.
"""

from __future__ import annotations

import json
from pathlib import Path

from .guard import check_write_target


class PositionLabels:
    def __init__(self, path: Path, protected: list[Path] | None = None):
        self.path = path
        self.protected = protected or []
        self.data: dict[str, dict[str, str]] = {}
        self.load()

    def load(self) -> None:
        self.data = {}
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(raw, dict):
            self.data = {k: dict(v) for k, v in raw.items() if isinstance(v, dict)}

    def save(self) -> None:
        check_write_target(self.path, self.protected)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    def label(self, key: str, index: int) -> str:
        """Return the operator's name for this position, or a neutral fallback."""
        return self.data.get(key, {}).get(str(index), "") or f"Position {index}"

    def is_named(self, key: str, index: int) -> bool:
        return bool(self.data.get(key, {}).get(str(index), ""))

    def set_label(self, key: str, index: int, name: str) -> None:
        self.data.setdefault(key, {})[str(index)] = name.strip()

    def for_key(self, key: str) -> dict[str, str]:
        return self.data.get(key, {})
