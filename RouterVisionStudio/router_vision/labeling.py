"""Label store.

An operator marks inspection images as good or as a specific defect, and those
labels become the training set for the DINOv2 classifier.  Labels are keyed by
the image file name, which is a unique timestamp, so the store stays valid even
if folders move.

Nothing here writes into the machine data - the store lives beside the app.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .guard import check_write_target

# Two visual classes only. GOOD is the model class; PASS is reserved for the
# final panel/interlock decision. Legacy PASS labels are migrated on load.
DEFAULT_CLASSES = ["GOOD", "NG"]


@dataclass
class Label:
    image: str  # file name, e.g. 20260622_000003.bmp
    cls: str
    path: str = ""  # full path when it was labelled, for convenience
    product: str = ""  # ProductId of the panel: models are trained per product
    recipe: str = ""
    table: str = ""
    position: int = -1  # picture index within its run
    sn: str = ""
    note: str = ""
    labelled_at: str = ""


class LabelStore:
    def __init__(self, path: Path, protected: list[Path] | None = None):
        self.path = path
        self.protected = protected or []
        self.classes: list[str] = list(DEFAULT_CLASSES)
        self.labels: dict[str, Label] = {}
        self.load()

    # -- persistence -------------------------------------------------------
    def load(self) -> None:
        self.labels = {}
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.classes = list(DEFAULT_CLASSES)
        for key, value in (raw.get("labels") or {}).items():
            try:
                value = dict(value)
                if value.get("cls") == "PASS":
                    value["cls"] = "GOOD"
                label = Label(**value)
            except TypeError:
                continue
            if label.cls in self.classes:
                self.labels[key] = label

    def save(self) -> None:
        check_write_target(self.path, self.protected)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "classes": self.classes,
            "labels": {k: asdict(v) for k, v in self.labels.items()},
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # -- editing -----------------------------------------------------------
    def set(self, image_path: str | Path, cls: str, **meta) -> Label:
        p = Path(image_path)
        label = Label(
            image=p.name,
            cls=cls,
            path=str(p),
            labelled_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **{
                k: v
                for k, v in meta.items()
                if k in ("product", "recipe", "table", "position", "sn", "note")
            },
        )
        self.labels[p.name] = label
        return label

    def unset(self, image_path: str | Path) -> None:
        self.labels.pop(Path(image_path).name, None)

    def get(self, image_path: str | Path) -> Label | None:
        return self.labels.get(Path(image_path).name)

    def cls_of(self, image_path: str | Path) -> str:
        label = self.get(image_path)
        return label.cls if label else ""

    # -- reporting ---------------------------------------------------------
    @staticmethod
    def key_of(label: Label) -> str:
        """Same key the baselines use: ProductId and table."""
        return f"{label.product}|{label.table}"

    def keys(self) -> list[str]:
        """Every ProductId|table that has at least one label."""
        return sorted({self.key_of(label) for label in self.labels.values() if label.product})

    def _for_key(self, key: str | None) -> list[Label]:
        if not key:
            return list(self.labels.values())
        return [label for label in self.labels.values() if self.key_of(label) == key]

    def counts(self, key: str | None = None) -> dict[str, int]:
        return dict(Counter(label.cls for label in self._for_key(key)))

    def usable_for_training(
        self, min_per_class: int = 5, key: str | None = None
    ) -> tuple[bool, str]:
        counts = self.counts(key)
        missing = [c for c in DEFAULT_CLASSES if counts.get(c, 0) == 0]
        if missing:
            return False, f"Need labels for both classes; none yet for {', '.join(missing)}."
        thin = {c: counts[c] for c in DEFAULT_CLASSES if counts[c] < min_per_class}
        if thin:
            detail = ", ".join(f"{c}={n}" for c, n in sorted(thin.items()))
            return False, f"Need at least {min_per_class} images per class; short on {detail}."
        total = sum(counts.get(c, 0) for c in DEFAULT_CLASSES)
        detail = ", ".join(f"{c} {counts[c]}" for c in DEFAULT_CLASSES)
        return True, f"{total} images ({detail})."

    def training_items(self, key: str | None = None) -> list[tuple[str, str]]:
        """(path, class) pairs for every usable label whose file still exists.

        With a ProductId|table key, only that combination's labels: a model
        trained across products and tables would learn to tell those apart
        rather than a good cut from a bad one.
        """
        items = []
        for label in self._for_key(key):
            if label.cls not in DEFAULT_CLASSES or not label.path:
                continue
            if Path(label.path).exists():
                items.append((label.path, label.cls))
        return items
