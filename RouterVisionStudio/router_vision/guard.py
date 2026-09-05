"""Write guard.

This tool is a reader. It must never create, modify or delete anything inside
the machine's data folders - those belong to the router software, and a stray
write there could corrupt a production record.

Everything the app produces (config.json, baselines.json, exported CSVs) lands
outside those folders. This module makes that a rule the code enforces rather
than a convention someone has to remember: any write path is checked against
the configured data locations first, so even a mis-typed path or an operator
picking the wrong folder in a save dialog is refused.
"""

from __future__ import annotations

from pathlib import Path

# Attributes of AppConfig that point at machine-owned data.
DATA_DIR_ATTRS = ("picture_dir", "result_dir", "recipe_dir")
DATA_FILE_ATTRS = ("eqp_cfg_path",)


class ProtectedPathError(Exception):
    """Raised when something tries to write inside the machine data."""


def _resolve(value: str | Path) -> Path | None:
    if not str(value).strip():
        return None
    try:
        return Path(value).expanduser().resolve(strict=False)
    except (OSError, ValueError):
        return None


def protected_roots(cfg) -> list[Path]:
    """Folders that must stay read-only: the data folders and the Eqp.cfg folder."""
    roots: list[Path] = []
    for attr in DATA_DIR_ATTRS:
        p = _resolve(getattr(cfg, attr, ""))
        if p:
            roots.append(p)
    for attr in DATA_FILE_ATTRS:
        p = _resolve(getattr(cfg, attr, ""))
        if p:
            roots.append(p.parent)
    return roots


def is_protected(target: str | Path, roots: list[Path]) -> Path | None:
    """Return the protected root that contains `target`, or None if it is safe."""
    resolved = _resolve(target)
    if resolved is None:
        return None
    for root in roots:
        if resolved == root or root in resolved.parents:
            return root
    return None


def check_write_target(target: str | Path, roots: list[Path]) -> None:
    """Raise ProtectedPathError if writing to `target` would touch machine data."""
    hit = is_protected(target, roots)
    if hit is not None:
        raise ProtectedPathError(
            f"Refusing to write to:\n    {target}\n\n"
            f"That path is inside a machine data folder:\n    {hit}\n\n"
            "This tool only reads machine data. Choose a location outside it."
        )
