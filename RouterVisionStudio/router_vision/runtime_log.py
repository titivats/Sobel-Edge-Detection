"""Bounded UTF-8 runtime logs, kept outside the router's data folders."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .guard import check_write_target, protected_roots
from .production_store import json_safe


def open_runtime_log(directory, cfg, *, max_bytes=10 * 1024 * 1024, backups=10):
    path = Path(directory) / "avtr.jsonl"
    check_write_target(path, protected_roots(cfg))
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.Logger("avtr", level=logging.INFO)
    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def log_event(logger, event, *, level=logging.INFO, **fields):
    if logger is not None:
        logger.log(
            level,
            json.dumps(
                json_safe(
                    {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
                ),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )


def close_runtime_log(logger):
    if logger is not None:
        for handler in list(logger.handlers):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
