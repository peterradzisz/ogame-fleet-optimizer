"""Logging configuration for the OGame fleet optimizer.

Logs go to logs/ogame-optimizer.log (rotating, 5 MB x 5) + stderr.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


_LOG_DIR = Path(__file__).parent.parent.parent / "logs"
_CONFIGURED = False


def setup_logging(level: int = logging.INFO, log_dir: Path | None = None) -> None:
    """Configure root logger with file + console handlers. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    target_dir = log_dir or _LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    log_file = target_dir / "ogame-optimizer.log"

    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        root.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console.setLevel(level)
    root.addHandler(console)

    _CONFIGURED = True
    root.info("Logging initialized -> %s", log_file)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger. Auto-initializes root if not yet configured."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
