"""Logging helpers with optional run_id context."""

from __future__ import annotations

import logging
import uuid


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


def configure_logging(level: int = logging.INFO, run_id: str | None = None) -> str:
    resolved_run_id = run_id or uuid.uuid4().hex[:8]
    root = logging.getLogger()

    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [run=%(run_id)s] %(name)s: %(message)s",
        )

    for handler in root.handlers:
        handler.addFilter(_RunIdFilter(resolved_run_id))

    root.setLevel(level)
    return resolved_run_id


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
