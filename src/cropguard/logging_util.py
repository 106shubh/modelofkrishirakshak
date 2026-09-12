"""Structured logging helpers."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload["fields"] = extra
        return json.dumps(payload)


def setup_logging(level: int = logging.INFO, fmt: str = "json") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    root.addHandler(handler)
    # Quiet noisy third-party loggers
    for noisy in ("PIL", "matplotlib", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_extra(logger: logging.Logger, level: int, msg: str, **fields: object) -> None:
    """Log a message with structured extra fields."""
    logger.log(level, msg, extra={"extra_fields": fields})