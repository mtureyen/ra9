import json
import logging
import sys
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """JSON-formatted log output for observability."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event_type"):
            entry["event_type"] = record.event_type
        if hasattr(record, "event_data"):
            entry["event_data"] = record.event_data
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])
        return json.dumps(entry)


def get_logger(name: str) -> logging.Logger:
    """Get a structured logger for the given module name."""
    logger = logging.getLogger(f"emotive.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event_type: str, event_data: dict | None = None) -> None:
    """Log a structured event. Bridge to event_bus wired in Stage 4."""
    extra = {"event_type": event_type, "event_data": event_data or {}}
    record = logger.makeRecord(
        name=logger.name,
        level=logging.INFO,
        fn="",
        lno=0,
        msg=f"{event_type}",
        args=(),
        exc_info=None,
    )
    record.event_type = extra["event_type"]
    record.event_data = extra["event_data"]
    logger.handle(record)
