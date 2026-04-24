"""
logger.py (NEW)
---------------
Centralised logger. One call to `get_logger(__name__)` anywhere in the project
gives you a named logger that writes to both stdout and logs/search.log with a
consistent format. Safe to call repeatedly (handlers are de-duplicated).
"""
import logging
import os
from logging.handlers import RotatingFileHandler

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATEFMT = "%H:%M:%S"
_configured = False


def _configure(log_file: str, level: str):
    global _configured
    if _configured:
        return
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Stream handler
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(sh)

    # Rotating file handler (5 MB x 3)
    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3,
                             encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(fh)

    # Silence noisy libraries
    for noisy in ("urllib3", "sentence_transformers", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str):
    """Import-safe: only configures logging on first call from any module."""
    try:
        import config  # local import to avoid circular issues
        _configure(config.LOG_FILE, config.LOG_LEVEL)
    except Exception:
        # Config not importable yet (e.g. during config parsing itself) - fall back.
        if not _configured:
            logging.basicConfig(level=logging.INFO, format=_FMT, datefmt=_DATEFMT)
    return logging.getLogger(name)
