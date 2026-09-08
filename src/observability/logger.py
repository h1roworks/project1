"""Structured logging.

Phase A3 placeholder: provides get_logger() writing to stderr.
Full JSON Lines tracing lands in Phase F (F2).
"""

from __future__ import annotations

import logging
import sys

_LOGGER_NAME = "smart_knowledge_hub"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a configured logger writing to stderr."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
