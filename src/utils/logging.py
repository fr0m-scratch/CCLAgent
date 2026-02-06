from __future__ import annotations

import logging
import os
import sys

def setup_logger(name: str, level: str | int | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    default_level = os.getenv("CCL_AGENT_LOG_LEVEL", "INFO").upper()
    resolved_level = level if level is not None else default_level
    logger.setLevel(resolved_level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
