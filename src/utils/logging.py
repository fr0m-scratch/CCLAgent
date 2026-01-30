import logging
import os
import sys

_DEFAULT_LEVEL = os.getenv("CCL_AGENT_LOG_LEVEL", "INFO").upper()


def setup_logger(name: str, level: str | int | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    resolved_level = level if level is not None else _DEFAULT_LEVEL
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
