"""Console + file logging."""

import logging
from pathlib import Path

FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def get_logger(name: str, file: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(level.upper())
    logger.propagate = False
    file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(FORMAT)
    for handler in (logging.StreamHandler(), logging.FileHandler(file, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
