"""Logging setup for weather-edge."""

import logging
import sys
from pathlib import Path


def setup_logging(config: dict) -> logging.Logger:
    """Configure and return the root logger."""
    level = getattr(logging, config["general"].get("log_level", "INFO").upper())
    log_dir = Path(config["general"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("weather_edge")
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s: %(message)s")
        console.setFormatter(fmt)
        logger.addHandler(console)

        # File handler
        fh = logging.FileHandler(log_dir / "weather_edge.log")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
