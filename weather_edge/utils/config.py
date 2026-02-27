"""Configuration loader with environment variable interpolation."""

import os
import re
from pathlib import Path

import yaml


def _interpolate_env(value: str) -> str:
    """Replace ${VAR_NAME} patterns with environment variable values."""
    pattern = re.compile(r"\$\{(\w+)\}")
    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")
    return pattern.sub(replacer, value)


def _walk_and_interpolate(obj):
    """Recursively interpolate environment variables in config values."""
    if isinstance(obj, dict):
        return {k: _walk_and_interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_interpolate(item) for item in obj]
    if isinstance(obj, str):
        return _interpolate_env(obj)
    return obj


def load_config(config_path: str | Path | None = None) -> dict:
    """Load YAML config and interpolate environment variables.

    Looks for config.yaml in the project root by default.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return _walk_and_interpolate(raw)


def get_city_by_name(config: dict, city_name: str) -> dict | None:
    """Look up a city entry from config by name (case-insensitive partial match)."""
    query = city_name.lower()
    for city in config["cities"]:
        if query in city["name"].lower():
            return city
    return None


def ensure_dirs(config: dict) -> None:
    """Create data and log directories if they don't exist."""
    for key in ("data_dir", "cache_dir", "log_dir"):
        path = Path(config["general"][key])
        path.mkdir(parents=True, exist_ok=True)
