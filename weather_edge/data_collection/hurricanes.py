"""HURDAT2 hurricane historical data parser and analysis."""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger("weather_edge.data_collection.hurricanes")

HURDAT2_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040824.txt"


@dataclass
class StormTrack:
    storm_id: str
    name: str
    entries: int
    records: list[dict] = field(default_factory=list)


def download_hurdat2(url: str = HURDAT2_URL, cache_dir: str | Path = "data/cache") -> str:
    """Download the HURDAT2 text file, caching locally."""
    cache_path = Path(cache_dir) / "hurdat2.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info("Using cached HURDAT2 at %s", cache_path)
        return cache_path.read_text()

    logger.info("Downloading HURDAT2 from %s", url)
    resp = httpx.get(url, timeout=120, follow_redirects=True)
    resp.raise_for_status()
    text = resp.text
    cache_path.write_text(text)
    return text


def parse_hurdat2(raw_text: str) -> list[StormTrack]:
    """Parse HURDAT2 format into structured StormTrack objects."""
    storms: list[StormTrack] = []
    lines = raw_text.strip().split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        # Header line: ID, NAME, ENTRIES
        header_match = re.match(r"^(\w+),\s*(\w+)\s*,\s*(\d+),?\s*$", line)
        if header_match:
            storm_id = header_match.group(1)
            name = header_match.group(2).strip()
            entries = int(header_match.group(3))
            storm = StormTrack(storm_id=storm_id, name=name, entries=entries)
            i += 1

            for _ in range(entries):
                if i >= len(lines):
                    break
                rec_line = lines[i].strip()
                parts = [p.strip() for p in rec_line.split(",")]
                if len(parts) >= 8:
                    try:
                        record = {
                            "date": parts[0],
                            "time": parts[1],
                            "record_id": parts[2],
                            "status": parts[3],
                            "lat": _parse_coord(parts[4]),
                            "lon": _parse_coord(parts[5]),
                            "max_wind_kt": _safe_int(parts[6]),
                            "min_pressure_mb": _safe_int(parts[7]),
                        }
                        storm.records.append(record)
                    except (ValueError, IndexError):
                        pass
                i += 1

            storms.append(storm)
        else:
            i += 1

    logger.info("Parsed %d storms from HURDAT2", len(storms))
    return storms


def _parse_coord(s: str) -> float:
    """Parse '25.1N' or '80.3W' to signed float."""
    s = s.strip()
    if not s:
        return 0.0
    direction = s[-1]
    value = float(s[:-1])
    if direction in ("S", "W"):
        value = -value
    return value


def _safe_int(s: str) -> int | None:
    s = s.strip()
    if s == "-999" or s == "" or s == "-99":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def storms_to_dataframe(storms: list[StormTrack]) -> pd.DataFrame:
    """Convert storm list to a flat DataFrame of all track records."""
    rows = []
    for storm in storms:
        for rec in storm.records:
            row = {"storm_id": storm.storm_id, "name": storm.name, **rec}
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"] + df["time"], format="%Y%m%d%H%M", errors="coerce")
        df["year"] = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
    return df


def compute_us_landfall_stats(storms: list[StormTrack]) -> pd.DataFrame:
    """Compute yearly US landfall counts by Saffir-Simpson category.

    A landfall is approximated as a track point crossing lat 24-49N, lon 66-98W
    with status 'HU' (hurricane) or 'TS' (tropical storm), using the 'L' record
    identifier when present.
    """
    df = storms_to_dataframe(storms)
    if df.empty:
        return pd.DataFrame()

    # Filter for records that are likely US landfalls
    us_mask = (
        (df["lat"] >= 24) & (df["lat"] <= 49)
        & (df["lon"] >= -98) & (df["lon"] <= -66)
        & (df["status"].isin(["HU", "TS"]))
        & (df["record_id"] == "L")
    )
    landfalls = df[us_mask].copy()

    if landfalls.empty:
        return pd.DataFrame()

    # Saffir-Simpson category from max wind
    landfalls["category"] = landfalls["max_wind_kt"].apply(_wind_to_category)

    # Yearly summary: count of landfalls by category
    yearly = (
        landfalls
        .groupby(["year", "category"])
        .agg(count=("storm_id", "nunique"))
        .reset_index()
    )
    return yearly


def _wind_to_category(wind_kt: int | None) -> str:
    """Convert max sustained wind (knots) to Saffir-Simpson category."""
    if wind_kt is None:
        return "TS"
    if wind_kt < 64:
        return "TS"
    if wind_kt < 83:
        return "Cat1"
    if wind_kt < 96:
        return "Cat2"
    if wind_kt < 113:
        return "Cat3"
    if wind_kt < 137:
        return "Cat4"
    return "Cat5"


def get_historical_landfall_rates(
    storms: list[StormTrack],
    min_year: int = 1950,
) -> dict[str, float]:
    """Return average annual landfall rate by category since min_year."""
    df = storms_to_dataframe(storms)
    if df.empty:
        return {}

    us_mask = (
        (df["lat"] >= 24) & (df["lat"] <= 49)
        & (df["lon"] >= -98) & (df["lon"] <= -66)
        & (df["status"].isin(["HU", "TS"]))
        & (df["record_id"] == "L")
        & (df["year"] >= min_year)
    )
    landfalls = df[us_mask].copy()
    if landfalls.empty:
        return {}

    landfalls["category"] = landfalls["max_wind_kt"].apply(_wind_to_category)
    n_years = df[df["year"] >= min_year]["year"].nunique()
    if n_years == 0:
        return {}

    rates = {}
    for cat in ["TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5"]:
        count = landfalls[landfalls["category"] == cat]["storm_id"].nunique()
        rates[cat] = count / n_years

    # Also compute "Cat4+" rate (what markets usually ask about)
    major = landfalls[landfalls["category"].isin(["Cat4", "Cat5"])]["storm_id"].nunique()
    rates["Cat4+"] = major / n_years

    return rates
