"""ENSO (El Nino / La Nina) status and CPC seasonal outlook data."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger("weather_edge.data_collection.enso")

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"


def fetch_oni_index(url: str = ONI_URL) -> pd.DataFrame:
    """Fetch the Oceanic Nino Index (ONI) time series from CPC.

    ONI is the primary indicator used to classify ENSO phases:
      - El Nino: ONI >= 0.5 for 5 consecutive overlapping 3-month periods
      - La Nina: ONI <= -0.5 for 5 consecutive overlapping 3-month periods
      - Neutral: otherwise

    Returns DataFrame with columns: year, season, total, anomaly, oni_phase.
    """
    logger.info("Fetching ONI index from %s", url)
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()

    # Parse fixed-width or whitespace-delimited text
    df = pd.read_csv(
        io.StringIO(resp.text),
        sep=r"\s+",
        skiprows=1,
        names=["season", "year", "total", "anomaly"],
        dtype={"year": int},
    )

    # Classify phase based on anomaly
    df["oni_phase"] = df["anomaly"].apply(_classify_oni)
    return df


def _classify_oni(anomaly: float) -> str:
    """Classify a single ONI value into ENSO phase."""
    if anomaly >= 1.5:
        return "strong_el_nino"
    if anomaly >= 0.5:
        return "el_nino"
    if anomaly <= -1.5:
        return "strong_la_nina"
    if anomaly <= -0.5:
        return "la_nina"
    return "neutral"


def get_current_enso_phase(oni_df: pd.DataFrame) -> dict:
    """Return the most recent ENSO status.

    Returns dict with keys: season, year, anomaly, phase.
    """
    if oni_df.empty:
        return {"season": "N/A", "year": 0, "anomaly": 0.0, "phase": "unknown"}

    last = oni_df.iloc[-1]
    return {
        "season": last["season"],
        "year": int(last["year"]),
        "anomaly": float(last["anomaly"]),
        "phase": last["oni_phase"],
    }


def get_enso_years_by_phase(oni_df: pd.DataFrame) -> dict[str, list[int]]:
    """Group years by their dominant ENSO phase.

    Uses the DJF (Dec-Jan-Feb) season as the canonical classification for each
    winter, which is the standard meteorological convention.
    """
    djf = oni_df[oni_df["season"] == "DJF"].copy()
    result: dict[str, list[int]] = {}
    for _, row in djf.iterrows():
        phase = row["oni_phase"]
        year = int(row["year"])
        result.setdefault(phase, []).append(year)
    return result


def fetch_cpc_temperature_outlook() -> dict:
    """Fetch CPC seasonal temperature outlook summary.

    Returns a dict with outlook metadata. Full parsing of the probabilistic
    maps requires GIS data; this provides the text-based summary.
    """
    url = "https://www.cpc.ncep.noaa.gov/products/predictions/long_range/fnet_temp.php"
    logger.info("Fetching CPC temperature outlook")
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        return {"status": "fetched", "content_length": len(resp.text), "url": url}
    except Exception:
        logger.exception("Failed to fetch CPC outlook")
        return {"status": "error", "url": url}


def fetch_cpc_precipitation_outlook() -> dict:
    """Fetch CPC seasonal precipitation outlook summary."""
    url = "https://www.cpc.ncep.noaa.gov/products/predictions/long_range/fnet_prcp.php"
    logger.info("Fetching CPC precipitation outlook")
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        return {"status": "fetched", "content_length": len(resp.text), "url": url}
    except Exception:
        logger.exception("Failed to fetch CPC precipitation outlook")
        return {"status": "error", "url": url}
