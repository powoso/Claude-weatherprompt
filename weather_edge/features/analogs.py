"""Analog year matching — find historically similar climate setups."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("weather_edge.features.analogs")


def find_analog_years(
    oni_df: pd.DataFrame,
    historical: pd.DataFrame,
    target_year: int,
    target_month: int,
    top_n: int = 10,
    enso_weight: float = 0.4,
    temp_anomaly_weight: float = 0.3,
    precip_anomaly_weight: float = 0.3,
    min_year: int = 1970,
) -> list[dict]:
    """Find the top-N most climatologically similar historical years.

    Similarity is computed as a weighted distance across:
      1. ENSO phase (ONI anomaly similarity)
      2. Temperature anomaly for the preceding months
      3. Precipitation anomaly for the preceding months

    Returns list of dicts with year, distance, and component scores.
    """
    # Get ONI values for target year's preceding winter
    target_oni = _get_oni_for_year(oni_df, target_year)

    # Get recent temp/precip anomalies for the target year
    target_temp_anom = _compute_anomaly(
        historical, target_year, target_month, "temperature_2m_max"
    )
    target_precip_anom = _compute_anomaly(
        historical, target_year, target_month, "precipitation_sum"
    )

    candidates = []
    all_years = sorted(historical.index.year.unique())

    for year in all_years:
        if year == target_year or year < min_year:
            continue

        year_oni = _get_oni_for_year(oni_df, year)
        year_temp_anom = _compute_anomaly(
            historical, year, target_month, "temperature_2m_max"
        )
        year_precip_anom = _compute_anomaly(
            historical, year, target_month, "precipitation_sum"
        )

        # Compute weighted distance
        oni_dist = abs(target_oni - year_oni) if (target_oni is not None and year_oni is not None) else 2.0
        temp_dist = abs(target_temp_anom - year_temp_anom) if (target_temp_anom is not None and year_temp_anom is not None) else 5.0
        precip_dist = abs(target_precip_anom - year_precip_anom) if (target_precip_anom is not None and year_precip_anom is not None) else 2.0

        # Normalize each component to 0-1 range approximately
        oni_norm = oni_dist / 3.0  # ONI ranges roughly -2 to +2
        temp_norm = temp_dist / 15.0  # Temp anomalies in F
        precip_norm = precip_dist / 3.0  # Precip anomaly in inches

        distance = (
            enso_weight * oni_norm
            + temp_anomaly_weight * temp_norm
            + precip_anomaly_weight * precip_norm
        )

        candidates.append({
            "year": year,
            "distance": distance,
            "oni_value": year_oni,
            "oni_distance": oni_dist,
            "temp_anomaly": year_temp_anom,
            "precip_anomaly": year_precip_anom,
        })

    # Sort by distance, return top_n
    candidates.sort(key=lambda x: x["distance"])
    top = candidates[:top_n]

    logger.info(
        "Analog years for %d (month %d): %s",
        target_year, target_month, [c["year"] for c in top],
    )
    return top


def _get_oni_for_year(oni_df: pd.DataFrame, year: int) -> float | None:
    """Get the DJF ONI value for a given year (Dec of previous year through Feb)."""
    if oni_df.empty:
        return None
    djf = oni_df[(oni_df["year"] == year) & (oni_df["season"] == "DJF")]
    if not djf.empty:
        return float(djf.iloc[0]["anomaly"])
    # Fallback to most recent season
    year_data = oni_df[oni_df["year"] == year]
    if not year_data.empty:
        return float(year_data.iloc[-1]["anomaly"])
    return None


def _compute_anomaly(
    historical: pd.DataFrame,
    year: int,
    month: int,
    column: str,
) -> float | None:
    """Compute the anomaly for a year/month vs the long-term mean."""
    if column not in historical.columns:
        return None

    df = historical.dropna(subset=[column])
    month_data = df[df.index.month == month]
    if month_data.empty:
        return None

    long_term_mean = month_data[column].mean()
    year_data = month_data[month_data.index.year == year]
    if year_data.empty:
        return None

    year_mean = year_data[column].mean()
    return year_mean - long_term_mean


def analog_outcome_distribution(
    historical: pd.DataFrame,
    analog_years: list[dict],
    month: int,
    column: str = "temperature_2m_max",
) -> pd.Series:
    """Get the distribution of outcomes from analog years for a given month.

    Returns a Series of daily values from the analog years' target month,
    weighted by inverse distance.
    """
    all_values = []
    all_weights = []

    for analog in analog_years:
        year = analog["year"]
        dist = analog["distance"]
        weight = 1.0 / (dist + 0.01)  # Inverse distance weight

        year_month = historical[
            (historical.index.year == year) & (historical.index.month == month)
        ]
        if column in year_month.columns:
            vals = year_month[column].dropna().values
            all_values.extend(vals)
            all_weights.extend([weight] * len(vals))

    if not all_values:
        return pd.Series(dtype=float)

    return pd.Series(all_values)
