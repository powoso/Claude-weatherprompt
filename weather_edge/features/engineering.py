"""Feature engineering for prediction models."""

from __future__ import annotations

import logging
from datetime import date, datetime

import numpy as np
import pandas as pd

logger = logging.getLogger("weather_edge.features.engineering")


def compute_current_anomaly(
    historical: pd.DataFrame,
    current_year: int | None = None,
    window_days: int = 30,
    column: str = "temperature_2m_max",
) -> dict:
    """Compute how the current year deviates from climatological norms.

    Compares the most recent `window_days` of the current year against
    the long-term average for the same calendar period.
    """
    if current_year is None:
        current_year = date.today().year

    df = historical.dropna(subset=[column])
    today_doy = date.today().timetuple().tm_yday

    # Current year's recent window
    current = df[df.index.year == current_year]
    if current.empty:
        return {"anomaly": None, "current_mean": None, "climo_mean": None}

    recent = current[current.index.dayofyear.isin(range(max(1, today_doy - window_days), today_doy + 1))]
    if recent.empty:
        return {"anomaly": None, "current_mean": None, "climo_mean": None}

    current_mean = recent[column].mean()

    # Climatological mean for the same day-of-year range
    climo = df[
        (df.index.year != current_year)
        & (df.index.dayofyear.isin(range(max(1, today_doy - window_days), today_doy + 1)))
    ]
    climo_mean = climo[column].mean() if not climo.empty else None

    anomaly = current_mean - climo_mean if climo_mean is not None else None

    return {
        "anomaly": anomaly,
        "current_mean": float(current_mean),
        "climo_mean": float(climo_mean) if climo_mean is not None else None,
        "window_days": window_days,
    }


def compute_time_decay(contract_end_date: str, reference_date: date | None = None) -> dict:
    """Compute time decay factor for a contract.

    Markets with more time remaining have more uncertainty.
    Returns days_remaining and a decay factor (0-1 where 1 = far out).
    """
    if reference_date is None:
        reference_date = date.today()

    try:
        if "T" in contract_end_date:
            end = datetime.fromisoformat(contract_end_date.replace("Z", "+00:00")).date()
        else:
            end = date.fromisoformat(contract_end_date)
    except (ValueError, TypeError):
        return {"days_remaining": None, "decay_factor": None}

    days_remaining = (end - reference_date).days
    if days_remaining < 0:
        return {"days_remaining": 0, "decay_factor": 0.0}

    # Decay factor: approaches 0 as contract nears expiry
    # Using sqrt scaling so decay is gradual at first, steeper near end
    max_days = 365  # Normalize against a year
    decay_factor = min(1.0, np.sqrt(days_remaining / max_days))

    return {
        "days_remaining": days_remaining,
        "decay_factor": float(decay_factor),
    }


def urban_heat_island_adjustment(
    temperature_f: float,
    uhi_offset_f: float,
    apply_to: str = "station_to_urban",
) -> float:
    """Apply urban heat island adjustment.

    station_to_urban: Add UHI to station reading (station is usually at airport).
    urban_to_station: Subtract UHI to go from urban temp to what station might read.

    Many weather markets reference airport stations, so urban areas are hotter.
    """
    if apply_to == "station_to_urban":
        return temperature_f + uhi_offset_f
    return temperature_f - uhi_offset_f


def ensemble_spread_uncertainty(
    forecast_temps: list[float],
) -> dict:
    """Compute uncertainty metrics from ensemble forecast spread.

    A wider spread means more uncertainty in the forecast.
    """
    if not forecast_temps or len(forecast_temps) < 2:
        return {"spread": None, "std": None, "mean": None}

    arr = np.array(forecast_temps)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "spread": float(np.max(arr) - np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def compute_forecast_bias(
    historical: pd.DataFrame,
    forecast_column: str = "temperature_2m_max",
    observed_column: str = "temperature_2m_max",
    window_days: int = 30,
) -> dict:
    """Estimate systematic forecast bias from recent verification.

    This helps correct for consistent over/under-forecasting.
    In practice, you'd compare past NWS forecasts vs observations.
    Here we compute the recent trend as a proxy.
    """
    df = historical.dropna(subset=[forecast_column])
    if len(df) < window_days:
        return {"bias": 0.0, "n_samples": len(df)}

    recent = df.tail(window_days)
    current_year = date.today().year

    # Compare recent actual temps to long-term mean for same period
    doy_range = recent.index.dayofyear
    climo = df[
        (df.index.year < current_year)
        & (df.index.dayofyear.isin(doy_range))
    ]

    if climo.empty:
        return {"bias": 0.0, "n_samples": 0}

    recent_mean = recent[forecast_column].mean()
    climo_mean = climo[forecast_column].mean()

    return {
        "bias": float(recent_mean - climo_mean),
        "recent_mean": float(recent_mean),
        "climo_mean": float(climo_mean),
        "n_samples": len(recent),
    }
