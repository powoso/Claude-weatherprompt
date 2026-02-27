"""Historical base rate calculations for weather events."""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("weather_edge.features.base_rates")


def temperature_exceedance_rate(
    historical: pd.DataFrame,
    threshold_f: float,
    month: int | None = None,
    months: list[int] | None = None,
    column: str = "temperature_2m_max",
) -> dict:
    """Calculate how often temperature exceeds a threshold.

    Args:
        historical: DataFrame with date index and temperature columns (Fahrenheit).
        threshold_f: Temperature threshold in Fahrenheit.
        month: Single month filter (1-12).
        months: List of months to include. Overrides `month`.
        column: Column name for the temperature data.

    Returns:
        Dict with base_rate, count, total_days, and yearly_rates.
    """
    df = historical.copy()
    df = df.dropna(subset=[column])

    if months:
        df = df[df.index.month.isin(months)]
    elif month:
        df = df[df.index.month == month]

    total_days = len(df)
    if total_days == 0:
        return {"base_rate": 0.0, "count": 0, "total_days": 0, "yearly_rates": {}}

    exceed = df[df[column] >= threshold_f]
    count = len(exceed)
    base_rate = count / total_days

    # Per-year rates (how many years had at least one exceedance)
    df["year"] = df.index.year
    yearly = df.groupby("year").apply(
        lambda g: int((g[column] >= threshold_f).any()), include_groups=False
    )
    year_count = int(yearly.sum())
    total_years = len(yearly)
    yearly_rate = year_count / total_years if total_years > 0 else 0.0

    return {
        "base_rate_daily": base_rate,
        "base_rate_yearly": yearly_rate,
        "exceed_days": count,
        "total_days": total_days,
        "exceed_years": year_count,
        "total_years": total_years,
        "yearly_rates": yearly.to_dict(),
    }


def snowfall_occurrence_rate(
    historical: pd.DataFrame,
    month: int | None = None,
    months: list[int] | None = None,
    min_snowfall_inches: float = 0.1,
    column: str = "snowfall_sum",
) -> dict:
    """Calculate historical snowfall frequency.

    Returns base rate of snow days and yearly probability of any snow.
    """
    df = historical.copy()
    df = df.dropna(subset=[column])

    if months:
        df = df[df.index.month.isin(months)]
    elif month:
        df = df[df.index.month == month]

    total_days = len(df)
    if total_days == 0:
        return {"base_rate": 0.0, "count": 0, "total_days": 0}

    snow_days = df[df[column] >= min_snowfall_inches]
    count = len(snow_days)

    # Per-year rates
    df["year"] = df.index.year
    yearly = df.groupby("year").apply(
        lambda g: int((g[column] >= min_snowfall_inches).any()), include_groups=False
    )
    year_count = int(yearly.sum())
    total_years = len(yearly)

    return {
        "base_rate_daily": count / total_days,
        "base_rate_yearly": year_count / total_years if total_years > 0 else 0.0,
        "snow_days": count,
        "total_days": total_days,
        "snow_years": year_count,
        "total_years": total_years,
    }


def precipitation_exceedance_rate(
    historical: pd.DataFrame,
    threshold_inches: float,
    month: int | None = None,
    months: list[int] | None = None,
    column: str = "precipitation_sum",
) -> dict:
    """Calculate how often precipitation exceeds a daily threshold."""
    df = historical.copy()
    df = df.dropna(subset=[column])

    if months:
        df = df[df.index.month.isin(months)]
    elif month:
        df = df[df.index.month == month]

    total_days = len(df)
    if total_days == 0:
        return {"base_rate": 0.0, "count": 0, "total_days": 0}

    exceed = df[df[column] >= threshold_inches]

    return {
        "base_rate_daily": len(exceed) / total_days,
        "exceed_days": len(exceed),
        "total_days": total_days,
    }


def fit_temperature_distribution(
    historical: pd.DataFrame,
    month: int,
    column: str = "temperature_2m_max",
) -> dict:
    """Fit a parametric distribution to monthly temperature data.

    Tries normal, skew-normal, and generalized extreme value distributions,
    returning the best fit by AIC.
    """
    df = historical.copy()
    df = df.dropna(subset=[column])
    df = df[df.index.month == month]
    values = df[column].values

    if len(values) < 30:
        return {"distribution": "insufficient_data", "n_samples": len(values)}

    results = {}

    # Normal
    mu, sigma = stats.norm.fit(values)
    ll = stats.norm.logpdf(values, mu, sigma).sum()
    results["normal"] = {"params": {"mu": mu, "sigma": sigma}, "aic": 2 * 2 - 2 * ll}

    # Skew-normal
    try:
        a, loc, scale = stats.skewnorm.fit(values)
        ll = stats.skewnorm.logpdf(values, a, loc, scale).sum()
        results["skewnorm"] = {
            "params": {"a": a, "loc": loc, "scale": scale},
            "aic": 2 * 3 - 2 * ll,
        }
    except Exception:
        pass

    # GEV (for extreme value analysis)
    try:
        c, loc, scale = stats.genextreme.fit(values)
        ll = stats.genextreme.logpdf(values, c, loc, scale).sum()
        results["gev"] = {
            "params": {"c": c, "loc": loc, "scale": scale},
            "aic": 2 * 3 - 2 * ll,
        }
    except Exception:
        pass

    # Pick best by AIC
    best_name = min(results, key=lambda k: results[k]["aic"])
    best = results[best_name]

    return {
        "distribution": best_name,
        "params": best["params"],
        "aic": best["aic"],
        "n_samples": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "all_fits": results,
    }


def compute_climatological_stats(
    historical: pd.DataFrame,
    column: str = "temperature_2m_max",
) -> pd.DataFrame:
    """Compute monthly climatological statistics.

    Returns a DataFrame indexed by month (1-12) with mean, std, percentiles.
    """
    df = historical.copy()
    df = df.dropna(subset=[column])
    df["month"] = df.index.month

    stats_df = df.groupby("month")[column].agg(
        mean="mean",
        std="std",
        min="min",
        p10=lambda x: np.percentile(x, 10),
        p25=lambda x: np.percentile(x, 25),
        median="median",
        p75=lambda x: np.percentile(x, 75),
        p90=lambda x: np.percentile(x, 90),
        max="max",
        count="count",
    )
    return stats_df
