"""Snowfall probability models using logistic regression."""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("weather_edge.probability.snowfall")


def logistic_snow_probability(
    historical: pd.DataFrame,
    target_month: int,
    deadline_month: int | None = None,
    column: str = "snowfall_sum",
    min_snowfall_inches: float = 0.1,
    enso_adjustment: float = 0.0,
    days_remaining: int | None = None,
) -> dict:
    """Estimate probability of snowfall using logistic approach.

    For contracts like "Will it snow in Austin before March?", computes:
    1. Historical base rate of snow in the target period
    2. Adjusts for current climate state (ENSO, temperature anomaly)
    3. Accounts for time remaining in the contract window

    Args:
        historical: DataFrame with snowfall data.
        target_month: Month(s) to evaluate.
        deadline_month: Last month in contract window (inclusive).
        column: Snowfall column name.
        min_snowfall_inches: Minimum to count as "snow".
        enso_adjustment: Probability adjustment for ENSO (-0.1 to +0.1).
        days_remaining: Days left in contract window.

    Returns:
        Dict with probability and details.
    """
    df = historical.copy()
    df = df.dropna(subset=[column])

    # Determine month range
    if deadline_month is None:
        deadline_month = target_month
    if deadline_month >= target_month:
        months = list(range(target_month, deadline_month + 1))
    else:
        # Wraps across year boundary (e.g., Nov through Feb)
        months = list(range(target_month, 13)) + list(range(1, deadline_month + 1))

    df = df[df.index.month.isin(months)]

    # Compute per-year snow occurrence
    df_copy = df.copy()
    df_copy["year"] = df_copy.index.year
    yearly_snow = df_copy.groupby("year")[column].apply(
        lambda x: int((x >= min_snowfall_inches).any())
    )

    total_years = len(yearly_snow)
    snow_years = int(yearly_snow.sum())

    if total_years == 0:
        return {"probability": None, "error": "insufficient_data"}

    base_rate = snow_years / total_years

    # Adjust for ENSO
    adjusted_prob = np.clip(base_rate + enso_adjustment, 0.01, 0.99)

    # Adjust for time remaining if we're partway through the window
    if days_remaining is not None and days_remaining >= 0:
        # Full window days
        full_window_days = sum(
            __import__("calendar").monthrange(date.today().year, m)[1] for m in months
        )
        if full_window_days > 0 and days_remaining < full_window_days:
            # If snow hasn't happened yet and fewer days remain,
            # probability of NOT snowing goes up
            daily_no_snow = (1 - adjusted_prob) ** (1 / full_window_days)
            adjusted_prob = 1 - daily_no_snow ** days_remaining

    # Confidence interval using Wilson score
    ci_lower, ci_upper = _wilson_confidence_interval(snow_years, total_years, z=1.645)

    return {
        "probability": float(adjusted_prob),
        "base_rate": float(base_rate),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "snow_years": snow_years,
        "total_years": total_years,
        "months_evaluated": months,
        "enso_adjustment": enso_adjustment,
        "days_remaining": days_remaining,
    }


def _wilson_confidence_interval(
    successes: int,
    total: int,
    z: float = 1.645,  # 90% CI
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return 0.0, 1.0

    p_hat = successes / total
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denominator

    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_seasonal_snow(
    historical: pd.DataFrame,
    months: list[int],
    threshold_inches: float = 0.0,
    analog_years: list[int] | None = None,
    n_bootstrap: int = 5000,
    column: str = "snowfall_sum",
) -> dict:
    """Bootstrap simulation for seasonal/cumulative snow contracts.

    Draws random seasons from historical data (optionally weighted toward
    analog years) and computes probability of exceeding a cumulative threshold.
    """
    df = historical.copy()
    df = df.dropna(subset=[column])
    df = df[df.index.month.isin(months)]
    df_copy = df.copy()
    df_copy["year"] = df_copy.index.year

    # Compute seasonal totals per year
    seasonal = df_copy.groupby("year")[column].sum()

    if len(seasonal) < 5:
        return {"probability": None, "error": "insufficient_data"}

    years = seasonal.index.values
    totals = seasonal.values

    # Build sampling weights (upweight analog years if provided)
    weights = np.ones(len(years))
    if analog_years:
        for i, y in enumerate(years):
            if y in analog_years:
                weights[i] = 3.0  # Triple weight for analog years
    weights /= weights.sum()

    # Bootstrap
    rng = np.random.default_rng()
    exceed_count = 0
    sampled_totals = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(years), p=weights)
        sampled_totals.append(totals[idx])
        if totals[idx] > threshold_inches:
            exceed_count += 1

    probability = exceed_count / n_bootstrap
    sampled_arr = np.array(sampled_totals)

    return {
        "probability": float(probability),
        "mean_total": float(sampled_arr.mean()),
        "median_total": float(np.median(sampled_arr)),
        "p10": float(np.percentile(sampled_arr, 10)),
        "p90": float(np.percentile(sampled_arr, 90)),
        "threshold_inches": threshold_inches,
        "n_bootstrap": n_bootstrap,
        "n_years": len(years),
        "used_analog_weights": analog_years is not None,
    }
