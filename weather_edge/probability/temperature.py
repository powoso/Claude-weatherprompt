"""Temperature threshold probability engine using Monte Carlo simulation."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("weather_edge.probability.temperature")


def simulate_temperature_exceedance(
    historical: pd.DataFrame,
    threshold_f: float,
    month: int | None = None,
    months: list[int] | None = None,
    days_remaining: int | None = None,
    n_simulations: int = 10000,
    column: str = "temperature_2m_max",
    uhi_adjustment_f: float = 0.0,
    forecast_bias_f: float = 0.0,
    enso_adjustment_f: float = 0.0,
) -> dict:
    """Monte Carlo simulation for temperature threshold contracts.

    For contracts like "Will NYC exceed 100F in July?", this:
    1. Fits a distribution to historical daily max temps for the target month(s)
    2. Applies bias corrections (UHI, forecast bias, ENSO)
    3. Simulates `days_remaining` draws from the corrected distribution
    4. Computes the probability that at least one day exceeds the threshold

    Returns dict with probability, confidence interval, and simulation details.
    """
    df = historical.copy()
    df = df.dropna(subset=[column])

    if months:
        df = df[df.index.month.isin(months)]
    elif month:
        df = df[df.index.month == month]

    values = df[column].values
    if len(values) < 30:
        return {"probability": None, "error": "insufficient_data", "n_samples": len(values)}

    # Fit distribution with bias corrections
    adjusted_values = values + uhi_adjustment_f + forecast_bias_f + enso_adjustment_f

    # Try skew-normal first, fall back to normal
    try:
        a, loc, scale = stats.skewnorm.fit(adjusted_values)
        dist = stats.skewnorm(a, loc=loc, scale=scale)
        dist_name = "skewnorm"
    except Exception:
        mu, sigma = stats.norm.fit(adjusted_values)
        dist = stats.norm(loc=mu, scale=sigma)
        dist_name = "normal"

    # Determine simulation days
    if days_remaining is None:
        if month:
            # Full month
            import calendar
            days_remaining = calendar.monthrange(date.today().year, month)[1]
        elif months:
            days_remaining = sum(
                __import__("calendar").monthrange(date.today().year, m)[1] for m in months
            )
        else:
            days_remaining = 30

    # Monte Carlo: for each simulation, draw `days_remaining` temperatures
    # and check if any exceed the threshold
    rng = np.random.default_rng()
    exceedance_count = 0

    # Vectorized approach for efficiency
    simulated = dist.rvs(size=(n_simulations, days_remaining), random_state=rng)
    max_temps = simulated.max(axis=1)
    exceedance_count = int((max_temps >= threshold_f).sum())

    probability = exceedance_count / n_simulations

    # Bootstrap confidence interval
    ci_lower, ci_upper = _bootstrap_ci(max_temps, threshold_f, n_boot=2000)

    return {
        "probability": probability,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "distribution": dist_name,
        "dist_params": {"loc": float(dist.mean()), "scale": float(dist.std())},
        "days_simulated": days_remaining,
        "n_simulations": n_simulations,
        "threshold_f": threshold_f,
        "adjustments": {
            "uhi": uhi_adjustment_f,
            "forecast_bias": forecast_bias_f,
            "enso": enso_adjustment_f,
        },
        "historical_samples": len(values),
    }


def _bootstrap_ci(
    max_temps: np.ndarray,
    threshold: float,
    n_boot: int = 2000,
    ci: float = 0.90,
) -> tuple[float, float]:
    """Bootstrap confidence interval for exceedance probability."""
    rng = np.random.default_rng(42)
    probs = []
    n = len(max_temps)
    for _ in range(n_boot):
        sample = rng.choice(max_temps, size=n, replace=True)
        probs.append((sample >= threshold).mean())
    alpha = (1 - ci) / 2
    lower = float(np.percentile(probs, 100 * alpha))
    upper = float(np.percentile(probs, 100 * (1 - alpha)))
    return lower, upper


def estimate_enso_temperature_effect(
    historical: pd.DataFrame,
    oni_df: pd.DataFrame,
    month: int,
    current_oni: float,
    column: str = "temperature_2m_max",
) -> float:
    """Estimate the ENSO-driven temperature adjustment.

    Computes the regression of monthly mean temperature on ONI values,
    then uses the current ONI to predict the temperature deviation.
    """
    if oni_df.empty:
        return 0.0

    df = historical.copy()
    df = df.dropna(subset=[column])
    df = df[df.index.month == month]

    # Monthly means per year
    yearly_means = df.groupby(df.index.year)[column].mean().reset_index()
    yearly_means.columns = ["year", "temp_mean"]

    # Match with DJF ONI (standard for ENSO classification)
    djf = oni_df[oni_df["season"] == "DJF"][["year", "anomaly"]].copy()
    merged = yearly_means.merge(djf, on="year", how="inner")

    if len(merged) < 10:
        return 0.0

    # Linear regression: temp = a + b * ONI
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        merged["anomaly"], merged["temp_mean"]
    )

    if p_value > 0.1:
        # Not statistically significant
        return 0.0

    # Predicted deviation from mean based on current ONI
    climo_mean = merged["temp_mean"].mean()
    predicted = intercept + slope * current_oni
    adjustment = predicted - climo_mean

    logger.info(
        "ENSO temp effect for month %d: ONI=%.2f -> adjustment=%.2f°F (r²=%.3f, p=%.4f)",
        month, current_oni, adjustment, r_value**2, p_value,
    )
    return float(adjustment)
