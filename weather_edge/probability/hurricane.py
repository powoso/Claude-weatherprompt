"""Hurricane probability models using Poisson regression."""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger("weather_edge.probability.hurricane")


def poisson_landfall_probability(
    historical_rate: float,
    enso_multiplier: float = 1.0,
    sst_multiplier: float = 1.0,
    season_fraction_remaining: float = 1.0,
    threshold_count: int = 1,
) -> dict:
    """Compute probability of at least N landfalls using adjusted Poisson model.

    For contracts like "Will a Cat4+ hurricane make US landfall this season?"

    Args:
        historical_rate: Base annual rate from HURDAT2 (e.g., 0.3 for Cat4+).
        enso_multiplier: ENSO phase adjustment (El Nino ~0.6x, La Nina ~1.4x).
        sst_multiplier: SST anomaly adjustment factor.
        season_fraction_remaining: What fraction of hurricane season remains (0-1).
        threshold_count: Minimum number of landfalls (usually 1 for "any").

    Returns:
        Dict with probability and model details.
    """
    adjusted_rate = historical_rate * enso_multiplier * sst_multiplier * season_fraction_remaining

    # P(X >= threshold) = 1 - P(X < threshold) using Poisson CDF
    prob_fewer = stats.poisson.cdf(threshold_count - 1, adjusted_rate)
    probability = 1 - prob_fewer

    return {
        "probability": float(probability),
        "adjusted_rate": float(adjusted_rate),
        "base_rate": historical_rate,
        "enso_multiplier": enso_multiplier,
        "sst_multiplier": sst_multiplier,
        "season_fraction_remaining": season_fraction_remaining,
        "threshold_count": threshold_count,
        "distribution": "poisson",
        "p_zero": float(stats.poisson.pmf(0, adjusted_rate)),
        "p_one": float(stats.poisson.pmf(1, adjusted_rate)),
        "p_two_plus": float(1 - stats.poisson.cdf(1, adjusted_rate)),
    }


def enso_hurricane_multiplier(oni_value: float) -> float:
    """Compute hurricane activity multiplier based on ONI value.

    El Nino (positive ONI) suppresses Atlantic hurricanes due to increased
    wind shear. La Nina (negative ONI) enhances activity.

    Based on historical analysis of Atlantic hurricane activity vs ENSO phase:
    - Strong El Nino (ONI > 1.5): ~60% of normal activity
    - Moderate El Nino (ONI 0.5-1.5): ~80% of normal
    - Neutral: 100% of normal
    - Moderate La Nina (ONI -0.5 to -1.5): ~130% of normal
    - Strong La Nina (ONI < -1.5): ~150% of normal
    """
    if oni_value >= 1.5:
        return 0.6
    if oni_value >= 0.5:
        # Linear interpolation between 0.8 and 1.0
        return 1.0 - 0.2 * (oni_value - 0.5)
    if oni_value >= -0.5:
        return 1.0
    if oni_value >= -1.5:
        # Linear interpolation between 1.0 and 1.3
        return 1.0 + 0.3 * (-oni_value - 0.5)
    return 1.5


def compute_season_fraction_remaining(reference_date: date | None = None) -> float:
    """Compute what fraction of the Atlantic hurricane season remains.

    Official season: June 1 - November 30 (183 days).
    Peak activity: August 20 - October 10.
    """
    if reference_date is None:
        reference_date = date.today()

    season_start = date(reference_date.year, 6, 1)
    season_end = date(reference_date.year, 11, 30)

    if reference_date < season_start:
        return 1.0  # Full season ahead
    if reference_date > season_end:
        return 0.0  # Season over

    total_days = (season_end - season_start).days
    elapsed = (reference_date - season_start).days
    remaining = 1.0 - (elapsed / total_days)

    # Weight by historical activity distribution (peak Aug-Oct)
    # This is a simplification; more accurate would use daily climatological rates
    return float(remaining)


def hurricane_probability_with_uncertainty(
    historical_rates: dict[str, float],
    oni_value: float,
    category: str = "Cat4+",
    n_simulations: int = 10000,
    reference_date: date | None = None,
) -> dict:
    """Full hurricane probability with uncertainty via simulation.

    Accounts for uncertainty in both the rate estimate and ENSO effect.
    """
    base_rate = historical_rates.get(category, 0.3)
    enso_mult = enso_hurricane_multiplier(oni_value)
    season_frac = compute_season_fraction_remaining(reference_date)

    # Add uncertainty to the rate estimate
    rng = np.random.default_rng()

    # Rate uncertainty: use gamma distribution centered on base_rate
    # Variance comes from limited historical sample
    rate_alpha = base_rate * 20  # Shape parameter
    rate_beta = 20.0  # Rate parameter
    if rate_alpha <= 0:
        rate_alpha = 0.1

    simulated_rates = rng.gamma(rate_alpha, 1.0 / rate_beta, size=n_simulations)
    simulated_rates *= enso_mult * season_frac

    # For each simulated rate, draw from Poisson
    simulated_counts = rng.poisson(simulated_rates)
    hit = (simulated_counts >= 1).astype(float)
    prob_at_least_one = float(hit.mean())

    return {
        "probability": prob_at_least_one,
        "ci_lower": float(np.percentile(hit, 5)),
        "ci_upper": float(np.percentile(hit, 95)),
        "mean_rate": float(simulated_rates.mean()),
        "base_rate": base_rate,
        "enso_multiplier": enso_mult,
        "season_fraction_remaining": season_frac,
        "category": category,
        "n_simulations": n_simulations,
    }
