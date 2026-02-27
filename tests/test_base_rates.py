"""Tests for historical base rate calculations."""

import numpy as np
import pandas as pd
import pytest

from weather_edge.features.base_rates import (
    compute_climatological_stats,
    fit_temperature_distribution,
    precipitation_exceedance_rate,
    snowfall_occurrence_rate,
    temperature_exceedance_rate,
)


@pytest.fixture
def sample_historical():
    """Create a synthetic historical dataset for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1990-01-01", "2023-12-31", freq="D")
    n = len(dates)

    # Seasonal temperature pattern (Northern Hemisphere)
    day_of_year = dates.dayofyear
    seasonal = 50 + 30 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    noise = rng.normal(0, 8, n)

    df = pd.DataFrame({
        "temperature_2m_max": seasonal + noise,
        "temperature_2m_min": seasonal + noise - 15,
        "precipitation_sum": rng.exponential(0.2, n),
        "snowfall_sum": np.where(
            (seasonal + noise < 35) & (rng.random(n) < 0.3),
            rng.exponential(1.0, n),
            0.0,
        ),
    }, index=dates)
    return df


def test_temperature_exceedance_rate(sample_historical):
    result = temperature_exceedance_rate(
        sample_historical, threshold_f=90, month=7,
    )
    assert 0 < result["base_rate_daily"] < 1
    assert result["total_days"] > 0
    assert result["total_years"] > 0
    assert 0 <= result["base_rate_yearly"] <= 1


def test_temperature_exceedance_rate_extreme_threshold(sample_historical):
    # Very high threshold - should have very low rate
    result = temperature_exceedance_rate(
        sample_historical, threshold_f=120, month=7,
    )
    assert result["base_rate_daily"] < 0.01


def test_temperature_exceedance_rate_low_threshold(sample_historical):
    # Very low threshold - should have very high rate
    result = temperature_exceedance_rate(
        sample_historical, threshold_f=0, month=7,
    )
    assert result["base_rate_daily"] > 0.99


def test_snowfall_occurrence_rate(sample_historical):
    result = snowfall_occurrence_rate(
        sample_historical, month=1,
    )
    assert result["total_days"] > 0
    assert 0 <= result["base_rate_daily"] <= 1
    assert 0 <= result["base_rate_yearly"] <= 1


def test_precipitation_exceedance_rate(sample_historical):
    result = precipitation_exceedance_rate(
        sample_historical, threshold_inches=0.5, month=6,
    )
    assert result["total_days"] > 0
    assert 0 <= result["base_rate_daily"] <= 1


def test_fit_temperature_distribution(sample_historical):
    result = fit_temperature_distribution(sample_historical, month=7)
    assert result["distribution"] in ("normal", "skewnorm", "gev")
    assert result["n_samples"] > 100
    assert "mean" in result
    assert "std" in result
    assert result["std"] > 0


def test_climatological_stats(sample_historical):
    stats = compute_climatological_stats(sample_historical)
    assert len(stats) == 12  # One row per month
    assert "mean" in stats.columns
    assert "std" in stats.columns
    # July should be warmer than January
    assert stats.loc[7, "mean"] > stats.loc[1, "mean"]


def test_empty_data():
    empty = pd.DataFrame(
        {"temperature_2m_max": pd.array([], dtype="float64")},
        index=pd.DatetimeIndex([], name="date"),
    )
    result = temperature_exceedance_rate(empty, threshold_f=100, month=7)
    # Empty data returns the base_rate key from the zero-day branch
    assert result["total_days"] == 0
    assert result.get("base_rate_daily", result.get("base_rate", 0)) == 0.0
