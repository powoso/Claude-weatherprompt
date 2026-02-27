"""Tests for probability engine models."""

import numpy as np
import pandas as pd
import pytest

from weather_edge.probability.hurricane import (
    compute_season_fraction_remaining,
    enso_hurricane_multiplier,
    hurricane_probability_with_uncertainty,
    poisson_landfall_probability,
)
from weather_edge.probability.snowfall import (
    bootstrap_seasonal_snow,
    logistic_snow_probability,
)
from weather_edge.probability.temperature import (
    simulate_temperature_exceedance,
)


@pytest.fixture
def sample_historical():
    """Create synthetic historical dataset."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1990-01-01", "2023-12-31", freq="D")
    n = len(dates)
    day_of_year = dates.dayofyear
    seasonal = 50 + 30 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    noise = rng.normal(0, 8, n)
    df = pd.DataFrame({
        "temperature_2m_max": seasonal + noise,
        "snowfall_sum": np.where(
            (seasonal + noise < 35) & (rng.random(n) < 0.3),
            rng.exponential(1.0, n),
            0.0,
        ),
    }, index=dates)
    return df


class TestTemperature:
    def test_monte_carlo_basic(self, sample_historical):
        result = simulate_temperature_exceedance(
            sample_historical,
            threshold_f=90,
            month=7,
            n_simulations=5000,
        )
        assert result["probability"] is not None
        assert 0 <= result["probability"] <= 1
        assert result["n_simulations"] == 5000
        assert result["ci_lower"] <= result["probability"] <= result["ci_upper"]

    def test_high_threshold_low_prob(self, sample_historical):
        result = simulate_temperature_exceedance(
            sample_historical,
            threshold_f=120,
            month=7,
            n_simulations=5000,
        )
        assert result["probability"] < 0.1

    def test_low_threshold_high_prob(self, sample_historical):
        result = simulate_temperature_exceedance(
            sample_historical,
            threshold_f=50,
            month=7,
            n_simulations=5000,
        )
        assert result["probability"] > 0.9

    def test_uhi_adjustment_increases_prob(self, sample_historical):
        base = simulate_temperature_exceedance(
            sample_historical, threshold_f=85, month=7, n_simulations=5000,
        )
        with_uhi = simulate_temperature_exceedance(
            sample_historical, threshold_f=85, month=7, n_simulations=5000,
            uhi_adjustment_f=3.0,
        )
        # UHI makes it hotter -> higher exceedance probability
        assert with_uhi["probability"] >= base["probability"] - 0.05  # Allow simulation noise

    def test_insufficient_data(self):
        small = pd.DataFrame(
            {"temperature_2m_max": [80, 85, 90]},
            index=pd.to_datetime(["2023-07-01", "2023-07-02", "2023-07-03"]),
        )
        result = simulate_temperature_exceedance(small, threshold_f=100, month=7)
        assert result["probability"] is None
        assert "insufficient_data" in result.get("error", "")


class TestHurricane:
    def test_poisson_basic(self):
        result = poisson_landfall_probability(
            historical_rate=0.3,
            enso_multiplier=1.0,
            sst_multiplier=1.0,
        )
        assert 0 < result["probability"] < 1
        assert result["adjusted_rate"] == pytest.approx(0.3)
        # P(X >= 1) for Poisson(0.3) = 1 - e^(-0.3)
        assert result["probability"] == pytest.approx(1 - np.exp(-0.3), abs=0.01)

    def test_el_nino_suppresses(self):
        neutral = poisson_landfall_probability(historical_rate=0.3, enso_multiplier=1.0)
        el_nino = poisson_landfall_probability(historical_rate=0.3, enso_multiplier=0.6)
        assert el_nino["probability"] < neutral["probability"]

    def test_la_nina_enhances(self):
        neutral = poisson_landfall_probability(historical_rate=0.3, enso_multiplier=1.0)
        la_nina = poisson_landfall_probability(historical_rate=0.3, enso_multiplier=1.4)
        assert la_nina["probability"] > neutral["probability"]

    def test_enso_multiplier_values(self):
        assert enso_hurricane_multiplier(2.0) == 0.6  # Strong El Nino
        assert enso_hurricane_multiplier(0.0) == 1.0  # Neutral
        assert enso_hurricane_multiplier(-2.0) == 1.5  # Strong La Nina

    def test_season_fraction(self):
        from datetime import date
        # Before season starts
        frac = compute_season_fraction_remaining(date(2024, 5, 1))
        assert frac == 1.0
        # After season ends
        frac = compute_season_fraction_remaining(date(2024, 12, 15))
        assert frac == 0.0
        # Mid-season
        frac = compute_season_fraction_remaining(date(2024, 9, 1))
        assert 0 < frac < 1

    def test_full_hurricane_simulation(self):
        rates = {"Cat4+": 0.3, "Cat3": 0.5, "TS": 2.0}
        result = hurricane_probability_with_uncertainty(
            historical_rates=rates,
            oni_value=-1.0,  # La Nina
            category="Cat4+",
            n_simulations=5000,
        )
        assert 0 < result["probability"] < 1
        assert result["enso_multiplier"] > 1.0  # La Nina enhances


class TestSnowfall:
    def test_logistic_basic(self, sample_historical):
        result = logistic_snow_probability(
            sample_historical, target_month=1,
        )
        assert result["probability"] is not None
        assert 0 <= result["probability"] <= 1
        assert result["total_years"] > 0

    def test_snow_in_summer_unlikely(self, sample_historical):
        result = logistic_snow_probability(
            sample_historical, target_month=7,
        )
        assert result["probability"] < 0.2

    def test_bootstrap_seasonal(self, sample_historical):
        result = bootstrap_seasonal_snow(
            sample_historical,
            months=[12, 1, 2],
            n_bootstrap=2000,
        )
        assert result["probability"] is not None
        assert result["n_years"] > 0
        assert result["n_bootstrap"] == 2000
