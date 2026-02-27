"""Unified probability engine that dispatches to contract-specific models."""

from __future__ import annotations

import logging
import re
from datetime import date

import pandas as pd

from weather_edge.data_collection.markets import MarketContract
from weather_edge.features.engineering import compute_time_decay
from weather_edge.probability.hurricane import (
    enso_hurricane_multiplier,
    hurricane_probability_with_uncertainty,
    poisson_landfall_probability,
)
from weather_edge.probability.snowfall import (
    bootstrap_seasonal_snow,
    logistic_snow_probability,
)
from weather_edge.probability.temperature import (
    estimate_enso_temperature_effect,
    simulate_temperature_exceedance,
)

logger = logging.getLogger("weather_edge.probability.engine")


def evaluate_contract(
    contract: MarketContract,
    historical: pd.DataFrame | None = None,
    oni_df: pd.DataFrame | None = None,
    current_oni: float = 0.0,
    hurricane_rates: dict[str, float] | None = None,
    city_config: dict | None = None,
    config: dict | None = None,
) -> dict:
    """Evaluate a market contract and produce a model probability.

    This is the main entry point that dispatches to the appropriate
    probability model based on contract category and content.
    """
    category = contract.category
    title_lower = contract.title.lower()
    result = {"contract_id": contract.market_id, "title": contract.title, "category": category}

    # Time decay
    time_info = compute_time_decay(contract.end_date)
    result["time_decay"] = time_info

    n_sims = 10000
    if config:
        n_sims = config.get("probability", {}).get("monte_carlo_simulations", 10000)

    try:
        if category == "temperature":
            result.update(_evaluate_temperature(
                contract, historical, oni_df, current_oni, city_config, n_sims
            ))
        elif category == "hurricane":
            result.update(_evaluate_hurricane(
                contract, hurricane_rates, current_oni
            ))
        elif category == "snowfall":
            result.update(_evaluate_snowfall(
                contract, historical, current_oni, time_info
            ))
        elif category == "precipitation":
            result.update(_evaluate_precipitation(
                contract, historical, time_info
            ))
        else:
            result["probability"] = None
            result["error"] = f"unsupported_category: {category}"
    except Exception:
        logger.exception("Error evaluating contract %s", contract.market_id)
        result["probability"] = None
        result["error"] = "evaluation_failed"

    return result


def _evaluate_temperature(
    contract: MarketContract,
    historical: pd.DataFrame | None,
    oni_df: pd.DataFrame | None,
    current_oni: float,
    city_config: dict | None,
    n_simulations: int,
) -> dict:
    """Evaluate a temperature threshold contract."""
    if historical is None or historical.empty:
        return {"probability": None, "error": "no_historical_data"}

    # Parse threshold from title
    threshold = _extract_temperature_threshold(contract.title)
    month = _extract_month(contract.title)
    months = _extract_months(contract.title)

    if threshold is None:
        return {"probability": None, "error": "could_not_parse_threshold"}

    # ENSO temperature adjustment
    enso_adj = 0.0
    if oni_df is not None and month is not None:
        enso_adj = estimate_enso_temperature_effect(
            historical, oni_df, month, current_oni
        )

    # UHI adjustment
    uhi = 0.0
    if city_config:
        uhi = city_config.get("uhi_adjustment_f", 0.0)

    # Time remaining
    time_info = compute_time_decay(contract.end_date)
    days_remaining = time_info.get("days_remaining")

    return simulate_temperature_exceedance(
        historical=historical,
        threshold_f=threshold,
        month=month,
        months=months,
        days_remaining=days_remaining,
        n_simulations=n_simulations,
        uhi_adjustment_f=uhi,
        enso_adjustment_f=enso_adj,
    )


def _evaluate_hurricane(
    contract: MarketContract,
    hurricane_rates: dict[str, float] | None,
    current_oni: float,
) -> dict:
    """Evaluate a hurricane contract."""
    if hurricane_rates is None:
        return {"probability": None, "error": "no_hurricane_data"}

    # Determine category from title
    title_lower = contract.title.lower()
    if "cat 5" in title_lower or "category 5" in title_lower:
        category = "Cat5"
    elif "cat 4" in title_lower or "category 4" in title_lower or "major" in title_lower:
        category = "Cat4+"
    elif "cat 3" in title_lower or "category 3" in title_lower:
        category = "Cat3"
    else:
        category = "Cat4+"  # Default for "major hurricane" contracts

    return hurricane_probability_with_uncertainty(
        historical_rates=hurricane_rates,
        oni_value=current_oni,
        category=category,
    )


def _evaluate_snowfall(
    contract: MarketContract,
    historical: pd.DataFrame | None,
    current_oni: float,
    time_info: dict,
) -> dict:
    """Evaluate a snowfall contract."""
    if historical is None or historical.empty:
        return {"probability": None, "error": "no_historical_data"}

    month = _extract_month(contract.title)
    deadline_month = _extract_deadline_month(contract.title)

    # ENSO effect on snow: La Nina generally means colder/more snow in south,
    # El Nino means warmer/less snow
    enso_adj = 0.0
    if current_oni >= 0.5:
        enso_adj = -0.05  # El Nino reduces snow probability
    elif current_oni <= -0.5:
        enso_adj = 0.05  # La Nina increases snow probability

    return logistic_snow_probability(
        historical=historical,
        target_month=month or 12,
        deadline_month=deadline_month,
        enso_adjustment=enso_adj,
        days_remaining=time_info.get("days_remaining"),
    )


def _evaluate_precipitation(
    contract: MarketContract,
    historical: pd.DataFrame | None,
    time_info: dict,
) -> dict:
    """Evaluate a precipitation contract (basic implementation)."""
    if historical is None or historical.empty:
        return {"probability": None, "error": "no_historical_data"}

    # For now, use historical base rates
    month = _extract_month(contract.title)
    df = historical.dropna(subset=["precipitation_sum"])
    if month:
        df = df[df.index.month == month]

    if df.empty:
        return {"probability": None, "error": "no_data_for_period"}

    # Simple threshold extraction and base rate
    threshold = _extract_precipitation_threshold(contract.title)
    if threshold is None:
        threshold = 0.1  # Default: any measurable precipitation

    rate = (df["precipitation_sum"] >= threshold).mean()
    return {
        "probability": float(rate),
        "base_rate": float(rate),
        "threshold_inches": threshold,
        "n_days": len(df),
        "method": "historical_base_rate",
    }


# --- Parsers for contract titles ---

def _extract_temperature_threshold(title: str) -> float | None:
    """Extract temperature threshold from contract title."""
    patterns = [
        r"(\d+)\s*°?\s*[Ff]",
        r"exceed\s+(\d+)",
        r"above\s+(\d+)",
        r"over\s+(\d+)",
        r"reach\s+(\d+)",
        r"hit\s+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _extract_month(title: str) -> int | None:
    """Extract a month reference from contract title."""
    month_names = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    title_lower = title.lower()
    for name, num in month_names.items():
        if name in title_lower:
            return num
    return None


def _extract_months(title: str) -> list[int] | None:
    """Extract multiple month references (e.g., 'June through August')."""
    month = _extract_month(title)
    if month is None:
        return None

    # Look for range patterns
    title_lower = title.lower()
    range_patterns = [
        r"(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+(?:through|to|thru|-)\s+"
        r"(january|february|march|april|may|june|july|august|september|october|november|december)",
    ]
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    for pattern in range_patterns:
        match = re.search(pattern, title_lower)
        if match:
            start = month_map[match.group(1)]
            end = month_map[match.group(2)]
            if start <= end:
                return list(range(start, end + 1))
            return list(range(start, 13)) + list(range(1, end + 1))

    return None


def _extract_deadline_month(title: str) -> int | None:
    """Extract a deadline month (e.g., 'before March' -> 3)."""
    patterns = [
        r"before\s+(january|february|march|april|may|june|july|august|september|october|november|december)",
        r"by\s+(january|february|march|april|may|june|july|august|september|october|november|december)",
    ]
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    title_lower = title.lower()
    for pattern in patterns:
        match = re.search(pattern, title_lower)
        if match:
            return month_map.get(match.group(1))
    return _extract_month(title)


def _extract_precipitation_threshold(title: str) -> float | None:
    """Extract precipitation threshold from title."""
    patterns = [
        r"(\d+\.?\d*)\s*(?:inch|in\.?|\")",
        r"exceed\s+(\d+\.?\d*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None
