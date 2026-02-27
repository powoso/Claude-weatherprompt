"""Current weather forecast collection from NWS and GFS."""

from __future__ import annotations

import logging
from datetime import datetime

import httpx
import pandas as pd

logger = logging.getLogger("weather_edge.data_collection.forecasts")


def fetch_nws_forecast(
    office: str,
    grid_x: int,
    grid_y: int,
    user_agent: str = "WeatherEdge/1.0 (weather-edge@example.com)",
) -> pd.DataFrame:
    """Fetch 7-day forecast from the NWS API for a given gridpoint.

    Returns a DataFrame with period-level forecast data.
    """
    url = f"https://api.weather.gov/gridpoints/{office}/{grid_x},{grid_y}/forecast"
    headers = {"User-Agent": user_agent, "Accept": "application/geo+json"}

    logger.info("Fetching NWS forecast: %s", url)
    resp = httpx.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    periods = data.get("properties", {}).get("periods", [])
    if not periods:
        logger.warning("No forecast periods returned for %s/%s,%s", office, grid_x, grid_y)
        return pd.DataFrame()

    rows = []
    for p in periods:
        rows.append({
            "name": p.get("name"),
            "start_time": p.get("startTime"),
            "end_time": p.get("endTime"),
            "is_daytime": p.get("isDaytime"),
            "temperature_f": p.get("temperature"),
            "wind_speed": p.get("windSpeed"),
            "wind_direction": p.get("windDirection"),
            "short_forecast": p.get("shortForecast"),
            "detailed_forecast": p.get("detailedForecast"),
            "precip_probability": p.get("probabilityOfPrecipitation", {}).get("value"),
        })
    return pd.DataFrame(rows)


def fetch_nws_hourly_forecast(
    office: str,
    grid_x: int,
    grid_y: int,
    user_agent: str = "WeatherEdge/1.0 (weather-edge@example.com)",
) -> pd.DataFrame:
    """Fetch hourly forecast from NWS for finer-grained temperature data."""
    url = f"https://api.weather.gov/gridpoints/{office}/{grid_x},{grid_y}/forecast/hourly"
    headers = {"User-Agent": user_agent, "Accept": "application/geo+json"}

    logger.info("Fetching NWS hourly forecast: %s", url)
    resp = httpx.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    periods = data.get("properties", {}).get("periods", [])
    if not periods:
        return pd.DataFrame()

    rows = []
    for p in periods:
        rows.append({
            "start_time": pd.to_datetime(p.get("startTime")),
            "temperature_f": p.get("temperature"),
            "wind_speed": p.get("windSpeed"),
            "wind_direction": p.get("windDirection"),
            "short_forecast": p.get("shortForecast"),
            "precip_probability": p.get("probabilityOfPrecipitation", {}).get("value"),
            "humidity_pct": p.get("relativeHumidity", {}).get("value"),
        })
    return pd.DataFrame(rows)


def fetch_city_forecast(city: dict, config: dict) -> dict:
    """Fetch both 7-day and hourly forecasts for a configured city.

    Returns dict with keys 'seven_day' and 'hourly', each a DataFrame.
    """
    office = city["nws_office"]
    gx, gy = city["nws_gridpoint"].split(",")
    user_agent = config["data_collection"]["forecasts"]["nws_user_agent"]

    result = {}
    try:
        result["seven_day"] = fetch_nws_forecast(office, int(gx), int(gy), user_agent)
    except Exception:
        logger.exception("Failed to fetch 7-day forecast for %s", city["name"])
        result["seven_day"] = pd.DataFrame()

    try:
        result["hourly"] = fetch_nws_hourly_forecast(office, int(gx), int(gy), user_agent)
    except Exception:
        logger.exception("Failed to fetch hourly forecast for %s", city["name"])
        result["hourly"] = pd.DataFrame()

    return result


def fetch_all_city_forecasts(config: dict) -> dict[str, dict]:
    """Fetch forecasts for all configured cities.

    Returns dict: city_name -> {seven_day: DataFrame, hourly: DataFrame}.
    """
    results = {}
    for city in config["cities"]:
        logger.info("Fetching forecasts for %s", city["name"])
        results[city["name"]] = fetch_city_forecast(city, config)
    return results
