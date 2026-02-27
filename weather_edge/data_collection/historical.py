"""Historical weather data collection from Open-Meteo archive and NOAA."""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger("weather_edge.data_collection.historical")

# Open-Meteo allows max 366 days per request, so chunk multi-year fetches.
_MAX_DAYS_PER_REQUEST = 365


def fetch_open_meteo_historical(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch daily historical weather data from the Open-Meteo archive API.

    Returns a DataFrame indexed by date with columns for each requested variable.
    Automatically chunks requests into <=365-day windows.
    """
    if variables is None:
        variables = [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "snowfall_sum",
            "windspeed_10m_max",
        ]

    base_url = "https://archive-api.open-meteo.com/v1/archive"
    frames: list[pd.DataFrame] = []
    current_start = start_date

    while current_start < end_date:
        chunk_end = min(current_start + timedelta(days=_MAX_DAYS_PER_REQUEST), end_date)
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": current_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "daily": ",".join(variables),
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "timezone": "America/New_York",
        }
        logger.info(
            "Open-Meteo request: lat=%.2f lon=%.2f %s to %s",
            lat, lon, current_start, chunk_end,
        )

        resp = httpx.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        if "time" not in daily:
            logger.warning("No 'time' key in Open-Meteo response, skipping chunk")
            current_start = chunk_end + timedelta(days=1)
            continue

        df = pd.DataFrame(daily)
        df["date"] = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"]).set_index("date")
        frames.append(df)

        current_start = chunk_end + timedelta(days=1)
        time.sleep(0.3)  # Rate-limit courtesy

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def fetch_all_cities_historical(
    config: dict,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch historical data for all configured cities.

    Returns a dict mapping city name -> DataFrame.
    """
    years_back = config["data_collection"]["historical"]["years_back"]
    if end_date is None:
        end_date = date.today() - timedelta(days=5)  # Archive lag
    if start_date is None:
        start_date = end_date.replace(year=end_date.year - years_back)

    variables = config["data_collection"]["historical"]["variables"]
    results = {}

    for city in config["cities"]:
        name = city["name"]
        logger.info("Fetching historical data for %s", name)
        try:
            df = fetch_open_meteo_historical(
                lat=city["lat"],
                lon=city["lon"],
                start_date=start_date,
                end_date=end_date,
                variables=variables,
            )
            results[name] = df
            logger.info("Got %d rows for %s", len(df), name)
        except Exception:
            logger.exception("Failed to fetch historical data for %s", name)

    return results


def save_historical(data: dict[str, pd.DataFrame], data_dir: str | Path) -> None:
    """Persist historical DataFrames as parquet files."""
    out_dir = Path(data_dir) / "historical"
    out_dir.mkdir(parents=True, exist_ok=True)
    for city_name, df in data.items():
        slug = city_name.lower().replace(" ", "_")
        path = out_dir / f"{slug}.parquet"
        df.to_parquet(path)
        logger.info("Saved %s -> %s", city_name, path)


def load_historical(city_name: str, data_dir: str | Path) -> pd.DataFrame:
    """Load previously-saved historical data for a city."""
    slug = city_name.lower().replace(" ", "_")
    path = Path(data_dir) / "historical" / f"{slug}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No historical data at {path}")
    return pd.read_parquet(path)
