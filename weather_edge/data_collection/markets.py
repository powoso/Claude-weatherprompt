"""Prediction market data collection from Polymarket and Kalshi."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime

import httpx

logger = logging.getLogger("weather_edge.data_collection.markets")

WEATHER_KEYWORDS = [
    "temperature", "heat", "cold", "snow", "hurricane", "tropical storm",
    "rainfall", "precipitation", "tornado", "freeze", "frost", "blizzard",
    "wind", "drought", "flood", "weather", "climate", "el nino", "la nina",
    "celsius", "fahrenheit", "degrees",
]


@dataclass
class MarketContract:
    """Represents a single prediction market contract."""
    source: str  # "polymarket" or "kalshi"
    market_id: str
    title: str
    description: str
    market_price: float  # Current implied probability (0-1)
    volume: float
    end_date: str
    category: str  # inferred contract type
    raw: dict  # full API response for this contract


def fetch_polymarket_weather_markets() -> list[MarketContract]:
    """Fetch weather-related markets from Polymarket's Gamma API.

    Polymarket's public API exposes market data without authentication.
    We search for markets matching weather-related keywords.
    """
    base_url = "https://gamma-api.polymarket.com/markets"
    contracts: list[MarketContract] = []

    for keyword in ["weather", "temperature", "hurricane", "snow", "heat"]:
        try:
            params = {
                "tag": keyword,
                "closed": "false",
                "limit": 50,
            }
            resp = httpx.get(base_url, params=params, timeout=30)
            if resp.status_code != 200:
                # Try text search fallback
                params = {"search": keyword, "closed": "false", "limit": 50}
                resp = httpx.get(base_url, params=params, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                markets = data if isinstance(data, list) else data.get("markets", data.get("data", []))
                for m in markets:
                    if not _is_weather_market(m.get("question", "") + " " + m.get("description", "")):
                        continue
                    price = _extract_polymarket_price(m)
                    contracts.append(MarketContract(
                        source="polymarket",
                        market_id=str(m.get("id", m.get("conditionId", ""))),
                        title=m.get("question", ""),
                        description=m.get("description", ""),
                        market_price=price,
                        volume=float(m.get("volume", 0) or 0),
                        end_date=m.get("endDate", m.get("end_date_iso", "")),
                        category=_classify_contract(m.get("question", "")),
                        raw=m,
                    ))
        except Exception:
            logger.exception("Error fetching Polymarket keyword=%s", keyword)

    # Deduplicate by market_id
    seen = set()
    unique = []
    for c in contracts:
        if c.market_id not in seen:
            seen.add(c.market_id)
            unique.append(c)
    return unique


def _extract_polymarket_price(market: dict) -> float:
    """Extract best available price/probability from a Polymarket market."""
    # Try various fields
    for field in ("outcomePrices", "outcome_prices"):
        prices = market.get(field)
        if prices:
            if isinstance(prices, str):
                # Sometimes returned as JSON string like "[0.65, 0.35]"
                try:
                    import json
                    prices = json.loads(prices)
                except (json.JSONDecodeError, TypeError):
                    continue
            if isinstance(prices, list) and len(prices) > 0:
                return float(prices[0])
    # Fallback to best bid/ask midpoint
    best_bid = market.get("bestBid", market.get("best_bid"))
    best_ask = market.get("bestAsk", market.get("best_ask"))
    if best_bid is not None and best_ask is not None:
        return (float(best_bid) + float(best_ask)) / 2
    return 0.5  # Unknown, default to 50%


def fetch_kalshi_weather_markets(config: dict) -> list[MarketContract]:
    """Fetch weather-related markets from Kalshi.

    Kalshi's public API provides event and market data. Some endpoints
    require authentication for trading but market browsing is often public.
    """
    base_url = config["data_collection"]["markets"]["kalshi"]["api_base"]
    contracts: list[MarketContract] = []

    try:
        # Kalshi organizes by events; search for weather-related events
        url = f"{base_url}/events"
        params = {"status": "open", "limit": 100}
        headers = {"Accept": "application/json"}

        # Add auth if available
        api_key = config["data_collection"]["markets"]["kalshi"].get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = httpx.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code != 200:
            logger.warning("Kalshi events endpoint returned %d", resp.status_code)
            return contracts

        data = resp.json()
        events = data.get("events", [])

        for event in events:
            event_title = event.get("title", "") + " " + event.get("sub_title", "")
            if not _is_weather_market(event_title):
                continue

            # Get markets within this event
            for market in event.get("markets", []):
                price = float(market.get("last_price", 0.5) or 0.5) / 100  # Kalshi uses cents
                contracts.append(MarketContract(
                    source="kalshi",
                    market_id=market.get("ticker", ""),
                    title=market.get("title", event.get("title", "")),
                    description=market.get("subtitle", ""),
                    market_price=price,
                    volume=float(market.get("volume", 0) or 0),
                    end_date=market.get("close_time", market.get("expiration_time", "")),
                    category=_classify_contract(market.get("title", event.get("title", ""))),
                    raw=market,
                ))
    except Exception:
        logger.exception("Error fetching Kalshi markets")

    return contracts


def fetch_all_weather_markets(config: dict) -> list[MarketContract]:
    """Fetch weather contracts from all configured market sources."""
    contracts: list[MarketContract] = []

    logger.info("Scanning Polymarket for weather markets...")
    contracts.extend(fetch_polymarket_weather_markets())
    logger.info("Found %d Polymarket weather contracts", len(contracts))

    logger.info("Scanning Kalshi for weather markets...")
    kalshi = fetch_kalshi_weather_markets(config)
    contracts.extend(kalshi)
    logger.info("Found %d Kalshi weather contracts", len(kalshi))

    logger.info("Total weather contracts found: %d", len(contracts))
    return contracts


def _is_weather_market(text: str) -> bool:
    """Check if a market title/description is weather-related."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in WEATHER_KEYWORDS)


def _classify_contract(title: str) -> str:
    """Classify a weather contract into a category type."""
    title_lower = title.lower()
    if any(w in title_lower for w in ("hurricane", "tropical storm", "cyclone")):
        return "hurricane"
    if any(w in title_lower for w in ("snow", "blizzard", "ice storm")):
        return "snowfall"
    if any(w in title_lower for w in ("temperature", "heat", "cold", "degrees", "°f", "°c", "freeze")):
        return "temperature"
    if any(w in title_lower for w in ("rain", "precipitation", "flood", "drought")):
        return "precipitation"
    if any(w in title_lower for w in ("tornado", "wind")):
        return "severe_weather"
    return "other_weather"
