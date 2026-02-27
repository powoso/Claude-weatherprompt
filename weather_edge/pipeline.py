"""Main pipeline orchestrator — ties all components together."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from weather_edge.alerts.notifier import send_alerts
from weather_edge.data_collection.enso import fetch_oni_index, get_current_enso_phase
from weather_edge.data_collection.forecasts import fetch_all_city_forecasts
from weather_edge.data_collection.historical import (
    fetch_all_cities_historical,
    load_historical,
    save_historical,
)
from weather_edge.data_collection.hurricanes import (
    download_hurdat2,
    get_historical_landfall_rates,
    parse_hurdat2,
)
from weather_edge.data_collection.markets import fetch_all_weather_markets
from weather_edge.edge.calibration import CalibrationTracker
from weather_edge.edge.detector import EdgeTracker, detect_edges
from weather_edge.features.analogs import find_analog_years
from weather_edge.probability.engine import evaluate_contract
from weather_edge.utils.config import ensure_dirs, get_city_by_name

logger = logging.getLogger("weather_edge.pipeline")


def run_ingest(config: dict) -> None:
    """Step 1: Download and cache all historical data sources."""
    ensure_dirs(config)
    data_dir = config["general"]["data_dir"]
    cache_dir = config["general"]["cache_dir"]

    # Historical weather
    logger.info("=== Ingesting historical weather data ===")
    historical = fetch_all_cities_historical(config)
    save_historical(historical, data_dir)

    # HURDAT2
    logger.info("=== Ingesting HURDAT2 hurricane data ===")
    hurdat2_url = config["data_collection"]["historical"]["hurdat2_url"]
    raw = download_hurdat2(url=hurdat2_url, cache_dir=cache_dir)
    storms = parse_hurdat2(raw)
    rates = get_historical_landfall_rates(storms)
    logger.info("Hurricane landfall rates: %s", rates)

    # Save rates
    rates_path = Path(data_dir) / "hurricane_rates.parquet"
    pd.DataFrame([rates]).to_parquet(rates_path)

    # ONI / ENSO
    logger.info("=== Ingesting ENSO data ===")
    oni_df = fetch_oni_index()
    oni_path = Path(cache_dir) / "oni_data.parquet"
    oni_df.to_parquet(oni_path)
    current = get_current_enso_phase(oni_df)
    logger.info("Current ENSO phase: %s (ONI=%.2f)", current["phase"], current["anomaly"])

    logger.info("=== Ingest complete ===")


def run_scan(config: dict) -> list:
    """Step 2: Scan markets, evaluate contracts, detect edges."""
    ensure_dirs(config)
    data_dir = config["general"]["data_dir"]
    cache_dir = config["general"]["cache_dir"]

    # Load ENSO data
    oni_path = Path(cache_dir) / "oni_data.parquet"
    oni_df = pd.DataFrame()
    current_oni = 0.0
    if oni_path.exists():
        oni_df = pd.read_parquet(oni_path)
        phase = get_current_enso_phase(oni_df)
        current_oni = phase["anomaly"]

    # Load hurricane rates
    rates_path = Path(data_dir) / "hurricane_rates.parquet"
    hurricane_rates = None
    if rates_path.exists():
        rates_df = pd.read_parquet(rates_path)
        if not rates_df.empty:
            hurricane_rates = rates_df.iloc[0].to_dict()

    # Fetch current market contracts
    logger.info("=== Scanning prediction markets ===")
    contracts = fetch_all_weather_markets(config)
    if not contracts:
        logger.info("No weather contracts found on markets")
        return []

    # Evaluate each contract
    logger.info("=== Evaluating %d contracts ===", len(contracts))
    evaluations = []
    for contract in contracts:
        # Try to match contract to a city for historical data
        historical = None
        city_config = None
        for city in config["cities"]:
            if city["name"].lower() in contract.title.lower():
                city_config = city
                try:
                    historical = load_historical(city["name"], data_dir)
                except FileNotFoundError:
                    pass
                break

        result = evaluate_contract(
            contract=contract,
            historical=historical,
            oni_df=oni_df,
            current_oni=current_oni,
            hurricane_rates=hurricane_rates,
            city_config=city_config,
            config=config,
        )
        evaluations.append(result)

    # Detect edges
    logger.info("=== Detecting edges ===")
    edge_config = config.get("edge", {})
    opportunities = detect_edges(
        evaluations=evaluations,
        market_contracts=contracts,
        min_edge=edge_config.get("min_edge_threshold", 0.05),
        kelly_fraction=edge_config.get("kelly_fraction", 0.5),
        max_position_pct=edge_config.get("max_position_pct", 0.10),
        bankroll=edge_config.get("initial_bankroll", 1000.0),
    )

    # Track edges over time
    tracker = EdgeTracker(Path(data_dir) / "edge_history.parquet")
    tracker.record(opportunities, evaluations)

    # Log predictions for calibration
    cal_tracker = CalibrationTracker(Path(data_dir) / "calibration.parquet")
    for opp in opportunities:
        cal_tracker.record_prediction(
            contract_id=opp.contract_id,
            title=opp.title,
            category=opp.category,
            model_prob=opp.model_prob,
            market_prob=opp.market_prob,
            prediction_date=date.today().isoformat(),
        )

    # Send alerts
    alert_min_edge = config.get("alerts", {}).get("new_opportunity_min_edge", 0.08)
    big_opportunities = [o for o in opportunities if o.edge >= alert_min_edge]
    if big_opportunities:
        send_alerts(big_opportunities, config, "new_opportunity")

    # Print summary
    logger.info("=== Scan Results ===")
    logger.info("Contracts scanned: %d", len(contracts))
    logger.info("Opportunities found: %d", len(opportunities))
    for opp in opportunities:
        logger.info(
            "  [%s] %s | Model: %.1f%% | Market: %.1f%% | Edge: %+.1f%% | Kelly: $%.2f",
            opp.source.upper(),
            opp.title[:60],
            opp.model_prob * 100,
            opp.market_prob * 100,
            opp.edge * 100,
            opp.suggested_bet_size,
        )

    return opportunities


def run_update(config: dict) -> None:
    """Full pipeline: fetch latest forecasts, rerun models, update everything."""
    logger.info("=== Running full update ===")

    # Fetch latest forecasts
    logger.info("Fetching latest NWS forecasts...")
    try:
        forecasts = fetch_all_city_forecasts(config)
        logger.info("Got forecasts for %d cities", len(forecasts))
    except Exception:
        logger.exception("Failed to fetch forecasts")

    # Run market scan
    run_scan(config)

    logger.info("=== Update complete ===")


def run_base_rates(
    config: dict,
    city_name: str,
    metric: str = "temperature_2m_max",
    threshold: float | None = None,
    month: int | None = None,
) -> dict:
    """Compute base rates for a specific city/metric/threshold combo."""
    data_dir = config["general"]["data_dir"]

    try:
        historical = load_historical(city_name, data_dir)
    except FileNotFoundError:
        return {"error": f"No historical data for {city_name}. Run 'ingest' first."}

    from weather_edge.features.base_rates import (
        compute_climatological_stats,
        fit_temperature_distribution,
        temperature_exceedance_rate,
    )

    result = {"city": city_name, "metric": metric}

    # Climatological stats
    result["climatology"] = compute_climatological_stats(historical, metric).to_dict()

    # Distribution fit
    if month:
        result["distribution_fit"] = fit_temperature_distribution(historical, month, metric)

    # Exceedance rate
    if threshold is not None:
        result["exceedance"] = temperature_exceedance_rate(
            historical, threshold, month=month, column=metric,
        )

    return result
