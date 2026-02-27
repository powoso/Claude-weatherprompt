"""Flask web application for Weather Edge dashboard."""

from __future__ import annotations

import calendar
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from weather_edge.edge.calibration import CalibrationTracker
from weather_edge.edge.detector import EdgeTracker
from weather_edge.utils.config import load_config

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

_config: dict | None = None


def get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    config = get_config()
    cities = [c["name"] for c in config["cities"]]
    return render_template("index.html", cities=cities, config=config)


# ---------------------------------------------------------------------------
# API: Opportunities
# ---------------------------------------------------------------------------

@app.route("/api/opportunities")
def api_opportunities():
    config = get_config()
    data_dir = config["general"]["data_dir"]
    tracker = EdgeTracker(Path(data_dir) / "edge_history.parquet")

    try:
        history = tracker.load()
    except Exception:
        history = pd.DataFrame()

    if history.empty:
        return jsonify(_sample_opportunities())

    latest_ts = history["timestamp"].max()
    latest = history[history["timestamp"] == latest_ts].copy()

    opps = []
    for _, row in latest.iterrows():
        opps.append({
            "contract_id": row.get("contract_id", ""),
            "title": row.get("title", ""),
            "model_prob": _safe_float(row.get("model_prob")),
            "market_prob": _safe_float(row.get("market_prob")),
            "edge": _safe_float(row.get("edge")),
            "days_remaining": _safe_int(row.get("days_remaining")),
        })

    return jsonify({
        "opportunities": opps,
        "last_scan": latest_ts,
        "count": len(opps),
        "is_sample": False,
    })


@app.route("/api/convergence/<contract_id>")
def api_convergence(contract_id):
    config = get_config()
    data_dir = config["general"]["data_dir"]
    tracker = EdgeTracker(Path(data_dir) / "edge_history.parquet")

    try:
        data = tracker.get_convergence_data(contract_id)
    except Exception:
        data = pd.DataFrame()

    if data.empty:
        return jsonify({"timestamps": [], "model_prob": [], "market_prob": []})

    return jsonify({
        "timestamps": data["timestamp"].tolist(),
        "model_prob": data["model_prob"].tolist(),
        "market_prob": data["market_prob"].tolist(),
    })


# ---------------------------------------------------------------------------
# API: Base Rates
# ---------------------------------------------------------------------------

@app.route("/api/base-rates")
def api_base_rates():
    config = get_config()
    city_name = request.args.get("city", "New York City")
    metric = request.args.get("metric", "temperature_2m_max")
    month = int(request.args.get("month", 7))

    data_dir = config["general"]["data_dir"]
    slug = city_name.lower().replace(" ", "_")
    hist_path = Path(data_dir) / "historical" / f"{slug}.parquet"

    if not hist_path.exists():
        return jsonify({"error": "no_data", "city": city_name})

    df = pd.read_parquet(hist_path)
    df_month = df[df.index.month == month]

    if metric not in df_month.columns or df_month[metric].dropna().empty:
        return jsonify({"error": "no_metric_data", "city": city_name, "metric": metric})

    values = df_month[metric].dropna()
    month_name = calendar.month_name[month]

    # Histogram data
    counts, bin_edges = np.histogram(values, bins=40)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()

    # Threshold exceedance rates
    if "temperature" in metric:
        thresholds = list(range(70, 120, 5))
    elif "snowfall" in metric:
        thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    else:
        thresholds = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]

    rates = []
    for t in thresholds:
        daily_rate = float((values >= t).mean())
        yearly = df_month.groupby(df_month.index.year)[metric].max()
        yearly_rate = float((yearly >= t).mean())
        rates.append({
            "threshold": t,
            "daily_rate": daily_rate,
            "yearly_rate": yearly_rate,
        })

    # Year-by-year averages
    yearly_avg = df_month.groupby(df_month.index.year)[metric].mean()

    # Stats
    stats = {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "count": int(len(values)),
    }

    return jsonify({
        "city": city_name,
        "metric": metric,
        "month": month,
        "month_name": month_name,
        "histogram": {"bins": bin_centers, "counts": counts.tolist()},
        "thresholds": rates,
        "yearly_avg": {"years": yearly_avg.index.tolist(), "values": yearly_avg.values.tolist()},
        "stats": stats,
    })


# ---------------------------------------------------------------------------
# API: ENSO
# ---------------------------------------------------------------------------

@app.route("/api/enso")
def api_enso():
    config = get_config()
    cache_dir = config["general"]["cache_dir"]
    enso_path = Path(cache_dir) / "oni_data.parquet"

    if not enso_path.exists():
        return jsonify({"error": "no_data"})

    oni_df = pd.read_parquet(enso_path)
    last = oni_df.iloc[-1]

    # Trim to last ~50 years for chart
    recent = oni_df.tail(200)

    return jsonify({
        "current": {
            "phase": str(last.get("oni_phase", "unknown")),
            "anomaly": float(last.get("anomaly", 0)),
            "season": str(last.get("season", "")),
            "year": int(last.get("year", 0)),
        },
        "history": {
            "labels": [f"{r['season']} {int(r['year'])}" for _, r in recent.iterrows()],
            "values": recent["anomaly"].tolist(),
        },
    })


# ---------------------------------------------------------------------------
# API: Calibration
# ---------------------------------------------------------------------------

@app.route("/api/calibration")
def api_calibration():
    config = get_config()
    data_dir = config["general"]["data_dir"]
    tracker = CalibrationTracker(Path(data_dir) / "calibration.parquet")

    try:
        records = tracker.load()
    except Exception:
        records = pd.DataFrame()

    resolved = records.dropna(subset=["outcome"]) if not records.empty else pd.DataFrame()

    brier = tracker.compute_brier_score()
    roi = tracker.compute_roi()

    if resolved.empty:
        # Return sample calibration data
        bins = np.arange(0.05, 1.0, 0.1).tolist()
        actual = [b * 0.9 + 0.02 for b in bins]
        return jsonify({
            "is_sample": True,
            "calibration": {"predicted": bins, "actual": actual, "counts": [12] * len(bins)},
            "brier_score": None,
            "roi": roi,
            "total_predictions": len(records),
            "resolved_predictions": 0,
        })

    cal = tracker.compute_calibration()
    cal_data = {
        "predicted": cal["predicted_mean"].tolist() if not cal.empty else [],
        "actual": cal["actual_rate"].tolist() if not cal.empty else [],
        "counts": cal["count"].tolist() if not cal.empty else [],
    }

    return jsonify({
        "is_sample": False,
        "calibration": cal_data,
        "brier_score": brier,
        "roi": roi,
        "total_predictions": len(records),
        "resolved_predictions": len(resolved),
    })


# ---------------------------------------------------------------------------
# API: Bankroll
# ---------------------------------------------------------------------------

@app.route("/api/bankroll")
def api_bankroll():
    config = get_config()
    edge_config = config.get("edge", {})
    data_dir = config["general"]["data_dir"]

    initial = edge_config.get("initial_bankroll", 1000.0)
    tracker = CalibrationTracker(Path(data_dir) / "calibration.parquet")

    try:
        roi = tracker.compute_roi()
    except Exception:
        roi = {"total_wagered": 0, "total_pnl": 0, "roi_pct": 0, "n_bets": 0, "win_rate": 0}

    return jsonify({
        "initial_bankroll": initial,
        "current_bankroll": initial + roi["total_pnl"],
        "total_pnl": roi["total_pnl"],
        "roi_pct": roi["roi_pct"],
        "n_bets": roi.get("n_bets", 0),
        "win_rate": roi.get("win_rate", 0),
        "kelly_fraction": edge_config.get("kelly_fraction", 0.5),
        "max_position_pct": edge_config.get("max_position_pct", 0.10),
        "efficiency_tiers": edge_config.get("efficiency_tiers", {}),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def _safe_int(v) -> int | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return int(v)


def _sample_opportunities():
    return {
        "is_sample": True,
        "last_scan": None,
        "count": 3,
        "opportunities": [
            {
                "contract_id": "sample_1",
                "title": "Will NYC exceed 100\u00b0F in July 2026?",
                "model_prob": 0.123,
                "market_prob": 0.080,
                "edge": 0.043,
                "days_remaining": 124,
                "source": "polymarket",
                "category": "temperature",
                "ev": 0.54,
                "kelly_bet": 21.50,
            },
            {
                "contract_id": "sample_2",
                "title": "Category 4+ hurricane US landfall 2026?",
                "model_prob": 0.285,
                "market_prob": 0.350,
                "edge": -0.065,
                "days_remaining": 278,
                "source": "kalshi",
                "category": "hurricane",
                "ev": -0.19,
                "kelly_bet": 0.00,
            },
            {
                "contract_id": "sample_3",
                "title": "Snow in Austin before March 2026?",
                "model_prob": 0.452,
                "market_prob": 0.380,
                "edge": 0.072,
                "days_remaining": 2,
                "source": "polymarket",
                "category": "snowfall",
                "ev": 0.19,
                "kelly_bet": 36.00,
            },
        ],
    }


def create_app(config_path: str | None = None) -> Flask:
    """Factory for creating the Flask app with optional config path."""
    global _config
    _config = load_config(config_path)
    return app


if __name__ == "__main__":
    app.run(debug=True, port=5000)
