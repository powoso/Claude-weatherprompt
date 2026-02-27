"""Streamlit dashboard for weather prediction market edge finder."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather_edge.edge.calibration import CalibrationTracker
from weather_edge.edge.detector import EdgeTracker
from weather_edge.utils.config import load_config

# --- Page config ---
st.set_page_config(
    page_title="Weather Edge",
    page_icon="=",
    layout="wide",
)


@st.cache_data(ttl=300)
def get_config():
    return load_config()


def main():
    st.title("Weather Prediction Market Edge Finder")
    config = get_config()

    tab_opps, tab_base, tab_enso, tab_cal, tab_bank = st.tabs([
        "Active Opportunities",
        "Historical Base Rates",
        "ENSO & Seasonal Outlook",
        "Model Calibration",
        "Bankroll Management",
    ])

    with tab_opps:
        render_opportunities_tab(config)

    with tab_base:
        render_base_rates_tab(config)

    with tab_enso:
        render_enso_tab(config)

    with tab_cal:
        render_calibration_tab(config)

    with tab_bank:
        render_bankroll_tab(config)


def render_opportunities_tab(config: dict):
    """Active markets with model prob vs market price and EV."""
    st.header("Active Market Opportunities")

    # Load edge history if available
    tracker = EdgeTracker()
    try:
        history = tracker.load()
    except Exception:
        history = pd.DataFrame()

    if history.empty:
        st.info(
            "No opportunity data yet. Run `weather-edge scan` to populate. "
            "Showing sample data structure below."
        )
        _show_sample_opportunities()
        return

    # Latest snapshot
    latest_ts = history["timestamp"].max()
    latest = history[history["timestamp"] == latest_ts].copy()

    st.metric("Opportunities Found", len(latest))
    st.caption(f"Last scan: {latest_ts}")

    # Edge table
    display_cols = ["title", "model_prob", "market_prob", "edge", "days_remaining"]
    available = [c for c in display_cols if c in latest.columns]
    if available:
        styled = latest[available].copy()
        for col in ["model_prob", "market_prob", "edge"]:
            if col in styled.columns:
                styled[col] = styled[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        st.dataframe(styled, use_container_width=True)

    # Convergence chart for selected contract
    contract_ids = latest["contract_id"].unique()
    if len(contract_ids) > 0:
        selected = st.selectbox("Track convergence for:", contract_ids)
        conv_data = tracker.get_convergence_data(selected)
        if not conv_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=conv_data["timestamp"], y=conv_data["model_prob"],
                name="Model Probability", line=dict(color="blue"),
            ))
            fig.add_trace(go.Scatter(
                x=conv_data["timestamp"], y=conv_data["market_prob"],
                name="Market Price", line=dict(color="red"),
            ))
            fig.update_layout(
                title="Model vs Market Price Over Time",
                yaxis_title="Probability",
                yaxis=dict(tickformat=".0%"),
            )
            st.plotly_chart(fig, use_container_width=True)


def _show_sample_opportunities():
    """Show sample data structure for when no real data exists."""
    sample = pd.DataFrame({
        "Title": [
            "Will NYC exceed 100F in July 2026?",
            "Cat 4+ hurricane US landfall 2026?",
            "Snow in Austin before March 2026?",
        ],
        "Model Prob": ["12.3%", "28.5%", "45.2%"],
        "Market Price": ["8.0%", "35.0%", "38.0%"],
        "Edge": ["+4.3%", "-6.5%", "+7.2%"],
        "EV": ["+0.54", "-0.19", "+0.19"],
        "Kelly Bet": ["$21.50", "$0.00", "$36.00"],
    })
    st.dataframe(sample, use_container_width=True)


def render_base_rates_tab(config: dict):
    """Historical base rate visualizations for any city/threshold combo."""
    st.header("Historical Base Rates")

    col1, col2, col3 = st.columns(3)
    city_names = [c["name"] for c in config["cities"]]
    with col1:
        city = st.selectbox("City", city_names, key="br_city")
    with col2:
        metric = st.selectbox("Metric", [
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "snowfall_sum",
        ], key="br_metric")
    with col3:
        month = st.selectbox("Month", list(range(1, 13)),
                             format_func=lambda m: date(2000, m, 1).strftime("%B"),
                             key="br_month")

    # Try to load historical data
    data_dir = config["general"]["data_dir"]
    slug = city.lower().replace(" ", "_")
    hist_path = Path(data_dir) / "historical" / f"{slug}.parquet"

    if hist_path.exists():
        df = pd.read_parquet(hist_path)
        df_month = df[df.index.month == month]

        if metric in df_month.columns and not df_month[metric].dropna().empty:
            values = df_month[metric].dropna()

            # Distribution plot
            fig = px.histogram(
                values, nbins=50,
                title=f"{metric} Distribution for {city} in {date(2000, month, 1).strftime('%B')}",
                labels={"value": metric, "count": "Days"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Threshold analysis
            st.subheader("Threshold Exceedance Rates")
            if "temperature" in metric:
                thresholds = list(range(80, 120, 5))
            elif "snowfall" in metric:
                thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            else:
                thresholds = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]

            rates = []
            for t in thresholds:
                daily_rate = (values >= t).mean()
                yearly = df_month.groupby(df_month.index.year)[metric].max()
                yearly_rate = (yearly >= t).mean()
                rates.append({
                    "Threshold": t,
                    "Daily Rate": f"{daily_rate:.2%}",
                    "Yearly Rate (any day)": f"{yearly_rate:.2%}",
                })
            st.dataframe(pd.DataFrame(rates), use_container_width=True)

            # Year-by-year time series
            st.subheader("Year-by-Year Monthly Averages")
            yearly_avg = df_month.groupby(df_month.index.year)[metric].mean()
            fig2 = px.line(
                x=yearly_avg.index, y=yearly_avg.values,
                title=f"Average {metric} in {date(2000, month, 1).strftime('%B')} by Year",
                labels={"x": "Year", "y": metric},
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning(f"No data for {metric} in {city}")
    else:
        st.info(
            f"No historical data for {city}. "
            "Run `weather-edge ingest` to download historical data."
        )


def render_enso_tab(config: dict):
    """Current ENSO status and seasonal outlook summary."""
    st.header("ENSO Status & Seasonal Outlook")

    enso_path = Path(config["general"]["data_dir"]) / "cache" / "oni_data.parquet"

    if enso_path.exists():
        oni_df = pd.read_parquet(enso_path)

        # Current status
        last = oni_df.iloc[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Phase", last.get("oni_phase", "N/A"))
        col2.metric("ONI Anomaly", f"{last.get('anomaly', 0):.2f}°C")
        col3.metric("Season", f"{last.get('season', 'N/A')} {last.get('year', '')}")

        # ONI time series
        fig = px.line(
            oni_df, x=oni_df.index, y="anomaly",
            title="Oceanic Nino Index (ONI) Time Series",
            labels={"anomaly": "ONI (°C)", "index": "Period"},
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                      annotation_text="El Nino threshold")
        fig.add_hline(y=-0.5, line_dash="dash", line_color="blue",
                      annotation_text="La Nina threshold")
        fig.add_hline(y=0, line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        # Impact summary
        st.subheader("ENSO Impact on Weather Markets")
        st.markdown("""
        **Key ENSO effects on US weather:**
        - **El Nino**: Wetter/cooler in the South, drier/warmer in the North.
          Suppresses Atlantic hurricanes. More snow in southern states.
        - **La Nina**: Drier in the South, wetter/cooler in the North.
          Enhances Atlantic hurricane activity. Less snow in southern states.
        - **Neutral**: Close to climatological normals.

        Markets often underweight ENSO effects, especially for:
        - Seasonal temperature outlooks
        - Hurricane season activity
        - Southern US snowfall probability
        """)
    else:
        st.info("No ENSO data yet. Run `weather-edge ingest` to download.")


def render_calibration_tab(config: dict):
    """Model calibration charts and accuracy tracking."""
    st.header("Model Calibration")

    tracker = CalibrationTracker()
    try:
        records = tracker.load()
    except Exception:
        records = pd.DataFrame()

    if records.empty or records.dropna(subset=["outcome"]).empty:
        st.info("No resolved predictions yet. Calibration data will appear after contracts resolve.")
        _show_sample_calibration()
        return

    # Calibration curve
    cal_data = tracker.compute_calibration()
    if not cal_data.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cal_data["predicted_mean"], y=cal_data["actual_rate"],
            mode="markers+lines", name="Model",
            marker=dict(size=cal_data["count"] * 2),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Perfect Calibration", line=dict(dash="dash", color="gray"),
        ))
        fig.update_layout(
            title="Calibration Curve",
            xaxis_title="Predicted Probability",
            yaxis_title="Actual Outcome Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Brier score
    brier = tracker.compute_brier_score()
    if brier is not None:
        st.metric("Brier Score", f"{brier:.4f}", help="Lower is better. 0 = perfect, 0.25 = random.")


def _show_sample_calibration():
    """Show example calibration chart with synthetic data."""
    bins = np.arange(0.05, 1.0, 0.1)
    perfect = bins
    # Simulated slight overconfidence
    actual = bins * 0.9 + 0.02

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bins, y=actual, mode="markers+lines", name="Example Model"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Perfect", line=dict(dash="dash", color="gray")))
    fig.update_layout(title="Example Calibration Curve (sample data)",
                      xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig, use_container_width=True)


def render_bankroll_tab(config: dict):
    """Bankroll management panel."""
    st.header("Bankroll Management")

    edge_config = config.get("edge", {})
    initial = edge_config.get("initial_bankroll", 1000.0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Bankroll", f"${initial:,.2f}")

    tracker = CalibrationTracker()
    try:
        roi = tracker.compute_roi()
    except Exception:
        roi = {"total_wagered": 0, "total_pnl": 0, "roi_pct": 0, "n_bets": 0}

    current_bankroll = initial + roi["total_pnl"]
    col2.metric("Current Bankroll", f"${current_bankroll:,.2f}",
                delta=f"${roi['total_pnl']:+,.2f}")
    col3.metric("ROI", f"{roi['roi_pct']:+.1f}%")

    st.divider()

    # Position sizing settings
    st.subheader("Position Sizing Parameters")
    scol1, scol2, scol3 = st.columns(3)
    scol1.metric("Kelly Fraction", f"{edge_config.get('kelly_fraction', 0.5):.0%}",
                 help="Using half-Kelly for conservative sizing")
    scol2.metric("Max Position", f"{edge_config.get('max_position_pct', 0.1):.0%}",
                 help="Maximum percentage of bankroll on a single bet")
    scol3.metric("Total Bets", roi.get("n_bets", 0))

    if roi.get("n_bets", 0) > 0:
        st.metric("Win Rate", f"{roi.get('win_rate', 0):.1%}")

    # Efficiency tiers
    st.subheader("Market Efficiency Tiers")
    tiers = edge_config.get("efficiency_tiers", {})
    for tier_name, markets in tiers.items():
        label = tier_name.replace("_", " ").title()
        st.markdown(f"**{label}**: {', '.join(markets)}")


if __name__ == "__main__":
    main()
