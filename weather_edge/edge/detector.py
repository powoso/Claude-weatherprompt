"""Edge detection: compare model probabilities vs market prices."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("weather_edge.edge.detector")


@dataclass
class Opportunity:
    """A detected +EV betting opportunity."""
    contract_id: str
    source: str
    title: str
    category: str
    model_prob: float
    market_prob: float
    edge: float  # model_prob - market_prob
    expected_value: float  # edge / market_prob
    kelly_fraction: float
    suggested_bet_size: float
    confidence_interval: tuple[float, float]
    detected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def detect_edges(
    evaluations: list[dict],
    market_contracts: list,
    min_edge: float = 0.05,
    kelly_fraction: float = 0.5,
    max_position_pct: float = 0.10,
    bankroll: float = 1000.0,
) -> list[Opportunity]:
    """Compare model evaluations against market prices and find +EV opportunities.

    Args:
        evaluations: List of dicts from probability engine (must include 'probability').
        market_contracts: List of MarketContract objects with market prices.
        min_edge: Minimum edge threshold to flag.
        kelly_fraction: Fraction of full Kelly to use (0.5 = half-Kelly).
        max_position_pct: Maximum position as fraction of bankroll.
        bankroll: Current bankroll for sizing.

    Returns:
        List of Opportunity objects sorted by edge descending.
    """
    # Build lookup from contract_id to market data
    market_lookup = {}
    for contract in market_contracts:
        market_lookup[contract.market_id] = contract

    opportunities: list[Opportunity] = []

    for evaluation in evaluations:
        contract_id = evaluation.get("contract_id")
        model_prob = evaluation.get("probability")

        if model_prob is None or contract_id is None:
            continue

        contract = market_lookup.get(contract_id)
        if contract is None:
            continue

        market_prob = contract.market_price
        if market_prob <= 0 or market_prob >= 1:
            continue

        # Edge = model probability - market implied probability
        edge = model_prob - market_prob

        if abs(edge) < min_edge:
            continue

        # Expected value
        if edge > 0:
            # Market underpricing YES: buy YES
            ev = (model_prob * (1 / market_prob - 1)) - ((1 - model_prob) * 1)
        else:
            # Market overpricing YES: buy NO (equivalent to shorting YES)
            # Reframe: model_prob of YES is lower than market thinks
            # Probability of NO = 1 - model_prob
            no_market = 1 - market_prob
            no_model = 1 - model_prob
            edge = no_model - no_market  # Edge on the NO side
            if edge < min_edge:
                continue
            ev = (no_model * (1 / no_market - 1)) - ((1 - no_model) * 1)
            model_prob = no_model
            market_prob = no_market

        # Kelly criterion sizing
        kelly_bet = kelly_criterion(model_prob, market_prob, kelly_fraction)
        max_bet = bankroll * max_position_pct
        suggested_bet = min(kelly_bet * bankroll, max_bet)

        ci = (
            evaluation.get("ci_lower", model_prob - 0.05),
            evaluation.get("ci_upper", model_prob + 0.05),
        )

        opp = Opportunity(
            contract_id=contract_id,
            source=contract.source,
            title=contract.title,
            category=contract.category,
            model_prob=model_prob,
            market_prob=market_prob,
            edge=edge,
            expected_value=ev,
            kelly_fraction=kelly_bet,
            suggested_bet_size=suggested_bet,
            confidence_interval=ci,
        )
        opportunities.append(opp)

    # Sort by edge descending
    opportunities.sort(key=lambda o: o.edge, reverse=True)

    logger.info("Found %d opportunities with edge >= %.2f", len(opportunities), min_edge)
    return opportunities


def kelly_criterion(
    model_prob: float,
    market_prob: float,
    fraction: float = 0.5,
) -> float:
    """Compute Kelly criterion bet fraction.

    For a binary bet at price `market_prob`:
      f* = (p * b - q) / b
    where:
      p = model probability of winning
      q = 1 - p
      b = (1 / market_prob) - 1 = decimal odds - 1

    Returns fraction of bankroll to bet (already adjusted by `fraction`).
    """
    if market_prob <= 0 or market_prob >= 1 or model_prob <= 0 or model_prob >= 1:
        return 0.0

    b = (1.0 / market_prob) - 1.0  # Net odds
    p = model_prob
    q = 1.0 - p

    f_star = (p * b - q) / b
    if f_star <= 0:
        return 0.0

    return f_star * fraction


class EdgeTracker:
    """Track edge evolution over time for convergence analysis."""

    def __init__(self, storage_path: str | Path = "data/edge_history.parquet"):
        self.storage_path = Path(storage_path)
        self._history: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if self._history is not None:
            return self._history
        if self.storage_path.exists():
            self._history = pd.read_parquet(self.storage_path)
        else:
            self._history = pd.DataFrame(columns=[
                "timestamp", "contract_id", "title", "model_prob",
                "market_prob", "edge", "days_remaining",
            ])
        return self._history

    def record(self, opportunities: list[Opportunity], evaluations: list[dict]) -> None:
        """Record current snapshot of opportunities for time-series tracking."""
        now = datetime.utcnow().isoformat()
        rows = []
        eval_lookup = {e.get("contract_id"): e for e in evaluations}

        for opp in opportunities:
            days_rem = None
            ev = eval_lookup.get(opp.contract_id, {})
            td = ev.get("time_decay", {})
            days_rem = td.get("days_remaining")

            rows.append({
                "timestamp": now,
                "contract_id": opp.contract_id,
                "title": opp.title,
                "model_prob": opp.model_prob,
                "market_prob": opp.market_prob,
                "edge": opp.edge,
                "days_remaining": days_rem,
            })

        if not rows:
            return

        new_df = pd.DataFrame(rows)
        history = self.load()
        self._history = pd.concat([history, new_df], ignore_index=True)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._history.to_parquet(self.storage_path)
        logger.info("Recorded %d edge snapshots", len(rows))

    def get_convergence_data(self, contract_id: str) -> pd.DataFrame:
        """Get time-series of model vs market price for a specific contract."""
        history = self.load()
        return history[history["contract_id"] == contract_id].sort_values("timestamp")
