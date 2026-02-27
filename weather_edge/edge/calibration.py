"""Model calibration tracking — compare predictions vs outcomes."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("weather_edge.edge.calibration")


class CalibrationTracker:
    """Track model predictions vs actual outcomes for calibration analysis."""

    def __init__(self, storage_path: str | Path = "data/calibration.parquet"):
        self.storage_path = Path(storage_path)
        self._records: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if self._records is not None:
            return self._records
        if self.storage_path.exists():
            self._records = pd.read_parquet(self.storage_path)
        else:
            self._records = pd.DataFrame(columns=[
                "contract_id", "title", "category", "model_prob",
                "market_prob_at_prediction", "prediction_date",
                "resolution_date", "outcome", "bet_placed", "bet_size",
                "pnl",
            ])
        return self._records

    def record_prediction(
        self,
        contract_id: str,
        title: str,
        category: str,
        model_prob: float,
        market_prob: float,
        prediction_date: str,
        bet_placed: bool = False,
        bet_size: float = 0.0,
    ) -> None:
        """Log a new prediction."""
        records = self.load()
        new_row = pd.DataFrame([{
            "contract_id": contract_id,
            "title": title,
            "category": category,
            "model_prob": model_prob,
            "market_prob_at_prediction": market_prob,
            "prediction_date": prediction_date,
            "resolution_date": None,
            "outcome": None,
            "bet_placed": bet_placed,
            "bet_size": bet_size,
            "pnl": None,
        }])
        self._records = pd.concat([records, new_row], ignore_index=True)
        self._save()

    def resolve_prediction(
        self,
        contract_id: str,
        outcome: int,  # 1 = YES happened, 0 = NO
        resolution_date: str,
    ) -> None:
        """Record the actual outcome for a previously logged prediction."""
        records = self.load()
        mask = (records["contract_id"] == contract_id) & (records["outcome"].isna())
        if not mask.any():
            logger.warning("No pending prediction found for %s", contract_id)
            return

        idx = records[mask].index[-1]  # Most recent pending
        records.loc[idx, "outcome"] = outcome
        records.loc[idx, "resolution_date"] = resolution_date

        # Compute P&L if bet was placed
        if records.loc[idx, "bet_placed"]:
            bet_size = records.loc[idx, "bet_size"]
            market_price = records.loc[idx, "market_prob_at_prediction"]
            if outcome == 1:
                pnl = bet_size * ((1 / market_price) - 1)
            else:
                pnl = -bet_size
            records.loc[idx, "pnl"] = pnl

        self._records = records
        self._save()
        logger.info("Resolved %s -> outcome=%d", contract_id, outcome)

    def compute_calibration(self, n_bins: int = 10) -> pd.DataFrame:
        """Compute calibration curve: predicted probability vs actual outcome rate.

        Returns DataFrame with columns: bin_center, predicted_mean, actual_rate, count.
        """
        records = self.load()
        resolved = records.dropna(subset=["outcome"])
        if resolved.empty:
            return pd.DataFrame()

        bins = np.linspace(0, 1, n_bins + 1)
        resolved = resolved.copy()
        resolved["prob_bin"] = pd.cut(resolved["model_prob"], bins=bins, include_lowest=True)

        cal = resolved.groupby("prob_bin", observed=False).agg(
            predicted_mean=("model_prob", "mean"),
            actual_rate=("outcome", "mean"),
            count=("outcome", "count"),
        ).reset_index()

        return cal

    def compute_brier_score(self) -> float | None:
        """Compute Brier score (lower is better, 0 = perfect)."""
        records = self.load()
        resolved = records.dropna(subset=["outcome"])
        if resolved.empty:
            return None
        return float(((resolved["model_prob"] - resolved["outcome"]) ** 2).mean())

    def compute_roi(self) -> dict:
        """Compute return on investment from all tracked bets."""
        records = self.load()
        bets = records[records["bet_placed"] == True]  # noqa: E712
        resolved = bets.dropna(subset=["pnl"])

        if resolved.empty:
            return {"total_wagered": 0, "total_pnl": 0, "roi_pct": 0, "n_bets": 0}

        total_wagered = resolved["bet_size"].sum()
        total_pnl = resolved["pnl"].sum()
        roi_pct = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0

        return {
            "total_wagered": float(total_wagered),
            "total_pnl": float(total_pnl),
            "roi_pct": float(roi_pct),
            "n_bets": len(resolved),
            "win_rate": float((resolved["pnl"] > 0).mean()),
        }

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if self._records is not None:
            self._records.to_parquet(self.storage_path)
