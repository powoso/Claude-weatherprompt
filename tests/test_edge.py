"""Tests for edge detection and Kelly criterion."""

import pytest

from weather_edge.data_collection.markets import MarketContract
from weather_edge.edge.detector import detect_edges, kelly_criterion


def _make_contract(market_id, title, price, category="temperature"):
    return MarketContract(
        source="test",
        market_id=market_id,
        title=title,
        description="",
        market_price=price,
        volume=1000,
        end_date="2026-07-31",
        category=category,
        raw={},
    )


class TestKellyCriterion:
    def test_positive_edge(self):
        # Model says 60%, market says 40% -> positive edge
        f = kelly_criterion(model_prob=0.6, market_prob=0.4)
        assert f > 0

    def test_no_edge(self):
        # Model agrees with market
        f = kelly_criterion(model_prob=0.5, market_prob=0.5)
        assert f == pytest.approx(0.0)

    def test_negative_edge(self):
        # Model says 30%, market says 50% -> negative edge
        f = kelly_criterion(model_prob=0.3, market_prob=0.5)
        assert f == 0.0

    def test_half_kelly(self):
        full = kelly_criterion(model_prob=0.7, market_prob=0.5, fraction=1.0)
        half = kelly_criterion(model_prob=0.7, market_prob=0.5, fraction=0.5)
        assert half == pytest.approx(full * 0.5)

    def test_extreme_edge(self):
        # Very high model confidence
        f = kelly_criterion(model_prob=0.95, market_prob=0.1)
        assert 0 < f <= 1


class TestEdgeDetection:
    def test_finds_positive_edge(self):
        contracts = [_make_contract("1", "NYC temp > 100F July", 0.10)]
        evaluations = [{"contract_id": "1", "probability": 0.20, "ci_lower": 0.15, "ci_upper": 0.25}]
        opps = detect_edges(evaluations, contracts, min_edge=0.05)
        assert len(opps) == 1
        assert opps[0].edge == pytest.approx(0.10)

    def test_filters_small_edge(self):
        contracts = [_make_contract("1", "NYC temp > 100F July", 0.10)]
        evaluations = [{"contract_id": "1", "probability": 0.12}]
        opps = detect_edges(evaluations, contracts, min_edge=0.05)
        assert len(opps) == 0

    def test_skips_missing_probability(self):
        contracts = [_make_contract("1", "test", 0.50)]
        evaluations = [{"contract_id": "1", "probability": None}]
        opps = detect_edges(evaluations, contracts)
        assert len(opps) == 0

    def test_max_position_sizing(self):
        contracts = [_make_contract("1", "test", 0.10)]
        evaluations = [{"contract_id": "1", "probability": 0.50}]
        opps = detect_edges(
            evaluations, contracts, min_edge=0.05,
            bankroll=1000.0, max_position_pct=0.10,
        )
        if opps:
            assert opps[0].suggested_bet_size <= 100.0  # 10% of 1000

    def test_multiple_opportunities_sorted(self):
        contracts = [
            _make_contract("1", "small edge", 0.40),
            _make_contract("2", "big edge", 0.20),
        ]
        evaluations = [
            {"contract_id": "1", "probability": 0.50},
            {"contract_id": "2", "probability": 0.45},
        ]
        opps = detect_edges(evaluations, contracts, min_edge=0.05)
        if len(opps) >= 2:
            assert opps[0].edge >= opps[1].edge
