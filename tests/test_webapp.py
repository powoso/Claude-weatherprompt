"""Tests for Flask web application."""

import json

import pytest

from weather_edge.webapp.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Weather Edge" in resp.data


def test_index_contains_cities(client):
    resp = client.get("/")
    assert b"New York City" in resp.data
    assert b"Austin" in resp.data


def test_api_opportunities(client):
    resp = client.get("/api/opportunities")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "opportunities" in data
    assert "count" in data
    assert isinstance(data["opportunities"], list)


def test_api_opportunities_sample_data(client):
    resp = client.get("/api/opportunities")
    data = json.loads(resp.data)
    # Without real data, should return sample
    if data.get("is_sample"):
        assert data["count"] == 3
        opp = data["opportunities"][0]
        assert "title" in opp
        assert "model_prob" in opp
        assert "market_prob" in opp


def test_api_base_rates_no_data(client):
    resp = client.get("/api/base-rates?city=New+York+City&metric=temperature_2m_max&month=7")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    # Without ingested data, should return error
    assert "error" in data or "stats" in data


def test_api_enso(client):
    resp = client.get("/api/enso")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    # Without data, should return error
    assert "error" in data or "current" in data


def test_api_calibration(client):
    resp = client.get("/api/calibration")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "calibration" in data
    assert "roi" in data


def test_api_bankroll(client):
    resp = client.get("/api/bankroll")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "initial_bankroll" in data
    assert "kelly_fraction" in data
    assert data["initial_bankroll"] == 1000.0


def test_api_convergence_unknown_id(client):
    resp = client.get("/api/convergence/nonexistent")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["timestamps"] == []
