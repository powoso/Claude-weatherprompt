"""Tests for configuration loading."""

import os

import pytest

from weather_edge.utils.config import get_city_by_name, load_config


@pytest.fixture
def config():
    return load_config()


def test_load_config(config):
    assert "general" in config
    assert "cities" in config
    assert "probability" in config
    assert "edge" in config


def test_cities_present(config):
    assert len(config["cities"]) >= 5
    names = [c["name"] for c in config["cities"]]
    assert "New York City" in names
    assert "Houston" in names


def test_city_has_required_fields(config):
    for city in config["cities"]:
        assert "name" in city
        assert "lat" in city
        assert "lon" in city
        assert "nws_office" in city
        assert "uhi_adjustment_f" in city


def test_get_city_by_name(config):
    nyc = get_city_by_name(config, "new york")
    assert nyc is not None
    assert nyc["name"] == "New York City"

    austin = get_city_by_name(config, "austin")
    assert austin is not None
    assert austin["state"] == "TX"

    assert get_city_by_name(config, "nonexistent") is None


def test_env_interpolation(config, monkeypatch):
    monkeypatch.setenv("NOAA_CDO_TOKEN", "test_token_123")
    cfg = load_config()
    token = cfg["data_collection"]["historical"]["noaa_cdo_token"]
    assert token == "test_token_123"
