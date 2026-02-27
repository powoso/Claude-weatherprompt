"""Tests for HURDAT2 parsing."""

import pytest

from weather_edge.data_collection.hurricanes import (
    _wind_to_category,
    parse_hurdat2,
    storms_to_dataframe,
)

SAMPLE_HURDAT2 = """\
AL012020,            ARTHUR,     16,
20200516, 1200,  , TS, 28.4N,  79.0W,  40, 1006,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200516, 1800,  , TS, 29.0N,  78.5W,  45, 1003,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200517, 0000,  , TS, 29.7N,  78.0W,  50, 1000,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200517, 0600,  , TS, 30.5N,  77.4W,  55,  997,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200517, 1200,  , TS, 31.4N,  76.7W,  55,  995,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200517, 1800,  , TS, 32.5N,  75.7W,  55,  993,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200518, 0000,  , TS, 33.7N,  74.7W,  55,  990,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200518, 0600,  , TS, 35.0N,  73.6W,  55,  988,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200518, 1200,  , TS, 36.4N,  72.6W,  60,  986,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200518, 1800,  , EX, 37.8N,  71.6W,  55,  986,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200519, 0000,  , EX, 39.1N,  70.2W,  55,  986,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200519, 0600,  , EX, 40.4N,  68.4W,  50,  988,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200519, 1200,  , EX, 41.4N,  66.3W,  50,  989,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200519, 1800,  , EX, 41.8N,  63.8W,  50,  990,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200520, 0000,  , EX, 42.2N,  61.1W,  50,  990,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200520, 0600,  , EX, 43.5N,  57.0W,  45,  993,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
"""


def test_parse_hurdat2():
    storms = parse_hurdat2(SAMPLE_HURDAT2)
    assert len(storms) == 1
    assert storms[0].name == "ARTHUR"
    assert storms[0].storm_id == "AL012020"
    assert len(storms[0].records) == 16


def test_storms_to_dataframe():
    storms = parse_hurdat2(SAMPLE_HURDAT2)
    df = storms_to_dataframe(storms)
    assert len(df) == 16
    assert "max_wind_kt" in df.columns
    assert "lat" in df.columns
    assert df["year"].iloc[0] == 2020


def test_wind_to_category():
    assert _wind_to_category(50) == "TS"
    assert _wind_to_category(65) == "Cat1"
    assert _wind_to_category(85) == "Cat2"
    assert _wind_to_category(100) == "Cat3"
    assert _wind_to_category(120) == "Cat4"
    assert _wind_to_category(140) == "Cat5"
    assert _wind_to_category(None) == "TS"
