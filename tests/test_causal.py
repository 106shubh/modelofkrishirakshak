"""Tests for the causal rule layer and weather service."""
from __future__ import annotations

import pytest
from cropguard.data.metadata import WeatherVector
from cropguard.inference.rules import CausalRuleLayer
from cropguard.inference.weather import WeatherService


def test_rule_fungal_high_risk():
    rules = CausalRuleLayer()
    # High humidity + high rainfall = high fungal risk
    w = WeatherVector(temperature_c=25.0, humidity_pct=90.0, rainfall_mm_7d=60.0)
    res = rules.evaluate(crop_id=1, weather=w, growth_stage="fruiting")
    assert res.risk_tier == "high"
    assert "High fungal pressure" in res.reason
    assert any("humidity" in f.lower() for f in res.contributing_factors)


def test_rule_late_blight_risk():
    rules = CausalRuleLayer()
    # Cool (15C) and wet = Late Blight risk
    w = WeatherVector(temperature_c=15.0, humidity_pct=95.0, rainfall_mm_24h=15.0)
    res = rules.evaluate(crop_id=1, weather=w, growth_stage="vegetative", predicted_class="Tomato___Late_blight")
    assert res.risk_tier == "high"
    assert "Late Blight" in res.reason


def test_rule_spider_mites_risk():
    rules = CausalRuleLayer()
    # Hot and dry = Mite risk
    w = WeatherVector(temperature_c=35.0, humidity_pct=30.0)
    res = rules.evaluate(crop_id=1, weather=w, growth_stage="vegetative", predicted_class="Tomato___Spider_mites Two-spotted_spider_mite")
    assert res.risk_tier == "high"
    assert "Pest outbreak risk (Mites)" in res.reason


def test_weather_service_synthesis():
    from cropguard.config import WeatherConfig

    svc = WeatherService(redis_client=None, weather_cfg=WeatherConfig(provider="synthesize"))
    # Synthesis should be deterministic for a region/month (no network)
    w1 = svc.get_weather("pune", month=7, use_cache=False)
    w2 = svc.get_weather("pune", month=7, use_cache=False)
    assert w1.temperature_c == w2.temperature_c
    assert w1.humidity_pct == w2.humidity_pct


def test_rule_vulnerable_stage():
    rules = CausalRuleLayer()
    # Normal weather but vulnerable stage = medium risk
    w = WeatherVector(temperature_c=25.0, humidity_pct=50.0)
    res = rules.evaluate(crop_id=1, weather=w, growth_stage="fruiting")
    assert res.risk_tier == "medium"
    assert any("fruiting" in f.lower() for f in res.contributing_factors)
