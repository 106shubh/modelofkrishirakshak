"""Tests for the weather service: providers, parsing, caching, fallback."""
from __future__ import annotations

import json

import pytest

from cropguard.config import WeatherConfig
from cropguard.data.metadata import synthesize_weather
from cropguard.inference.weather import (
    WeatherService,
    _parse_imd,
    _parse_open_meteo,
    make_redis_from_env,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class FakeHttp:
    """Records requests; returns a canned payload (or raises if none)."""

    def __init__(self, payload: dict | None = None):
        self.payload = payload
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        if self.payload is None:
            raise RuntimeError("network down")
        return FakeResponse(self.payload)


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value
        self.ttls[key] = ttl


def _open_meteo_payload() -> dict:
    """Matches the real API shape: instant `current` + daily precip sums."""
    return {
        "current": {"temperature_2m": 23.4, "relative_humidity_2m": 88.0},
        "daily": {"precipitation_sum": [0.0] * 6 + [12.0, 3.0]},  # 8 days, last = 24h
    }


def _open_meteo_hourly_only_payload() -> dict:
    """Legacy/fallback shape without `current` (parser must still map it)."""
    return {
        "hourly_analysis": {
            "temperature_2m": [20.0] * 30 + [22.5, 23.0],
            "relative_humidity_2m": [50.0] * 20 + [90.0] * 24,  # last 24 all 90
        },
        "daily": {"precipitation_sum": [0.0] * 6 + [12.0, 3.0]},
    }


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_parse_open_meteo():
    v = _parse_open_meteo(_open_meteo_payload())
    assert v.temperature_c == 23.4  # instant current value
    assert v.humidity_pct == 88.0
    assert v.rainfall_mm_24h == 3.0  # last daily value
    assert v.rainfall_mm_7d == 15.0  # sum of trailing 7 days


def test_parse_open_meteo_hourly_fallback():
    """Without `current`, the parser falls back to hourly arrays."""
    v = _parse_open_meteo(_open_meteo_hourly_only_payload())
    assert v.temperature_c == 23.0  # last hourly value
    assert v.humidity_pct == 90.0  # mean of last 24h RH


def test_parse_open_meteo_empty_payload_is_zeroes():
    v = _parse_open_meteo({})
    assert (v.temperature_c, v.humidity_pct, v.rainfall_mm_24h, v.rainfall_mm_7d) == (0.0, 0.0, 0.0, 0.0)


def test_parse_imd():
    v = _parse_imd({"current_wx": [{"temp": 28.4, "rh": 71.0, "rainfall_24h": 5.5, "rainfall_7d": 40.0}]})
    assert (v.temperature_c, v.humidity_pct, v.rainfall_mm_24h, v.rainfall_mm_7d) == (28.4, 71.0, 5.5, 40.0)


# --------------------------------------------------------------------------- #
# Provider dispatch
# --------------------------------------------------------------------------- #


def test_open_meteo_fetch_via_injected_client():
    http = FakeHttp(_open_meteo_payload())
    svc = WeatherService(weather_cfg=WeatherConfig(provider="open-meteo"), http_client=http)
    v = svc.get_weather("nashik", month=7, use_cache=False)
    assert v.temperature_c == 23.4  # instant current value
    call = http.calls[0]
    assert call["url"].startswith("https://api.open-meteo.com")
    assert call["params"]["latitude"] == 19.997  # nashik centroid
    assert "current" in call["params"]  # instant-conditions request shape
    assert "api_key" not in str(call["params"])  # open-meteo needs no credentials


def test_imd_fetch_requires_key():
    svc = WeatherService(weather_cfg=WeatherConfig(provider="imd", api_key=None))
    with pytest.raises(ValueError):
        svc._fetch_from_api("pune")


def test_imd_fetch_with_key():
    http = FakeHttp({"current_wx": [{"temp": 30.0, "rh": 55.0, "rainfall": 2.0}]})
    svc = WeatherService(
        weather_cfg=WeatherConfig(provider="imd", api_key="test-key"), http_client=http
    )
    v = svc.get_weather("pune", use_cache=False)
    assert v.temperature_c == 30.0
    assert http.calls[0]["headers"]["X-API-Key"] == "test-key"


def test_unknown_region_falls_back_to_synthesis():
    http = FakeHttp(None)  # would fail anyway
    svc = WeatherService(weather_cfg=WeatherConfig(provider="open-meteo"), http_client=http)
    v = svc.get_weather("atlantis", month=7, use_cache=False)
    assert v.to_dict() == synthesize_weather("atlantis", 7).to_dict()  # prior, not zeros


# --------------------------------------------------------------------------- #
# Fallback + cache
# --------------------------------------------------------------------------- #


def test_fetch_failure_falls_back_to_synthesis():
    svc = WeatherService(
        weather_cfg=WeatherConfig(provider="open-meteo"), http_client=FakeHttp(None)
    )
    v = svc.get_weather("pune", month=7, use_cache=False)
    assert v.to_dict() == synthesize_weather("pune", 7).to_dict()


def test_cache_hit_skips_fetch():
    """A warm cache entry for the current taluka+time-bucket must prevent any
    network call (NFR2: API cost scales with geography, not farmer count)."""
    import datetime

    now = datetime.datetime.now()
    bucket = f"pune:{now.year}:{now.month}:{now.day}:{now.hour // 3}"
    cached = {"temperature_c": 11.1, "humidity_pct": 22.2, "rainfall_mm_24h": 3.3, "rainfall_mm_7d": 4.4}

    redis = FakeRedis()
    redis.store[f"weather:{bucket}"] = json.dumps(cached)

    http = FakeHttp(None)  # .get raises — proves the fetch path is never hit
    svc = WeatherService(
        redis_client=redis,
        weather_cfg=WeatherConfig(provider="open-meteo"),
        http_client=http,
    )
    v = svc.get_weather("pune", use_cache=True)
    assert v.temperature_c == 11.1
    assert http.calls == []


def test_cache_write_after_fetch():
    redis = FakeRedis()
    svc = WeatherService(
        redis_client=redis,
        weather_cfg=WeatherConfig(provider="synthesize"),  # deterministic, no network
    )
    v = svc.get_weather("pune", month=7, use_cache=True)
    assert len(redis.store) == 1
    key = next(iter(redis.store))
    assert key.startswith("weather:pune:")
    assert json.loads(redis.store[key]) == v.to_dict()
    assert redis.ttls[key] == svc.ttl  # FR3/NFR2: per-bucket TTL, not per farmer


# --------------------------------------------------------------------------- #
# Redis factory
# --------------------------------------------------------------------------- #


def test_redis_factory_unset_env(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert make_redis_from_env() is None


def test_redis_factory_unreachable(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")  # nothing listens here
    assert make_redis_from_env() is None  # degrades gracefully, never raises


def test_redis_factory_missing_package(monkeypatch):
    """No REDIS_URL handling when the redis package is absent (lazy import)."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("redis not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert make_redis_from_env() is None
