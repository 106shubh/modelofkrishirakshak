"""Weather service with taluka-level caching (FR3/NFR2).

Providers (config weather.provider):
  - open-meteo: https://api.open-meteo.com — no API key; free for non-commercial
    use (10k calls/day); CC BY 4.0 attribution required in any public dashboard.
  - imd: api.imd.gov.in district-wise current weather — requires an approved key
    (weather.api_key / CROPGUARD_WEATHER__API_KEY). Mapping is provisional until
    validated against a real credential (Inquiry 1).
  - synthesize: deterministic seasonal climatology prior (offline default).

All fetch failures fall back to synthesis, and every result is cached per
taluka + 3h time bucket so API cost scales with geography, not farmer count.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any

from ..config import WeatherConfig
from ..data.metadata import WeatherVector, synthesize_weather

log = logging.getLogger(__name__)

# District centroids (lat, lon) for the REGIONS vocabulary — taluka-level
# granularity is a V2 upgrade once a taluka→coords table exists (Inquiry 1).
_DISTRICT_COORDS: dict[str, tuple[float, float]] = {
    "nashik": (19.997, 73.79), "pune": (18.52, 73.86), "aurangabad": (19.88, 75.34),
    "nagpur": (21.15, 79.09), "amravati": (20.93, 77.75), "solapur": (17.66, 75.91),
    "kolhapur": (16.70, 74.24), "latur": (18.40, 76.58), "akola": (20.70, 77.00),
    "wardha": (20.75, 78.60), "jalgaon": (21.00, 75.56), "sangli": (16.85, 74.58),
}


def _empty_vector() -> WeatherVector:
    return WeatherVector(temperature_c=0.0, humidity_pct=0.0, rainfall_mm_24h=0.0, rainfall_mm_7d=0.0)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _parse_open_meteo(payload: dict[str, Any]) -> WeatherVector:
    """Map an Open-Meteo forecast response onto WeatherVector.

    `current` carries instant temp/RH; daily carries precipitation_sum for the
    trailing 7 days (past_days=7 request). Hourly arrays are the fallback path.
    """
    v = _empty_vector()
    cur = payload.get("current") or {}
    if cur.get("temperature_2m") is not None:
        v.temperature_c = round(float(cur["temperature_2m"]), 1)
    if cur.get("relative_humidity_2m") is not None:
        v.humidity_pct = round(float(cur["relative_humidity_2m"]), 1)

    if v.temperature_c == 0.0 or v.humidity_pct == 0.0:
        # Fallback: hourly arrays (last observed value / last-24h mean RH).
        hourly = payload.get("hourly_analysis") or payload.get("hourly") or {}
        temps = hourly.get("temperature_2m") or []
        rhs = hourly.get("relative_humidity_2m") or []
        if temps and v.temperature_c == 0.0:
            v.temperature_c = round(float(temps[-1]), 1)
        if rhs and v.humidity_pct == 0.0:
            v.humidity_pct = round(_mean([float(x) for x in rhs[-24:]]), 1)

    daily = payload.get("daily") or {}
    rain = daily.get("precipitation_sum") or []
    if rain:
        v.rainfall_mm_24h = round(float(rain[-1]), 1)
        v.rainfall_mm_7d = round(sum(float(x) for x in rain[-7:]), 1)
    return v


def _parse_imd(payload: dict[str, Any]) -> WeatherVector:
    """Map an IMD district-wise current-weather response (provisional).

    IMD's official field names are only published to approved accounts; this
    adapter follows the shapes seen in their public API reference and must be
    validated against a real key before trusting the numbers.
    """
    v = _empty_vector()
    records = payload.get("current_wx") or payload.get("observations") or []
    rec = records[0] if records else payload
    v.temperature_c = round(float(rec.get("temp") or rec.get("temperature") or 0.0), 1)
    v.humidity_pct = round(float(rec.get("rh") or rec.get("humidity") or 0.0), 1)
    rain24 = rec.get("rainfall_24h") or rec.get("rainfall") or 0.0
    v.rainfall_mm_24h = round(float(rain24), 1)
    v.rainfall_mm_7d = round(float(rec.get("rainfall_7d") or 0.0), 1)
    return v


def make_redis_from_env() -> Any | None:
    """Build a Redis client from REDIS_URL, or None when unset/unavailable.

    The redis package is a lazy dependency: only imported here, so the service
    runs without it and the cache simply becomes process-local (i.e. disabled
    unless a client is injected).
    """
    url = os.environ.get("REDIS_URL", "").strip()
    if not url:
        return None
    try:
        import redis  # lazy: only needed when REDIS_URL is configured

        client = redis.Redis.from_url(url, socket_timeout=2.0, socket_connect_timeout=2.0)
        client.ping()
        log.info("Redis cache connected via REDIS_URL")
        return client
    except Exception as e:
        log.warning("Redis unavailable (%s) — proceeding without shared cache", e)
        return None


class WeatherService:
    """Fetches and caches weather by taluka + time window."""

    def __init__(
        self,
        redis_client: Any | None = None,
        weather_cfg: WeatherConfig | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.redis = redis_client
        self.cfg = weather_cfg or WeatherConfig()
        self.http = http_client
        self.ttl = self.cfg.cache_ttl_sec

    def __init__(
        self,
        redis_client: Any | None = None,
        weather_cfg: WeatherConfig | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.redis = redis_client
        self.cfg = weather_cfg or WeatherConfig()
        self.http = http_client
        self.ttl = self.cfg.cache_ttl_sec
        self.last_source = "unknown"  # "api" | "synthesize" | "cache" (provenance)

    def get_weather(
        self,
        region: str,
        month: int | None = None,
        use_cache: bool = True,
    ) -> WeatherVector:
        """Fetch weather for a taluka (region). Falls back to synthesis."""
        now = datetime.datetime.now()
        month = month or now.month
        # Time bucket: taluka + year + month + day + (hour // 3) — NFR2 caps
        # API spend per geography+window, not per farmer.
        bucket = f"{region}:{now.year}:{now.month}:{now.day}:{now.hour // 3}"
        cache_key = f"weather:{bucket}"

        if use_cache and self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    log.debug(f"Cache hit for {cache_key}")
                    data = json.loads(cached)
                    self.last_source = "cache"
                    return WeatherVector(**data)
            except Exception as e:
                log.warning(f"Redis cache read failed: {e}")

        try:
            weather = self._fetch_from_api(region)
            self.last_source = "api"
        except Exception as e:
            log.warning(f"Weather API fetch failed for {region}, falling back to synthesis: {e}")
            weather = synthesize_weather(region, month)
            self.last_source = "synthesize"

        if use_cache and self.redis:
            try:
                self.redis.setex(cache_key, self.ttl, json.dumps(weather.to_dict()))
            except Exception as e:
                log.warning(f"Redis cache write failed: {e}")

        return weather

    def _fetch_from_api(self, region: str) -> WeatherVector:
        """Dispatch to the configured provider. Raises on failure."""
        provider = self.cfg.provider
        if provider == "open-meteo":
            return self._fetch_open_meteo(region)
        if provider == "imd":
            return self._fetch_imd(region)
        if provider == "synthesize":
            return synthesize_weather(region, datetime.datetime.now().month)
        raise ValueError(f"Unknown weather provider: {provider}")

    # ------------------------------------------------------------------ #
    # Providers
    # ------------------------------------------------------------------ #

    def _post(self, url: str, params: dict[str, Any] | None, headers: dict[str, str]) -> dict[str, Any]:
        """One HTTP GET via the injected client or a lazily-created httpx client."""
        if self.http is not None:
            resp = self.http.get(url, params=params, headers=headers, timeout=self.cfg.timeout_sec)
            resp.raise_for_status()
            return resp.json()
        import httpx  # lazy: keeps torch-heavy import paths lean

        with httpx.Client(timeout=self.cfg.timeout_sec) as client:
            resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()

    def _fetch_open_meteo(self, region: str) -> WeatherVector:
        """Open-Meteo forecast API — no key, observed analysis + trailing 7d rain."""
        key = region.lower().strip()
        if key not in _DISTRICT_COORDS:
            raise ValueError(f"No coordinates for region '{region}'")
        lat, lon = _DISTRICT_COORDS[key]
        payload = self._post(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m",
                "daily": "precipitation_sum",
                "past_days": 7,
                "forecast_days": 1,
                "timezone": "Asia/Kolkata",
            },
            headers={"User-Agent": "cropguard/0.1 (agricultural advisory MVP)"},
        )
        return _parse_open_meteo(payload)

    def _fetch_imd(self, region: str) -> WeatherVector:
        """IMD district-wise current weather — requires an approved API key."""
        if not self.cfg.api_key:
            raise ValueError("weather.provider=imd but weather.api_key is not set")
        payload = self._post(
            "https://api.imd.gov.in/api/v1/current_wx",
            params={"district": region.lower().strip()},
            headers={"Accept": "application/json", "X-API-Key": self.cfg.api_key},
        )
        return _parse_imd(payload)
