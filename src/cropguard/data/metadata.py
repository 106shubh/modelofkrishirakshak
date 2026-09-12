"""Metadata encoding: crop / growth stage / region / weather → dense vector.

The context encoder gives the multimodal model information a leaf photo cannot
carry: which crop it is, where in Maharashtra it grows, what the weather has
been like. Weather synthesis is a documented *prior* (deterministic seasonal
climatology), never a live causal input.
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn

from ..taxonomy import CROPS, GROWTH_STAGES, REGIONS

# Normalization ranges for raw weather values.
WEATHER_KEYS = ("temperature_c", "humidity_pct", "rainfall_mm_24h", "rainfall_mm_7d")
WEATHER_MIN = {"temperature_c": 5.0, "humidity_pct": 20.0, "rainfall_mm_24h": 0.0, "rainfall_mm_7d": 0.0}
WEATHER_MAX = {"temperature_c": 48.0, "humidity_pct": 100.0, "rainfall_mm_7d": 200.0}
WEATHER_MAX["rainfall_mm_24h"] = WEATHER_MAX["rainfall_mm_7d"] / 2.0

_FALLBACK_REGIONS = REGIONS


def _stable_digest(text: str) -> int:
    """Deterministic 31-bit digest (builtin hash() is salted per process)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def crop_id_tensor(crop_name_or_id: str | int) -> int:
    """Resolve a crop name or id to a canonical integer id in [1, len(CROPS)]."""
    if isinstance(crop_name_or_id, int):
        if crop_name_or_id not in CROPS:
            raise ValueError(f"Unknown crop_id: {crop_name_or_id}")
        return crop_name_or_id
    for cid, name in CROPS.items():
        if name == crop_name_or_id:
            return cid
    raise ValueError(f"Unknown crop: {crop_name_or_id}")


def stage_index(stage: str) -> int:
    """Map a growth-stage name to its index in GROWTH_STAGES."""
    try:
        return GROWTH_STAGES.index(stage.lower().strip())
    except ValueError:
        raise ValueError(f"Unknown stage '{stage}'. Choose from {GROWTH_STAGES}") from None


def region_index(region: str) -> int:
    """Map a Maharashtra district name to its index in REGIONS."""
    try:
        return REGIONS.index(region.lower().strip())
    except ValueError:
        raise ValueError(f"Unknown region '{region}'. Choose from {REGIONS}") from None


def normalize_weather(w: dict[str, float]) -> list[float]:
    """Clamp raw weather values and scale each key to [0, 1]."""
    out: list[float] = []
    for key in WEATHER_KEYS:
        lo = WEATHER_MIN[key]
        hi = WEATHER_MAX[key]
        raw = float(w.get(key, lo))
        out.append(min(1.0, max(0.0, (raw - lo) / (hi - lo))))
    return out


class WeatherVector:
    """A validated weather observation or synthesized prior."""

    __slots__ = WEATHER_KEYS

    def __init__(self, **values: float) -> None:
        for key in WEATHER_KEYS:
            setattr(self, key, float(values.get(key, 0.0)))

    def to_dict(self) -> dict[str, float]:
        return {key: getattr(self, key) for key in WEATHER_KEYS}

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(normalize_weather(self.to_dict()), dtype=torch.float32)


# District-level monsoon climatology (coarse priors for Maharashtra).
_REGION_RAININESS = {
    "nashik": 1.00, "pune": 0.85, "aurangabad": 0.80, "nagpur": 0.95,
    "amravati": 0.95, "solapur": 0.65, "kolhapur": 1.15, "latur": 0.75,
    "akola": 0.85, "wardha": 0.90, "jalgaon": 0.75, "sangli": 0.90,
}

# Monthly rainfall weight (relative), June–Sept monsoon peak.
_MONTH_RAIN = [0.02, 0.03, 0.05, 0.10, 0.25, 1.00, 1.20, 1.10, 0.60, 0.20, 0.05, 0.02]
# Monthly mean temperature (°C) for interior Maharashtra.
_MONTH_TEMP = [24.0, 26.5, 30.5, 33.5, 34.5, 30.0, 26.5, 26.5, 28.5, 29.5, 27.5, 25.0]


def synthesize_weather(region: str, month: int, seed: str = "") -> WeatherVector:
    """Deterministic seasonal weather prior for a district and month.

    The seed (e.g. a file digest) jitters values within the month so two
    samples from the same district/month are not identical.
    """
    if not 1 <= int(month) <= 12:
        raise ValueError(f"month must be in 1..12, got {month}")
    region_key = region.lower().strip()
    raininess = _REGION_RAININESS.get(region_key, 1.0)
    jitter = (_stable_digest(f"{region_key}|{seed}") % 1000) / 1000.0

    temp = _MONTH_TEMP[month - 1] + (jitter - 0.5) * 4.0
    humidity = 45.0 + 35.0 * _MONTH_RAIN[month - 1] + jitter * 15.0
    rain7 = 120.0 * _MONTH_RAIN[month - 1] * raininess * (0.6 + 0.8 * jitter)
    rain24 = rain7 * (0.3 + 0.4 * ((jitter * 7.0) % 1.0))

    return WeatherVector(
        temperature_c=round(temp, 1),
        humidity_pct=round(min(humidity, 98.0), 1),
        rainfall_mm_24h=round(rain24, 1),
        rainfall_mm_7d=round(rain7, 1),
    )


class MetadataEncoder(nn.Module):
    """Embeds crop/stage/region ids and weather into a dense context vector."""

    def __init__(
        self,
        context_dim: int = 128,
        dropout: float = 0.0,
        weather_dim: int = 4,
    ) -> None:
        super().__init__()
        n_crops = len(CROPS) + 1
        self.crop_emb = nn.Embedding(n_crops, 16)
        self.stage_emb = nn.Embedding(len(GROWTH_STAGES) + 1, 8)
        self.region_emb = nn.Embedding(len(REGIONS) + 1, 16)
        self.weather_proj = nn.Linear(weather_dim, 16)
        self.mlp = nn.Sequential(
            nn.Linear(16 + 8 + 16 + 16, context_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

    def forward(
        self,
        crop_id: torch.Tensor,
        stage_idx: torch.Tensor,
        region_idx: torch.Tensor,
        weather: torch.Tensor,
    ) -> torch.Tensor:
        """All inputs (B,) or (B,1) long/float; returns (B, context_dim)."""
        crop_id = crop_id.reshape(-1)
        stage_idx = stage_idx.reshape(-1)
        region_idx = region_idx.reshape(-1)
        weather = weather.reshape(weather.shape[0], -1).float()

        # Clamp out-of-vocabulary ids to the reserved "unknown" slot.
        crop_id = crop_id.clamp(0, self.crop_emb.num_embeddings - 1)
        stage_idx = stage_idx.clamp(0, self.stage_emb.num_embeddings - 1)
        region_idx = region_idx.clamp(0, self.region_emb.num_embeddings - 1)

        parts = [
            self.crop_emb(crop_id),
            self.stage_emb(stage_idx),
            self.region_emb(region_idx),
            self.weather_proj(weather),
        ]
        return self.mlp(torch.cat(parts, dim=-1))
