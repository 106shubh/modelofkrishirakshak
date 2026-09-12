"""Inference engine: dual-model routing, severity normalization, and calibration.

Severity model (Issue 3): both vision paths feed one normalized severity_score
in [0, 1], so disease and pest readings are comparable before the severity_tier
lookup. MVP uses calibrated max-class confidence as the normalized signal —
temperature scaling puts disease and pest confidence on one shared scale, which
is exactly the comparability Issue 3 demands. V2 replaces the proxy per path:
Grad-CAM lesion area (disease) / bbox coverage (pest) mapped through this same
normalize() into [0, 1].
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from ..config import Config
from ..taxonomy import TAXONOMY

log = logging.getLogger(__name__)

# Confidence at/below this is treated as "no actionable evidence of damage" for
# severity purposes (an uncertain positive is not evidence of a severe case).
SEVERITY_EVIDENCE_FLOOR = 0.5


class ModelRouter:
    """Routes requests to disease or pest models.

    Issue 2 / Inquiry 3 (MVP default): farmer hint is optional. When a hint is
    given we run only that model (1x cost); when absent we run both and pick by
    confidence (2x cost, no farmer friction). The backend contract is unchanged:
    one predicted_pest_disease_id is returned either way.

NFR4: the pest slot holds ANY adapter matching the router's callable contract
(model(image, **ctx) → dict) — including models.pest_detector.PestDetectionAdapter,
which wraps a YOLOv8 detector (boxes+scores) into classifier-shaped output.
Confidence is read from probs when present, else from logits (adapter path).
    """

    def __init__(self, disease_model: nn.Module, pest_model: nn.Module | None = None) -> None:
        self.disease_model = disease_model
        self.pest_model = pest_model or disease_model  # single shared model until V2 dual checkpoints

    def predict(
        self,
        image: torch.Tensor,
        type_hint: str | None = None,
        **context: Any,
    ) -> dict[str, Any]:
        """Runs the appropriate model(s) and returns combined results."""
        if type_hint == "disease":
            return self.disease_model(image, **context)
        if type_hint == "pest":
            return self.pest_model(image, **context)

        # No hint: run both only if they are actually different models.
        if self.pest_model is self.disease_model:
            return self.disease_model(image, **context)

        out_d = self.disease_model(image, **context)
        out_p = self.pest_model(image, **context)
        return out_d if _confidence(out_d) >= _confidence(out_p) else out_p


def _confidence(out: dict[str, Any]) -> float:
    """Top confidence from classifier probs or adapter pseudo-logits."""
    probs = out.get("probs")
    if probs is not None:
        return float(probs.max().item())
    return float(torch.sigmoid(out["logits"]).max().item())


def normalize_severity(raw: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp and min-max scale a raw severity signal into [0, 1].

    One seam both paths must pass through before writing severity_score, so
    Grad-CAM lesion area and bbox coverage become comparable once their (lo, hi)
    calibration constants exist (V2).
    """
    # ponytail: real per-path (lo, hi) constants need labeled field data; until
    # then the identity mapping (lo=0, hi=1) is the ceiling — upgrade path is
    # fitting constants from field severity labels in V2.
    if hi <= lo:
        raise ValueError(f"severity normalization needs hi > lo, got ({lo}, {hi})")
    return min(1.0, max(0.0, (raw - lo) / (hi - lo)))


class SeverityEngine:
    """Writes one normalized severity_score and a severity_tier from it."""

    def __init__(self, cfg: Config) -> None:
        self.bands = cfg.eval.severity_bands

    def calculate(
        self,
        class_name: str,
        confidence: float | None = None,
        raw_severity: float | None = None,
    ) -> tuple[str, float]:
        """Returns (severity_tier, normalized severity_score in [0, 1]).

        raw_severity: path-specific raw signal (V2: lesion area / bbox coverage).
        MVP: None → calibrated confidence is the severity proxy; identity-normalized.
        Exactly one of confidence / raw_severity must be given for non-healthy classes.
        """
        if TAXONOMY.is_healthy(class_name):
            # A healthy read carries no damage evidence regardless of confidence —
            # banding it as "high" severity would corrupt the IPM tier lookup.
            return "low", 0.0

        if raw_severity is not None:
            score = normalize_severity(raw_severity)
        elif confidence is not None:
            # ponytail: confidence-as-severity proxy. Overstates severity on an
            # uncertain positive only up to the evidence floor; replace with
            # lesion-area/bbox signal + per-path (lo, hi) constants in V2.
            score = normalize_severity(max(0.0, confidence - SEVERITY_EVIDENCE_FLOOR) / (1.0 - SEVERITY_EVIDENCE_FLOOR))
        else:
            raise ValueError("severity needs confidence or raw_severity")

        if score < self.bands[0]:
            tier = "low"
        elif score < self.bands[1]:
            tier = "medium"
        else:
            tier = "high"
        return tier, score


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Applies temperature scaling for confidence calibration (MVP 6)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / temperature
