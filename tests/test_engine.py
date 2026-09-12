"""Tests for the inference engine: severity normalization + calibration."""
from __future__ import annotations

import pytest
import torch

from cropguard.config import Config
from cropguard.inference.engine import (
    SEVERITY_EVIDENCE_FLOOR,
    SeverityEngine,
    normalize_severity,
    temperature_scale,
)


def test_normalize_severity_identity_and_clamp():
    assert normalize_severity(-0.5) == 0.0
    assert normalize_severity(0.25) == 0.25
    assert normalize_severity(2.0) == 1.0


def test_normalize_severity_minmax_window():
    # A raw signal living on [0, 100] (e.g. bbox coverage %) maps into [0, 1].
    assert normalize_severity(50.0, lo=0.0, hi=100.0) == 0.5
    assert normalize_severity(-10.0, lo=0.0, hi=100.0) == 0.0


def test_normalize_severity_rejects_degenerate_window():
    with pytest.raises(ValueError):
        normalize_severity(0.5, lo=1.0, hi=1.0)


def test_severity_healthy_is_always_low():
    """Issue 3 fix: a confident healthy read must not band as high severity."""
    eng = SeverityEngine(Config())
    tier, score = eng.calculate("Tomato___healthy", confidence=0.95)
    assert tier == "low"
    assert score == 0.0


def test_severity_tiers_from_confidence_proxy():
    eng = SeverityEngine(Config())
    # confidence 0.55 → 0.1 above the evidence floor → low band
    tier, score = eng.calculate("Tomato___Late_blight", confidence=0.55)
    assert tier == "low"
    assert score == pytest.approx((0.55 - SEVERITY_EVIDENCE_FLOOR) / 0.5)
    # 0.75 → medium band
    tier, score = eng.calculate("Tomato___Late_blight", confidence=0.75)
    assert tier == "medium"
    # 0.95 → high band
    tier, _ = eng.calculate("Tomato___Late_blight", confidence=0.95)
    assert tier == "high"


def test_severity_raw_signal_path():
    """V2 seam: a path-specific raw signal flows through the same normalization."""
    eng = SeverityEngine(Config())
    tier, score = eng.calculate("Tomato___Late_blight", raw_severity=0.70)
    assert tier == "high"  # bands [0.34, 0.67): 0.70 is above the top edge
    assert score == pytest.approx(0.70)
    tier, score = eng.calculate("Tomato___Late_blight", raw_severity=0.50)
    assert tier == "medium"
    # Out-of-window raw values clamp, not explode.
    tier, score = eng.calculate("Tomato___Late_blight", raw_severity=65.0)
    assert tier == "high" and score == 1.0


def test_temperature_scale_rejects_bad_temperature():
    logits = torch.randn(2, 4)
    with pytest.raises(ValueError):
        temperature_scale(logits, temperature=0.0)
    with pytest.raises(ValueError):
        temperature_scale(logits, temperature=-1.0)


def test_temperature_scale_softens_with_t_gt_1():
    logits = torch.tensor([[4.0, 0.0, 0.0, 0.0]])
    sharp = torch.softmax(logits, dim=-1).max().item()
    soft = torch.softmax(temperature_scale(logits, temperature=2.0), dim=-1).max().item()
    assert soft < sharp  # T > 1 spreads the distribution
