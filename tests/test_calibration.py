"""Tests for confidence calibration (MVP-6: fitted temperature)."""
from __future__ import annotations

import torch
import torch.nn as nn

from cropguard.inference.service import _resolve_temperature
from cropguard.training.evaluate import _fit_T, fit_temperature
from cropguard.training.train import build_model
from cropguard.taxonomy import TAXONOMY


def _nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(nn.CrossEntropyLoss()(logits, targets))


def test_fit_T_softens_overconfident_logits():
    """Overconfident wrong predictions ⇒ optimal T > 1 and lower NLL."""
    torch.manual_seed(0)
    n, c = 64, 4
    targets = torch.randint(0, c, (n,))
    # Big-margin logits that are often wrong → overconfidence.
    wrong = (targets + 1) % c
    logits = torch.full((n, c), 0.1)
    logits[torch.arange(n), wrong] = 12.0

    nll_before = _nll(logits, targets)
    T = _fit_T(logits, targets)
    nll_after = _nll(logits / T, targets)

    assert T > 1.0
    assert nll_after < nll_before


def test_fit_T_stays_near_one_when_calibrated():
    """Labels sampled from the model's own softmax ⇒ perfectly calibrated ⇒ T ≈ 1."""
    torch.manual_seed(0)
    n, c = 256, 4
    logits = torch.randn(n, c)
    targets = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
    T = _fit_T(logits, targets)
    assert 0.5 < T < 2.0


def test_fit_temperature_clamps_degenerate_split(tiny_cfg, monkeypatch):
    """A degenerate split drives the raw optimum to T→∞; the public fitter must
    clamp instead of shipping an absurd temperature to the service."""
    import cropguard.training.evaluate as ev

    monkeypatch.setattr(ev, "_fit_T", lambda *a, **k: 1111106.5)
    device = torch.device("cpu")
    model = build_model(tiny_cfg, num_classes=4, pretrained=False)
    T = ev.fit_temperature(tiny_cfg, model, device, split="val")
    assert T == 4.0


def test_fit_temperature_on_tiny_split(tiny_cfg):
    device = torch.device("cpu")
    model = build_model(tiny_cfg, num_classes=len(TAXONOMY.classes), pretrained=False)
    T = fit_temperature(tiny_cfg, model, device, split="val")
    assert T > 0.0
    assert torch.isfinite(torch.tensor(T))


def test_resolve_temperature_prefers_config(tmp_path):
    from cropguard.config import Config

    ckpt = tmp_path / "m.pth"
    torch.save({"calibration_temperature": 3.3}, ckpt)

    cfg = Config()
    cfg.eval.temperature = 2.0
    assert _resolve_temperature(cfg, ckpt) == 2.0  # explicit config wins

    cfg.eval.temperature = None
    assert _resolve_temperature(cfg, ckpt) == 3.3  # checkpoint-stamped T next

    empty = tmp_path / "empty.pth"
    torch.save({}, empty)
    assert _resolve_temperature(cfg, empty) == 1.0  # legacy checkpoint default
