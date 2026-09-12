"""Tests for the FastAPI inference service."""
from __future__ import annotations

import base64
import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from cropguard.config import Config
from cropguard.inference.service import create_app
from cropguard.taxonomy import TAXONOMY
from cropguard.training.train import train


@pytest.fixture
def client(tiny_cfg: Config):
    """App with a trained tiny checkpoint loaded."""
    train(tiny_cfg, tiny_cfg.training.out_dir)
    app = create_app(tiny_cfg)
    app.state.model = _load(tiny_cfg, app.state.device)
    app.state.ready = True
    with TestClient(app) as c:
        yield c


def _load(cfg: Config, device: torch.device):
    from cropguard.inference.service import _load_model

    return _load_model(cfg, device)


def _jpeg_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(90, 140, 60)).save(buf, format="JPEG")
    return buf.getvalue()


def _jpeg_b64(size: tuple[int, int] = (64, 64)) -> str:
    """JPEG bytes as base64 — how image_bytes travels in a JSON body."""
    return base64.b64encode(_jpeg_bytes(size)).decode("ascii")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["num_classes"] == len(TAXONOMY.classes)
    assert "nashik" in body["regions"]
    assert body["hardware"]["torch"]


def test_predict_json(client):
    r = client.post("/predict", json={"image_bytes": _jpeg_b64(), "region": "nashik", "month": 7})
    assert r.status_code == 200
    body = r.json()
    assert body["class_name"] in TAXONOMY.classes
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["pest_disease_id"] == TAXONOMY.pest_disease_id(body["class_name"])
    assert body["context"]["region"] == "nashik"
    assert len(body["probabilities"]) == len(TAXONOMY.classes)
    # Issue 4/5: advisories always present (possibly empty); causal block carries
    # the cold-start history flag.
    assert isinstance(body["advisories"], list)
    ca = body["causal_analysis"]
    assert ca["cold_start"] == (ca["history_count"] == 0)
    assert ca["calibration_temperature"] > 0


def test_predict_rejects_bad_image(client):
    r = client.post("/predict", json={"image_bytes": base64.b64encode(b"garbage").decode()})
    assert r.status_code == 400


def test_predict_rejects_too_large(client):
    cfg_max = client.app.state.config.inference.max_image_bytes
    r = client.post(
        "/predict",
        json={"image_bytes": base64.b64encode(b"x" * (cfg_max + 1)).decode()},
    )
    assert r.status_code == 413


def test_predict_file_endpoint(client):
    r = client.post(
        "/predict/file",
        files={"file": ("leaf.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"region": "pune"},
    )
    assert r.status_code == 200
    assert r.json()["class_name"] in TAXONOMY.classes


def test_predict_file_rejects_bad_type(client):
    r = client.post(
        "/predict/file",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400


def test_predict_high_risk_rule_attaches_advisory(client):
    """Issue 4: a high-risk rule flag must surface in farmer-facing advisories,
    even when the (untrained) model is overconfident and nothing escalates."""
    r = client.post(
        "/predict",
        json={
            "image_bytes": _jpeg_b64(),
            "region": "nashik",
            "month": 7,  # monsoon synthesis → high humidity + rain
            "weather": {
                "temperature_c": 15.0,
                "humidity_pct": 95.0,
                "rainfall_mm_24h": 12.0,
                "rainfall_mm_7d": 80.0,
            },
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["causal_analysis"]["flagged"] is True
    assert any("Rule layer advisory" in a for a in body["advisories"])


def test_predict_before_ready(tmp_path):
    """When the model fails to load, /predict must 503 (not 500)."""
    cfg = Config()
    cfg.training.out_dir = tmp_path / "empty"  # no checkpoint → startup can't load
    app = create_app(cfg)
    app.state.ready = False
    app.state.model = None
    with TestClient(app) as c:
        r = c.post("/predict", json={"image_bytes": _jpeg_b64()})
        assert r.status_code == 503
