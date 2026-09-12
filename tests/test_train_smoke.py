"""End-to-end smoke tests: train → checkpoint → evaluate → predict.

Uses the tiny synthetic dataset and tiny images so it runs on CPU in seconds.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from cropguard.config import Config
from cropguard.data.dataset import CropDiseaseDataModule
from cropguard.taxonomy import TAXONOMY
from cropguard.training.predict import predict, severity_band
from cropguard.training.train import build_model, train


@pytest.fixture
def trained_checkpoint(tiny_cfg: Config):
    """Run a minimal training loop and return (cfg, checkpoint_path)."""
    tiny_cfg.training.epochs = 1
    train(tiny_cfg, tiny_cfg.training.out_dir)
    ckpt = tiny_cfg.training.out_dir / "best_model.pth"
    assert ckpt.exists()
    return tiny_cfg, ckpt


def test_train_writes_checkpoint(trained_checkpoint):
    cfg, ckpt = trained_checkpoint
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in blob
    assert blob["epoch"] == 0


def test_evaluate_smoke(trained_checkpoint):
    from cropguard.training.evaluate import evaluate

    cfg, ckpt = trained_checkpoint
    metrics = evaluate(cfg, ckpt)
    # §1: dual-split contract — lab-condition keys always present.
    assert 0.0 <= metrics["lab_accuracy"] <= 100.0
    assert 0.0 <= metrics["lab_abstain_rate"] <= 1.0


def test_predict_single_image(trained_checkpoint):
    cfg, ckpt = trained_checkpoint
    device = torch.device("cpu")
    model = build_model(cfg, num_classes=len(TAXONOMY.classes), pretrained=False).to(device)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])

    # Grab a real (synthetic) image from the dataset folder.
    img_path = next(iter(cfg.data.root.rglob("*.jpg")))
    result = predict(
        model=model,
        image_path=img_path,
        crop_id=1,
        stage="vegetative",
        region="nashik",
        month=7,
        weather=None,
        device=device,
        cfg=cfg,
    )
    assert "error" not in result
    assert result["class_name"] in TAXONOMY.classes
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["severity"] in {"low", "medium", "high"}
    assert isinstance(result["abstain"], bool)
    assert len(result["probabilities"]) == len(TAXONOMY.classes)


def test_predict_rejects_invalid_image(tiny_cfg: Config, tmp_path: Path):
    device = torch.device("cpu")
    model = build_model(tiny_cfg, num_classes=len(TAXONOMY.classes), pretrained=False)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"garbage")
    result = predict(
        model=model,
        image_path=bad,
        crop_id=None, stage=None, region=None, month=None, weather=None,
        device=device, cfg=tiny_cfg,
    )
    assert result == {"error": "invalid image"}


def test_severity_band():
    bands = [0.34, 0.67]
    assert severity_band(0.10, bands) == "low"
    assert severity_band(0.50, bands) == "medium"
    assert severity_band(0.90, bands) == "high"
