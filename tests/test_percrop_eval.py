"""§20 per-crop evaluation: minority-crop performance must never hide inside an average."""
from __future__ import annotations

import pytest
import torch

from cropguard.config import Config
from cropguard.taxonomy import TAXONOMY
from cropguard.training.evaluate import _metrics_over_loader
from cropguard.training.train import train


def test_percrop_metrics_structure(tiny_cfg: Config):
    """_metrics_over_loader returns per-crop accuracy/n + a worst-crop entry."""
    from cropguard.training.train import build_model

    model = build_model(tiny_cfg, num_classes=len(TAXONOMY.classes), pretrained=False)
    dm = __import__("cropguard.data.datamodule", fromlist=["CropDiseaseDataModule"]).CropDiseaseDataModule(tiny_cfg)
    loaders = dm.loaders(batch_size=4)
    metrics = _metrics_over_loader(model, loaders["test"], torch.device("cpu"), threshold=0.65)

    assert set(metrics) >= {"accuracy", "per_crop", "worst_crop"}
    assert metrics["per_crop"], "tiny fixture spans 3 crops — breakdown must be non-empty"
    for crop, m in metrics["per_crop"].items():
        assert isinstance(m["n"], int) and m["n"] > 0
        assert 0.0 <= m["accuracy"] <= 100.0
    # per-crop n must sum to the total n
    assert sum(m["n"] for m in metrics["per_crop"].values()) == metrics["n"]

    wc = metrics["worst_crop"]
    assert wc is not None and wc["crop"] in metrics["per_crop"]
    assert wc["accuracy"] == min(m["accuracy"] for m in metrics["per_crop"].values())


def test_evaluate_reports_percrop(tiny_cfg: Config, tmp_path):
    """evaluate() exposes lab_per_crop / lab_worst_crop at top level."""
    from cropguard.training.evaluate import evaluate

    train(tiny_cfg, tiny_cfg.training.out_dir)
    ckpt = tiny_cfg.training.out_dir / "checkpoint_epoch_1.pth"
    assert ckpt.exists()
    metrics = evaluate(tiny_cfg, ckpt)
    assert "lab_per_crop" in metrics and metrics["lab_per_crop"]
    assert metrics["lab_worst_crop"] is not None
    assert "lab_accuracy" in metrics  # legacy keys untouched
