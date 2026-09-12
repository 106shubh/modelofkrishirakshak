"""Tests for §1 supervised upgrades: v2_s backbone, label smoothing, dual-split eval."""
from __future__ import annotations

import shutil

import pytest

from cropguard.config import Config
from cropguard.models import build_backbone, feature_dim
from cropguard.training.evaluate import evaluate
from cropguard.training.train import build_model
from cropguard.taxonomy import TAXONOMY


def test_efficientnet_v2_s_registered():
    assert feature_dim("efficientnet_v2_s") == 1280
    m = build_backbone("efficientnet_v2_s", pretrained=False)
    out = m(torch_zeros := __import__("torch").randn(1, 3, 64, 64))
    assert out.shape == (1, 1280)


def test_build_model_supports_v2_s(tiny_cfg):
    tiny_cfg.model.backbone = "efficientnet_v2_s"
    model = build_model(tiny_cfg, num_classes=len(TAXONOMY.classes), pretrained=False)
    import torch

    out = model(torch.randn(1, 3, 64, 64))
    assert out["logits"].shape == (1, len(TAXONOMY.classes))


def test_label_smoothing_config_wiring(tiny_cfg):
    """cfg.eval.label_smoothing must flow into the training criterion (§1)."""
    import inspect

    from cropguard.training import train as train_mod

    src = inspect.getsource(train_mod)
    assert "label_smoothing=cfg.eval.label_smoothing" in src, (
        "criterion must use cfg.eval.label_smoothing"
    )
    cfg = Config()
    cfg.eval.label_smoothing = 0.1
    assert 0.0 < cfg.eval.label_smoothing < 1.0


def _any_checkpoint(out_dir):
    """best_model.pth only exists when val_acc improved; fall back to the epoch
    checkpoint (an untrained net can score exactly 0 on the tiny val split)."""
    best = out_dir / "best_model.pth"
    return best if best.exists() else out_dir / "checkpoint_epoch_1.pth"


def test_dual_split_eval_reports_lab_only_without_field_root(tiny_cfg, tmp_path):
    """Without field_root, evaluate() reports lab metrics only (no crash)."""
    tiny_cfg.training.epochs = 1
    from cropguard.training.train import train

    train(tiny_cfg, tiny_cfg.training.out_dir)
    metrics = evaluate(tiny_cfg, _any_checkpoint(tiny_cfg.training.out_dir))
    assert "lab_accuracy" in metrics
    assert "field_accuracy" not in metrics
    assert "domain_gap" not in metrics


def test_dual_split_eval_reports_field_split_and_domain_gap(tiny_cfg, tmp_path):
    """With a field split present, lab and field accuracies must be reported
    SEPARATELY (§1 acceptance criterion) plus the domain gap."""
    import torch
    from PIL import Image

    tiny_cfg.training.epochs = 1
    from cropguard.training.train import train

    train(tiny_cfg, tiny_cfg.training.out_dir)

    # Build a "field" split in the <plant>/color layout (resolve_split_paths
    # needs the variant layer for split paths to resolve).
    field_root = tmp_path / "fieldplant" / "color"
    field_root.mkdir(parents=True)
    size = 48
    import numpy as np

    pattern = np.arange(size, dtype="int") % 255
    for name in ("Tomato___healthy", "Tomato___Late_blight", "Potato___Early_blight", "Grape___Black_rot"):
        d = field_root / name
        d.mkdir(parents=True)
        for i in range(4):
            # Structurally distinct stripes per image — identical-looking
            # images would collapse under the perceptual-hash dedup.
            band = ((pattern * (i + 3)) % 255).astype("uint8")
            img = np.stack([band, band, band], axis=-1)
            Image.fromarray(img).save(d / f"f_{i:03d}.jpg", quality=92)

    tiny_cfg.data.field_root = field_root
    metrics = evaluate(tiny_cfg, _any_checkpoint(tiny_cfg.training.out_dir))
    assert "lab_accuracy" in metrics
    assert "field_accuracy" in metrics
    assert "domain_gap" in metrics
    assert metrics["domain_gap"] == pytest.approx(
        metrics["field_accuracy"] - metrics["lab_accuracy"]
    )
    # The two splits are reported independently — never blended.
    assert set(metrics) >= {"lab_accuracy", "lab_loss", "lab_abstain_rate", "field_accuracy", "field_loss", "field_abstain_rate"}
