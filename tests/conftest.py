"""Shared pytest fixtures: tiny synthetic dataset + config overrides."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cropguard.config import Config


def _make_tiny_plantvillage(root: Path, per_class: int = 6) -> Path:
    """Create a tiny synthetic dataset mirroring the PlantVillage layout.

    Images are structurally distinct: each class gets a different stripe
    orientation/frequency, each sample a different stripe phase. Distinct
    structure matters because dataset build dedups by perceptual hash —
    brightness-only variation hashes identically and collapses the class.
    """
    names = [
        "Tomato___healthy",
        "Tomato___Late_blight",
        "Potato___Early_blight",
        "Grape___Black_rot",
    ]
    size = 48
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    # Widely-spaced scales: small period steps collapse to identical aHash bits.
    scales = [6, 8, 11, 15, 20, 26]
    for ci, name in enumerate(names):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        orientations = {
            0: lambda s: xx % s,
            1: lambda s: yy % s,
            2: lambda s: (xx + yy) % s,
            3: lambda s: (((xx // s) + (yy // s)) % 2) * 10,  # checkerboard
        }
        orient = orientations[ci % len(orientations)]
        for i in range(per_class):
            p = orient(scales[i % len(scales)])
            img = np.stack([p * 20, p * 20 + 10, p * 20 + 30], axis=-1)
            img = np.clip(img, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(d / f"img_{i:03d}.jpg", quality=92)
    return root


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """A tiny synthetic PlantVillage-style color dataset."""
    return _make_tiny_plantvillage(tmp_path / "PlantVillage" / "color")


@pytest.fixture
def tiny_cfg(tiny_dataset: Path, tmp_path: Path) -> Config:
    """Config pointing at the tiny dataset with fast settings."""
    cfg = Config()
    cfg.data.root = tiny_dataset
    cfg.data.image_size = 32
    cfg.data.workers = 0
    cfg.data.train_subset_per_class = None
    cfg.data.val_subset_per_class = 2
    cfg.data.min_samples_per_class = 4
    cfg.model.pretrained = False
    cfg.weather.provider = "synthesize"  # hermetic: no live API calls in tests
    cfg.training.amp = "off"
    cfg.training.out_dir = tmp_path / "checkpoints"
    cfg.training.epochs = 1
    cfg.training.batch_size = 4
    return cfg
