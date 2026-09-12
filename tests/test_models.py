"""Tests for model definitions and forward passes."""
from __future__ import annotations

import pytest
import torch

from cropguard.models import BaselineClassifier, MultimodalClassifier, build_backbone, feature_dim
from cropguard.taxonomy import TAXONOMY

N = len(TAXONOMY.classes)
BATCH = 2
IMG = 64


def _context():
    return dict(
        crop_id=torch.tensor([1, 2]),
        stage_idx=torch.tensor([0, 3]),
        region_idx=torch.tensor([4, 11]),
        weather=torch.rand(BATCH, 4),
    )


@pytest.mark.parametrize("fusion", ["gated", "concat", "image_only"])
def test_multimodal_with_context(fusion):
    m = MultimodalClassifier(num_classes=N, fusion=fusion, pretrained=False)
    out = m(torch.randn(BATCH, 3, IMG, IMG), **_context())
    assert out["logits"].shape == (BATCH, N)
    assert torch.allclose(out["probs"].sum(-1), torch.ones(BATCH), atol=1e-5)


@pytest.mark.parametrize("fusion", ["gated", "concat", "image_only"])
def test_multimodal_without_context(fusion):
    """Missing context must fall back to the image-only path, not crash."""
    m = MultimodalClassifier(num_classes=N, fusion=fusion, pretrained=False)
    out = m(torch.randn(BATCH, 3, IMG, IMG))
    assert out["logits"].shape == (BATCH, N)
    assert out["gate"] is None and out["context_vec"] is None


def test_gated_fusion_interpolates():
    m = MultimodalClassifier(num_classes=N, fusion="gated", pretrained=False)
    ctx = _context()
    out = m(torch.randn(BATCH, 3, IMG, IMG), **ctx)
    assert out["gate"] is not None
    assert out["gate"].shape == (BATCH, feature_dim("efficientnet_b0"))
    assert ((out["gate"] > 0) & (out["gate"] < 1)).all()


def test_baseline_accepts_and_ignores_context():
    m = BaselineClassifier(num_classes=N, pretrained=False)
    out = m(torch.randn(BATCH, 3, IMG, IMG), **_context())
    assert out["logits"].shape == (BATCH, N)
    assert out["gate"] is None


def test_invalid_fusion_rejected():
    with pytest.raises(ValueError):
        MultimodalClassifier(num_classes=N, fusion="espionage")


def test_invalid_backbone_rejected():
    with pytest.raises(ValueError):
        build_backbone("alexnetClassic")
    with pytest.raises(ValueError):
        feature_dim("alexnetClassic")


def test_features_only_path():
    m = MultimodalClassifier(num_classes=N, fusion="gated", pretrained=False)
    feats = m(torch.randn(BATCH, 3, IMG, IMG), **_context(), features_only=True)
    assert feats.shape == (BATCH, feature_dim("efficientnet_b0"))
