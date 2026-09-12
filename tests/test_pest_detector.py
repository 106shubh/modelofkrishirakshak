"""Tests for the pest-detection head: detector, adapter, router integration."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from cropguard.config import Config
from cropguard.inference.engine import ModelRouter
from cropguard.models.pest_detector import (
    Detection,
    HeuristicSpotDetector,
    PestDetectionAdapter,
    bbox_coverage,
    build_pest_detector,
    denormalize_boxes,
)


def _leaf_with_spot(radius: int = 30) -> torch.Tensor:
    """Green leaf image with one big dark spot, as a normalized (1,3,H,W) tensor."""
    from cropguard.data.transforms import IMAGENET_MEAN, IMAGENET_STD

    size = 128
    img = np.full((size, size, 3), 0, dtype=np.uint8)
    img[..., 1] = 140  # green leaf
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    mask = (xx - size // 2) ** 2 + (yy - size // 2) ** 2 < radius**2
    img[mask] = 15  # dark spot
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0)


# --------------------------------------------------------------------------- #
# bbox utilities
# --------------------------------------------------------------------------- #


def test_denormalize_boxes_clamps():
    boxes = torch.tensor([[0.0, 0.0, 0.5, 0.5], [-0.2, 0.9, 1.5, 1.2]])
    px = denormalize_boxes(boxes, width=100, height=200)
    assert px[0].tolist() == [0.0, 0.0, 50.0, 100.0]
    assert px[1].tolist() == [0.0, 180.0, 100.0, 200.0]  # clamped to image


def test_bbox_coverage():
    full = [Detection(0, "x", 0.9, (0.0, 0.0, 50.0, 50.0))]
    assert bbox_coverage(full, 100, 100) == pytest.approx(0.25)
    assert bbox_coverage([], 100, 100) == 0.0
    # Union caps at 1.0
    huge = [Detection(0, "x", 0.9, (0.0, 0.0, 200.0, 200.0))]
    assert bbox_coverage(huge, 100, 100) == 1.0


# --------------------------------------------------------------------------- #
# Heuristic stand-in detector
# --------------------------------------------------------------------------- #


def test_heuristic_detector_finds_dark_spot():
    img = _leaf_with_spot(radius=30)[0]  # (3,H,W)
    # Un-normalize to a PIL image the way the adapter does.
    from cropguard.data.transforms import IMAGENET_MEAN, IMAGENET_STD

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    arr = ((img.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    dets = HeuristicSpotDetector().detect(Image.fromarray(arr))
    assert len(dets) == 1
    x1, y1, x2, y2 = dets[0].xyxy
    assert (x2 - x1) * (y2 - y1) > 1000  # a real blob, not a speck
    assert 0.0 < dets[0].confidence <= 1.0


def test_heuristic_detector_clean_leaf_no_detections():
    img = Image.new("RGB", (128, 128), color=(90, 140, 60))
    assert HeuristicSpotDetector().detect(img) == []


# --------------------------------------------------------------------------- #
# Adapter: detector output → router contract (NFR4)
# --------------------------------------------------------------------------- #


def test_adapter_maps_onto_router_contract():
    adapter = PestDetectionAdapter(HeuristicSpotDetector())
    out = adapter.predict(_leaf_with_spot())
    # Router-contract keys present…
    assert "logits" in out and "probs" in out and "features" in out
    # …plus detector-specific extras.
    assert out["detections"] and out["raw_severity"] > 0.0
    assert out["image_size"] == (128, 128)
    # Pseudo-logits: sigmoid max ≈ top detection confidence.
    top_conf = max(d.confidence for d in out["detections"])
    assert torch.sigmoid(out["logits"]).max().item() == pytest.approx(top_conf, abs=1e-3)


def test_adapter_clean_image_zero_confidence():
    adapter = PestDetectionAdapter(HeuristicSpotDetector())
    from cropguard.data.transforms import eval_transform

    t = eval_transform(128)(Image.new("RGB", (256, 256), (90, 140, 60))).unsqueeze(0)
    out = adapter.predict(t)
    assert out["detections"] == []
    assert out["raw_severity"] == 0.0
    assert torch.sigmoid(out["logits"]).max().item() < 0.5  # honest low confidence


# --------------------------------------------------------------------------- #
# Router integration (dual-model contract, Issue 2 + NFR4)
# --------------------------------------------------------------------------- #


class _FakeClassifier:
    def __call__(self, image, type_hint=None, **ctx):
        n = 4
        logits = torch.tensor([[4.0] + [-1.0] * (n - 1)])
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
            "features": torch.zeros(1, 8),
        }


def test_router_picks_higher_confidence_side():
    adapter = PestDetectionAdapter(HeuristicSpotDetector())
    # Disease classifier confident (sigmoid(4)≈0.98) beats the adapter's spot conf.
    router = ModelRouter(_FakeClassifier(), adapter)
    out = router.predict(_leaf_with_spot())
    assert out.get("detections") is None  # classifier side won

    # With a clean image the adapter's confidence ≈ 0 → classifier still wins,
    # but forcing type_hint=pest returns the detector output.
    out_p = router.predict(_leaf_with_spot(), type_hint="pest")
    assert out_p.get("detections") is not None


def test_build_pest_detector_factory():
    cfg = Config()
    assert build_pest_detector(cfg) is None  # off
    cfg.model.pest_detector = "heuristic"
    assert isinstance(build_pest_detector(cfg), PestDetectionAdapter)
    cfg.model.pest_detector = "yolo"
    with pytest.raises(ValueError):
        build_pest_detector(cfg)  # yolo without weights
    with pytest.raises(ValueError):
        Config.model_validate({"model": {"pest_detector": "moonwalk"}})
