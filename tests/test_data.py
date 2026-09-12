"""Tests for data validation, metadata encoding, and weather synthesis."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from cropguard.data.metadata import (
    WEATHER_KEYS,
    MetadataEncoder,
    crop_id_tensor,
    normalize_weather,
    region_index,
    stage_index,
    synthesize_weather,
)
from cropguard.data.validation import (
    load_validated_image,
    perceptual_hash,
    perceptual_hash_distance,
)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def test_perceptual_hash_identical_images():
    img = Image.new("RGB", (64, 64), color=(120, 90, 40))
    h1, h2 = perceptual_hash(img), perceptual_hash(img)
    assert perceptual_hash_distance(h1, h2) == 0.0


def test_perceptual_hash_different_images():
    # aHash compares structure: half-white vs full-white differ on half the bits.
    half = np.zeros((64, 64), dtype=np.uint8)
    half[:, :32] = 255
    a = Image.fromarray(half).convert("RGB")
    b = Image.fromarray(np.full((64, 64), 255, dtype=np.uint8)).convert("RGB")
    assert perceptual_hash_distance(perceptual_hash(a), perceptual_hash(b)) > 0.1


def test_load_validated_image_accepts_good(tmp_path):
    p = tmp_path / "ok.jpg"
    Image.new("RGB", (64, 48), color=(10, 200, 30)).save(p)
    out = load_validated_image(p)
    assert out is not None
    img, size, sha = out
    assert img.size == (64, 48) and size == (64, 48)
    assert len(sha) == 64


def test_load_validated_image_rejects_bad(tmp_path):
    assert load_validated_image(tmp_path / "missing.jpg") is None
    p = tmp_path / "corrupt.jpg"
    p.write_bytes(b"not an image at all")
    assert load_validated_image(p) is None


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #


def test_crop_id_tensor_name_and_int():
    assert crop_id_tensor("tomato") == 1
    assert crop_id_tensor(2) == 2
    with pytest.raises(ValueError):
        crop_id_tensor("banana")
    with pytest.raises(ValueError):
        crop_id_tensor(999)


def test_stage_and_region_indices():
    assert stage_index("seedling") == 0
    assert stage_index("Fruiting") == 3
    assert region_index("nashik") >= 0
    with pytest.raises(ValueError):
        stage_index("ancient")
    with pytest.raises(ValueError):
        region_index("paris")


def test_normalize_weather_bounds():
    vec = normalize_weather({"temperature_c": 5.0, "humidity_pct": 100.0,
                             "rainfall_mm_24h": 999.0, "rainfall_mm_7d": -5.0})
    assert all(0.0 <= v <= 1.0 for v in vec)
    assert len(vec) == len(WEATHER_KEYS)


def test_synthesize_weather_deterministic_and_seasonal():
    a = synthesize_weather("nashik", 7, "seed1")
    b = synthesize_weather("nashik", 7, "seed1")
    c = synthesize_weather("nashik", 12, "seed1")
    assert a.to_dict() == b.to_dict()  # deterministic
    assert a.to_dict()["rainfall_mm_7d"] > c.to_dict()["rainfall_mm_7d"]  # monsoon > dec
    assert all(k in a.to_dict() for k in WEATHER_KEYS)
    with pytest.raises(ValueError):
        synthesize_weather("nashik", 13)


def test_metadata_encoder_shapes():
    enc = MetadataEncoder(context_dim=32)
    out = enc(
        torch.tensor([1, 2]),
        torch.tensor([0, 3]),
        torch.tensor([4, 11]),
        torch.rand(2, 4),
    )
    assert out.shape == (2, 32)


def test_metadata_encoder_clamps_oov_ids():
    enc = MetadataEncoder(context_dim=16)
    out = enc(
        torch.tensor([999]),
        torch.tensor([999]),
        torch.tensor([999]),
        torch.rand(1, 4),
    )
    assert torch.isfinite(out).all()
