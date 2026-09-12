"""Image loading, integrity validation, and perceptual hashing.

Used both at dataset build time (near-duplicate filtering) and at inference
time (rejecting corrupt/undersized inputs before they reach the model).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

MIN_SIZE = 32  # smallest edge we accept (px)


def perceptual_hash(img: Image.Image, size: int = 8) -> np.ndarray:
    """Average-hash: mean-thresholded grayscale thumbnail, flattened to bits.

    Returns a 64-bit hash (8x8). Flat images hash to all-zeros by definition;
    that is fine — they compare equal to each other, which dedup wants.
    """
    gray = img.convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32)
    return (arr > arr.mean()).astype(np.uint8).ravel()


def perceptual_hash_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Hamming distance between two perceptual hashes, as a fraction in [0, 1]."""
    if h1.shape != h2.shape:
        raise ValueError(f"Hash shape mismatch: {h1.shape} vs {h2.shape}")
    return float((h1 != h2).mean())


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of a file's bytes (deterministic across processes)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_validated_image(
    path: str | Path,
) -> tuple[Image.Image, tuple[int, int], str] | None:
    """Load an image after basic integrity validation.

    Returns (RGB image, (width, height), sha256) or None when the file is
    missing, corrupt, truncated, or smaller than MIN_SIZE.
    """
    p = Path(path)
    try:
        if not p.is_file():
            log.warning("Image not found: %s", p)
            return None
        with Image.open(p) as probe:
            probe.verify()  # raises on corrupt/truncated files
        img = Image.open(p).convert("RGB")
        w, h = img.size
        if w < MIN_SIZE or h < MIN_SIZE:
            log.warning("Image too small (%dx%d): %s", w, h, p)
            return None
        return img, (w, h), file_sha256(p)
    except Exception as exc:  # noqa: BLE001 — any decode failure means "invalid"
        log.warning("Invalid image %s: %s", p, exc)
        return None
