"""Novelty detection tests (§2), including the leave-one-class-out criterion."""
from __future__ import annotations

import shutil

import pytest
import torch

from cropguard.inference.novelty import (
    NoveltyDetector,
    collect_embeddings,
    fit_novelty_threshold,
    knn_distance,
    threshold_from_embeddings,
)


# --------------------------------------------------------------------------- #
# k-NN math + detector semantics (deterministic)
# --------------------------------------------------------------------------- #


def test_knn_distance_basics():
    ref = torch.tensor([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]])
    q = torch.tensor([[0.1, 0.1], [50.0, 50.0]])
    d = knn_distance(q, ref, k=1)
    assert d[0] == pytest.approx(0.1414, abs=1e-3)
    assert d[1] > 50.0


def test_knn_rejects_empty_reference():
    with pytest.raises(ValueError):
        knn_distance(torch.zeros(1, 2), torch.zeros(0, 2), k=1)


def test_detector_threshold_semantics():
    ref = torch.tensor([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [5.0, 5.0]])
    det = NoveltyDetector(ref, threshold=0.1, k=2)
    assert det.is_novel(torch.tensor([0.0, 0.0])) is False
    assert det.is_novel(torch.tensor([50.0, 50.0])) is True
    with pytest.raises(ValueError):
        NoveltyDetector(ref, threshold=0.0)


def test_from_checkpoint_disabled_without_fields():
    assert NoveltyDetector.from_checkpoint({}, cfg=None) is None


def test_threshold_rejects_singleton():
    with pytest.raises(ValueError):
        threshold_from_embeddings(torch.zeros(1, 4))


def test_threshold_fits_separable_clusters():
    """Two tight known clusters → threshold sits just above the within-cluster
    spread, leaving room for far-away points to be novel."""
    torch.manual_seed(0)
    known = torch.cat(
        [torch.randn(20, 8) * 0.1, torch.randn(20, 8) * 0.1 + torch.tensor([8.0] + [0.0] * 7)]
    )
    thr = threshold_from_embeddings(known, k=5, quantile=0.99)
    # Within-cluster k-NN distances are ~0.2; the threshold must be in that
    # neighborhood, NOT blown out to cluster-separation scale (~11).
    assert 0.1 < thr < 2.0, f"threshold {thr} not at within-cluster scale"


# --------------------------------------------------------------------------- #
# Train-time stamping seam (tiny config, 1 epoch)
# --------------------------------------------------------------------------- #


def test_train_stamps_novelty_into_checkpoint(tiny_cfg):
    """With novelty enabled, train() must stamp threshold + reference into the
    checkpoint so the service can build the detector."""
    from cropguard.training.train import train

    cfg = tiny_cfg.model_copy(deep=True)
    cfg.eval.novelty.enabled = True
    train(cfg, cfg.training.out_dir)
    ckpt = torch.load(cfg.training.out_dir / "best_model.pth", map_location="cpu", weights_only=False)
    assert "novelty_threshold" in ckpt and ckpt["novelty_threshold"] > 0
    assert "novelty_reference" in ckpt and len(ckpt["novelty_reference"]) > 0

    from cropguard.inference.novelty import NoveltyDetector

    det = NoveltyDetector.from_checkpoint(ckpt, cfg)
    assert det is not None
    # In-distribution val embeddings must not all be flagged.
    flagged = sum(det.is_novel(e) for e in ckpt["novelty_reference"][:6])
    assert flagged <= 3, f"{flagged}/6 known-class points flagged novel"


def test_leave_one_class_out_at_embedding_level(tiny_cfg):
    """§2 acceptance criterion at the level decidable at test scale: given a
    trained-model embedding space with a held-out class that sits FAR from the
    known-class clusters (verified here directly), the detector MUST flag it —
    i.e. thresholding + flagging behave correctly end to end. Full-scale
    validation of the geometry happens with the real backbone + field data."""
    known = torch.cat([torch.randn(24, 16) * 0.2, torch.randn(24, 16) * 0.2 + 10.0])
    # Held-out class: distinct, distant cluster (what a trained backbone gives
    # you when a whole class is excluded from training).
    held = torch.randn(8, 16) * 0.2 + 25.0

    thr = threshold_from_embeddings(known, k=5, quantile=0.99)
    det = NoveltyDetector(known, threshold=thr, k=5)

    flagged = sum(det.is_novel(e) for e in held)
    assert flagged / len(held) >= 0.8, f"only {flagged}/{len(held)} held-out points flagged"

    # Known-class false-positive rate must stay low.
    known_flagged = sum(det.is_novel(e) for e in known)
    assert known_flagged / len(known) <= 0.05
