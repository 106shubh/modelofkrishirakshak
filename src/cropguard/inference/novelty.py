"""Novelty detection over the trained embedding space (§2 unsupervised).

Gives the abstention system an escalation reason that is genuinely different
from "low confidence": an image far from every known-class cluster is flagged
reason='novel_presentation' ("this doesn't match anything we know"), whereas
low confidence says "we're unsure WHICH known class this is". Officers need
those two statements distinguished.

Method: k-NN distance in the penultimate embedding space, fitted ONLY on
known-class training examples. Simple, inspectable, and its acceptance test is
the doc's own: leave one class out of training entirely and confirm the
detector flags it as novel instead of misclassifying it confidently.

The learned-model verdict is never overridden here — novelty is a third input
(alongside confidence and the rule layer) to the escalation decision.
"""
from __future__ import annotations

import logging

import torch
from torch import nn

log = logging.getLogger(__name__)


@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over a loader, returning (embeddings (N,D), targets (N,))."""
    model.eval()
    feats, targets = [], []
    for batch in loader:
        images = batch["image"].to(device)
        out = model(
            images,
            crop_id=batch["crop_id"].to(device),
            stage_idx=batch["stage_idx"].to(device),
            region_idx=batch["region_idx"].to(device),
            weather=batch["weather"].to(device),
        )
        feats.append(out["features"].cpu())
        targets.append(batch["target"])
    return torch.cat(feats), torch.cat(targets)


def threshold_from_embeddings(
    embeddings: torch.Tensor, k: int = 5, quantile: float = 0.99
) -> float:
    """Novelty threshold = `quantile` of mean k-NN distances among known-class
    embeddings (self-matches excluded via the diagonal)."""
    if len(embeddings) < 2:
        raise ValueError("novelty fit needs at least 2 known-class embeddings")
    d = torch.cdist(embeddings, embeddings)
    d.fill_diagonal_(float("inf"))
    k = min(k, len(embeddings) - 1)
    baseline = d.topk(k, largest=False).values.mean(dim=1)
    return float(torch.quantile(baseline, quantile))


@torch.no_grad()
def fit_novelty_threshold(
    cfg,
    model: nn.Module,
    device: torch.device,
    split: str = "val",
    quantile: float = 0.99,
) -> float:
    """Fit the novelty threshold over the held-out split (known classes only).
    Stamped into the checkpoint at train time; the service treats missing
    values as "novelty disabled"."""
    from ..data.datamodule import CropDiseaseDataModule

    datamodule = CropDiseaseDataModule(cfg)
    loaders = datamodule.loaders(batch_size=cfg.training.batch_size)
    if split not in loaders:
        raise ValueError(f"Unknown split '{split}'. Available: {sorted(loaders)}")

    embeddings, _ = collect_embeddings(model, loaders[split], device)
    threshold = threshold_from_embeddings(embeddings, k=cfg.eval.novelty.k, quantile=quantile)
    log.info(
        "Novelty threshold %.4f fitted on '%s' split (quantile=%.2f, n=%d)",
        threshold, split, quantile, len(embeddings),
    )
    return threshold


@torch.no_grad()
def knn_distance(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int = 5,
    batch_size: int = 512,
) -> torch.Tensor:
    """Mean L2 distance from each query to its k nearest reference points.

    Batched to keep memory bounded: (N,D) x (M,D) full pairwise would blow up
    at real corpus sizes.
    """
    if reference.numel() == 0:
        raise ValueError("novelty reference set is empty")
    k = min(k, len(reference))
    out = []
    for i in range(0, len(query), batch_size):
        q = query[i : i + batch_size]
        d = torch.cdist(q, reference)  # (B, M)
        out.append(d.topk(k, largest=False).values.mean(dim=1))
    return torch.cat(out)


class NoveltyDetector:
    """Thresholds embedding-space distances against a fitted reference set."""

    def __init__(self, reference_embeddings: torch.Tensor, threshold: float, k: int = 5) -> None:
        if threshold <= 0:
            raise ValueError(f"novelty threshold must be > 0, got {threshold}")
        self.reference = reference_embeddings
        self.threshold = threshold
        self.k = k

    @classmethod
    def from_checkpoint(cls, ckpt: dict, cfg) -> "NoveltyDetector | None":
        """Build from a checkpoint blob; None when not fitted (disabled)."""
        ref = ckpt.get("novelty_reference")
        thr = ckpt.get("novelty_threshold")
        if ref is None or thr is None or not cfg.eval.novelty.enabled:
            return None
        return cls(ref, float(thr), k=cfg.eval.novelty.k)

    @torch.no_grad()
    def is_novel(self, embedding: torch.Tensor) -> bool:
        """True when the embedding's mean k-NN distance exceeds the threshold."""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        dist = knn_distance(embedding, self.reference, k=self.k)
        return bool(dist.item() > self.threshold)
