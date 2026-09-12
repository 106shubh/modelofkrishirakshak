"""Baseline: image-only classifier (visual evidence only, no context)."""
from __future__ import annotations

import torch
from torch import nn

from .backbones import FeatureBackbone


class BaselineClassifier(nn.Module):
    """Pretrained backbone + linear head. The ablation baseline."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = FeatureBackbone(backbone, pretrained=pretrained, dropout=dropout)
        self.head = nn.Linear(self.backbone.feat_dim, num_classes)

    def forward(
        self,
        image: torch.Tensor,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        feats = self.backbone(image)
        logits = self.head(feats)
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
            "features": feats,
            "gate": None,
            "context_vec": None,
        }

    def feature_forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return the feature vector for Grad-CAM / severity head usage."""
        return self.backbone(image)