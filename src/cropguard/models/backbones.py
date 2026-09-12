"""Vision backbone factory (torchvision).

Supported: efficientnet_b0/b1, convnext_tiny, vit_b_16, swin_t.
Each returns a backbone whose classifier head has been stripped so it emits a
feature vector, plus the name of its last convolutional layer (for Grad-CAM).
"""
from __future__ import annotations

import logging

import torch
from torch import nn
from torchvision import models

log = logging.getLogger(__name__)

_CANDIDATES = {
    "efficientnet_b0": (models.efficientnet_b0, "features.7.0.block.5"),
    "efficientnet_b1": (models.efficientnet_b1, "features.7.0.block.5"),
    "efficientnet_v2_s": (models.efficientnet_v2_s, "features.7.0.block.5"),
    "convnext_tiny": (models.convnext_tiny, "features.9"),
    "vit_b_16": (models.vit_b_16, None),  # attention — Grad-CAM handled separately
    "swin_t": (models.swin_t, None),
}

# Feature dimensions of the penultimate representation per backbone.
_FEATURE_DIMS = {
    "efficientnet_b0": 1280,
    "efficientnet_b1": 1280,
    "efficientnet_v2_s": 1280,
    "convnext_tiny": 768,
    "vit_b_16": 768,
    "swin_t": 768,
}


def feature_dim(backbone: str) -> int:
    if backbone not in _FEATURE_DIMS:
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose from {sorted(_FEATURE_DIMS)}")
    return _FEATURE_DIMS[backbone]


def last_conv_layer(backbone: str) -> str | None:
    return _CANDIDATES[backbone][1]


def build_backbone(backbone: str, pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
    """Return a feature-extractor backbone (no classification head)."""
    if backbone not in _CANDIDATES:
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose from {sorted(_CANDIDATES)}")
    builder, _ = _CANDIDATES[backbone]
    weights = "DEFAULT" if pretrained else None
    model = builder(weights=weights)

    if backbone.startswith("efficientnet"):
        in_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
    elif backbone.startswith("convnext"):
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Identity()
    elif backbone.startswith("vit"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Identity()
    elif backbone.startswith("swin"):
        in_features = model.head.in_features
        model.head = nn.Identity()
    else:  # pragma: no cover
        raise ValueError(backbone)

    if not pretrained:
        log.warning("Backbone %s initialized randomly (no pretrained weights)", backbone)
    return model


class FeatureBackbone(nn.Module):
    """Thin wrapper: backbone + optional dropout after features (for MC-dropout)."""

    def __init__(self, backbone: str, pretrained: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.net = build_backbone(backbone, pretrained=pretrained)
        self.feat_dim = feature_dim(backbone)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.last_conv = last_conv_layer(backbone)

    def forward(self, x: torch.Tensor, features_only: bool = False) -> torch.Tensor:
        """x: (B,3,H,W) → (B, feat_dim)."""
        f = self.net(x)
        if features_only:
            return f
        return self.dropout(f)

    # ------------------------------------------------------------------ #
    # §3.2 two-stage fine-tuning: freeze/unfreeze schedule primitives
    # ------------------------------------------------------------------ #

    def _blocks(self) -> list[nn.Module]:
        """Progressive blocks, input-side → output-side. The LAST block is the
        first one unfrozen in stage 2."""
        if hasattr(self.net, "features"):
            return list(self.net.features)
        return [self.net]  # opaque backbones: all-or-nothing granularity

    def freeze_all(self) -> None:
        """Stage 1: backbone fully frozen — only fusion + head receive grads."""
        for p in self.net.parameters():
            p.requires_grad = False

    def unfreeze_last_block(self) -> None:
        """Stage 2a: unfreeze only the final block (gradual unfreeze)."""
        blocks = self._blocks()
        if len(blocks) <= 1:
            return  # opaque: leave all-frozen until full unfreeze
        for p in blocks[-1].parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        """Stage 2b: full unfreeze."""
        for p in self.net.parameters():
            p.requires_grad = True

    def trainable_param_groups(
        self, base_lr: float, unfreeze_lr_factor: float
    ) -> list[dict]:
        """Optimizer param groups honoring the two-stage LRs: newly unfrozen
        last block trains at ``base_lr * 0.25``, everything else trainable at
        ``base_lr * unfreeze_lr_factor`` (spec §3.2)."""
        blocks = self._blocks()
        last_block_params = (
            {id(p) for p in blocks[-1].parameters()} if len(blocks) > 1 else set()
        )
        groups: dict[str, list] = {"last": [], "rest": []}
        for p in self.net.parameters():
            if not p.requires_grad:
                continue
            groups["last" if id(p) in last_block_params else "rest"].append(p)
        out = []
        if groups["last"]:
            out.append({"params": groups["last"], "lr": base_lr * 0.25})
        if groups["rest"]:
            out.append({"params": groups["rest"], "lr": base_lr * unfreeze_lr_factor})
        return out