"""Multimodal classifier: image features + contextual metadata."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.metadata import MetadataEncoder
from .backbones import FeatureBackbone


class MultimodalClassifier(nn.Module):
    """Fuses image features with crop/stage/region/weather context."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
        fusion: str = "gated",
        context_dim: int = 128,
    ) -> None:
        super().__init__()
        if fusion not in {"gated", "concat", "image_only"}:
            raise ValueError(f"Unsupported fusion: {fusion}")

        self.backbone = FeatureBackbone(backbone, pretrained=pretrained, dropout=dropout)
        self.metadata = MetadataEncoder(context_dim=context_dim, dropout=dropout)
        self.fusion = fusion
        self.context_dim = context_dim

        if fusion == "gated":
            # Gate spans the fused feature dim; a projection maps context into
            # image-feature space so the two can be interpolated.
            self.context_proj = nn.Linear(context_dim, self.backbone.feat_dim)
            self.gate = nn.Sequential(
                nn.Linear(self.backbone.feat_dim + context_dim, self.backbone.feat_dim),
                nn.Sigmoid(),
            )
            self.head = nn.Linear(self.backbone.feat_dim, num_classes)
        elif fusion == "concat":
            # image_only still has a MetadataEncoder attribute (unused) so the
            # attribute surface is identical across variants; head takes image
            # features only. concat head takes the concatenated vector.
            self.head = nn.Linear(self.backbone.feat_dim + context_dim, num_classes)
        else:  # image_only
            self.head = nn.Linear(self.backbone.feat_dim, num_classes)

    def forward(
        self,
        image: torch.Tensor,
        crop_id: torch.Tensor | None = None,
        stage_idx: torch.Tensor | None = None,
        region_idx: torch.Tensor | None = None,
        weather: torch.Tensor | None = None,
        features_only: bool = False,
        **_: object,
    ) -> dict[str, torch.Tensor | None]:
        image_features = self.backbone(image)
        gate = None
        context = None

        if crop_id is None or stage_idx is None or region_idx is None or weather is None:
            # No context supplied → image-only path. For the concat variant the
            # head expects the concatenated width, so pad with zeros for the
            # missing context rather than crashing.
            fused = image_features
            if self.fusion == "concat":
                fused = torch.cat(
                    [fused, torch.zeros(image.shape[0], self.context_dim, device=image.device)],
                    dim=-1,
                )
        else:
            context = self.metadata(crop_id, stage_idx, region_idx, weather)
            if self.fusion == "gated":
                gate = self.gate(torch.cat([image_features, context], dim=-1))
                # gate ∈ (0,1)^feat_dim interpolates image evidence vs context.
                context_projected = self.context_proj(context)
                fused = gate * image_features + (1.0 - gate) * context_projected
            elif self.fusion == "concat":
                fused = torch.cat([image_features, context], dim=-1)
            else:
                fused = image_features

        if features_only:
            return fused

        logits = self.head(fused)
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=-1),
            "features": image_features,
            "gate": gate,
            "context_vec": context,
        }

    # §3.2 two-stage fine-tuning surface (delegates to FeatureBackbone).
    def freeze_backbone(self) -> None:
        self.backbone.freeze_all()

    def unfreeze_last_block(self) -> None:
        self.backbone.unfreeze_last_block()

    def unfreeze_backbone(self) -> None:
        self.backbone.unfreeze_all()

    def backbone_param_groups(self, base_lr: float, unfreeze_lr_factor: float) -> list[dict]:
        return self.backbone.trainable_param_groups(base_lr, unfreeze_lr_factor)
