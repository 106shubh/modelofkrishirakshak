"""Model definitions."""
from cropguard.models.backbones import FeatureBackbone, build_backbone, feature_dim
from cropguard.models.baseline import BaselineClassifier
from cropguard.models.multimodal import MultimodalClassifier

__all__ = [
    "BaselineClassifier",
    "FeatureBackbone",
    "MultimodalClassifier",
    "build_backbone",
    "feature_dim",
]
