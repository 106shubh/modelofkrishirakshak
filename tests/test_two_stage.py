"""§3.2 two-stage fine-tuning: freeze/unfreeze schedule primitives + train() integration.

The spec's acceptance: after the frozen stage only head/fusion params receive
grads; stage 2 unfreezes the last block first at lr*0.25, then everything at
lr*unfreeze_lr_factor — via a config-driven schedule, no code-path branching.
"""
from __future__ import annotations

import pytest
import torch

from cropguard.config import Config
from cropguard.models.baseline import BaselineClassifier
from cropguard.models.multimodal import MultimodalClassifier
from cropguard.training.train import train


def _trainable(model) -> set[str]:
    return {n for n, p in model.named_parameters() if p.requires_grad}


# --------------------------------------------------------------------------- #
# Model primitives
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_cls", [BaselineClassifier, MultimodalClassifier])
def test_freeze_then_unfreeze_schedule(model_cls):
    model = model_cls(num_classes=4, backbone="efficientnet_b0", pretrained=False)
    model.eval()

    model.freeze_backbone()
    assert _trainable(model) == {"head.weight", "head.bias"} or all(
        n.startswith(("head", "metadata", "context_proj", "gate")) for n in _trainable(model)
    ), f"non-head params trainable: {_trainable(model)}"

    model.unfreeze_last_block()
    last = list(model.backbone.net.features)[-1]
    assert all(p.requires_grad for p in last.parameters()), "last block must be trainable"
    first = list(model.backbone.net.features)[0]
    assert not any(p.requires_grad for p in first.parameters()), "stem must stay frozen"

    model.unfreeze_backbone()
    assert len(_trainable(model)) == len(list(model.named_parameters()))


def test_param_group_lrs():
    model = MultimodalClassifier(num_classes=4, backbone="efficientnet_b0", pretrained=False)

    # "last" stage: only the final block is trainable → one group @ lr*0.25.
    model.freeze_backbone()
    model.unfreeze_last_block()
    groups = model.backbone_param_groups(base_lr=1e-3, unfreeze_lr_factor=0.1)
    assert {round(g["lr"], 8) for g in groups} == {1e-3 * 0.25}, groups
    assert all(len(g["params"]) > 0 for g in groups)

    # "full" stage: everything trainable → last block @ lr*0.25, rest @ lr*factor.
    model.unfreeze_backbone()
    groups = model.backbone_param_groups(base_lr=1e-3, unfreeze_lr_factor=0.1)
    lrs = {round(g["lr"], 8): len(g["params"]) for g in groups}
    assert 1e-3 * 0.25 in lrs and 1e-3 * 0.1 in lrs, lrs
    assert all(len(g["params"]) > 0 for g in groups)


def test_frozen_backbone_blocks_grads():
    model = MultimodalClassifier(num_classes=4, backbone="efficientnet_b0", pretrained=False)
    model.freeze_backbone()
    model.train()
    x = torch.randn(2, 3, 32, 32)
    out = model(x, crop_id=torch.tensor([1, 1]), stage_idx=torch.tensor([0, 0]),
                region_idx=torch.tensor([0, 0]), weather=torch.zeros(2, 4))
    out["logits"].sum().backward()
    stem = list(model.backbone.net.features)[0]
    conv_w = stem[0].weight if isinstance(stem, torch.nn.Sequential) else None
    assert conv_w is None or conv_w.grad is None or conv_w.grad.abs().sum() == 0


# --------------------------------------------------------------------------- #
# train() integration: schedule runs end-to-end on the tiny fixture
# --------------------------------------------------------------------------- #


def test_train_runs_two_stage_schedule(tiny_cfg: Config, tmp_path):
    tiny_cfg.training.epochs = 3
    tiny_cfg.training.freeze_epochs = 1
    tiny_cfg.training.unfreeze_lr_factor = 0.1
    out = tmp_path / "two_stage_ckpt"
    train(tiny_cfg, out)  # must not raise; stages transition internally
    assert (out / "best_model.pth").exists()


def test_zero_freeze_epochs_unchanged(tiny_cfg: Config, tmp_path):
    """freeze_epochs=0 keeps the legacy single-stage path (backward compat)."""
    tiny_cfg.training.freeze_epochs = 0
    out = tmp_path / "legacy_ckpt"
    train(tiny_cfg, out)
    assert (out / "best_model.pth").exists()
